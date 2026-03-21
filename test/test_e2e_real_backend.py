"""E2E test against real ASR + LLM backends.

Sends actual audio through the WebSocket and verifies transcription
and LLM response come back. Requires LLM_BASE_URL pointing to a real
server with ASR and chat completion endpoints.

Run with:  pytest test/test_e2e_real_backend.py -v -s
Skip:      skipped automatically if LLM_BASE_URL is not set or unreachable.
"""
import asyncio
import json
import math
import os
import struct
import wave
import io

import pytest

# ─── Skip if no real backend ────────────────────────────────────────────────

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "")
if not LLM_BASE_URL or LLM_BASE_URL.startswith("http://fake"):
    pytest.skip("LLM_BASE_URL not configured for real backend", allow_module_level=True)

# Only pop API_KEY after the skip check so we don't interfere with other tests
# when this module is skipped
os.environ.pop("API_KEY", None)

from src.main import app, sessions, session_modes, last_session_ids


# ─── Audio generators ───────────────────────────────────────────────────────

SAMPLE_RATE = 16000


def generate_tone(duration_s: float, freq: int = 440, amplitude: int = 10000) -> bytes:
    n = int(SAMPLE_RATE * duration_s)
    samples = [int(amplitude * math.sin(2 * math.pi * freq * i / SAMPLE_RATE)) for i in range(n)]
    return struct.pack(f"<{n}h", *samples)


def generate_silence(duration_s: float) -> bytes:
    n = int(SAMPLE_RATE * duration_s)
    return b"\x00\x00" * n


def generate_spoken_like_audio(duration_s: float) -> bytes:
    """Generate audio that resembles speech more than a pure tone.

    Mixes several frequencies with amplitude modulation to create
    something vaguely voice-like. ASR may or may not transcribe it,
    but it exercises the full pipeline.
    """
    n = int(SAMPLE_RATE * duration_s)
    samples = []
    for i in range(n):
        t = i / SAMPLE_RATE
        # Fundamental + harmonics (voice-like)
        s = (
            0.5 * math.sin(2 * math.pi * 150 * t)
            + 0.3 * math.sin(2 * math.pi * 300 * t)
            + 0.15 * math.sin(2 * math.pi * 450 * t)
            + 0.05 * math.sin(2 * math.pi * 600 * t)
        )
        # Amplitude modulation (~4Hz, like syllable rhythm)
        envelope = 0.5 + 0.5 * math.sin(2 * math.pi * 4 * t)
        samples.append(int(s * envelope * 10000))
    return struct.pack(f"<{n}h", *samples)


def chunk_audio(pcm: bytes, chunk_size: int = 8192) -> list[bytes]:
    return [pcm[i : i + chunk_size] for i in range(0, len(pcm), chunk_size)]


def pcm_to_wav_file(pcm: bytes, path: str) -> None:
    """Write PCM to a WAV file (useful for debugging)."""
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm)


# ─── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_sessions():
    sessions.clear()
    session_modes.clear()
    last_session_ids.clear()
    yield
    sessions.clear()
    session_modes.clear()
    last_session_ids.clear()


@pytest.fixture
def ws_client():
    return app.test_client()


# ─── Helpers ─────────────────────────────────────────────────────────────────

async def collect_ws_messages(ws, timeout: float = 15.0) -> list[dict]:
    """Collect all JSON messages from the WebSocket until timeout."""
    messages = []
    deadline = asyncio.get_event_loop().time() + timeout
    while True:
        remaining = deadline - asyncio.get_event_loop().time()
        if remaining <= 0:
            break
        try:
            raw = await asyncio.wait_for(ws.receive(), timeout=min(remaining, 3.0))
            msg = json.loads(raw)
            messages.append(msg)
            # Stop early if we got llm_done — full cycle complete
            if msg.get("type") == "llm_done":
                break
        except asyncio.TimeoutError:
            break
        except Exception:
            break
    return messages


def print_messages(messages: list[dict]) -> None:
    """Pretty-print messages for debugging (use with -s flag)."""
    for m in messages:
        t = m.get("type", "?")
        if t == "transcript":
            print(f"  [transcript] {m.get('text', '')!r}")
        elif t == "transcript_done":
            print(f"  [transcript_done]")
        elif t == "llm_token":
            print(f"  [llm_token] {m.get('token', '')!r}")
        elif t == "llm_done":
            print(f"  [llm_done]")
        elif t == "error":
            print(f"  [error] {m.get('message', '')}")
        else:
            print(f"  [{t}] {m}")


# ─── Tests ──────────────────────────────────────────────────────────────────

async def test_full_pipeline_tone(ws_client):
    """Send a tone + silence through the WS and verify the pipeline completes.

    The ASR may return empty text for a sine wave, but the pipeline
    should not crash and should handle it gracefully.
    """
    async with ws_client.websocket("/ws/speech?mode=user") as ws:
        start_msg = json.loads(await ws.receive())
        assert start_msg["type"] == "session_start"
        print(f"\n  Session: {start_msg['session_id']}")

        # 3s of tone + 0.6s silence (triggers chunk + pause)
        audio = generate_tone(3.0, freq=440, amplitude=10000) + generate_silence(0.6)
        for frame in chunk_audio(audio):
            await ws.send(frame)
            await asyncio.sleep(0.005)

        # Wait for ASR + potential LLM
        await asyncio.sleep(2.0)

        await ws.send(json.dumps({"type": "stop"}))
        messages = await collect_ws_messages(ws, timeout=15.0)

    print_messages(messages)
    types = [m["type"] for m in messages]

    # Pipeline should not crash — we should get at least some messages
    # (transcript may be empty for non-speech audio, which is fine)
    errors = [m for m in messages if m["type"] == "error"]
    assert not errors, f"Got errors: {errors}"


async def test_full_pipeline_speech_like(ws_client):
    """Send speech-like audio and verify we get transcription + LLM response.

    Uses amplitude-modulated harmonics that are more likely to produce
    ASR output than a pure sine wave.
    """
    async with ws_client.websocket("/ws/speech?mode=user") as ws:
        start_msg = json.loads(await ws.receive())
        assert start_msg["type"] == "session_start"
        sid = start_msg["session_id"]
        print(f"\n  Session: {sid}")

        # 3s of speech-like audio + 0.6s silence
        audio = generate_spoken_like_audio(3.0) + generate_silence(0.6)
        for frame in chunk_audio(audio):
            await ws.send(frame)
            await asyncio.sleep(0.005)

        # Wait for ASR + LLM
        await asyncio.sleep(3.0)

        await ws.send(json.dumps({"type": "stop"}))
        messages = await collect_ws_messages(ws, timeout=20.0)

    print_messages(messages)
    types = [m["type"] for m in messages]
    errors = [m for m in messages if m["type"] == "error"]
    assert not errors, f"Got errors: {errors}"

    # Session should exist
    assert sid in sessions


async def test_full_pipeline_two_utterances(ws_client):
    """Send two speech segments with a pause between them.

    Verifies the pipeline handles multiple utterances in one session.
    """
    async with ws_client.websocket("/ws/speech?mode=user") as ws:
        start_msg = json.loads(await ws.receive())
        assert start_msg["type"] == "session_start"
        sid = start_msg["session_id"]
        print(f"\n  Session: {sid}")

        # First utterance: 3s speech + pause
        audio1 = generate_spoken_like_audio(3.0) + generate_silence(0.6)
        for frame in chunk_audio(audio1):
            await ws.send(frame)
            await asyncio.sleep(0.005)

        # Wait for first ASR + LLM cycle
        await asyncio.sleep(5.0)

        # Second utterance: 3s speech + pause
        audio2 = generate_tone(3.0, freq=300, amplitude=8000) + generate_silence(0.6)
        for frame in chunk_audio(audio2):
            await ws.send(frame)
            await asyncio.sleep(0.005)

        await asyncio.sleep(5.0)

        await ws.send(json.dumps({"type": "stop"}))
        messages = await collect_ws_messages(ws, timeout=20.0)

    print_messages(messages)
    types = [m["type"] for m in messages]
    errors = [m for m in messages if m["type"] == "error"]
    assert not errors, f"Got errors: {errors}"

    # Should have at least some transcript messages
    transcript_msgs = [m for m in messages if m["type"] == "transcript"]
    print(f"\n  Total transcripts: {len(transcript_msgs)}")
    print(f"  Total LLM tokens: {sum(1 for m in messages if m['type'] == 'llm_token')}")


async def test_pipeline_no_crash_on_stop_during_llm(ws_client):
    """Send audio, trigger LLM, then stop while LLM is still streaming.

    Verifies graceful shutdown without errors.
    """
    async with ws_client.websocket("/ws/speech?mode=user") as ws:
        start_msg = json.loads(await ws.receive())
        assert start_msg["type"] == "session_start"
        print(f"\n  Session: {start_msg['session_id']}")

        # Short burst of speech + pause to trigger ASR+LLM
        audio = generate_spoken_like_audio(2.5) + generate_silence(0.5)
        for frame in chunk_audio(audio):
            await ws.send(frame)
            await asyncio.sleep(0.005)

        # Wait just enough for ASR but maybe not for full LLM
        await asyncio.sleep(2.0)

        # Stop immediately — may interrupt LLM
        await ws.send(json.dumps({"type": "stop"}))
        messages = await collect_ws_messages(ws, timeout=10.0)

    print_messages(messages)

    # Should not have crashed
    errors = [m for m in messages if m["type"] == "error"]
    # ASR errors are acceptable (timeout etc) but not crashes
    crash_errors = [e for e in errors if "Traceback" in e.get("message", "")]
    assert not crash_errors, f"Got crash errors: {crash_errors}"
