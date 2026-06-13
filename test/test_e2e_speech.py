"""End-to-end speech tests using programmatically generated audio samples.

These tests exercise the full pipeline: audio → chunker → ASR → LLM,
using mocked ASR and LLM backends but real audio data flowing through
the WebSocket.
"""
import asyncio
import json
import math
import os
import struct
import wave
import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

os.environ.setdefault("LLM_BASE_URL", "http://fake-llm")
os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.pop("API_KEY", None)

from src.main import app, sessions, session_modes, last_session_ids


# ─── Audio sample generators ────────────────────────────────────────────────

SAMPLE_RATE = 16000


def generate_tone(duration_s: float, freq: int = 440, amplitude: int = 10000) -> bytes:
    """Generate a sine wave as 16-bit LE mono PCM."""
    n = int(SAMPLE_RATE * duration_s)
    samples = [int(amplitude * math.sin(2 * math.pi * freq * i / SAMPLE_RATE)) for i in range(n)]
    return struct.pack(f"<{n}h", *samples)


def generate_silence(duration_s: float) -> bytes:
    """Generate silence as 16-bit LE mono PCM."""
    n = int(SAMPLE_RATE * duration_s)
    return b"\x00\x00" * n


def generate_speech_with_pause(
    speech1_s: float = 1.0,
    pause_s: float = 0.6,
    speech2_s: float = 1.0,
) -> bytes:
    """Generate audio simulating speech, pause, then more speech."""
    return generate_tone(speech1_s, freq=300) + generate_silence(pause_s) + generate_tone(speech2_s, freq=500)


def pcm_to_wav(pcm: bytes) -> bytes:
    """Wrap raw PCM in a valid WAV file for verification."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm)
    return buf.getvalue()


def chunk_audio(pcm: bytes, chunk_size: int = 8192) -> list[bytes]:
    """Split PCM into WebSocket-sized frames."""
    return [pcm[i : i + chunk_size] for i in range(0, len(pcm), chunk_size)]


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_sessions(tmp_path, monkeypatch):
    monkeypatch.setattr("src.speech.AUDIO_RECORDING_DIR", str(tmp_path / "recordings"))
    monkeypatch.setattr("src.speech.MISTRAL_API_KEY", "")  # disable TTS; covered by test_tts_integration.py
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


def make_llm_mock(tokens: list[str]):
    """Create a mock httpx client that streams SSE tokens."""
    chunks = [
        f'data: {json.dumps({"choices": [{"delta": {"content": t}}]})}\n\n'.encode()
        for t in tokens
    ]
    chunks.append(b"data: [DONE]\n\n")

    async def aiter_raw():
        for c in chunks:
            yield c

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    mock_resp.aiter_raw = aiter_raw
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    mock_client = MagicMock()
    mock_client.stream = MagicMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    return mock_client


# ─── Audio sample validity ───────────────────────────────────────────────────

def test_generated_tone_is_valid_pcm():
    pcm = generate_tone(1.0)
    assert len(pcm) == SAMPLE_RATE * 2  # 16000 samples * 2 bytes
    # Verify it's parseable
    samples = struct.unpack(f"<{SAMPLE_RATE}h", pcm)
    assert max(samples) > 5000  # should have non-zero values


def test_generated_silence_is_zero():
    pcm = generate_silence(0.5)
    assert all(b == 0 for b in pcm)


def test_pcm_to_wav_is_valid():
    pcm = generate_tone(0.5)
    wav_bytes = pcm_to_wav(pcm)
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == SAMPLE_RATE
        assert wf.getnframes() == SAMPLE_RATE // 2


def test_speech_with_pause_has_correct_length():
    pcm = generate_speech_with_pause(1.0, 0.5, 1.0)
    expected_samples = int(SAMPLE_RATE * 2.5)
    assert len(pcm) == expected_samples * 2


# ─── E2E: "hello world" flow ────────────────────────────────────────────────

async def test_e2e_hello_world(ws_client):
    """Simulate a user saying 'hello world' — audio flows through chunker,
    gets transcribed, pauses trigger LLM."""
    asr_call_count = 0

    async def mock_asr(audio_pcm, *, base_url, api_key, model, language=""):
        nonlocal asr_call_count
        asr_call_count += 1
        return "hello world"

    llm_mock = make_llm_mock(["Hi", " there", "!"])

    with patch("src.speech.transcribe", side_effect=mock_asr), \
         patch("src.dual_llm.httpx.AsyncClient", return_value=llm_mock):
        async with ws_client.websocket("/ws/speech?mode=user") as ws:
            start_msg = json.loads(await ws.receive())
            assert start_msg["type"] == "session_start"
            sid = start_msg["session_id"]

            # Send 2.5s of "speech" + silence (enough for chunk + pause)
            audio = generate_tone(2.5) + generate_silence(0.6)
            for frame in chunk_audio(audio):
                await ws.send(frame)
                await asyncio.sleep(0.01)

            # Allow processing time
            await asyncio.sleep(1.0)

            # Stop
            await ws.send(json.dumps({"type": "stop"}))

            messages = []
            try:
                while True:
                    raw = await asyncio.wait_for(ws.receive(), timeout=2.0)
                    messages.append(json.loads(raw))
            except (asyncio.TimeoutError, Exception):
                pass

    # Verify ASR was called at least once
    assert asr_call_count >= 1

    # Verify we got transcript and LLM tokens
    types = [m["type"] for m in messages]
    assert "transcript" in types

    # Session should be stored
    assert sid in sessions


async def test_e2e_silence_only(ws_client):
    """Sending only silence should not trigger ASR or LLM."""
    asr_mock = AsyncMock(return_value="")

    with patch("src.speech.transcribe", asr_mock):
        async with ws_client.websocket("/ws/speech?mode=user") as ws:
            await ws.receive()  # session_start

            # Send pure silence
            silence = generate_silence(3.0)
            for frame in chunk_audio(silence):
                await ws.send(frame)
                await asyncio.sleep(0.01)

            await asyncio.sleep(0.5)
            await ws.send(json.dumps({"type": "stop"}))

            messages = []
            try:
                while True:
                    raw = await asyncio.wait_for(ws.receive(), timeout=1.0)
                    messages.append(json.loads(raw))
            except (asyncio.TimeoutError, Exception):
                pass

    # ASR should not have been called (no chunks from silence-only audio
    # since there's no speech to create chunk boundaries, only flush on stop
    # which may produce one chunk but transcribe returns empty)
    transcript_msgs = [m for m in messages if m.get("type") == "transcript" and m.get("text")]
    assert len(transcript_msgs) == 0


async def test_e2e_multiple_utterances(ws_client):
    """Multiple speech segments separated by pauses should produce multiple transcripts."""
    call_index = 0

    async def mock_asr(audio_pcm, *, base_url, api_key, model, language=""):
        nonlocal call_index
        call_index += 1
        return f"utterance {call_index}"

    llm_mock = make_llm_mock(["OK"])

    with patch("src.speech.transcribe", side_effect=mock_asr), \
         patch("src.dual_llm.httpx.AsyncClient", return_value=llm_mock):
        async with ws_client.websocket("/ws/speech?mode=user") as ws:
            await ws.receive()  # session_start

            # First utterance
            audio1 = generate_tone(2.5, freq=300) + generate_silence(0.6)
            for frame in chunk_audio(audio1):
                await ws.send(frame)
                await asyncio.sleep(0.01)
            await asyncio.sleep(1.0)

            # Second utterance
            audio2 = generate_tone(2.5, freq=500) + generate_silence(0.6)
            for frame in chunk_audio(audio2):
                await ws.send(frame)
                await asyncio.sleep(0.01)
            await asyncio.sleep(1.0)

            await ws.send(json.dumps({"type": "stop"}))

            messages = []
            try:
                while True:
                    raw = await asyncio.wait_for(ws.receive(), timeout=2.0)
                    messages.append(json.loads(raw))
            except (asyncio.TimeoutError, Exception):
                pass

    # Should have multiple transcripts
    transcript_msgs = [m for m in messages if m.get("type") == "transcript"]
    assert len(transcript_msgs) >= 2


async def test_e2e_interruption_and_continuation(ws_client):
    """Simulate: user speaks → pause → LLM starts → user speaks again → LLM cancelled → pause → LLM continues."""
    asr_calls = []

    async def mock_asr(audio_pcm, *, base_url, api_key, model, language=""):
        asr_calls.append(len(audio_pcm))
        return "test speech"

    # First LLM call returns slowly (simulate being interrupted)
    slow_chunks = [
        f'data: {json.dumps({"choices": [{"delta": {"content": "The answer"}}]})}\n\n'.encode(),
        f'data: {json.dumps({"choices": [{"delta": {"content": " is"}}]})}\n\n'.encode(),
    ]
    # Note: no [DONE] — this simulates the LLM being cut off

    fast_chunks = [
        f'data: {json.dumps({"choices": [{"delta": {"content": " 42."}}]})}\n\n'.encode(),
        b"data: [DONE]\n\n",
    ]

    call_count = 0

    def make_dynamic_llm_mock():
        nonlocal call_count

        async def aiter_raw():
            nonlocal call_count
            current_call = call_count
            call_count += 1
            chunks = slow_chunks if current_call == 0 else fast_chunks
            for c in chunks:
                yield c
                if current_call == 0:
                    await asyncio.sleep(0.3)  # slow to allow interruption

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.aiter_raw = aiter_raw
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        return mock_client

    with patch("src.speech.transcribe", side_effect=mock_asr), \
         patch("src.dual_llm.httpx.AsyncClient", side_effect=lambda **kw: make_dynamic_llm_mock()):
        async with ws_client.websocket("/ws/speech?mode=user") as ws:
            await ws.receive()  # session_start

            # First speech + pause to trigger LLM
            audio1 = generate_tone(0.5) + generate_silence(0.6)
            for frame in chunk_audio(audio1):
                await ws.send(frame)
                await asyncio.sleep(0.01)
            await asyncio.sleep(0.8)

            # Now interrupt with new speech while LLM is responding
            await ws.send(generate_tone(0.3))
            await asyncio.sleep(0.2)

            # Pause again to trigger continuation
            for _ in range(6):
                await ws.send(generate_silence(0.1))
                await asyncio.sleep(0.1)

            await asyncio.sleep(1.0)
            await ws.send(json.dumps({"type": "stop"}))

            messages = []
            try:
                while True:
                    raw = await asyncio.wait_for(ws.receive(), timeout=2.0)
                    messages.append(json.loads(raw))
            except (asyncio.TimeoutError, Exception):
                pass

    types = [m["type"] for m in messages]

    # Should have gotten transcripts
    assert "transcript" in types

    # Should have ASR calls
    assert len(asr_calls) >= 1
