"""Tests for src/speech.py — WebSocket speech handler integration tests."""
import asyncio
import json
import math
import os
import struct
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

os.environ.setdefault("LLM_BASE_URL", "http://fake-llm")
os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.pop("API_KEY", None)

from src.main import app, sessions, session_modes, last_session_ids


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _make_pcm_tone(duration_s: float, amplitude: int = 10000, freq: int = 440) -> bytes:
    n_samples = int(16000 * duration_s)
    samples = []
    for i in range(n_samples):
        value = int(amplitude * math.sin(2 * math.pi * freq * i / 16000))
        samples.append(value)
    return struct.pack(f"<{n_samples}h", *samples)


def _make_pcm_silence(duration_s: float) -> bytes:
    n_samples = int(16000 * duration_s)
    return b"\x00\x00" * n_samples


def _patch_stream_chat(*tokens):
    """Patch the memorizer engine seam to stream the given content tokens."""
    async def _stream(messages, *, reasoning_effort=None, tools=None, usage_out=None):
        for token in tokens:
            yield {"choices": [{"delta": {"content": token}}]}

    return patch("src.llm_engine.stream_chat", _stream)


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


# ─── Tests ───────────────────────────────────────────────────────────────────

async def test_websocket_session_start(ws_client):
    """Connecting to the WebSocket should receive a session_start message."""
    async with ws_client.websocket("/ws/speech?mode=user") as ws:
        # Should get session_start immediately
        msg = json.loads(await ws.receive())
        assert msg["type"] == "session_start"
        assert "session_id" in msg

        # Send stop to cleanly close
        await ws.send(json.dumps({"type": "stop"}))


async def test_websocket_uses_provided_session_id(ws_client):
    """If session_id is provided, it should be used."""
    # Pre-create a session
    sessions["existing-session"] = [{"role": "system", "content": "test"}]
    session_modes["existing-session"] = "user"

    async with ws_client.websocket("/ws/speech?mode=user&session_id=existing-session") as ws:
        msg = json.loads(await ws.receive())
        assert msg["type"] == "session_start"
        assert msg["session_id"] == "existing-session"
        await ws.send(json.dumps({"type": "stop"}))


async def test_websocket_audio_triggers_asr(ws_client):
    """Sending audio that produces a chunk should trigger ASR."""
    asr_mock = AsyncMock(return_value="hello world")

    with patch("src.speech.transcribe", asr_mock):
        async with ws_client.websocket("/ws/speech?mode=user") as ws:
            # Read session_start
            await ws.receive()

            # Send 2.5s of tone (enough for a chunk) + silence for cut point
            await ws.send(_make_pcm_tone(2.5))
            await ws.send(_make_pcm_silence(0.2))

            # Allow async processing
            await asyncio.sleep(0.1)

            # Send stop to trigger flush and finalize
            await ws.send(json.dumps({"type": "stop"}))

            # Collect messages
            messages = []
            try:
                while True:
                    raw = await asyncio.wait_for(ws.receive(), timeout=2.0)
                    messages.append(json.loads(raw))
            except (asyncio.TimeoutError, Exception):
                pass

    # ASR should have been called
    assert asr_mock.await_count >= 1

    # Should have received transcript messages
    transcript_msgs = [m for m in messages if m.get("type") == "transcript"]
    assert len(transcript_msgs) >= 1
    assert transcript_msgs[0]["text"] == "hello world"


async def test_websocket_pause_triggers_llm(ws_client):
    """After a speech pause, the LLM should be triggered."""
    asr_mock = AsyncMock(return_value="tell me a joke")

    with patch("src.speech.transcribe", asr_mock), _patch_stream_chat("Why", " did"):
        async with ws_client.websocket("/ws/speech?mode=user") as ws:
            await ws.receive()  # session_start

            # Send speech then enough silence for pause detection
            await ws.send(_make_pcm_tone(0.5))
            # Wait a moment, then send silence over 0.5s (> 0.4s pause threshold)
            for _ in range(6):
                await ws.send(_make_pcm_silence(0.1))
                await asyncio.sleep(0.05)

            # Allow processing
            await asyncio.sleep(0.5)

            await ws.send(json.dumps({"type": "stop"}))

            messages = []
            try:
                while True:
                    raw = await asyncio.wait_for(ws.receive(), timeout=2.0)
                    messages.append(json.loads(raw))
            except (asyncio.TimeoutError, Exception):
                pass

    # Check for LLM tokens
    token_msgs = [m for m in messages if m.get("type") == "llm_token"]
    if token_msgs:
        tokens = "".join(m["token"] for m in token_msgs)
        assert "Why" in tokens


async def test_dictation_mode_streams_transcript_without_llm(ws_client):
    """In dictation mode, transcripts stream but the LLM is never invoked."""
    asr_mock = AsyncMock(return_value="take a note")

    async def _boom(messages, *, reasoning_effort=None, tools=None, usage_out=None):
        raise AssertionError("LLM must not be called in dictation mode")
        yield  # pragma: no cover - makes this an async generator

    with patch("src.speech.transcribe", asr_mock), \
         patch("src.llm_engine.stream_chat", _boom):
        async with ws_client.websocket("/ws/speech?dictation=1") as ws:
            await ws.receive()  # session_start

            await ws.send(_make_pcm_tone(0.5))
            for _ in range(6):
                await ws.send(_make_pcm_silence(0.1))
                await asyncio.sleep(0.05)
            await asyncio.sleep(0.4)
            await ws.send(json.dumps({"type": "stop"}))

            messages = []
            try:
                while True:
                    raw = await asyncio.wait_for(ws.receive(), timeout=2.0)
                    messages.append(json.loads(raw))
            except (asyncio.TimeoutError, Exception):
                pass

    types = [m.get("type") for m in messages]
    assert "transcript" in types          # transcripts streamed
    assert "transcript_done" in types     # pause finalized
    assert "llm_token" not in types       # but the LLM never ran


async def test_websocket_stop_without_audio(ws_client):
    """Sending stop immediately should work without errors."""
    async with ws_client.websocket("/ws/speech?mode=user") as ws:
        msg = json.loads(await ws.receive())
        assert msg["type"] == "session_start"
        await ws.send(json.dumps({"type": "stop"}))


async def test_websocket_creates_session_in_store(ws_client):
    """WebSocket connection should create a session in the session store."""
    async with ws_client.websocket("/ws/speech?mode=user") as ws:
        msg = json.loads(await ws.receive())
        sid = msg["session_id"]
        await ws.send(json.dumps({"type": "stop"}))

    assert sid in sessions


# ─── WebSocket auth (key gate) ───────────────────────────────────────────────

async def test_ws_open_when_no_api_key(ws_client, monkeypatch):
    """Standalone (no API_KEY): the socket stays open — connect succeeds."""
    monkeypatch.setattr("src.speech.API_KEY", "")
    async with ws_client.websocket("/ws/speech?mode=user") as ws:
        msg = json.loads(await ws.receive())
        assert msg["type"] == "session_start"


async def test_ws_rejects_missing_key_when_configured(ws_client, monkeypatch):
    monkeypatch.setattr("src.speech.API_KEY", "secret")
    async with ws_client.websocket("/ws/speech?mode=user") as ws:
        msg = json.loads(await ws.receive())
        assert msg["type"] == "error"  # unauthorized, no session_start


async def test_ws_rejects_wrong_key(ws_client, monkeypatch):
    monkeypatch.setattr("src.speech.API_KEY", "secret")
    async with ws_client.websocket("/ws/speech?mode=user&key=nope") as ws:
        msg = json.loads(await ws.receive())
        assert msg["type"] == "error"


async def test_ws_accepts_correct_key(ws_client, monkeypatch):
    monkeypatch.setattr("src.speech.API_KEY", "secret")
    async with ws_client.websocket("/ws/speech?mode=user&key=secret") as ws:
        msg = json.loads(await ws.receive())
        assert msg["type"] == "session_start"


# ─── Dual-LLM toggle ─────────────────────────────────────────────────────────

def test_voice_llm_engine_respects_dual_toggle(monkeypatch):
    """_voice_llm_stream picks dual_stream or single_stream based on the toggle."""
    import src.speech as speech

    chosen = []
    monkeypatch.setattr(speech, "dual_stream", lambda **kw: chosen.append("dual") or iter(()))
    monkeypatch.setattr(speech, "single_stream", lambda **kw: chosen.append("single") or iter(()))

    monkeypatch.setattr(speech, "DUAL_LLM_ENABLED", True)
    speech._voice_llm_stream([{"role": "user", "content": "hi"}])

    monkeypatch.setattr(speech, "DUAL_LLM_ENABLED", False)
    speech._voice_llm_stream([{"role": "user", "content": "hi"}])

    assert chosen == ["dual", "single"]


async def test_pause_triggers_llm_with_dual_disabled(ws_client, monkeypatch):
    """With dual-LLM disabled, voice mode still streams tokens via single_stream."""
    monkeypatch.setattr("src.speech.DUAL_LLM_ENABLED", False)
    asr_mock = AsyncMock(return_value="hi there")

    with patch("src.speech.transcribe", asr_mock), _patch_stream_chat("Hello", " back"):
        async with ws_client.websocket("/ws/speech?mode=user") as ws:
            await ws.receive()  # session_start
            await ws.send(_make_pcm_tone(0.5))
            for _ in range(6):
                await ws.send(_make_pcm_silence(0.1))
                await asyncio.sleep(0.05)
            await asyncio.sleep(0.5)
            await ws.send(json.dumps({"type": "stop"}))

            messages = []
            try:
                while True:
                    raw = await asyncio.wait_for(ws.receive(), timeout=2.0)
                    messages.append(json.loads(raw))
            except (asyncio.TimeoutError, Exception):
                pass

    token_msgs = [m for m in messages if m.get("type") == "llm_token"]
    if token_msgs:
        assert "Hello" in "".join(m["token"] for m in token_msgs)
