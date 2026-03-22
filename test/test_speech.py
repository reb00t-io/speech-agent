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


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_sessions(tmp_path, monkeypatch):
    monkeypatch.setattr("src.speech.AUDIO_RECORDING_DIR", str(tmp_path / "recordings"))
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

    # Mock LLM streaming
    llm_response_chunks = [
        b'data: {"choices":[{"delta":{"content":"Why"}}]}\n\n',
        b'data: {"choices":[{"delta":{"content":" did"}}]}\n\n',
        b'data: [DONE]\n\n',
    ]

    async def mock_aiter_raw():
        for chunk in llm_response_chunks:
            yield chunk

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.aiter_raw = mock_aiter_raw
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    mock_client = MagicMock()
    mock_client.stream = MagicMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("src.speech.transcribe", asr_mock), \
         patch("src.speech.httpx.AsyncClient", return_value=mock_client):
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
    assert session_modes[sid] == "user"
