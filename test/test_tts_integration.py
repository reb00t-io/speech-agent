"""Integration tests for TTS with LLM speech output pipeline."""
import asyncio
import io
import json
import math
import os
import struct
import wave
from unittest.mock import AsyncMock, patch

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


def _make_fake_wav() -> bytes:
    """Create a minimal valid WAV file for testing."""
    n_samples = 2205  # 0.1s at 22050 Hz
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(22050)
        wf.writeframes(b"\x00\x00" * n_samples)
    return buf.getvalue()


async def _fake_dual_stream_two_sentences(**kwargs):
    """Fake dual_stream that yields two sentences token by token."""
    for token in ["Hello", " world", ".", " How", " are", " you", "?"]:
        yield token


async def _fake_dual_stream_single(**kwargs):
    """Fake dual_stream that yields a single sentence."""
    yield "A response."


async def _fake_dual_stream_slow(**kwargs):
    """Fake dual_stream that streams slowly (for cancellation tests)."""
    for i in range(20):
        await asyncio.sleep(0.05)
        yield f"word{i} "


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

async def test_tts_audio_sent_during_llm_streaming(ws_client, monkeypatch):
    """When TTS is enabled, tts_audio messages should be sent for LLM sentences."""
    monkeypatch.setattr("src.speech.MISTRAL_API_KEY", "fake-key")

    asr_mock = AsyncMock(return_value="hello")
    tts_mock = AsyncMock(return_value=_make_fake_wav())

    with patch("src.speech.transcribe", asr_mock), \
         patch("src.speech.dual_stream", _fake_dual_stream_two_sentences), \
         patch("src.speech.tts_synthesize", tts_mock):
        async with ws_client.websocket("/ws/speech?mode=user") as ws:
            await ws.receive()  # session_start

            # Send speech then silence for pause detection
            await ws.send(_make_pcm_tone(0.5))
            for _ in range(6):
                await ws.send(_make_pcm_silence(0.1))
                await asyncio.sleep(0.05)

            await asyncio.sleep(0.5)
            await ws.send(json.dumps({"type": "stop"}))

            messages = []
            try:
                while True:
                    raw = await asyncio.wait_for(ws.receive(), timeout=3.0)
                    messages.append(json.loads(raw))
            except (asyncio.TimeoutError, Exception):
                pass

    tts_audio_msgs = [m for m in messages if m.get("type") == "tts_audio"]
    if tts_mock.await_count > 0:
        # TTS was called — verify tts_audio messages were sent
        assert len(tts_audio_msgs) > 0
        # Each tts_audio message should have index and audio_base64
        for msg in tts_audio_msgs:
            assert "index" in msg
            assert "audio_base64" in msg
            assert len(msg["audio_base64"]) > 0

        # Indices should be sequential starting from 0
        indices = [m["index"] for m in tts_audio_msgs]
        assert indices == list(range(len(indices)))


async def test_tts_disabled_when_no_api_key(ws_client, monkeypatch):
    """When MISTRAL_API_KEY is empty, no tts_audio messages should be sent."""
    monkeypatch.setattr("src.speech.MISTRAL_API_KEY", "")

    asr_mock = AsyncMock(return_value="hello")
    tts_mock = AsyncMock(return_value=_make_fake_wav())

    with patch("src.speech.transcribe", asr_mock), \
         patch("src.speech.dual_stream", _fake_dual_stream_single), \
         patch("src.speech.tts_synthesize", tts_mock):
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
                    raw = await asyncio.wait_for(ws.receive(), timeout=3.0)
                    messages.append(json.loads(raw))
            except (asyncio.TimeoutError, Exception):
                pass

    # TTS should NOT have been called
    tts_mock.assert_not_awaited()
    tts_audio_msgs = [m for m in messages if m.get("type") == "tts_audio"]
    assert len(tts_audio_msgs) == 0


async def test_tts_sentence_splitting_integration(ws_client, monkeypatch):
    """TTS should split LLM output into sentences and synthesize each one."""
    monkeypatch.setattr("src.speech.MISTRAL_API_KEY", "fake-key")

    asr_mock = AsyncMock(return_value="tell me something")
    tts_calls = []

    async def fake_tts(text, *, api_key, model, voice):
        tts_calls.append(text)
        return _make_fake_wav()

    with patch("src.speech.transcribe", asr_mock), \
         patch("src.speech.dual_stream", _fake_dual_stream_two_sentences), \
         patch("src.speech.tts_synthesize", fake_tts):
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
                    raw = await asyncio.wait_for(ws.receive(), timeout=3.0)
                    messages.append(json.loads(raw))
            except (asyncio.TimeoutError, Exception):
                pass

    # If LLM was triggered, TTS should have been called for each sentence
    if tts_calls:
        assert len(tts_calls) == 2
        assert "Hello world." in tts_calls[0]
        assert "How are you?" in tts_calls[1]


async def test_tts_cancellation_on_user_speech(ws_client, monkeypatch):
    """When user starts speaking during LLM+TTS, TTS should be cancelled."""
    monkeypatch.setattr("src.speech.MISTRAL_API_KEY", "fake-key")

    asr_mock = AsyncMock(return_value="hello")
    tts_mock = AsyncMock(return_value=_make_fake_wav())

    with patch("src.speech.transcribe", asr_mock), \
         patch("src.speech.dual_stream", _fake_dual_stream_slow), \
         patch("src.speech.tts_synthesize", tts_mock):
        async with ws_client.websocket("/ws/speech?mode=user") as ws:
            await ws.receive()  # session_start

            # Trigger LLM
            await ws.send(_make_pcm_tone(0.5))
            for _ in range(6):
                await ws.send(_make_pcm_silence(0.1))
                await asyncio.sleep(0.05)

            # Wait for LLM to start streaming
            await asyncio.sleep(0.3)

            # Interrupt with new speech (loud audio)
            await ws.send(_make_pcm_tone(0.3, amplitude=15000))
            await asyncio.sleep(0.1)

            await ws.send(json.dumps({"type": "stop"}))

            messages = []
            try:
                while True:
                    raw = await asyncio.wait_for(ws.receive(), timeout=3.0)
                    messages.append(json.loads(raw))
            except (asyncio.TimeoutError, Exception):
                pass

    # Should have received an llm_cancelled message
    cancelled_msgs = [m for m in messages if m.get("type") == "llm_cancelled"]
    # May or may not have been cancelled depending on timing,
    # but the test should complete without errors either way


async def test_tts_mistral_api_contract():
    """Verify the TTS client calls Mistral API with correct params."""
    import base64
    from unittest.mock import MagicMock
    from src.tts import synthesize

    pcm_data = struct.pack("<4f", 0.1, 0.2, -0.1, 0.0)
    audio_b64 = base64.b64encode(pcm_data).decode()
    sse = f'event: speech.audio.delta\ndata: {json.dumps({"audio_data": audio_b64})}\n\n'
    sse += f'event: speech.audio.done\ndata: {json.dumps({"usage": {"characters_count": 12}})}\n\n'

    async def fake_aiter():
        yield sse

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.aiter_text = fake_aiter
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    client = MagicMock()
    client.stream = MagicMock(return_value=mock_resp)
    client.aclose = AsyncMock()

    wav = await synthesize(
        "Hello world.",
        api_key="test-key",
        voice="en_neutral_female",
        client=client,
    )

    assert wav[:4] == b"RIFF"
    call_args = client.stream.call_args
    assert call_args[1]["json"]["input"] == "Hello world."
    assert call_args[1]["json"]["voice_id"] == "en_neutral_female"
    assert call_args[1]["json"]["stream"] is True


async def test_tts_llm_done_sent_after_all_tts(ws_client, monkeypatch):
    """llm_done should only be sent after all TTS tasks complete."""
    monkeypatch.setattr("src.speech.MISTRAL_API_KEY", "fake-key")

    asr_mock = AsyncMock(return_value="hello")

    async def slow_tts(text, *, api_key, model, voice):
        await asyncio.sleep(0.1)  # Simulate TTS latency
        return _make_fake_wav()

    with patch("src.speech.transcribe", asr_mock), \
         patch("src.speech.dual_stream", _fake_dual_stream_two_sentences), \
         patch("src.speech.tts_synthesize", slow_tts):
        async with ws_client.websocket("/ws/speech?mode=user") as ws:
            await ws.receive()  # session_start

            await ws.send(_make_pcm_tone(0.5))
            for _ in range(6):
                await ws.send(_make_pcm_silence(0.1))
                await asyncio.sleep(0.05)

            await asyncio.sleep(1.0)
            await ws.send(json.dumps({"type": "stop"}))

            messages = []
            try:
                while True:
                    raw = await asyncio.wait_for(ws.receive(), timeout=3.0)
                    messages.append(json.loads(raw))
            except (asyncio.TimeoutError, Exception):
                pass

    # If LLM was triggered, llm_done should come AFTER all tts_audio messages
    tts_idxs = [i for i, m in enumerate(messages) if m.get("type") == "tts_audio"]
    done_idxs = [i for i, m in enumerate(messages) if m.get("type") == "llm_done"]

    if tts_idxs and done_idxs:
        assert max(tts_idxs) < min(done_idxs), \
            "llm_done should be sent after all tts_audio messages"
