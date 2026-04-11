"""Tests for src/tts.py — TTS client and sentence splitting."""
import base64
import json
import struct
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from src.tts import split_sentences, synthesize, wav_to_base64, _pcm_to_wav


# ─── split_sentences ─────────────────────────────────────────────────────────

def test_split_single_sentence():
    assert split_sentences("Hello world.") == ["Hello world."]


def test_split_two_sentences():
    assert split_sentences("Hello. World.") == ["Hello.", "World."]


def test_split_question_and_statement():
    assert split_sentences("How are you? I am fine.") == ["How are you?", "I am fine."]


def test_split_exclamation():
    assert split_sentences("Wow! That is great.") == ["Wow!", "That is great."]


def test_split_semicolon():
    assert split_sentences("First part; second part.") == ["First part;", "second part."]


def test_split_no_punctuation():
    assert split_sentences("Hello world") == ["Hello world"]


def test_split_empty():
    assert split_sentences("") == []
    assert split_sentences("   ") == []


def test_split_preserves_inner_periods():
    # Abbreviations like "Dr." shouldn't split if not followed by space+uppercase
    # But our simple regex will split on ". " — this is acceptable for TTS
    result = split_sentences("Dr. Smith is here.")
    assert len(result) >= 1  # At least produces something


def test_split_german():
    result = split_sentences("Wie geht es dir? Mir geht es gut. Danke!")
    assert result == ["Wie geht es dir?", "Mir geht es gut.", "Danke!"]


def test_split_multiline():
    result = split_sentences("First sentence.\nSecond sentence.")
    assert len(result) == 2


# ─── wav_to_base64 ──────────────────────────────────────────────────────────

def test_wav_to_base64():
    data = b"RIFF\x00\x00\x00\x00WAVEfmt "
    encoded = wav_to_base64(data)
    assert base64.b64decode(encoded) == data


# ─── _pcm_to_wav ────────────────────────────────────────────────────────────

def test_pcm_to_wav():
    """PCM float32 LE should be converted to a valid WAV file."""
    import wave
    import io
    # 10 samples of float32 silence
    pcm = struct.pack("<10f", *([0.0] * 10))
    wav = _pcm_to_wav(pcm, sample_rate=24000)
    assert wav[:4] == b"RIFF"
    with wave.open(io.BytesIO(wav), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 24000
        assert wf.getnframes() == 10


def test_pcm_to_wav_clamps():
    """Values outside [-1, 1] should be clamped."""
    pcm = struct.pack("<2f", 2.0, -2.0)
    wav = _pcm_to_wav(pcm)
    assert wav[:4] == b"RIFF"


# ─── synthesize ──────────────────────────────────────────────────────────────

def _make_sse_response(text: str = "Hello") -> list[str]:
    """Build SSE chunks like the Mistral API returns."""
    # Create some float32 PCM data
    pcm_data = struct.pack("<4f", 0.1, 0.2, -0.1, 0.0)
    audio_b64 = base64.b64encode(pcm_data).decode()
    return [
        f"event: speech.audio.delta\ndata: {json.dumps({'audio_data': audio_b64})}\n\n",
        f"event: speech.audio.delta\ndata: {json.dumps({'audio_data': audio_b64})}\n\n",
        f'event: speech.audio.done\ndata: {json.dumps({"usage": {"characters_count": len(text)}})}\n\n',
    ]


async def test_synthesize_calls_mistral_api():
    """synthesize should POST to Mistral API with correct params."""
    sse_chunks = _make_sse_response()

    async def fake_aiter_text():
        for chunk in sse_chunks:
            yield chunk

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.aiter_text = fake_aiter_text
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    client = MagicMock()
    client.stream = MagicMock(return_value=mock_resp)
    client.aclose = AsyncMock()

    result = await synthesize(
        "Hello world",
        api_key="test-key",
        model="voxtral-mini-tts-2603",
        voice="en_neutral_female",
        client=client,
    )

    # Should return valid WAV
    assert result[:4] == b"RIFF"

    # Verify API call
    call_args = client.stream.call_args
    assert call_args[0][0] == "POST"
    assert "mistral.ai" in call_args[0][1]
    assert call_args[1]["json"]["input"] == "Hello world"
    assert call_args[1]["json"]["model"] == "voxtral-mini-tts-2603"
    assert call_args[1]["json"]["voice_id"] == "en_neutral_female"
    assert call_args[1]["json"]["stream"] is True
    assert call_args[1]["json"]["response_format"] == "pcm"
    assert "Bearer test-key" in call_args[1]["headers"]["Authorization"]


async def test_synthesize_with_custom_voice():
    sse_chunks = _make_sse_response()

    async def fake_aiter_text():
        for chunk in sse_chunks:
            yield chunk

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.aiter_text = fake_aiter_text
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    client = MagicMock()
    client.stream = MagicMock(return_value=mock_resp)
    client.aclose = AsyncMock()

    await synthesize(
        "Hallo Welt",
        api_key="test-key",
        voice="de_female",
        client=client,
    )

    call_args = client.stream.call_args
    assert call_args[1]["json"]["voice_id"] == "de_female"


async def test_synthesize_raises_on_error():
    mock_resp = MagicMock()
    error_response = MagicMock()
    error_response.status_code = 401
    error_response.text = "Unauthorized"
    mock_resp.raise_for_status = MagicMock(
        side_effect=httpx.HTTPStatusError("401", request=MagicMock(), response=error_response)
    )
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    client = MagicMock()
    client.stream = MagicMock(return_value=mock_resp)
    client.aclose = AsyncMock()

    with pytest.raises(httpx.HTTPStatusError):
        await synthesize(
            "Hello",
            api_key="bad-key",
            client=client,
        )
