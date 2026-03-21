"""Tests for src/asr.py — ASR client."""
import struct
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from src.asr import _make_wav, transcribe


# ─── WAV header ──────────────────────────────────────────────────────────────

def test_make_wav_header_structure():
    pcm = b"\x00\x01" * 100  # 100 samples
    wav = _make_wav(pcm)

    # WAV starts with RIFF header
    assert wav[:4] == b"RIFF"
    assert wav[8:12] == b"WAVE"
    assert wav[12:16] == b"fmt "

    # Data chunk
    data_marker = wav.find(b"data")
    assert data_marker > 0
    data_size = struct.unpack_from("<I", wav, data_marker + 4)[0]
    assert data_size == len(pcm)

    # Actual PCM follows
    assert wav[data_marker + 8:] == pcm


def test_make_wav_sample_rate():
    pcm = b"\x00" * 200
    wav = _make_wav(pcm)
    # Sample rate is at offset 24 in the WAV header
    sample_rate = struct.unpack_from("<I", wav, 24)[0]
    assert sample_rate == 16000


def test_make_wav_mono_16bit():
    pcm = b"\x00" * 200
    wav = _make_wav(pcm)
    # Channels at offset 22
    channels = struct.unpack_from("<H", wav, 22)[0]
    assert channels == 1
    # Bits per sample at offset 34
    bits = struct.unpack_from("<H", wav, 34)[0]
    assert bits == 16


def test_make_wav_empty_pcm():
    wav = _make_wav(b"")
    assert wav[:4] == b"RIFF"
    data_marker = wav.find(b"data")
    data_size = struct.unpack_from("<I", wav, data_marker + 4)[0]
    assert data_size == 0


# ─── transcribe() ────────────────────────────────────────────────────────────

@pytest.fixture
def mock_client():
    """Create a mock httpx.AsyncClient."""
    client = AsyncMock(spec=httpx.AsyncClient)
    return client


async def test_transcribe_sends_correct_request(mock_client):
    response = MagicMock()
    response.status_code = 200
    response.text = "hello world"
    response.raise_for_status = MagicMock()
    mock_client.post = AsyncMock(return_value=response)

    result = await transcribe(
        b"\x00\x01" * 100,
        base_url="http://asr-server",
        api_key="test-key",
        model="whisper-1",
        client=mock_client,
    )

    assert result == "hello world"
    mock_client.post.assert_awaited_once()
    call_args = mock_client.post.call_args
    assert call_args[0][0] == "http://asr-server/v1/audio/transcriptions"
    assert call_args[1]["data"]["model"] == "whisper-1"
    assert call_args[1]["data"]["response_format"] == "text"
    assert "Authorization" in call_args[1]["headers"]


async def test_transcribe_strips_whitespace(mock_client):
    response = MagicMock()
    response.status_code = 200
    response.text = "  hello world  \n"
    response.raise_for_status = MagicMock()
    mock_client.post = AsyncMock(return_value=response)

    result = await transcribe(
        b"\x00" * 100,
        base_url="http://asr-server",
        api_key="",
        model="whisper-1",
        client=mock_client,
    )
    assert result == "hello world"


async def test_transcribe_no_api_key_omits_auth_header(mock_client):
    response = MagicMock()
    response.status_code = 200
    response.text = "ok"
    response.raise_for_status = MagicMock()
    mock_client.post = AsyncMock(return_value=response)

    await transcribe(
        b"\x00" * 100,
        base_url="http://asr-server",
        api_key="",
        model="whisper-1",
        client=mock_client,
    )

    headers = mock_client.post.call_args[1]["headers"]
    assert "Authorization" not in headers


async def test_transcribe_raises_on_http_error(mock_client):
    response = MagicMock()
    response.status_code = 500
    response.text = "Internal Server Error"
    response.raise_for_status = MagicMock(
        side_effect=httpx.HTTPStatusError("500", request=MagicMock(), response=response)
    )
    mock_client.post = AsyncMock(return_value=response)

    with pytest.raises(httpx.HTTPStatusError):
        await transcribe(
            b"\x00" * 100,
            base_url="http://asr-server",
            api_key="key",
            model="whisper-1",
            client=mock_client,
        )


async def test_transcribe_sends_wav_file(mock_client):
    response = MagicMock()
    response.status_code = 200
    response.text = "test"
    response.raise_for_status = MagicMock()
    mock_client.post = AsyncMock(return_value=response)

    pcm = b"\x00\x01" * 50
    await transcribe(
        pcm,
        base_url="http://asr-server",
        api_key="key",
        model="whisper-1",
        client=mock_client,
    )

    files_arg = mock_client.post.call_args[1]["files"]
    file_tuple = files_arg["file"]
    assert file_tuple[0] == "audio.wav"
    # Read the BytesIO and check WAV header
    wav_data = file_tuple[1].read()
    assert wav_data[:4] == b"RIFF"
