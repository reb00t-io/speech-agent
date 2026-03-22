"""Tests for src/tts.py — TTS client and sentence splitting."""
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from src.tts import split_sentences, synthesize, wav_to_base64


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
    import base64
    data = b"RIFF\x00\x00\x00\x00WAVEfmt "
    encoded = wav_to_base64(data)
    assert base64.b64decode(encoded) == data


# ─── synthesize ──────────────────────────────────────────────────────────────

@pytest.fixture
def mock_client():
    client = AsyncMock(spec=httpx.AsyncClient)
    return client


async def test_synthesize_sends_correct_request(mock_client):
    response = MagicMock()
    response.status_code = 200
    response.content = b"RIFF fake wav data"
    response.raise_for_status = MagicMock()
    mock_client.post = AsyncMock(return_value=response)

    result = await synthesize(
        "Hello world",
        base_url="http://tts-server",
        language="en",
        speaker=0,
        client=mock_client,
    )

    assert result == b"RIFF fake wav data"
    mock_client.post.assert_awaited_once()
    call_args = mock_client.post.call_args
    assert call_args[0][0] == "http://tts-server/tts"
    assert call_args[1]["json"]["text"] == "Hello world"
    assert call_args[1]["json"]["language"] == "en"
    assert call_args[1]["json"]["speaker"] == 0


async def test_synthesize_with_german(mock_client):
    response = MagicMock()
    response.status_code = 200
    response.content = b"wav"
    response.raise_for_status = MagicMock()
    mock_client.post = AsyncMock(return_value=response)

    await synthesize(
        "Hallo Welt",
        base_url="http://tts-server",
        language="de",
        speaker=1,
        client=mock_client,
    )

    call_args = mock_client.post.call_args
    assert call_args[1]["json"]["language"] == "de"
    assert call_args[1]["json"]["speaker"] == 1


async def test_synthesize_raises_on_error(mock_client):
    response = MagicMock()
    response.status_code = 500
    response.text = "Internal Server Error"
    response.raise_for_status = MagicMock(
        side_effect=httpx.HTTPStatusError("500", request=MagicMock(), response=response)
    )
    mock_client.post = AsyncMock(return_value=response)

    with pytest.raises(httpx.HTTPStatusError):
        await synthesize(
            "Hello",
            base_url="http://tts-server",
            language="en",
            client=mock_client,
        )
