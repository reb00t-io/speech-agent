"""End-to-end TTS test: synthesize text via Voxtral → send audio to ASR → verify transcript.

Requires:
    - MISTRAL_API_KEY env var set
    - ASR server at LLM_BASE_URL

Run with:
    pytest test/test_e2e_tts.py -v
"""
from __future__ import annotations

import io
import os
import struct
import wave

import pytest

# Use env vars
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
ASR_MODEL = os.environ.get("ASR_MODEL", "")
TTS_VOICE = os.environ.get("TTS_VOICE", "en_paul_neutral")


requires_tts = pytest.mark.skipif(
    not MISTRAL_API_KEY, reason="MISTRAL_API_KEY not set"
)
requires_asr = pytest.mark.skipif(
    not LLM_BASE_URL, reason="LLM_BASE_URL not set"
)


def _resample_wav_to_16k_pcm(wav_bytes: bytes) -> bytes:
    """Read a WAV file and resample to 16kHz 16-bit mono PCM for ASR.

    Uses simple linear interpolation (no external deps).
    """
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        assert wf.getnchannels() == 1, "Expected mono audio"
        assert wf.getsampwidth() == 2, "Expected 16-bit audio"
        src_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if src_rate == 16000:
        return raw

    # Unpack source samples
    src_samples = struct.unpack(f"<{n_frames}h", raw)
    dst_rate = 16000
    dst_n = int(n_frames * dst_rate / src_rate)
    ratio = src_rate / dst_rate

    # Linear interpolation
    dst_samples = []
    for i in range(dst_n):
        src_pos = i * ratio
        idx = int(src_pos)
        frac = src_pos - idx
        if idx + 1 < n_frames:
            val = src_samples[idx] * (1 - frac) + src_samples[idx + 1] * frac
        else:
            val = src_samples[min(idx, n_frames - 1)]
        dst_samples.append(int(val))

    return struct.pack(f"<{dst_n}h", *dst_samples)


async def _synthesize(text: str) -> bytes:
    """Call Voxtral TTS and return WAV bytes."""
    from src.tts import synthesize
    return await synthesize(
        text,
        api_key=MISTRAL_API_KEY,
        voice=TTS_VOICE,
    )


async def _transcribe(pcm_16k: bytes) -> str:
    """Call the ASR server with 16kHz PCM and return text."""
    from src.asr import transcribe
    return await transcribe(
        pcm_16k,
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        model=ASR_MODEL,
    )


def _words(text: str) -> set[str]:
    """Extract lowercase words from text for fuzzy comparison."""
    return {w.strip(".,!?;:'\"").lower() for w in text.split() if w.strip(".,!?;:'\"") }


# ─── Tests ───────────────────────────────────────────────────────────────────

@requires_tts
@requires_asr
async def test_e2e_tts_asr_roundtrip_english():
    """Synthesize English text, send to ASR, verify transcript roughly matches."""
    original = "The weather is nice today."

    wav_bytes = await _synthesize(original)
    assert len(wav_bytes) > 1000, "WAV too small — synthesis may have failed"

    pcm_16k = _resample_wav_to_16k_pcm(wav_bytes)
    transcript = await _transcribe(pcm_16k)

    original_words = _words(original)
    transcript_words = _words(transcript)

    # At least half the original words should appear in the transcript
    overlap = original_words & transcript_words
    ratio = len(overlap) / len(original_words) if original_words else 0
    assert ratio >= 0.5, (
        f"Transcript doesn't match. Original: {original!r}, "
        f"Transcript: {transcript!r}, Overlap: {overlap}"
    )


@requires_tts
@requires_asr
async def test_e2e_tts_asr_roundtrip_with_dash():
    """Verify em dashes are handled — text with dashes should still round-trip."""
    original = "Sure thing—just let me know what you need."

    wav_bytes = await _synthesize(original)
    assert len(wav_bytes) > 1000

    pcm_16k = _resample_wav_to_16k_pcm(wav_bytes)
    transcript = await _transcribe(pcm_16k)

    # Key content words should survive the round-trip
    for word in ["sure", "know", "need"]:
        assert word in transcript.lower(), (
            f"Expected '{word}' in transcript: {transcript!r}"
        )


@requires_tts
@requires_asr
async def test_e2e_tts_asr_roundtrip_longer():
    """Test a longer sentence to verify sustained synthesis quality."""
    original = "Artificial intelligence is transforming how we work and communicate with each other."

    wav_bytes = await _synthesize(original)
    assert len(wav_bytes) > 5000

    pcm_16k = _resample_wav_to_16k_pcm(wav_bytes)
    transcript = await _transcribe(pcm_16k)

    original_words = _words(original)
    transcript_words = _words(transcript)

    overlap = original_words & transcript_words
    ratio = len(overlap) / len(original_words) if original_words else 0
    assert ratio >= 0.4, (
        f"Transcript doesn't match. Original: {original!r}, "
        f"Transcript: {transcript!r}, Overlap ratio: {ratio:.0%}"
    )
