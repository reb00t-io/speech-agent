"""ASR client — sends audio to an OpenAI-compatible transcription endpoint."""
from __future__ import annotations

import io
import json
import struct
import logging
import time

import httpx

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
BITS_PER_SAMPLE = 16
NUM_CHANNELS = 1


def _make_wav(pcm: bytes) -> bytes:
    """Wrap raw 16-bit LE mono PCM in a WAV header."""
    data_size = len(pcm)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,            # file size - 8
        b"WAVE",
        b"fmt ",
        16,                        # chunk size
        1,                         # PCM format
        NUM_CHANNELS,
        SAMPLE_RATE,
        SAMPLE_RATE * NUM_CHANNELS * BITS_PER_SAMPLE // 8,  # byte rate
        NUM_CHANNELS * BITS_PER_SAMPLE // 8,                # block align
        BITS_PER_SAMPLE,
        b"data",
        data_size,
    )
    return header + pcm


async def transcribe(
    audio_pcm: bytes,
    *,
    base_url: str,
    api_key: str,
    model: str,
    language: str = "",
    client: httpx.AsyncClient | None = None,
) -> str:
    """Transcribe raw PCM audio via OpenAI-compatible ASR endpoint.

    Returns the transcribed text (may be empty string for silence).
    """
    wav_bytes = _make_wav(audio_pcm)

    files = {"file": ("audio.wav", io.BytesIO(wav_bytes), "audio/wav")}
    data = {"model": model, "response_format": "text"}
    if language:
        data["language"] = language
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    should_close = client is None
    if client is None:
        client = httpx.AsyncClient(timeout=30)

    audio_duration_s = len(audio_pcm) / (SAMPLE_RATE * NUM_CHANNELS * BITS_PER_SAMPLE // 8)
    t0 = time.monotonic()
    try:
        resp = await client.post(
            f"{base_url}/audio/transcriptions",
            files=files,
            data=data,
            headers=headers,
        )
        resp.raise_for_status()
        body = resp.text.strip()
        # Some endpoints return JSON even with response_format=text
        if body.startswith("{"):
            try:
                text = json.loads(body).get("text", "").strip()
            except (json.JSONDecodeError, AttributeError):
                text = body
        else:
            text = body
        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.info(
            "ASR: audio=%.1fs text=%d chars e2e=%.0fms result=%r",
            audio_duration_s, len(text), elapsed_ms, text[:120],
        )
        return text
    except httpx.HTTPStatusError as exc:
        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.error("ASR request failed in %.0fms: %s %s", elapsed_ms, exc.response.status_code, exc.response.text[:200])
        raise
    except Exception:
        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.exception("ASR request error after %.0fms", elapsed_ms)
        raise
    finally:
        if should_close:
            await client.aclose()
