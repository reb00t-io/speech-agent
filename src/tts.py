"""TTS client — sends text to a TTS server and returns WAV audio."""
from __future__ import annotations

import base64
import logging
import re
import time

import httpx

logger = logging.getLogger(__name__)

# Split on sentence-ending punctuation followed by whitespace or end-of-string.
# Keeps the punctuation with the sentence.
_SENTENCE_RE = re.compile(r'(?<=[.!?;:।。！？])\s+')


def split_sentences(text: str) -> list[str]:
    """Split text into sentences for TTS chunking."""
    parts = _SENTENCE_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


async def synthesize(
    text: str,
    *,
    base_url: str,
    language: str = "en",
    speaker: int = 0,
    client: httpx.AsyncClient | None = None,
) -> bytes:
    """Call the TTS server and return WAV bytes."""
    should_close = client is None
    if client is None:
        client = httpx.AsyncClient(timeout=60)

    t0 = time.monotonic()
    try:
        resp = await client.post(
            f"{base_url}/tts",
            json={"text": text, "language": language, "speaker": speaker},
        )
        resp.raise_for_status()
        wav_bytes = resp.content
        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.info(
            "TTS: text=%d chars e2e=%.0fms wav=%d bytes result=%r",
            len(text), elapsed_ms, len(wav_bytes), text[:80],
        )
        return wav_bytes
    except httpx.HTTPStatusError as exc:
        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.error("TTS request failed in %.0fms: %s %s", elapsed_ms, exc.response.status_code, exc.response.text[:200])
        raise
    except Exception:
        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.exception("TTS request error after %.0fms", elapsed_ms)
        raise
    finally:
        if should_close:
            await client.aclose()


def wav_to_base64(wav_bytes: bytes) -> str:
    """Encode WAV bytes as base64 for WebSocket transport."""
    return base64.b64encode(wav_bytes).decode("ascii")
