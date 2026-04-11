"""TTS client — synthesizes speech via Mistral Voxtral TTS API (streaming)."""
from __future__ import annotations

import base64
import io
import json
import logging
import re
import struct
import time

import httpx

logger = logging.getLogger(__name__)

MISTRAL_TTS_URL = "https://api.mistral.ai/v1/audio/speech"
VOXTRAL_SAMPLE_RATE = 24000

# Split on sentence-ending punctuation followed by whitespace or end-of-string.
# Keeps the punctuation with the sentence.
_SENTENCE_RE = re.compile(r'(?<=[.!?;:।。！？])\s+')


def split_sentences(text: str) -> list[str]:
    """Split text into sentences for TTS chunking."""
    parts = _SENTENCE_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def _pcm_to_wav(pcm: bytes, sample_rate: int = VOXTRAL_SAMPLE_RATE) -> bytes:
    """Wrap raw PCM float32 LE samples in a WAV header (16-bit mono)."""
    import wave

    # Voxtral PCM is float32 LE — convert to int16
    n_floats = len(pcm) // 4
    floats = struct.unpack(f"<{n_floats}f", pcm)
    int16_samples = []
    for f in floats:
        clamped = max(-1.0, min(1.0, f))
        int16_samples.append(int(clamped * 32767))
    int16_bytes = struct.pack(f"<{n_floats}h", *int16_samples)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int16_bytes)
    return buf.getvalue()


async def synthesize(
    text: str,
    *,
    api_key: str,
    model: str = "voxtral-mini-tts-2603",
    voice: str = "en_paul_neutral",
    client: httpx.AsyncClient | None = None,
) -> bytes:
    """Call Voxtral TTS API with streaming and return WAV bytes."""
    should_close = client is None
    if client is None:
        client = httpx.AsyncClient(timeout=120)

    t0 = time.monotonic()
    pcm_chunks: list[bytes] = []

    try:
        async with client.stream(
            "POST",
            MISTRAL_TTS_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            },
            json={
                "model": model,
                "input": text,
                "voice_id": voice,
                "response_format": "pcm",
                "stream": True,
            },
        ) as resp:
            resp.raise_for_status()
            ttfa = None
            buf = ""
            async for chunk in resp.aiter_text():
                buf += chunk
                # Parse SSE events from buffer
                while "\n\n" in buf:
                    event_text, buf = buf.split("\n\n", 1)
                    data_line = None
                    for line in event_text.split("\n"):
                        if line.startswith("data: "):
                            data_line = line[6:]
                    if not data_line:
                        continue
                    try:
                        payload = json.loads(data_line)
                    except json.JSONDecodeError:
                        continue
                    audio_b64 = payload.get("audio_data")
                    if audio_b64:
                        if ttfa is None:
                            ttfa = (time.monotonic() - t0) * 1000
                        pcm_chunks.append(base64.b64decode(audio_b64))

        pcm_data = b"".join(pcm_chunks)
        wav_bytes = _pcm_to_wav(pcm_data)
        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.info(
            "TTS: text=%d chars ttfa=%.0fms e2e=%.0fms wav=%d bytes result=%r",
            len(text), ttfa or 0, elapsed_ms, len(wav_bytes), text[:80],
        )
        return wav_bytes
    except httpx.HTTPStatusError as exc:
        elapsed_ms = (time.monotonic() - t0) * 1000
        try:
            error_text = exc.response.text[:200]
        except httpx.ResponseNotRead:
            error_text = "(streaming response not read)"
        logger.error("TTS request failed in %.0fms: %s %s", elapsed_ms, exc.response.status_code, error_text)
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
