"""Audio chunking with silence detection for speech mode."""
from __future__ import annotations

import struct
from dataclasses import dataclass

SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2  # 16-bit signed LE
CHUNK_MIN_SECONDS = 2.0
CHUNK_MIN_BYTES = int(SAMPLE_RATE * BYTES_PER_SAMPLE * CHUNK_MIN_SECONDS)
SILENCE_THRESHOLD_RMS = 500
SILENCE_WINDOW_SAMPLES = int(SAMPLE_RATE * 0.05)  # 50ms window
SILENCE_WINDOW_BYTES = SILENCE_WINDOW_SAMPLES * BYTES_PER_SAMPLE
PAUSE_SECONDS = 0.4


@dataclass
class ChunkEvent:
    type: str  # "chunk" or "pause"
    audio: bytes | None = None


def rms_int16(pcm: bytes) -> float:
    """Compute RMS of 16-bit signed LE PCM."""
    if len(pcm) < BYTES_PER_SAMPLE:
        return 0.0
    n_samples = len(pcm) // BYTES_PER_SAMPLE
    samples = struct.unpack(f"<{n_samples}h", pcm[: n_samples * BYTES_PER_SAMPLE])
    if not samples:
        return 0.0
    return (sum(s * s for s in samples) / n_samples) ** 0.5


def is_silent(pcm: bytes, threshold: float = SILENCE_THRESHOLD_RMS) -> bool:
    """Check if a PCM buffer is below the silence threshold."""
    return rms_int16(pcm) < threshold


class AudioChunker:
    """Accumulates raw PCM and emits chunk/pause events."""

    def __init__(
        self,
        *,
        chunk_min_seconds: float = CHUNK_MIN_SECONDS,
        silence_threshold: float = SILENCE_THRESHOLD_RMS,
        pause_seconds: float = PAUSE_SECONDS,
    ):
        self.buffer = bytearray()
        self.chunk_min_bytes = int(SAMPLE_RATE * BYTES_PER_SAMPLE * chunk_min_seconds)
        self.silence_threshold = silence_threshold
        self.pause_seconds = pause_seconds

        self.has_speech = False
        self.last_speech_time: float = 0.0
        self._pause_emitted = False

    @staticmethod
    def _find_speech_end(buf: bytearray) -> int:
        """Find the byte offset where speech ends (start of trailing silence).

        Scans backwards from the end of the buffer to find the last
        non-silent window. Returns the offset just after that window,
        so the emitted chunk contains all speech but minimal trailing silence.
        """
        length = len(buf)
        step = SILENCE_WINDOW_BYTES
        # Scan backwards in window-sized steps
        pos = length
        while pos - step >= 0:
            window = bytes(buf[pos - step : pos])
            if not is_silent(window):
                # Found speech — include this window plus one extra for safety
                return min(length, pos + step)
            pos -= step
        # Entire buffer is silent — return all of it
        return length

    def feed(self, pcm_data: bytes, current_time: float) -> list[ChunkEvent]:
        """Feed raw PCM bytes and the current monotonic time. Returns events."""
        events: list[ChunkEvent] = []
        self.buffer.extend(pcm_data)

        # Analyze the incoming data for speech activity
        tail = bytes(pcm_data)
        if not is_silent(tail, self.silence_threshold):
            self.has_speech = True
            self.last_speech_time = current_time
            self._pause_emitted = False

        # Check if we have enough audio for a chunk
        if len(self.buffer) >= self.chunk_min_bytes:
            # Check if the tail of the buffer is silent (good cut point)
            tail_window = bytes(self.buffer[-SILENCE_WINDOW_BYTES:]) if len(self.buffer) >= SILENCE_WINDOW_BYTES else bytes(self.buffer)
            if is_silent(tail_window, self.silence_threshold):
                # Find where speech ends — trim trailing silence to avoid
                # the ASR seeing a word fragment at the boundary
                cut = self._find_speech_end(self.buffer)
                events.append(ChunkEvent(type="chunk", audio=bytes(self.buffer[:cut])))
                self.buffer = bytearray(self.buffer[cut:])

        # Check for pause (0.4s silence after speech)
        if (
            self.has_speech
            and not self._pause_emitted
            and current_time - self.last_speech_time >= self.pause_seconds
        ):
            # Flush any remaining audio as a chunk first
            if len(self.buffer) > 0:
                events.append(ChunkEvent(type="chunk", audio=bytes(self.buffer)))
                self.buffer.clear()
            events.append(ChunkEvent(type="pause"))
            self._pause_emitted = True

        return events

    def flush(self) -> list[ChunkEvent]:
        """Flush remaining buffer as a final chunk."""
        events: list[ChunkEvent] = []
        if len(self.buffer) > 0:
            events.append(ChunkEvent(type="chunk", audio=bytes(self.buffer)))
            self.buffer.clear()
        return events

    def reset(self) -> None:
        """Reset all state."""
        self.buffer.clear()
        self.has_speech = False
        self.last_speech_time = 0.0
        self._pause_emitted = False
