"""Record audio chunks and full streams for later analysis."""
from __future__ import annotations

import json
import struct
import wave
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
NUM_CHANNELS = 1
SAMPLE_WIDTH = 2  # 16-bit


class AudioRecorder:
    """Records audio chunks with transcriptions and the full audio stream."""

    def __init__(self, output_dir: str | Path, session_id: str):
        self.session_id = session_id
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.session_dir = Path(output_dir) / f"{ts}_{session_id}"
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self._full_stream = bytearray()
        self._chunk_index = 0
        self._manifest: list[dict] = []

    def feed_audio(self, pcm: bytes) -> None:
        """Append raw PCM to the full stream recording."""
        self._full_stream.extend(pcm)

    def record_chunk(self, pcm: bytes, transcript: str, skipped: bool = False) -> None:
        """Save an individual chunk as a WAV with its transcription."""
        idx = self._chunk_index
        self._chunk_index += 1

        entry = {
            "index": idx,
            "audio_bytes": len(pcm),
            "audio_seconds": round(len(pcm) / (SAMPLE_RATE * SAMPLE_WIDTH), 2),
            "transcript": transcript,
            "skipped": skipped,
        }
        self._manifest.append(entry)

        if skipped:
            return

        wav_path = self.session_dir / f"chunk_{idx:04d}.wav"
        _write_wav(wav_path, pcm)

        txt_path = self.session_dir / f"chunk_{idx:04d}.txt"
        txt_path.write_text(transcript, encoding="utf-8")

    def finalize(self) -> None:
        """Write the full stream WAV and manifest."""
        if self._full_stream:
            full_path = self.session_dir / "full_stream.wav"
            _write_wav(full_path, bytes(self._full_stream))
            logger.info(
                "Audio recording saved: %s (%.1fs, %d chunks)",
                self.session_dir,
                len(self._full_stream) / (SAMPLE_RATE * SAMPLE_WIDTH),
                self._chunk_index,
            )

        manifest_path = self.session_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "session_id": self.session_id,
                    "sample_rate": SAMPLE_RATE,
                    "channels": NUM_CHANNELS,
                    "sample_width": SAMPLE_WIDTH,
                    "total_audio_seconds": round(len(self._full_stream) / (SAMPLE_RATE * SAMPLE_WIDTH), 2),
                    "chunks": self._manifest,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )


def _write_wav(path: Path, pcm: bytes) -> None:
    """Write raw PCM to a WAV file."""
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(NUM_CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm)
