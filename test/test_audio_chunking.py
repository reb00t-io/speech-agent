"""Tests for src/audio_chunking.py — audio chunking and silence detection."""
import struct

import pytest

from src.audio_chunking import (
    BYTES_PER_SAMPLE,
    CHUNK_MIN_SECONDS,
    PAUSE_SECONDS,
    SAMPLE_RATE,
    SILENCE_THRESHOLD_RMS,
    AudioChunker,
    ChunkEvent,
    is_silent,
    rms_int16,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _make_pcm_silence(duration_s: float) -> bytes:
    """Generate silent PCM (all zeros)."""
    n_samples = int(SAMPLE_RATE * duration_s)
    return b"\x00\x00" * n_samples


def _make_pcm_tone(duration_s: float, amplitude: int = 10000, freq: int = 440) -> bytes:
    """Generate a sine-wave PCM tone."""
    import math
    n_samples = int(SAMPLE_RATE * duration_s)
    samples = []
    for i in range(n_samples):
        value = int(amplitude * math.sin(2 * math.pi * freq * i / SAMPLE_RATE))
        samples.append(value)
    return struct.pack(f"<{n_samples}h", *samples)


def _make_pcm_constant(duration_s: float, value: int = 5000) -> bytes:
    """Generate constant-value PCM (non-silent)."""
    n_samples = int(SAMPLE_RATE * duration_s)
    return struct.pack(f"<{n_samples}h", *([value] * n_samples))


# ─── rms_int16 ───────────────────────────────────────────────────────────────

def test_rms_silence():
    assert rms_int16(_make_pcm_silence(0.01)) == 0.0


def test_rms_constant():
    pcm = _make_pcm_constant(0.01, value=1000)
    assert abs(rms_int16(pcm) - 1000) < 1


def test_rms_empty():
    assert rms_int16(b"") == 0.0


def test_rms_single_byte():
    assert rms_int16(b"\x00") == 0.0


def test_rms_tone():
    pcm = _make_pcm_tone(0.05, amplitude=10000)
    rms = rms_int16(pcm)
    # RMS of a sine wave = amplitude / sqrt(2) ≈ 7071
    assert 6800 < rms < 7300


# ─── is_silent ───────────────────────────────────────────────────────────────

def test_silence_is_silent():
    assert is_silent(_make_pcm_silence(0.05))


def test_loud_is_not_silent():
    assert not is_silent(_make_pcm_tone(0.05, amplitude=10000))


def test_quiet_is_silent():
    # Very low amplitude should be below threshold
    pcm = _make_pcm_constant(0.05, value=100)
    assert is_silent(pcm)


def test_custom_threshold():
    pcm = _make_pcm_constant(0.05, value=600)
    assert not is_silent(pcm, threshold=500)
    assert is_silent(pcm, threshold=700)


# ─── AudioChunker — basic chunking ──────────────────────────────────────────

def test_no_events_for_short_audio():
    chunker = AudioChunker()
    events = chunker.feed(_make_pcm_tone(0.5), 0.5)
    # Not enough audio for a chunk, no pause yet
    assert all(e.type != "chunk" for e in events)


def test_chunk_emitted_at_silence_boundary():
    chunker = AudioChunker()
    # Feed 2s of tone
    events = chunker.feed(_make_pcm_tone(2.0), 2.0)
    # Then feed silence (creates a silent tail for cut point)
    events += chunker.feed(_make_pcm_silence(0.1), 2.1)
    chunk_events = [e for e in events if e.type == "chunk"]
    assert len(chunk_events) >= 1
    assert chunk_events[0].audio is not None
    assert len(chunk_events[0].audio) > 0


def test_no_chunk_without_silence_at_tail():
    chunker = AudioChunker()
    # Feed >2s of continuous tone, no silence
    events = chunker.feed(_make_pcm_tone(3.0), 3.0)
    chunk_events = [e for e in events if e.type == "chunk"]
    # Should not emit — waiting for silence boundary
    assert len(chunk_events) == 0


def test_flush_emits_remaining_audio():
    chunker = AudioChunker()
    chunker.feed(_make_pcm_tone(1.0), 1.0)
    events = chunker.flush()
    assert len(events) == 1
    assert events[0].type == "chunk"
    assert len(events[0].audio) > 0


def test_flush_empty_buffer():
    chunker = AudioChunker()
    events = chunker.flush()
    assert events == []


# ─── AudioChunker — pause detection ─────────────────────────────────────────

def test_pause_detected_after_silence_duration():
    chunker = AudioChunker()
    t = 0.0

    # Feed speech
    speech = _make_pcm_tone(0.5, amplitude=10000)
    chunker.feed(speech, t)
    t += 0.5

    # Feed silence for > PAUSE_SECONDS
    silence_chunk = _make_pcm_silence(0.1)
    events = []
    for _ in range(6):  # 0.6s of silence
        t += 0.1
        events.extend(chunker.feed(silence_chunk, t))

    pause_events = [e for e in events if e.type == "pause"]
    assert len(pause_events) >= 1


def test_no_pause_without_prior_speech():
    chunker = AudioChunker()
    # Only silence — no speech detected, so no pause event
    events = []
    t = 0.0
    for _ in range(10):
        t += 0.1
        events.extend(chunker.feed(_make_pcm_silence(0.1), t))
    pause_events = [e for e in events if e.type == "pause"]
    assert len(pause_events) == 0


def test_pause_not_repeated():
    chunker = AudioChunker()
    t = 0.0

    # Speech then silence
    chunker.feed(_make_pcm_tone(0.5), t)
    t += 0.5

    events = []
    silence = _make_pcm_silence(0.1)
    for _ in range(10):  # 1s of silence
        t += 0.1
        events.extend(chunker.feed(silence, t))

    pause_events = [e for e in events if e.type == "pause"]
    assert len(pause_events) == 1  # Only one pause, not repeated


def test_pause_re_triggers_after_new_speech():
    chunker = AudioChunker()
    t = 0.0

    # First speech + pause
    chunker.feed(_make_pcm_tone(0.5), t)
    t += 0.5
    events = []
    for _ in range(6):
        t += 0.1
        events.extend(chunker.feed(_make_pcm_silence(0.1), t))
    assert any(e.type == "pause" for e in events)

    # New speech
    chunker.feed(_make_pcm_tone(0.5), t)
    t += 0.5

    # New pause
    events = []
    for _ in range(6):
        t += 0.1
        events.extend(chunker.feed(_make_pcm_silence(0.1), t))
    assert any(e.type == "pause" for e in events)


# ─── AudioChunker — reset ───────────────────────────────────────────────────

def test_reset_clears_state():
    chunker = AudioChunker()
    chunker.feed(_make_pcm_tone(1.0), 1.0)
    assert len(chunker.buffer) > 0
    chunker.reset()
    assert len(chunker.buffer) == 0
    assert not chunker.has_speech


# ─── AudioChunker — custom parameters ───────────────────────────────────────

def test_custom_chunk_min_seconds():
    chunker = AudioChunker(chunk_min_seconds=1.0)
    # Feed 1s of tone + silence
    events = chunker.feed(_make_pcm_tone(1.0), 1.0)
    events += chunker.feed(_make_pcm_silence(0.1), 1.1)
    chunk_events = [e for e in events if e.type == "chunk"]
    assert len(chunk_events) >= 1


def test_custom_pause_seconds():
    chunker = AudioChunker(pause_seconds=0.2)
    t = 0.0

    chunker.feed(_make_pcm_tone(0.3), t)
    t += 0.3

    events = []
    for _ in range(4):  # 0.4s of silence (> 0.2s threshold)
        t += 0.1
        events.extend(chunker.feed(_make_pcm_silence(0.1), t))

    assert any(e.type == "pause" for e in events)
