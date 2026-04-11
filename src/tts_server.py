"""Standalone TTS server using NeMo MagpieTTS.

Run with:
    python src/tts_server.py

Requires: nemo_toolkit[tts], torch, numpy, flask

Endpoints:
    GET  /health     — liveness check
    POST /tts        — synthesize text, return WAV audio
    POST /tts/file   — synthesize text, save WAV to disk, return path + duration
"""
from __future__ import annotations

import hashlib
import io
import os
import wave

from flask import Flask, Response, jsonify, request

app = Flask(__name__)

SAMPLE_RATE = 22050
TTS_PORT = int(os.environ.get("TTS_PORT", "7860"))

# Lazy-loaded heavy deps (torch, numpy, nemo)
np = None
torch = None
model = None


def _stub_nv_one_logger():
    """Stub out nv_one_logger — NVIDIA-internal package not available on PyPI."""
    import sys
    import types

    if "nv_one_logger" in sys.modules:
        return

    modules = [
        "nv_one_logger",
        "nv_one_logger.api",
        "nv_one_logger.api.config",
        "nv_one_logger.training_telemetry",
        "nv_one_logger.training_telemetry.api",
        "nv_one_logger.training_telemetry.api.callbacks",
        "nv_one_logger.training_telemetry.api.config",
        "nv_one_logger.training_telemetry.api.training_telemetry_provider",
        "nv_one_logger.training_telemetry.integration",
        "nv_one_logger.training_telemetry.integration.pytorch_lightning",
    ]
    for name in modules:
        sys.modules[name] = types.ModuleType(name)

    # Stub classes/functions expected by nemo
    sys.modules["nv_one_logger.api.config"].OneLoggerConfig = type(
        "OneLoggerConfig", (), {"__init__": lambda self, **kw: None},
    )
    sys.modules["nv_one_logger.training_telemetry.api.callbacks"].on_app_start = (
        lambda *a, **kw: None
    )
    sys.modules["nv_one_logger.training_telemetry.api.config"].TrainingTelemetryConfig = type(
        "TrainingTelemetryConfig", (), {"__init__": lambda self, **kw: None},
    )
    class _StubProvider:
        _inst = None

        def __init__(self, **kw):
            pass

        @classmethod
        def instance(cls):
            if cls._inst is None:
                cls._inst = cls()
            return cls._inst

        def with_base_config(self, *a, **kw):
            return self

        def __getattr__(self, name):
            """Return self for any chained method call."""
            return lambda *a, **kw: self

    sys.modules["nv_one_logger.training_telemetry.api.training_telemetry_provider"].TrainingTelemetryProvider = _StubProvider
    sys.modules["nv_one_logger.training_telemetry.integration.pytorch_lightning"].TimeEventCallback = type(
        "TimeEventCallback", (), {"__init__": lambda self, *a, **kw: None},
    )


_stub_nv_one_logger()


def load_model():
    global model, np, torch
    import numpy as _np
    import torch as _torch
    np = _np
    torch = _torch

    print("Loading MagpieTTS model...")
    from nemo.collections.tts.models import MagpieTTSModel

    model = MagpieTTSModel.from_pretrained("nvidia/magpie_tts_multilingual_357m")
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    print("Model loaded.")


def _normalize_text(text: str) -> str:
    """Normalize text before synthesis to avoid unknown char warnings."""
    # Replace em/en dashes with commas (natural pause)
    text = text.replace("—", ", ").replace("–", ", ")
    return text


def _synthesize(text: str, speaker: int = 0, language: str = "en"):
    """Synthesize speech from text. Returns (audio_np, duration_seconds)."""
    text = _normalize_text(text)
    with torch.no_grad():
        audio, audio_len = model.do_tts(
            text,
            language=language,
            apply_TN=True,
            speaker_index=speaker,
        )

    # audio may be (1, T) or (T,)
    if isinstance(audio, torch.Tensor):
        audio_np = audio.cpu().numpy().flatten()
    else:
        audio_np = np.array(audio).flatten()

    # Trim to audio_len if provided
    if isinstance(audio_len, torch.Tensor):
        length = int(audio_len.item())
        if 0 < length <= len(audio_np):
            audio_np = audio_np[:length]
    elif isinstance(audio_len, (int, float)) and 0 < int(audio_len) <= len(audio_np):
        audio_np = audio_np[: int(audio_len)]

    # Normalize
    audio_np = audio_np.astype(np.float32)
    peak = np.max(np.abs(audio_np))
    if peak > 0:
        audio_np = audio_np / peak * 0.95

    duration = len(audio_np) / SAMPLE_RATE
    return audio_np, duration


def _audio_to_wav_bytes(audio_np) -> bytes:
    """Convert float32 audio array to WAV bytes (PCM 16-bit, mono, 22050 Hz)."""
    audio_int16 = np.clip(audio_np, -1.0, 1.0)
    audio_int16 = (audio_int16 * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_int16.tobytes())
    return buf.getvalue()


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/tts", methods=["POST"])
def tts():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    if not text or not text.strip():
        return jsonify({"error": "text is required and must not be empty"}), 400

    speaker = data.get("speaker", 0)
    language = data.get("language", "en")

    audio_np, _ = _synthesize(text, speaker=speaker, language=language)
    wav_bytes = _audio_to_wav_bytes(audio_np)

    return Response(wav_bytes, mimetype="audio/wav")


@app.route("/tts/file", methods=["POST"])
def tts_file():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    if not text or not text.strip():
        return jsonify({"error": "text is required and must not be empty"}), 400

    speaker = data.get("speaker", 0)
    language = data.get("language", "en")

    audio_np, duration = _synthesize(text, speaker=speaker, language=language)
    wav_bytes = _audio_to_wav_bytes(audio_np)

    os.makedirs("output", exist_ok=True)
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    filepath = f"output/{text_hash}.wav"
    with open(filepath, "wb") as f:
        f.write(wav_bytes)

    return jsonify({"file": filepath, "duration_seconds": round(duration, 2)})


def main():
    load_model()
    print(f"TTS server ready on port {TTS_PORT}")
    app.run(host="0.0.0.0", port=TTS_PORT, threaded=False)


if __name__ == "__main__":
    main()
