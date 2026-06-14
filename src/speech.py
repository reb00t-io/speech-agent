"""WebSocket handler for speech mode."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import secrets
import time

from quart import websocket

try:
    from .asr import transcribe
    from .audio_chunking import AudioChunker, BYTES_PER_SAMPLE, SAMPLE_RATE, is_silent, rms_int16, SILENCE_THRESHOLD_RMS, SILENCE_WINDOW_BYTES
    from .audio_recording import AudioRecorder
    from .dual_llm import dual_stream
    from .tts import split_sentences, synthesize as tts_synthesize, wav_to_base64
except ImportError:
    from asr import transcribe
    from audio_chunking import AudioChunker, BYTES_PER_SAMPLE, SAMPLE_RATE, is_silent, rms_int16, SILENCE_THRESHOLD_RMS, SILENCE_WINDOW_BYTES
    from audio_recording import AudioRecorder
    from dual_llm import dual_stream
    from tts import split_sentences, synthesize as tts_synthesize, wav_to_base64

logger = logging.getLogger(__name__)

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-oss-120b")
ASR_MODEL = os.environ.get("ASR_MODEL", "")
ASR_LANGUAGE = os.environ.get("ASR_LANGUAGE", "")
AUDIO_RECORDING_DIR = os.environ.get("AUDIO_RECORDING_DIR", "data/audio_recordings")
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "")
TTS_MODEL = os.environ.get("TTS_MODEL", "voxtral-mini-tts-2603")
TTS_VOICE = os.environ.get("TTS_VOICE", "en_paul_neutral")

# Barge-in (interrupt while the assistant is speaking) tuning. To avoid false
# positives from the assistant's own TTS bleeding back through the mic, a
# barge-in requires audio that is both LOUDER than BARGE_IN_THRESHOLD_RMS and
# SUSTAINED for at least BARGE_IN_MIN_MS — a brief echo blip won't qualify.
# Use `or` so an empty value (e.g. docker-compose passing an unset host var as
# "") falls back to the default instead of crashing on float("").
BARGE_IN_THRESHOLD_RMS = float(os.environ.get("BARGE_IN_THRESHOLD_RMS") or 800)
BARGE_IN_MIN_MS = float(os.environ.get("BARGE_IN_MIN_MS") or 400)

logger.info(
    "Speech module: LLM_BASE_URL=%s ASR_MODEL=%s LLM_MODEL=%s ASR_LANGUAGE=%s barge_in_rms=%.0f barge_in_min_ms=%.0f",
    LLM_BASE_URL, ASR_MODEL, LLM_MODEL, ASR_LANGUAGE or "auto", BARGE_IN_THRESHOLD_RMS, BARGE_IN_MIN_MS,
)


class SpeechState:
    """Per-connection state for a speech WebSocket session."""

    def __init__(self, session_id: str, messages: list[dict]):
        self.session_id = session_id
        self.messages = messages
        self.chunker = AudioChunker()
        self.transcript_parts: list[str] = []
        self.llm_task: asyncio.Task | None = None
        self.partial_llm_response: str = ""
        self.is_speaking: bool = False
        # Dictation mode: stream transcripts to the client only, no LLM/TTS.
        self.dictation: bool = False
        # Accumulated duration (ms) of consecutive loud audio while the
        # assistant is active — used to require sustained speech for barge-in.
        self.barge_in_loud_ms: float = 0.0
        # True while the client is still playing TTS audio (reported by the
        # frontend). Lets us barge-in even after the LLM task has finished but
        # buffered speech is still playing in the browser.
        self.client_playing: bool = False
        self.recorder = AudioRecorder(AUDIO_RECORDING_DIR, session_id)
        # For short-chunk merging: track the last two transcribed chunks
        self.prev_chunk_audio: bytes | None = None  # second-to-last
        self.last_chunk_audio: bytes | None = None   # most recent


async def _send_json(data: dict) -> None:
    await websocket.send(json.dumps(data))


MIN_SPEECH_WINDOWS = 5  # at least 5 × 50ms = 250ms of non-silent audio
MERGE_THRESHOLD_BYTES = int(16000 * 2 * 1.0)  # 1 second of audio


def _chunk_has_speech(pcm: bytes) -> bool:
    """Check if a chunk contains enough speech (not just mic noise)."""
    step = SILENCE_WINDOW_BYTES
    loud_count = 0
    for i in range(0, len(pcm) - step + 1, step):
        if not is_silent(pcm[i : i + step], SILENCE_THRESHOLD_RMS):
            loud_count += 1
            if loud_count >= MIN_SPEECH_WINDOWS:
                return True
    return False


async def _do_transcribe(chunk_audio: bytes) -> str:
    """Run ASR on audio and return text (may be empty)."""
    return await transcribe(
        chunk_audio,
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        model=ASR_MODEL,
        language=ASR_LANGUAGE,
    )


async def _transcribe_chunk(chunk_audio: bytes, state: SpeechState) -> None:
    """Send audio chunk to ASR and forward transcript to client."""
    if not _chunk_has_speech(chunk_audio):
        logger.info("Skipping silent chunk: %d bytes (%.1fs)", len(chunk_audio), len(chunk_audio) / 32000)
        state.recorder.record_chunk(chunk_audio, "", skipped=True)
        return
    logger.info("Transcribing chunk: %d bytes (%.1fs audio)", len(chunk_audio), len(chunk_audio) / 32000)
    try:
        text = await _do_transcribe(chunk_audio)
        state.recorder.record_chunk(chunk_audio, text)
        if text:
            state.transcript_parts.append(text)
            state.prev_chunk_audio = state.last_chunk_audio
            state.last_chunk_audio = chunk_audio
            await _send_json({"type": "transcript", "text": text, "is_final": False})
    except Exception as exc:
        logger.error("ASR error: %s", exc)
        state.recorder.record_chunk(chunk_audio, f"ERROR: {exc}")
        await _send_json({"type": "error", "message": f"ASR error: {exc}"})


async def _maybe_merge_short_chunk(state: SpeechState) -> None:
    """If the last chunk before a pause is short and there was a previous chunk,
    merge them and re-transcribe to get a better result."""
    if state.last_chunk_audio is None or state.prev_chunk_audio is None:
        return
    if len(state.last_chunk_audio) >= MERGE_THRESHOLD_BYTES:
        return
    if len(state.transcript_parts) < 2:
        return

    merged_audio = state.prev_chunk_audio + state.last_chunk_audio
    logger.info(
        "Merging short chunk (%.1fs) with previous (%.1fs) → %.1fs",
        len(state.last_chunk_audio) / 32000,
        len(state.prev_chunk_audio) / 32000,
        len(merged_audio) / 32000,
    )
    try:
        merged_text = await _do_transcribe(merged_audio)
        state.recorder.record_chunk(merged_audio, f"[merged] {merged_text}")
        if merged_text:
            old_prev = state.transcript_parts[-2]
            old_last = state.transcript_parts[-1]
            state.transcript_parts[-2:] = [merged_text]
            logger.info(
                "Merged transcript: %r + %r → %r",
                old_prev, old_last, merged_text,
            )
            await _send_json({
                "type": "transcript_replace",
                "replace_last": 2,
                "text": merged_text,
            })
    except Exception as exc:
        logger.error("Merge transcription error: %s", exc)


async def _synth_sentence(text: str) -> bytes | None:
    """Synthesize one sentence to WAV bytes (None on failure)."""
    try:
        return await tts_synthesize(text, api_key=MISTRAL_API_KEY, model=TTS_MODEL, voice=TTS_VOICE)
    except Exception as exc:
        logger.error("TTS error: %s", exc)
        return None


async def _stream_llm(state: SpeechState) -> None:
    """Stream LLM response via dual-LLM system, with optional TTS.

    Sentences are synthesized concurrently (for low latency) but the resulting
    audio is sent to the client strictly in sentence order — otherwise a shorter
    later sentence could finish synthesis first and be spoken out of order.
    """
    tts_enabled = bool(MISTRAL_API_KEY)
    synth_tasks: list[asyncio.Task] = []
    tts_queue: asyncio.Queue = asyncio.Queue()
    emitter_task: asyncio.Task | None = None
    sentence_buf = ""

    async def _emit_in_order() -> None:
        index = 0
        while True:
            task = await tts_queue.get()
            if task is None:  # sentinel: no more sentences
                return
            wav_bytes = await task
            if wav_bytes:
                await _send_json({
                    "type": "tts_audio",
                    "index": index,
                    "audio_base64": wav_to_base64(wav_bytes),
                })
            index += 1

    def _queue_sentence(text: str) -> None:
        task = asyncio.create_task(_synth_sentence(text))
        synth_tasks.append(task)
        tts_queue.put_nowait(task)

    try:
        if tts_enabled:
            emitter_task = asyncio.create_task(_emit_in_order())

        async for token in dual_stream(
            messages=list(state.messages),
            model=LLM_MODEL,
            base_url=LLM_BASE_URL,
            api_key=LLM_API_KEY,
        ):
            state.partial_llm_response += token
            await _send_json({"type": "llm_token", "token": token})

            # TTS: queue complete sentences for ordered synthesis/playback
            if tts_enabled:
                sentence_buf += token
                sentences = split_sentences(sentence_buf)
                if len(sentences) > 1:
                    for s in sentences[:-1]:
                        _queue_sentence(s)
                    sentence_buf = sentences[-1]

        # TTS: flush remaining text
        if tts_enabled and sentence_buf.strip():
            _queue_sentence(sentence_buf.strip())

        # Signal end and wait for all queued audio to be emitted in order
        if tts_enabled:
            tts_queue.put_nowait(None)
            if emitter_task:
                await emitter_task

        # Full response complete — add to message history
        state.messages.append({"role": "assistant", "content": state.partial_llm_response})
        state.partial_llm_response = ""
        await _send_json({"type": "llm_done"})

    except asyncio.CancelledError:
        logger.info("LLM streaming cancelled (user resumed speaking)")
        raise
    except Exception as exc:
        logger.error("LLM streaming error: %s", exc)
        await _send_json({"type": "error", "message": f"LLM error: {exc}"})
    finally:
        # Cancel and reap any still-running synthesis/emitter tasks (e.g. on barge-in).
        if emitter_task and not emitter_task.done():
            emitter_task.cancel()
        for task in synth_tasks:
            if not task.done():
                task.cancel()
        for task in (emitter_task, *synth_tasks):
            if task is not None:
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass


async def _handle_pause(state: SpeechState) -> None:
    """Called when a speech pause is detected — trigger LLM (or finalize, in dictation)."""
    # Try merging a short trailing chunk with the previous one
    await _maybe_merge_short_chunk(state)

    user_text = " ".join(state.transcript_parts).strip()
    logger.info("Pause detected: transcript_parts=%d text=%r", len(state.transcript_parts), user_text[:100] if user_text else "")
    # Ignore empty or noise-only transcripts (e.g. ".", "...", "♪", emoji)
    if not user_text or not re.sub(r"[\s.\-,!?…♪🎵*]+", "", user_text):
        if user_text:
            logger.info("Ignoring noise transcript: %r", user_text)
        state.transcript_parts.clear()
        return

    # Send final marker (no text — frontend already has it from chunk transcripts)
    await _send_json({"type": "transcript_done"})

    # Reset chunk tracking for next utterance
    state.prev_chunk_audio = None
    state.last_chunk_audio = None

    # Dictation mode: stream transcripts only, never invoke the LLM or TTS.
    if state.dictation:
        state.transcript_parts.clear()
        return

    # Build messages for LLM
    if state.partial_llm_response:
        # We were interrupted — continue from where we left off
        state.messages.append({"role": "user", "content": user_text})
        state.messages.append({"role": "assistant", "content": state.partial_llm_response})
        state.messages.append({
            "role": "user",
            "content": "Continue your response from exactly where you stopped. Do not repeat anything you already said.",
        })
    else:
        state.messages.append({"role": "user", "content": user_text})

    state.transcript_parts.clear()
    logger.info("Starting LLM stream: %d messages in history", len(state.messages))
    state.llm_task = asyncio.create_task(_stream_llm(state))


async def _cancel_llm(state: SpeechState) -> None:
    """Cancel in-progress LLM if user starts speaking again."""
    if state.llm_task and not state.llm_task.done():
        state.llm_task.cancel()
        try:
            await state.llm_task
        except asyncio.CancelledError:
            pass
        # The client stops TTS on llm_cancelled; reflect that server-side too.
        state.client_playing = False
        await _send_json({
            "type": "llm_cancelled",
            "partial_response": state.partial_llm_response,
        })
        state.llm_task = None


async def _stop_client_audio(state: SpeechState) -> None:
    """Tell the client to stop TTS playback after the LLM has already finished.

    Used when the user barges in while buffered speech is still playing in the
    browser but there is no longer an LLM task to cancel."""
    state.client_playing = False
    await _send_json({"type": "stop_audio"})


def _assistant_is_active(state: SpeechState) -> bool:
    """Whether the assistant is still producing output the user can interrupt:
    either the LLM is streaming or the client is still playing TTS audio."""
    return bool(state.llm_task and not state.llm_task.done()) or state.client_playing


async def _interrupt_assistant(state: SpeechState) -> None:
    """Barge-in: stop whatever the assistant is currently producing."""
    if state.llm_task and not state.llm_task.done():
        await _cancel_llm(state)
    elif state.client_playing:
        await _stop_client_audio(state)


async def handle_speech_ws(
    *,
    sessions: dict[str, list[dict]],
    load_system_prompt,
    save_sessions,
    on_session_start=None,
) -> None:
    """Main WebSocket handler for speech mode. Call from a @app.websocket route."""
    args = websocket.args
    session_id = args.get("session_id") or secrets.token_urlsafe(16)

    if session_id not in sessions:
        sessions[session_id] = [{"role": "system", "content": load_system_prompt()}]
        if on_session_start:
            on_session_start(session_id)

    messages = sessions[session_id]
    state = SpeechState(session_id=session_id, messages=messages)
    state.dictation = args.get("dictation") == "1"

    await _send_json({"type": "session_start", "session_id": session_id})
    logger.info("Speech WS connected: session=%s dictation=%s", session_id, state.dictation)

    try:
        while True:
            msg = await websocket.receive()

            if isinstance(msg, bytes):
                # Audio data
                state.recorder.feed_audio(msg)
                now = time.monotonic()

                # Barge-in: if the assistant is still producing output (LLM
                # streaming or TTS still playing in the browser) and the user
                # speaks, interrupt. Require sustained, clearly-loud audio so
                # the assistant's own TTS echoing through the mic doesn't
                # falsely trigger a cancellation.
                if _assistant_is_active(state):
                    frame_rms = rms_int16(msg)
                    frame_ms = (len(msg) / (SAMPLE_RATE * BYTES_PER_SAMPLE)) * 1000.0 if msg else 0.0
                    loud = frame_rms >= BARGE_IN_THRESHOLD_RMS
                    state.barge_in_loud_ms = state.barge_in_loud_ms + frame_ms if loud else 0.0
                    logger.info(
                        "barge-in check: rms=%.0f thr=%.0f loud=%s sustained=%.0f/%.0fms (llm=%s tts=%s)",
                        frame_rms, BARGE_IN_THRESHOLD_RMS, loud,
                        state.barge_in_loud_ms, BARGE_IN_MIN_MS,
                        bool(state.llm_task and not state.llm_task.done()), state.client_playing,
                    )
                    if state.barge_in_loud_ms >= BARGE_IN_MIN_MS:
                        logger.info("Barge-in: sustained speech (rms=%.0f) — interrupting assistant", frame_rms)
                        state.barge_in_loud_ms = 0.0
                        await _interrupt_assistant(state)
                else:
                    state.barge_in_loud_ms = 0.0

                try:
                    events = state.chunker.feed(msg, now)
                    for event in events:
                        if event.type == "chunk" and event.audio:
                            await _transcribe_chunk(event.audio, state)
                        elif event.type == "pause":
                            await _handle_pause(state)
                except Exception:
                    logger.exception("Error processing audio frame")

            elif isinstance(msg, str):
                try:
                    data = json.loads(msg)
                except json.JSONDecodeError:
                    continue

                if data.get("type") == "playback_state":
                    # Frontend reports whether it is currently playing TTS audio.
                    state.client_playing = bool(data.get("playing"))
                    continue

                if data.get("type") == "stop":
                    # Finalize remaining audio
                    for event in state.chunker.flush():
                        if event.type == "chunk" and event.audio:
                            await _transcribe_chunk(event.audio, state)
                    # Trigger final LLM if there's pending transcript
                    if state.transcript_parts:
                        await _handle_pause(state)
                    # Wait for LLM to finish
                    if state.llm_task and not state.llm_task.done():
                        try:
                            await state.llm_task
                        except asyncio.CancelledError:
                            pass
                    break
            else:
                logger.warning("Unexpected WS message type: %s %r", type(msg).__name__, msg[:50] if isinstance(msg, (bytes, str)) else msg)
    except Exception:
        logger.exception("Speech WebSocket handler error")
    finally:
        try:
            state.recorder.finalize()
        except Exception:
            logger.exception("Failed to finalize audio recording")
        save_sessions()
