"""WebSocket handler for speech mode."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import time

import httpx
from quart import websocket

try:
    from .asr import transcribe
    from .audio_chunking import AudioChunker, ChunkEvent, is_silent, SILENCE_THRESHOLD_RMS, SILENCE_WINDOW_BYTES
    from .audio_recording import AudioRecorder
    from .dual_llm import dual_stream
    from .tts import split_sentences, synthesize as tts_synthesize, wav_to_base64
except ImportError:
    from asr import transcribe
    from audio_chunking import AudioChunker, ChunkEvent, is_silent, SILENCE_THRESHOLD_RMS, SILENCE_WINDOW_BYTES
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
TTS_BASE_URL = os.environ.get("TTS_BASE_URL", "")
TTS_LANGUAGE = os.environ.get("TTS_LANGUAGE", os.environ.get("ASR_LANGUAGE", "en"))
TTS_SPEAKER = int(os.environ.get("TTS_SPEAKER", "0"))

logger.info("Speech module: LLM_BASE_URL=%s ASR_MODEL=%s LLM_MODEL=%s ASR_LANGUAGE=%s", LLM_BASE_URL, ASR_MODEL, LLM_MODEL, ASR_LANGUAGE or "auto")


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


async def _tts_sentence(text: str, index: int) -> None:
    """Synthesize a sentence and send audio to the client."""
    try:
        wav_bytes = await tts_synthesize(
            text,
            base_url=TTS_BASE_URL,
            language=TTS_LANGUAGE,
            speaker=TTS_SPEAKER,
        )
        await _send_json({
            "type": "tts_audio",
            "index": index,
            "audio_base64": wav_to_base64(wav_bytes),
        })
    except Exception as exc:
        logger.error("TTS error for sentence %d: %s", index, exc)


async def _stream_llm(state: SpeechState) -> None:
    """Stream LLM response via dual-LLM system, with optional TTS."""
    tts_enabled = bool(TTS_BASE_URL)
    tts_tasks: list[asyncio.Task] = []
    sentence_buf = ""
    sentence_index = 0

    try:
        async for token in dual_stream(
            messages=list(state.messages),
            model=LLM_MODEL,
            base_url=LLM_BASE_URL,
            api_key=LLM_API_KEY,
        ):
            state.partial_llm_response += token
            await _send_json({"type": "llm_token", "token": token})

            # TTS: accumulate and send complete sentences
            if tts_enabled:
                sentence_buf += token
                sentences = split_sentences(sentence_buf)
                if len(sentences) > 1:
                    for s in sentences[:-1]:
                        task = asyncio.create_task(_tts_sentence(s, sentence_index))
                        tts_tasks.append(task)
                        sentence_index += 1
                    sentence_buf = sentences[-1]

        # TTS: flush remaining text
        if tts_enabled and sentence_buf.strip():
            task = asyncio.create_task(_tts_sentence(sentence_buf.strip(), sentence_index))
            tts_tasks.append(task)

        # Wait for all TTS tasks to finish
        if tts_tasks:
            await asyncio.gather(*tts_tasks, return_exceptions=True)

        # Full response complete — add to message history
        state.messages.append({"role": "assistant", "content": state.partial_llm_response})
        state.partial_llm_response = ""
        await _send_json({"type": "llm_done"})

    except asyncio.CancelledError:
        for task in tts_tasks:
            if not task.done():
                task.cancel()
        logger.info("LLM streaming cancelled (user resumed speaking)")
        raise
    except Exception as exc:
        logger.error("LLM streaming error: %s", exc)
        await _send_json({"type": "error", "message": f"LLM error: {exc}"})


async def _handle_pause(state: SpeechState) -> None:
    """Called when a speech pause is detected — trigger LLM."""
    # Try merging a short trailing chunk with the previous one
    await _maybe_merge_short_chunk(state)

    user_text = " ".join(state.transcript_parts).strip()
    logger.info("Pause detected: transcript_parts=%d text=%r", len(state.transcript_parts), user_text[:100] if user_text else "")
    if not user_text:
        return

    # Send final marker (no text — frontend already has it from chunk transcripts)
    await _send_json({"type": "transcript_done"})

    # Reset chunk tracking for next utterance
    state.prev_chunk_audio = None
    state.last_chunk_audio = None

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
        await _send_json({
            "type": "llm_cancelled",
            "partial_response": state.partial_llm_response,
        })
        state.llm_task = None


async def handle_speech_ws(
    *,
    sessions: dict[str, list[dict]],
    session_modes: dict[str, str],
    load_system_prompt,
    save_sessions,
    on_session_start=None,
) -> None:
    """Main WebSocket handler for speech mode. Call from a @app.websocket route."""
    args = websocket.args
    session_id = args.get("session_id") or secrets.token_urlsafe(16)
    mode = args.get("mode", "user")

    if session_id not in sessions:
        sessions[session_id] = [{"role": "system", "content": load_system_prompt(mode)}]
        session_modes[session_id] = mode
        if on_session_start:
            on_session_start(session_id, mode)

    messages = sessions[session_id]
    state = SpeechState(session_id=session_id, messages=messages)

    await _send_json({"type": "session_start", "session_id": session_id})
    logger.info("Speech WS connected: session=%s mode=%s", session_id, mode)

    try:
        while True:
            msg = await websocket.receive()

            if isinstance(msg, bytes):
                # Audio data
                state.recorder.feed_audio(msg)
                now = time.monotonic()

                # If LLM is streaming and we get new audio with speech, cancel it
                if state.llm_task and not state.llm_task.done():
                    if not is_silent(msg, SILENCE_THRESHOLD_RMS):
                        await _cancel_llm(state)

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
