"""WebSocket handler for speech mode."""
from __future__ import annotations

import asyncio
import codecs
import json
import logging
import os
import secrets
import time

import httpx
from quart import websocket

try:
    from .asr import transcribe
    from .audio_chunking import AudioChunker, ChunkEvent
except ImportError:
    from asr import transcribe
    from audio_chunking import AudioChunker, ChunkEvent

logger = logging.getLogger(__name__)

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-oss-120b")
ASR_MODEL = os.environ.get("ASR_MODEL", "whisper-1")


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


async def _send_json(data: dict) -> None:
    await websocket.send(json.dumps(data))


async def _transcribe_chunk(chunk_audio: bytes, state: SpeechState) -> None:
    """Send audio chunk to ASR and forward transcript to client."""
    try:
        text = await transcribe(
            chunk_audio,
            base_url=LLM_BASE_URL,
            api_key=LLM_API_KEY,
            model=ASR_MODEL,
        )
        if text:
            state.transcript_parts.append(text)
            await _send_json({"type": "transcript", "text": text, "is_final": False})
    except Exception as exc:
        logger.error("ASR error: %s", exc)
        await _send_json({"type": "error", "message": f"ASR error: {exc}"})


async def _stream_llm(state: SpeechState) -> None:
    """Stream LLM response, accumulating partial_llm_response."""
    try:
        body = {
            "model": LLM_MODEL,
            "stream": True,
            "messages": list(state.messages),
        }
        decoder = codecs.getincrementaldecoder("utf-8")()
        text_buf = ""

        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST",
                f"{LLM_BASE_URL}/chat/completions",
                headers={
                    "Accept": "text/event-stream",
                    "Accept-Encoding": "identity",
                    "Authorization": f"Bearer {LLM_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=body,
            ) as resp:
                resp.raise_for_status()
                async for raw_chunk in resp.aiter_raw():
                    if not raw_chunk:
                        continue
                    text_buf += decoder.decode(raw_chunk)
                    while "\n\n" in text_buf:
                        event, text_buf = text_buf.split("\n\n", 1)
                        for line in event.splitlines():
                            if not line.startswith("data: "):
                                continue
                            payload = line[6:]
                            if payload == "[DONE]":
                                return
                            try:
                                chunk = json.loads(payload)
                            except json.JSONDecodeError:
                                continue
                            choices = chunk.get("choices") or []
                            if not choices:
                                continue
                            delta = choices[0].get("delta") or {}
                            content = delta.get("content") or ""
                            if content:
                                state.partial_llm_response += content
                                await _send_json({"type": "llm_token", "token": content})

        await _send_json({"type": "llm_done"})
        # Full response complete — add to message history
        state.messages.append({"role": "assistant", "content": state.partial_llm_response})
        state.partial_llm_response = ""

    except asyncio.CancelledError:
        # Partial response preserved in state.partial_llm_response
        logger.info("LLM streaming cancelled (user resumed speaking)")
        raise
    except Exception as exc:
        logger.error("LLM streaming error: %s", exc)
        await _send_json({"type": "error", "message": f"LLM error: {exc}"})


async def _handle_pause(state: SpeechState) -> None:
    """Called when a speech pause is detected — trigger LLM."""
    user_text = " ".join(state.transcript_parts).strip()
    if not user_text:
        return

    # Send final transcript marker
    await _send_json({"type": "transcript", "text": user_text, "is_final": True})

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

    try:
        while True:
            msg = await websocket.receive()

            if isinstance(msg, bytes):
                # Audio data
                now = time.monotonic()

                # If LLM is streaming and we get new audio with speech, cancel it
                if state.llm_task and not state.llm_task.done():
                    from .audio_chunking import is_silent, SILENCE_THRESHOLD_RMS
                    if not is_silent(msg, SILENCE_THRESHOLD_RMS):
                        await _cancel_llm(state)

                events = state.chunker.feed(msg, now)
                for event in events:
                    if event.type == "chunk" and event.audio:
                        await _transcribe_chunk(event.audio, state)
                    elif event.type == "pause":
                        await _handle_pause(state)

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
    finally:
        save_sessions()
