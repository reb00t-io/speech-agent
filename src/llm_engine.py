"""Memorizer-backed LLM request engine.

Every LLM chat-completion call in the app goes through memorizer's ``Model`` —
the single request engine (no direct httpx fallback). A single shared ``Model``
instance is reused by all callers (the dual-LLM Router / System 1 / System 2, and
the text-chat tool loop) so they send the same request context and the inference
server's prefix cache is hit across calls.

Memorizer is synchronous (``requests``); this module bridges its streaming
``requests.Response`` to an async generator so the rest of the async app is
unchanged. See docs/memorizer_integration.md.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import AsyncGenerator

from memorizer import Model

logger = logging.getLogger(__name__)

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "") or "dummy"
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-oss-120b")
LLM_MAX_COMPLETION_TOKENS = int(os.environ.get("LLM_MAX_COMPLETION_TOKENS") or 1500)
# memorizer always sends ``reasoning_effort``; this is used when a caller does
# not specify one (e.g. System 2's deep pass).
LLM_DEFAULT_REASONING_EFFORT = os.environ.get("LLM_DEFAULT_REASONING_EFFORT") or "medium"

_model: Model | None = None


def get_model() -> Model:
    """Return the process-wide shared memorizer request engine.

    One instance is shared by every caller so the dual-LLM subsystems reuse the
    same request context (maximising the server-side prefix cache). The backing
    ``Context`` is not persisted and is not used to hold the conversation — we
    pass explicit ``messages`` per call — so the engine has no per-turn memory
    side effects. (Driving the conversation through memorizer's ``Context`` is a
    future enhancement; see docs/memorizer_integration.md.)
    """
    global _model
    if _model is None:
        _model = Model.create(
            model_id=LLM_MODEL,
            base_url=LLM_BASE_URL,
            api_key=LLM_API_KEY,
            system_prompt="",
            max_completion_tokens=LLM_MAX_COMPLETION_TOKENS,
            persist=False,
        )
        logger.info("LLM engine: memorizer Model(model_id=%s base_url=%s)", LLM_MODEL, LLM_BASE_URL)
    return _model


def reset_model() -> None:
    """Drop the cached Model (used by tests after patching config)."""
    global _model
    _model = None


async def stream_chat(
    messages: list[dict],
    *,
    reasoning_effort: str | None = None,
    tools: list[dict] | None = None,
    usage_out: dict | None = None,
) -> AsyncGenerator[dict, None]:
    """Stream an OpenAI chat completion via memorizer; yield parsed chunk dicts.

    Bridges memorizer's blocking ``requests`` stream to async with a worker
    thread + queue. If the consumer stops early (e.g. barge-in cancels the
    generator), the underlying response is closed so the worker unwinds.
    """
    model = get_model()
    effort = reasoning_effort or LLM_DEFAULT_REASONING_EFFORT
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    done = object()
    state: dict = {"response": None, "cancelled": False}

    def worker() -> None:
        try:
            response = model.stream(messages, tools=tools, reasoning_effort=effort)
            state["response"] = response
            for raw_line in response.iter_lines():
                if state["cancelled"]:
                    break
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                loop.call_soon_threadsafe(queue.put_nowait, chunk)
        except Exception as exc:  # surfaced to the async caller
            loop.call_soon_threadsafe(queue.put_nowait, exc)
        finally:
            try:
                if state["response"] is not None:
                    state["response"].close()
            except Exception:
                pass
            loop.call_soon_threadsafe(queue.put_nowait, done)

    loop.run_in_executor(None, worker)
    try:
        while True:
            item = await queue.get()
            if item is done:
                break
            if isinstance(item, Exception):
                raise item
            if usage_out is not None:
                usage = item.get("usage")
                if isinstance(usage, dict):
                    usage_out.update(usage)
            yield item
    finally:
        # Stop the worker if the consumer abandoned the stream (barge-in, error).
        state["cancelled"] = True
        response = state["response"]
        if response is not None:
            try:
                response.close()
            except Exception:
                pass


async def stream_content_tokens(
    messages: list[dict],
    *,
    reasoning_effort: str | None = None,
    usage_out: dict | None = None,
) -> AsyncGenerator[str, None]:
    """Convenience: stream just the assistant ``content`` token strings."""
    async for chunk in stream_chat(messages, reasoning_effort=reasoning_effort, usage_out=usage_out):
        choices = chunk.get("choices") or []
        if not choices:
            continue
        content = (choices[0].get("delta") or {}).get("content") or ""
        if content:
            yield content
