"""Persistent long-term memory via a shared memorizer Context.

This is a thin layer on top of memorizer's memory model (see
docs/memorizer_integration.md). One process-wide **persistent** Context
(``MEMORIZER_DATA_DIR``, default ``data/memory``) accumulates durable memory —
distilled facts/preferences, goals, episodic summaries, workspace — across
sessions and **survives restarts** (memorizer reloads from disk on create).

Two operations, both designed to keep latency off the user-facing path:

- :func:`inject` — prepend the durable memory (NOT the live conversation, which
  the caller already holds) to the request messages, so the assistant recalls
  what it knows. Cheap, in-memory render.
- :func:`observe` — record a finished user/assistant turn into memory. The
  expensive compression/summarisation runs in memorizer's own **background**
  daemon threads (and we call it off the request path), so it never blocks the
  reply.

Disabled with ``MEMORY_ENABLED=false`` (then both ops are no-ops).
"""
from __future__ import annotations

import logging
import os
import threading

logger = logging.getLogger(__name__)

ENABLED = (os.environ.get("MEMORY_ENABLED") or "true").strip().lower() not in {"0", "false", "no", "off"}
DATA_DIR = os.environ.get("MEMORIZER_DATA_DIR") or "data/memory"
MAX_RECALL_CHARS = int(os.environ.get("MEMORY_MAX_RECALL_CHARS") or 6000)
# durable sections to recall (skip working/short_term — that's the live convo)
_RECALL_SECTIONS = ("long_term_factual", "model_goal", "long_term_episodic", "workspace")

_model = None
_lock = threading.RLock()


def _get():
    """The shared persistent memorizer Model (lazy; reloads memory from disk)."""
    global _model
    if _model is None:
        from memorizer import Model

        try:
            from . import llm_engine
        except ImportError:  # flat layout in the container (run as `python main.py`)
            import llm_engine
        _model = Model.create(
            model_id=llm_engine.LLM_MODEL,
            base_url=llm_engine.LLM_BASE_URL,
            api_key=llm_engine.LLM_API_KEY,
            system_prompt="",
            max_completion_tokens=llm_engine.LLM_MAX_COMPLETION_TOKENS,
            data_dir=DATA_DIR,
            persist=True,
        )
        logger.info("memory: persistent Context at %s", DATA_DIR)
    return _model


def _as_text(content) -> str:
    if isinstance(content, list):  # multimodal -> text parts only
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text").strip()
    return (content or "").strip() if isinstance(content, str) else ""


def recall_text() -> str:
    """The distilled durable memory as a single string (capped), or ''."""
    if not ENABLED:
        return ""
    try:
        with _lock:
            ctx = _get().context
            chunks = []
            for attr in _RECALL_SECTIONS:
                sec = getattr(ctx, attr, None)
                if sec is None:
                    continue
                for m in sec.to_messages():
                    c = _as_text(m.get("content"))
                    if c:
                        chunks.append(c)
        blob = "\n\n".join(chunks).strip()
        return blob[:MAX_RECALL_CHARS]
    except Exception as e:
        logger.warning("memory recall failed: %s", e)
        return ""


def inject(messages: list[dict]) -> list[dict]:
    """Return messages with durable memory inserted after the leading system
    prompt. Does not mutate the caller's list (so it isn't persisted into the
    session and re-injected next turn)."""
    blob = recall_text()
    if not blob:
        return messages
    mem = {"role": "system", "content":
           "Long-term memory about the user and earlier conversations — use it to "
           "personalise your answer; do not repeat it verbatim:\n\n" + blob}
    i = 0
    while i < len(messages) and messages[i].get("role") == "system":
        i += 1
    return messages[:i] + [mem] + messages[i:]


def observe(user_text, assistant_text) -> None:
    """Record one finished turn into long-term memory (background; non-blocking)."""
    if not ENABLED:
        return
    user_text = _as_text(user_text)
    assistant_text = _as_text(assistant_text)
    if not (user_text or assistant_text):
        return

    def run():
        try:
            with _lock:
                m = _get()
                if user_text:
                    m.append("user", user_text)       # fires bg compression
                if assistant_text:
                    m.append("assistant", assistant_text)
                m.update_workspace_async()             # bg workspace refresh
        except Exception as e:
            logger.warning("memory observe failed: %s", e)

    threading.Thread(target=run, daemon=True, name="memory-observe").start()


def last_user_text(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return _as_text(m.get("content"))
    return ""
