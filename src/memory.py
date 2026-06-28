"""Persistent long-term memory via a shared memorizer Context + Qdrant store.

This is a thin layer on top of memorizer's memory model (see
docs/memorizer_integration.md). One process-wide **persistent** Context
(``MEMORIZER_DATA_DIR``, default ``data/memory``) accumulates durable memory —
distilled facts/preferences, goals, episodic summaries, workspace — across
sessions and **survives restarts** (memorizer reloads from disk on create).

With the memorizer ``store`` extra the same Model also owns a **Qdrant-backed
retrieval store** (hybrid dense + BM25 search). Compression archives raw turns
and episodic summaries into it under short ids (``m12``); the model pulls them
back on demand through the :data:`recall` tool. So there are three operations:

- :func:`inject` — prepend the *consolidated* durable memory (facts, goal,
  workspace, org block, …) to the request messages as a cheap, cacheable
  prefix. NOT the live conversation, which the caller already holds.
- :func:`recall` — execute the model's ``recall`` tool call (hybrid search or
  id fetch) against the store. This is the on-demand depth behind the compact
  prefix; the speech-agent advertises :func:`recall_tool_schema` in its own tool
  loop and routes the call here (the agentic loop stays in the app — memorizer
  only provides the tool schema + executor).
- :func:`observe` — record a finished user/assistant turn into memory. The
  expensive compression/summarisation (and the store writes) run in memorizer's
  own **background** daemon threads, off the request path, so they never block
  the reply.

Disabled with ``MEMORY_ENABLED=false`` (then every op is a no-op). The retrieval
store is on by default when memory is enabled; shared **organization** memory is
opt-in via ``MEMORY_ORG_ENABLED=true`` (supported for integrators, off here).
"""
from __future__ import annotations

import logging
import os
import threading

logger = logging.getLogger(__name__)


def _flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def _csv(name: str) -> list[str]:
    return [x.strip() for x in (os.environ.get(name) or "").split(",") if x.strip()]


ENABLED = _flag("MEMORY_ENABLED", True)
DATA_DIR = os.environ.get("MEMORIZER_DATA_DIR") or "data/memory"
MAX_RECALL_CHARS = int(os.environ.get("MEMORY_MAX_RECALL_CHARS") or 6000)

# Retrieval store (Qdrant hybrid search + the `recall` tool). On by default when
# memory is enabled; needs the memorizer `store` extra (qdrant-client).
STORE_ENABLED = _flag("MEMORY_STORE_ENABLED", True)
# Shared, role-gated organization memory. Off here; supported for integrators.
ORG_ENABLED = _flag("MEMORY_ORG_ENABLED", False)
ORG_PROFILE = os.environ.get("MEMORY_ORG_PROFILE") or None  # path / literal / None=template
MEMBER_ID = os.environ.get("MEMORY_MEMBER_ID") or None       # scopes personal recall
ROLE = os.environ.get("MEMORY_ROLE") or None                 # gates org read/write

# Embeddings for the store: same OpenAI-compatible endpoint as chat (LLM_BASE_URL).
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL") or "qwen3-embedding-4b"
EMBEDDING_DIMENSIONS = int(os.environ.get("EMBEDDING_DIMENSIONS") or 1024)
# Qdrant server URL (e.g. http://localhost:6333); empty → local on-disk store
# under DATA_DIR/qdrant (survives restarts via the data/ volume).
QDRANT_LOCATION = os.environ.get("QDRANT_LOCATION") or None

# Consolidated sections injected as the cacheable prefix. The shared Context
# spans every session, so short_term/working hold turns from OTHER sessions too;
# we dedupe against the live request so the current session isn't duplicated.
# Raw/episodic detail lives in the store behind `recall`, not here.
_RECALL_SECTIONS = ("org", "long_term_factual", "model_goal", "long_term_episodic",
                    "workspace", "short_term", "working")

_model = None
_lock = threading.RLock()


def _create_model(*, store: bool, org: bool):
    from memorizer import Model

    try:
        from . import llm_engine
    except ImportError:  # flat layout in the container (run as `python main.py`)
        import llm_engine
    return Model.create(
        model_id=llm_engine.LLM_MODEL,
        base_url=llm_engine.LLM_BASE_URL,
        api_key=llm_engine.LLM_API_KEY,
        system_prompt="",
        max_completion_tokens=llm_engine.LLM_MAX_COMPLETION_TOKENS,
        data_dir=DATA_DIR,
        persist=True,
        enable_memory=store,
        enable_org=org,
        org_profile=ORG_PROFILE,
        org_roles=_csv("MEMORY_ORG_ROLES") or None,
        org_writer_roles=_csv("MEMORY_ORG_WRITER_ROLES") or None,
        member_id=MEMBER_ID,
        role=ROLE,
        embedding_model=EMBEDDING_MODEL,
        embedding_dimensions=EMBEDDING_DIMENSIONS,
        qdrant_location=QDRANT_LOCATION,
    )


def _get():
    """The shared persistent memorizer Model (lazy; reloads memory from disk).

    The retrieval store is attached when ``MEMORY_STORE_ENABLED`` and the
    ``store`` extra are present; if store init fails (missing extra, unreachable
    Qdrant) we fall back to a store-less Model so inject/observe keep working.
    """
    global _model
    if _model is None:
        try:
            _model = _create_model(store=STORE_ENABLED, org=ORG_ENABLED)
            logger.info(
                "memory: persistent Context at %s (store=%s org=%s)",
                DATA_DIR, _model.memory is not None, _model.org_memory is not None,
            )
        except Exception as e:
            if not (STORE_ENABLED or ORG_ENABLED):
                raise
            logger.warning(
                "memory: retrieval store unavailable (%s); continuing without it", e
            )
            _model = _create_model(store=False, org=False)
            logger.info("memory: persistent Context at %s (store=False)", DATA_DIR)
    return _model


def reset_model() -> None:
    """Drop the cached Model, closing the Qdrant client first (local mode holds a
    directory lock, so it must be released before the path is reopened)."""
    global _model
    if _model is not None:
        for store in (getattr(_model, "memory", None), getattr(_model, "org_memory", None)):
            if store is not None:
                store.close()
    _model = None


def _as_text(content) -> str:
    if isinstance(content, list):  # multimodal -> text parts only
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text").strip()
    return (content or "").strip() if isinstance(content, str) else ""


def recall_messages() -> list[dict]:
    """All remembered messages across sections (oldest→newest), or []."""
    if not ENABLED:
        return []
    try:
        with _lock:
            ctx = _get().context
            out: list[dict] = []
            for attr in _RECALL_SECTIONS:
                sec = getattr(ctx, attr, None)
                if sec is not None:
                    out.extend(sec.to_messages())
        return out
    except Exception as e:
        logger.warning("memory recall failed: %s", e)
        return []


def inject(messages: list[dict]) -> list[dict]:
    """Return messages with recalled memory inserted after the leading system
    prompt — deduped against the live request (so the current session isn't
    repeated) and capped to the most recent MAX_RECALL_CHARS. Does not mutate
    the caller's list."""
    recalled = recall_messages()
    if not recalled:
        return messages
    seen = {_as_text(m.get("content")) for m in messages}
    picked: list[dict] = []
    for m in recalled:
        c = _as_text(m.get("content"))
        if c and c not in seen:
            seen.add(c)
            picked.append(m)
    if not picked:
        return messages
    # keep the most recent within the char budget
    kept: list[dict] = []
    total = 0
    for m in reversed(picked):
        c = _as_text(m.get("content"))
        if total + len(c) > MAX_RECALL_CHARS:
            break
        kept.append(m)
        total += len(c)
    kept.reverse()
    header = {"role": "system", "content":
              "Recalled memory from earlier conversations with this user — use it to "
              "personalise your answer; do not repeat it verbatim:"}
    i = 0
    while i < len(messages) and messages[i].get("role") == "system":
        i += 1
    return messages[:i] + [header] + kept + messages[i:]


def recall_tool_schema() -> dict | None:
    """The ``recall`` tool schema to advertise in the app's tool loop, or None.

    Memorizer owns the tool's *definition* and *execution* (the capability); the
    speech-agent keeps owning the agentic loop and merges this into its own tool
    list (see tool_schemas.get_tools). Returns None when memory or the store is
    disabled, so callers can simply skip it."""
    if not (ENABLED and STORE_ENABLED):
        return None
    try:
        from memorizer.store import RECALL_TOOL  # static schema; no store built here
    except Exception as e:  # store extra not installed
        logger.warning("memory: recall tool unavailable: %s", e)
        return None
    return RECALL_TOOL


def recall(args: dict) -> str:
    """Execute one ``recall`` tool call against the store; return tool-result text.

    Routed here by tool_executor when the model calls ``recall(query=…)`` or
    ``recall(id=…)``. Scoped to the configured member id / role. Synchronous
    (the store is ``requests``/Qdrant); call from a thread on the async path."""
    if not ENABLED:
        return "Memory is not available."
    try:
        with _lock:
            m = _get()
            if m.memory is None and m.org_memory is None:
                return "Memory is not available."
            from memorizer.store import execute_recall
            return execute_recall(
                args,
                agent_store=m.memory,
                org_store=m.org_memory,
                member_id=m.member_id,
                role=m.role,
                default_limit=getattr(m, "recall_limit", 5),
            )
    except Exception as e:
        logger.warning("memory recall failed: %s", e)
        return "Memory recall failed."


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
