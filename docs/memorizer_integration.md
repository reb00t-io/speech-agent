# Memorizer integration

The LLM **request engine** and the **long-term memory store** are both
[memorizer](https://github.com/reb00t-io/memorizer) (`reb00t-io/memorizer`),
pinned to **`memorizer[store]@v1.0.3`** in `pyproject.toml` (the `store` extra
pulls in `qdrant-client`). All LLM chat-completion calls go through it — there is
no direct `httpx`/requests fallback. ASR (`src/asr.py`) and TTS (`src/tts.py`)
keep their own HTTP clients; memorizer is the *LLM chat* engine and the *memory*
engine.

Two memorizer `Model` instances, by design:

- **`src/llm_engine.py`** — a stateless **request engine** (`persist=False`, no
  store). The tool loop and the voice subsystems stream through this so they
  share one request context (server-side prefix cache).
- **`src/memory.py`** — a **persistent** Context that *also* owns the Qdrant
  retrieval store (`enable_memory=True`). It backs `inject` / `observe` /
  `recall` and is what the `recall` tool queries.

## How it's wired

- **`src/llm_engine.py`** is the single seam. It builds one shared memorizer
  `Model` (`get_model()`, configured from the `LLM_*` env vars) and exposes:
  - `stream_chat(messages, *, reasoning_effort=None, tools=None, usage_out=None)`
    — async generator of parsed OpenAI chunk dicts.
  - `stream_content_tokens(...)` — convenience: just the content token strings.
- **One shared instance.** `get_model()` returns a process-wide singleton, so the
  dual-LLM Router / System 1 / System 2 (and the text-chat tool loop) all stream
  through the same engine and send the same request context — maximising the
  inference server's prefix cache.
- **Sync → async bridge.** memorizer is synchronous (`requests`). `stream_chat`
  runs `Model.stream()` in a worker thread and bridges its blocking
  `iter_lines()` to an async queue. If the consumer stops early (barge-in), the
  underlying response is closed so the worker unwinds.
- **Consumers:** `src/dual_llm.py` (voice; `_llm_stream_tokens` →
  `stream_content_tokens`) and `src/streaming.py::generate_stream` (text chat;
  `llm_engine.stream_chat`, with tools).

The live conversation is **not** held in the request engine's `Context`: callers
pass explicit `messages` per call (prefix-sharing). The persistent `memory`
Model is the one with per-turn side effects (`append()` → background
compression, workspace refresh, and store writes).

## Memory + the `recall` tool

`src/memory.py` is the seam over memorizer's memory model. Three operations,
all kept off the user-facing latency path:

- **`inject(messages)`** — prepend the *consolidated* durable memory (facts,
  goal, workspace, episodic summaries, org block, recent cross-session turns) as
  a cheap, cacheable system prefix. Deduped against the live request and capped
  to `MEMORY_MAX_RECALL_CHARS`.
- **`observe(user, assistant)`** — record a finished turn. memorizer's own
  **background daemon threads** do the summarisation/compression and, with the
  store enabled, archive raw turns + episodic summaries into Qdrant under short
  ids (`m12`).
- **`recall(args)` / `recall_tool_schema()`** — the on-demand depth behind the
  compact prefix. With `MEMORY_STORE_ENABLED`, memorizer keeps a Qdrant store
  with hybrid (dense `qwen3-embedding-4b` + BM25, RRF-fused) search. The model
  pulls memory with `recall(query="…")` or follows an inline `[m12]` pointer
  with `recall(id="m12")` to get the original detail it was compressed from.

### Who owns the tool loop (the design decision)

memorizer provides the `recall` tool's **schema** (`RECALL_TOOL`) and its
**executor** (`execute_recall`) — the *capability* — but **not** the agentic
loop. This app keeps owning the loop:

- `tool_schemas.get_tools()` merges `memory.recall_tool_schema()` into the
  advertised tool set (alongside builtins + plugins).
- `streaming.generate_stream()` runs the existing multi-round tool loop and
  streams through `llm_engine`.
- `tool_executor.execute_tool_call()` routes a `recall` call to
  `memory.recall()` (run in a thread — the store is synchronous), which calls
  memorizer's `execute_recall` against the persistent model's stores.

memorizer's own `Model.generate()` / chat `stream_completion()` are non-stream
reference drivers that only know about `recall`; using them would discard this
app's streaming, frontend/backend tool split, multimodal input, and other tools.
So the loop stays here. The **voice path** (`dual_llm`) has no tool loop and is
latency-sensitive, so it gets `inject()` but **no** `recall` tool.

### Embeddings + Qdrant

The store embeds via the same OpenAI-compatible endpoint as chat
(`LLM_BASE_URL/embeddings`, model `EMBEDDING_MODEL`, `EMBEDDING_DIMENSIONS`). No
extra service is required. Qdrant is **local on-disk** by default (under
`MEMORIZER_DATA_DIR/qdrant`, persisted via the `data/` volume); set
`QDRANT_LOCATION` to a server URL for a shared/org deployment. If the store
can't be built (extra missing, Qdrant unreachable) `memory.py` falls back to a
store-less Model so `inject`/`observe` keep working; recall returns a graceful
"unavailable" message.

### Organization memory

`MEMORY_ORG_ENABLED` turns on shared, role-gated cross-agent memory (governed by
`MEMORY_ORG_PROFILE`, with `MEMORY_ROLE` / `MEMORY_ORG_ROLES` /
`MEMORY_ORG_WRITER_ROLES`). It is **off in this deployment** but fully wired so
downstream integrators can enable it. Personal recall is scoped by
`MEMORY_MEMBER_ID` (empty = one shared personal store).

## Config

`src/llm_engine.py` reads (env):

| Var | Default | Purpose |
|---|---|---|
| `LLM_BASE_URL` | — | OpenAI-compatible endpoint (memorizer `base_url`) |
| `LLM_API_KEY` | `dummy` | bearer key |
| `LLM_MODEL` | `gpt-oss-120b` | model id |
| `LLM_MAX_COMPLETION_TOKENS` | `1500` | per-call completion budget |
| `LLM_DEFAULT_REASONING_EFFORT` | `medium` | effort when a caller doesn't set one (e.g. System 2's deep pass) |

`DUAL_LLM_ENABLED` (in `src/speech.py`) still toggles dual vs single voice
streaming.

`src/memory.py` reads (env):

| Var | Default | Purpose |
|---|---|---|
| `MEMORY_ENABLED` | `true` | master switch for the memory layer (inject/observe/recall) |
| `MEMORIZER_DATA_DIR` | `data/memory` | persistent Context + on-disk Qdrant location |
| `MEMORY_MAX_RECALL_CHARS` | `6000` | char budget for the injected memory prefix |
| `MEMORY_STORE_ENABLED` | `true` | attach the Qdrant store + `recall` tool |
| `EMBEDDING_MODEL` | `qwen3-embedding-4b` | embedding model (`LLM_BASE_URL/embeddings`) |
| `EMBEDDING_DIMENSIONS` | `1024` | requested embedding vector size |
| `QDRANT_LOCATION` | — | Qdrant server URL; empty = local on-disk |
| `MEMORY_ORG_ENABLED` | `false` | shared, role-gated org memory |
| `MEMORY_MEMBER_ID` | — | scopes personal recall |
| `MEMORY_ROLE` | — | this member's org role |
| `MEMORY_ORG_PROFILE` | — | org extraction-rules doc (path/literal); empty = template |
| `MEMORY_ORG_ROLES` | — | comma-separated known org roles |
| `MEMORY_ORG_WRITER_ROLES` | — | comma-separated roles allowed to write org facts |

## Packaging / the git submodule consumer

memorizer is a **public** repo, pinned as a release artifact:

```toml
"memorizer[store] @ git+https://github.com/reb00t-io/memorizer.git@v1.0.3"
```

The `store` extra adds `qdrant-client` (pure-Python wheel + grpcio/pydantic),
needed for the retrieval store and the `recall` tool.

A project that vendors this app as a git submodule gets a working dependency by
installing the app (`pip install -e <submodule>`), which resolves the pin — no
credentials needed (public repo). **`git` must be available wherever the install
runs**: the `Dockerfile` installs it (and removes it) around `pip install`, since
`python:3.13-slim` ships without git.

To bump memorizer: change the `@vX.Y.Z` tag in `pyproject.toml` (and reinstall).

## Known limitations / follow-ups

These were handled in our `httpx` payload before but are not expressed through
memorizer's request API, so they are currently dropped:

1. **Kimi reasoning toggle.** We used to translate `reasoning_effort` into
   `chat_template_kwargs.thinking` for Kimi models. memorizer sends
   `reasoning_effort` directly. v1.0.1 added a `thinking` parameter to
   `Model.create(...)`, so Kimi thinking can be set at model-construction time —
   wire `LLM_MODEL`-based detection into `get_model()` if per-deployment Kimi
   control is needed. (Per-call fast/slow toggling would need a memorizer API.)
2. **`response_format`.** The Router used `{"type":"json_object"}`. memorizer's
   `stream()` doesn't take it; the Router already falls back to extracting the
   first number, so this is cosmetic.
3. **Memory features.** Now wired (see "Memory + the `recall` tool" above):
   persistent Context + Qdrant retrieval store + the `recall` tool for text
   chat. Still open: the **voice** path stays prefix-only (no `recall` tool) for
   latency, and per-call Kimi fast/slow toggling (item 1) is unrelated.

## Tests

`test/test_llm_engine.py` covers the bridge (SSE → chunk dicts, tools/effort
pass-through, usage extraction, worker-exception propagation). `test/test_memory.py`
covers inject/observe, the `recall` tool schema gating, and recall execution
against a fake store; `test/test_plugins.py` covers `get_tools()` including
`recall` and the executor routing `recall` → `memory.recall`. All other suites
mock the engine via `patch("src.llm_engine.stream_chat", ...)`.
