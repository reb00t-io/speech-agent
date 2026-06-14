# Memorizer integration

The LLM **request engine** is [memorizer](https://github.com/reb00t-io/memorizer)
(`reb00t-io/memorizer`), pinned to **v1.0.1** in `pyproject.toml`. All LLM
chat-completion calls go through it — there is no direct `httpx`/requests
fallback. ASR (`src/asr.py`) and TTS (`src/tts.py`) keep their own HTTP clients;
memorizer is the *LLM chat* engine only.

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

The conversation is **not** held in memorizer's `Context`: we pass explicit
`messages` per call (same prefix-sharing as before). So the engine currently has
no per-turn memory side effects (no `append()`, no workspace/compression).

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

## Packaging / the git submodule consumer

memorizer is a **public** repo, pinned as a release artifact:

```toml
"memorizer @ git+https://github.com/reb00t-io/memorizer.git@v1.0.1"
```

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
3. **Memory features unused.** Driving the conversation through a memorizer
   `Context` (long-term/working memory, compression, goals) is the natural next
   step but has latency/cost implications for voice — left opt-in for later.

## Tests

`test/test_llm_engine.py` covers the bridge (SSE → chunk dicts, tools/effort
pass-through, usage extraction, worker-exception propagation). All other suites
mock the engine via `patch("src.llm_engine.stream_chat", ...)`.
