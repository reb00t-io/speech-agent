# Speech Agent

A chat agent with speech mode — talk to an LLM through your browser's microphone. Audio streams to the backend in real time, gets transcribed via ASR, and triggers streaming LLM responses on speech pauses. Supports interruption and continuation: if you start speaking while the LLM is responding, it stops and picks up where it left off after your next pause.

A single ChatGPT-style assistant: one full-page chat interface with text, image upload, dictation, and live voice conversation. There is no user/dev mode — every conversation uses the same system prompt and the full tool set.

## Features

- **Text chat** — type messages, get streamed LLM responses with markdown rendering
- **Voice conversation** — the round voice button starts a live, hands-free conversation (real-time ASR + spoken TTS replies); start speaking again to interrupt it
- **Dictation** — the mic button transcribes speech into the text box (`POST /v1/transcribe`) so you can edit before sending
- **Image upload** — attach images with the "+" button; they are sent to the model as base64 vision input
- **Web search shortcut** — toggle to instruct the model to use the `web_search` tool before answering
- **Deep research shortcut** — toggle to have the model research thoroughly and deliver a downloadable PDF report (via the `publish_document` tool)
- **Tool calling** — `web_search`, `fetch_url`, `python`, `bash`, `get_logs`, and `publish_document` (Markdown → PDF download link)
- **Interruption handling** — speak while the assistant is responding (text or voice) to cancel it; on the next pause it continues from where it stopped
- **Session persistence** — conversations are saved to disk and restored on reload
- **Auth** — optional password or API key authentication

## Quick Start

```bash
# 1. Allow direnv to load the environment (creates venv, installs deps)
direnv allow

# 2. Run locally
python src/main.py
# → http://localhost:$PORT

# 3. Or run with Docker
./scripts/build.sh
docker compose up
```

## Environment Variables

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `PORT` | yes | — | Server port |
| `LLM_BASE_URL` | yes | — | Base URL for LLM and ASR APIs (OpenAI-compatible) |
| `LLM_API_KEY` | no | `""` | API key for the LLM/ASR backend |
| `LLM_MODEL` | no | `gpt-oss-120b` | Model for chat completions (image input requires a vision-capable model, e.g. Kimi) |
| `LLM_MAX_COMPLETION_TOKENS` | no | `1500` | Per-call completion-token budget for the memorizer request engine |
| `LLM_DEFAULT_REASONING_EFFORT` | no | `medium` | `reasoning_effort` used when a caller doesn't set one |
| `ASR_MODEL` | no | `whisper-1` | Model for speech recognition (voice + dictation) |
| `ASR_LANGUAGE` | no | `""` | Optional ASR language hint (empty = auto-detect) |
| `API_KEY` | no | `""` | Bearer token for client authentication |
| `AUTH_MODE` | no | `none` | Auth mode: `none`, `password`, `auth0` |
| `AUTH_PASSWORD` | no | — | Required when `AUTH_MODE=password` |
| `MISTRAL_API_KEY` | no | `""` | Enables TTS for voice conversation when set |
| `BARGE_IN_THRESHOLD_RMS` | no | `800` | Min mic loudness (RMS) to interrupt the assistant; raise if its TTS echoes back through the mic |
| `BARGE_IN_MIN_MS` | no | `400` | Min duration of sustained loud audio before a barge-in fires (filters brief echo blips) |
| `DUAL_LLM_ENABLED` | no | `true` | Use the dual-LLM ("thinking fast and slow") orchestration for voice replies; set `false` for a single plain LLM stream |
| `SESSIONS_PATH` | no | `data/sessions.json` | Where chat history is persisted |
| `REQUEST_LOG_PATH` | no | `data/requests.log` | Request/response log file |
| `DOWNLOADS_DIR` | no | `data/downloads` | Where `publish_document` PDFs are stored and served from `/download/<token>.pdf` |
| `SYSTEM_PROMPT_PATH` | no | `config/system_prompt.json` | Override the system prompt with a mounted file (lets a deployment inject its own identity/instructions) |
| `AGENT_PLUGINS` | no | `""` | Comma-separated Python module names or file paths that contribute extra tools — see [Tool plugins](#tool-plugins) |

## Tool plugins

This is a **generic** agent. The core ships only domain-agnostic tools
(`web_search`, `fetch_url`, `python`, `bash`, `get_logs`, `publish_document`)
and a generic system prompt — run it standalone and it just works.

A *deployment* injects its own domain tools and identity without forking:

- **Tools** — set `AGENT_PLUGINS` to a comma-separated list of Python module
  names or file paths. Each plugin module exposes `register(registry)` and
  calls `registry.add_tool(schema, handler)`. A handler has the signature
  `async def handler(session, args: dict) -> dict` (see `src/plugins.py`).
- **Identity / instructions** — point `SYSTEM_PROMPT_PATH` at a mounted prompt
  file to replace the default prompt.

Example: the [netmon](https://github.com/reb00t-io/netmon) home hub mounts a
`home_tools.py` plugin (smart-home control over its HTTP API) and a "Verity"
system prompt at deploy time; neither lives in this repo.

## Project Structure

```
src/
  main.py              # Quart app entry point, routes, session management
  speech.py            # WebSocket handler for speech mode
  audio_chunking.py    # Audio chunker with silence detection
  asr.py               # ASR client (OpenAI-compatible transcription API)
  streaming.py         # SSE streaming for text chat, tool orchestration
  documents.py         # Markdown → PDF publishing + secure download paths
  tool_schemas.py      # Builtin (generic) tool definitions + plugin merge
  tool_executor.py     # Tool execution (bash, python, web tools, publish_document) + plugin dispatch
  plugins.py           # Tool-plugin seam: ToolRegistry + AGENT_PLUGINS loader
  web_tools.py         # Web search and URL fetching
  runtime_logs.py      # In-memory log capture
  templates/
    index.html         # Full-page chat UI (HTML/CSS)
  static/chat/
    chat.js            # Chat UI logic: streaming, images, dictation, voice, chips
    speech.js          # SpeechSession class (WebSocket + mic capture for voice)
    pcm-processor.js   # AudioWorklet for PCM capture
config/
  system_prompt.json   # The default (generic) system prompt; override via SYSTEM_PROMPT_PATH
docs/
  app_docs.md              # App documentation (injected into the system prompt)
  speech_mode_spec.md      # High-level speech mode spec
  speech_mode_detailed_spec.md  # Detailed speech mode spec
scripts/
  venv.rc              # Virtual env setup
  build.sh             # Docker build
  deploy.sh            # Remote deployment via SSH
test/
  test_main.py             # Text chat backend tests
  test_audio_chunking.py   # Audio chunking + silence detection tests
  test_asr.py              # ASR client tests
  test_speech.py           # WebSocket speech handler tests
  test_e2e_speech.py       # End-to-end speech pipeline tests
  e2e.sh                   # Docker-based smoke test
```

## How Speech Mode Works

```
Browser mic → AudioWorklet (PCM capture) → WebSocket binary frames
    → Backend AudioChunker (~2s chunks, silence-boundary cuts)
    → ASR (OpenAI /v1/audio/transcriptions)
    → Transcript streamed back to browser
    → On 0.4s pause: transcript sent to LLM
    → LLM response streamed token-by-token to browser
    → If user speaks during LLM response: cancel, preserve partial
    → On next pause: LLM continues from where it stopped
```

## Testing

```bash
# Run all tests
pytest

# Run specific test suites
pytest test/test_audio_chunking.py   # 21 tests — chunking, silence, pauses
pytest test/test_asr.py              # 9 tests — WAV encoding, API calls
pytest test/test_speech.py           # 6 tests — WebSocket lifecycle
pytest test/test_e2e_speech.py       # 8 tests — full pipeline with generated audio
pytest test/test_main.py             # 33 tests — text chat, auth, sessions, tools
```

## Docker

```bash
./scripts/build.sh
docker compose up
```

The `docker-compose.yml` passes through `PORT`, `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`, `ASR_MODEL`, `ASR_LANGUAGE`, `API_KEY`, `MISTRAL_API_KEY`, and the TTS settings from your environment, and persists sessions, request logs, and published PDFs under the mounted `/data` volume.
