# Speech Agent

A chat agent with speech mode — talk to an LLM through your browser's microphone. Audio streams to the backend in real time, gets transcribed via ASR, and triggers streaming LLM responses on speech pauses. Supports interruption and continuation: if you start speaking while the LLM is responding, it stops and picks up where it left off after your next pause.

## Features

- **Text chat** — type messages, get streamed LLM responses with markdown rendering
- **Speech mode** — click the mic button to talk; audio is chunked, transcribed, and sent to the LLM
- **Interruption handling** — speak while the LLM is responding to cancel it; on the next pause it continues from where it stopped
- **Smart audio chunking** — ~2s chunks with silence-boundary detection for clean ASR input
- **Pause detection** — 0.4s of silence after speech triggers the LLM
- **Tool calling** — the LLM can execute bash, python, web search, and other tools (mode-dependent)
- **Session persistence** — conversations are saved to disk and restored on reload
- **User/Dev modes** — different system prompts, tools, and documentation per mode
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
| `LLM_MODEL` | no | `gpt-oss-120b` | Model for chat completions |
| `ASR_MODEL` | no | `whisper-1` | Model for speech recognition |
| `API_KEY` | no | `""` | Bearer token for client authentication |
| `AUTH_MODE` | no | `none` | Auth mode: `none`, `password`, `auth0` |
| `AUTH_PASSWORD` | no | — | Required when `AUTH_MODE=password` |

## Project Structure

```
src/
  main.py              # Quart app entry point, routes, session management
  speech.py            # WebSocket handler for speech mode
  audio_chunking.py    # Audio chunker with silence detection
  asr.py               # ASR client (OpenAI-compatible transcription API)
  streaming.py         # SSE streaming for text chat, tool orchestration
  tool_schemas.py      # Tool definitions per mode
  tool_executor.py     # Tool execution (bash, python, web tools)
  web_tools.py         # Web search and URL fetching
  runtime_logs.py      # In-memory log capture
  templates/
    index.html         # Page shell + chat panel HTML/CSS
  static/chat/
    chat.js            # Chat panel logic + speech mode integration
    speech.js          # SpeechSession class (WebSocket + mic capture)
    pcm-processor.js   # AudioWorklet for PCM capture
config/
  dev_system_prompt.json   # Dev mode AI system prompt
  user_system_prompt.json  # User mode AI system prompt
docs/
  dev_docs.md              # Developer documentation (injected into dev prompt)
  user_docs.md             # User documentation (injected into user prompt)
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

The `docker-compose.yml` passes through `PORT`, `LLM_BASE_URL`, `LLM_API_KEY`, `ASR_MODEL`, and `API_KEY` from your environment.
