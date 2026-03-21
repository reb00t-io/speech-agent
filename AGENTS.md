# AGENTS.md

## 1. Mission & Priorities
**Role of the agent in this repository:**
- Develop and maintain a speech-enabled chat agent web application with real-time audio streaming, ASR transcription, and LLM interaction

**Decision priority order:**
- correctness > security > maintainability > performance > speed

**Global constraints or goals:**
- All LLM and ASR calls go through OpenAI-compatible APIs at `LLM_BASE_URL`
- Speech mode uses WebSocket for real-time bidirectional audio/text streaming
- Text chat mode uses SSE (server-sent events) for streaming responses
- Both modes share the same session store and message history

## 2. Executable Commands (Ground Truth)
All commands listed here must work.

- Install / setup:
  - `direnv allow` or `source scripts/venv.rc`
  - `pip install -e '.[dev]'`
- Dev server:
  - `python src/main.py` (requires `PORT` and `LLM_BASE_URL` env vars)
- Lint:
  - N/A (no linter configured)
- Format:
  - N/A (no formatter configured)
- Type check:
  - N/A (no type checker configured)
- Unit tests:
  - `pytest`
- Integration / e2e tests:
  - `pytest test/test_e2e_speech.py` (speech pipeline e2e with generated audio)
  - `./test/e2e.sh` (Docker-based smoke test)

## 3. Repository Map
**High-level structure:**
- `src/` — Application source (Python backend + JS frontend)
- `src/static/chat/` — Frontend JavaScript modules (chat, speech, audio worklet)
- `src/templates/` — Jinja2 HTML templates
- `config/` — System prompts for user/dev modes
- `docs/` — Documentation and specs
- `test/` — Pytest test suite
- `scripts/` — Build, deploy, and dev scripts

**Entry points:**
- Backend: `src/main.py` (Quart app, all HTTP + WebSocket routes)
- Frontend: `src/templates/index.html` → `src/static/chat/chat.js`
- Speech WebSocket: `src/speech.py` (handler for `/ws/speech`)

**Key configuration locations:**
- `config/user_system_prompt.json` — User mode AI system prompt template
- `config/dev_system_prompt.json` — Dev mode AI system prompt template
- `.envrc` — Environment variables (PORT, LLM_MODEL, ASR_MODEL, AUTH_MODE)
- `docker-compose.yml` — Container environment passthrough

## 4. Definition of Done
For any change, the following must hold:
- [ ] All existing tests pass (`pytest` — 77 tests)
- [ ] New tests added for new functionality
- [ ] No regressions in text chat mode when modifying speech mode (and vice versa)
- [ ] Docs updated if behavior or environment variables change

## 5. Code Style & Conventions (Repo-Specific)
Only list conventions that are easy to get wrong.

- Language(s) + version(s):
  - `Python 3.13` (backend), `ES2022+` modules (frontend)
- Formatter:
  - None configured; follow existing style (4-space indent Python, 4-space indent JS)
- Naming conventions:
  - Python: `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_CASE` for constants
  - JS: `camelCase` for functions/variables, `PascalCase` for classes
  - Private functions prefixed with `_` in Python
- Error handling pattern:
  - Backend: try/except with `logger.error()`, return JSON error responses
  - WebSocket: catch errors, send `{"type": "error", "message": ...}` to client
  - Frontend: catch in `try/catch`, display in UI
- Logging rules:
  - Use `logging.getLogger(__name__)` per module
  - Log errors and significant state changes; don't log audio data or full message contents
  - Request/response logging to `REQUEST_LOG_PATH` with sensitive header redaction

## 6. Boundaries & Guardrails
The agent must **not**:
- Commit `.envrc.local` or any file containing secrets
- Modify `test/test_main.py` existing tests without explicit request (they are the regression baseline)
- Remove the `os.environ.pop("API_KEY", None)` from test files (required for test isolation when `API_KEY` is set in shell)

When unsure:
- Prefer the smallest possible change
- Leave a TODO with context rather than guessing

## 7. Security & Privacy Constraints
- Sensitive data locations:
  - `.envrc.local` — API keys and passwords (gitignored)
  - `data/sessions.json` — Chat history (gitignored via volume mount)
  - `data/requests.log` — Request logs with redacted auth headers
- Redaction / handling rules:
  - Authorization, Cookie, Set-Cookie headers are redacted in request logs
  - Request body truncated to `REQUEST_LOG_BODY_LIMIT` (default 20000 chars)
- Approved crypto / storage patterns:
  - Session secret derived from `AUTH_PASSWORD` via SHA-256
- Threat model notes:
  - WebSocket `/ws/speech` does not require auth (same as the chat panel — relies on `AUTH_MODE` page-level gate)
  - Audio data is processed in-memory only; not persisted to disk

## 8. Common Pitfalls & Couplings
Things that are easy to break:
- If you touch `src/speech.py` WebSocket protocol, you must also update `src/static/chat/speech.js` (they share the message schema)
- If you add new env vars, update: `main.py`, `docker-compose.yml`, `.envrc`, and the README env var table
- Test files that import `src.main` must include `os.environ.pop("API_KEY", None)` before the import to avoid auth interference from shell env
- `src/audio_chunking.py` constants (SAMPLE_RATE, BYTES_PER_SAMPLE) must match the AudioWorklet settings in `pcm-processor.js` (16kHz, 16-bit mono)
- If you change the ASR or LLM URL paths, update both `src/asr.py` and `src/speech.py`

## 9. Examples & Canonical Patterns (Optional)

### Example: Add a new WebSocket message type
- Files to edit:
  - `src/speech.py` (add handler in the message dispatch)
  - `src/static/chat/speech.js` (add case in `_handleMessage`)
  - `docs/speech_mode_detailed_spec.md` (document the new message type)
- Tests to add:
  - `test/test_speech.py` (WebSocket integration test)
- Commands to run:
  - `pytest test/test_speech.py test/test_e2e_speech.py`

### Example: Add a new tool
- Files to edit:
  - `src/tool_schemas.py` (define the tool schema, add to appropriate mode)
  - `src/tool_executor.py` (implement execution)
- Tests to add:
  - `test/test_main.py` (tool availability per mode, execution test)
- Commands to run:
  - `pytest test/test_main.py`

## 10. Pull Requests & Branching
Default branch: main

When a PR is requested, create a branch agent/<branch_name> and create a PR from there using gh
