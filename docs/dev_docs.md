# bootstrap — Developer Guide

## Purpose

**bootstrap** is a lightweight Python web app template with a built-in assistant panel.

The assistant can operate in a user-facing mode or a development-oriented mode.

---

## Stack overview

- Quart application served by Uvicorn
- Jinja template in `src/templates/index.html`
- Frontend chat logic in `src/static/chat/chat.js`
- Backend streaming and tool orchestration in `src/streaming.py`
- Tool definitions in `src/tool_schemas.py`

---

## Chat behavior

- The frontend posts to `/v1/responses`.
- The backend forwards streaming chat-completions requests to the configured LLM backend.
- Tool calls are executed server-side and the backend continues the same assistant turn.
- The frontend receives a single assistant stream.

---

## Modes

- **user mode** uses the user system prompt and user docs.
- **dev mode** uses the developer system prompt and developer docs.
- **user mode** excludes the `bash` tool.
- **dev mode** includes all configured tools.

---

## Runtime configuration

Important environment variables:

- `PORT`
- `LLM_BASE_URL`
- `LLM_API_KEY`
- `LLM_MODEL`
- `API_KEY`
- `SESSIONS_PATH`
- `STREAM_PACE_SECONDS`

---

## Persistence

- Sessions are stored in JSON at `SESSIONS_PATH`.
- The latest session id is tracked in metadata.
- Session mode is also stored in metadata so the UI can restore the active mode.

---

## Frontend notes

- The chat panel supports desktop resize.
- On mobile, the panel appears as a bottom sheet.
- The chat header includes a mode toggle, clear button, and close button.

---

## Boundaries

- Visible history endpoints return only user and assistant messages.
- Tool messages remain backend-only.
- If the docs do not cover something, the assistant should say so rather than invent details.