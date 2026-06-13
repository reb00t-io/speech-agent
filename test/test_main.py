"""Tests for src/main.py — chat backend."""
import copy
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Set required env vars before importing the app module.
os.environ.setdefault("LLM_BASE_URL", "http://fake-llm")
os.environ.setdefault("LLM_API_KEY", "test-llm-key")
os.environ.pop("API_KEY", None)  # auth disabled by default

import src.main as main_module  # noqa: E402
from src.main import _load_system_prompt, app, last_session_ids, session_modes, sessions  # noqa: E402


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _sse(*tokens: str) -> list[bytes]:
    """Build raw SSE byte chunks as httpx aiter_raw() would yield them."""
    chunks = [
        f'data: {json.dumps({"choices": [{"delta": {"content": t}}]})}\n\n'.encode()
        for t in tokens
    ]
    chunks.append(b"data: [DONE]\n\n")
    return chunks


def mock_llm(chunks: list[bytes] | None = None, tokens: tuple[str, ...] | None = None):
    """Patch both httpx.AsyncClient and dual_stream for text chat tests.

    Fresh prompts go through dual_stream; tool continuations go through httpx.
    """
    if tokens is None and chunks is None:
        tokens = ("Hi", " there")
    if tokens is None:
        # Extract tokens from SSE chunks
        tokens = tuple(
            json.loads(line[6:]).get("choices", [{}])[0].get("delta", {}).get("content", "")
            for chunk in chunks
            for line in chunk.decode("utf-8", errors="replace").strip().split("\n")
            if line.startswith("data: ") and line[6:] != "[DONE]" and json.loads(line[6:]).get("choices", [{}])[0].get("delta", {}).get("content")
        )
    if chunks is None:
        chunks = _sse(*tokens)

    async def aiter_raw():
        for chunk in chunks:
            yield chunk

    @asynccontextmanager
    async def _stream(*args, **kwargs):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.aiter_raw = aiter_raw
        yield resp

    @asynccontextmanager
    async def _client(*args, **kwargs):
        client = MagicMock()
        client.stream = _stream
        yield client

    async def _fake_dual_stream(**kwargs):
        for t in tokens:
            yield t

    httpx_patch = patch("src.main.httpx.AsyncClient", _client)
    dual_patch = patch("src.streaming.dual_stream", _fake_dual_stream)

    class _Combined:
        def __enter__(self):
            self._h = httpx_patch.__enter__()
            self._d = dual_patch.__enter__()
            return self._h
        def __exit__(self, *args):
            dual_patch.__exit__(*args)
            httpx_patch.__exit__(*args)
    return _Combined()


def mock_llm_rounds(rounds: list[list[bytes]], *, capture_bodies: list[dict] | None = None):
    """Patch httpx.AsyncClient for multiple streamed completion rounds.

    Also disables the dual-LLM path so tool-calling tests go through
    the standard generate_stream pipeline.
    """
    round_iter = iter(rounds)

    @asynccontextmanager
    async def _stream(*args, **kwargs):
        if capture_bodies is not None:
            json_body = kwargs.get("json")
            if isinstance(json_body, dict):
                capture_bodies.append(copy.deepcopy(json_body))
        chunks = next(round_iter)

        async def aiter_raw():
            for chunk in chunks:
                yield chunk

        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.aiter_raw = aiter_raw
        yield resp

    @asynccontextmanager
    async def _client(*args, **kwargs):
        client = MagicMock()
        client.stream = _stream
        yield client

    httpx_patch = patch("src.main.httpx.AsyncClient", _client)
    # Disable dual-LLM so all rounds go through generate_stream
    dual_patch = patch("src.streaming.generate_dual_stream", None)

    class _Combined:
        def __enter__(self):
            self._h = httpx_patch.__enter__()
            self._d = dual_patch.__enter__()
            return self._h
        def __exit__(self, *args):
            dual_patch.__exit__(*args)
            httpx_patch.__exit__(*args)
    return _Combined()


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture()
def client():
    return app.test_client()


@pytest.fixture()
def request_log_path(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "requests.log"
    monkeypatch.setattr(main_module, "REQUEST_LOG_PATH", path)
    return path


@pytest.fixture(autouse=True)
def reset_sessions():
    sessions.clear()
    session_modes.clear()
    last_session_ids.clear()
    yield
    sessions.clear()
    session_modes.clear()
    last_session_ids.clear()


# ─── Auth ────────────────────────────────────────────────────────────────────

async def test_auth_skipped_when_api_key_unset(client):
    with mock_llm():
        resp = await client.post("/v1/responses", json={"prompt": "hello"})
    assert resp.status_code == 200


def _read_request_log(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


async def test_request_log_file_records_non_stream_request_and_response(client, request_log_path):
    resp = await client.get("/v1/sessions/latest?mode=user")
    assert resp.status_code == 200

    entries = _read_request_log(request_log_path)
    assert [entry["event"] for entry in entries] == ["request", "response"]
    assert entries[0]["method"] == "GET"
    assert entries[0]["path"] == "/v1/sessions/latest"
    assert entries[1]["status_code"] == 200
    assert '"session_id"' in entries[1]["body"]


async def test_request_log_file_records_streaming_response_chunks(client, request_log_path):
    with mock_llm(_sse("Hi", " there")):
        resp = await client.post("/v1/responses", json={"prompt": "hello"})
        await resp.get_data()

    entries = _read_request_log(request_log_path)
    events = [entry["event"] for entry in entries]
    assert events[0] == "request"
    assert "response_start" in events
    assert "response_chunk" in events
    assert events[-1] == "response_end"
    assert any("data: [DONE]" in entry.get("body", "") for entry in entries if entry["event"] == "response_chunk")


async def test_unwritable_request_log_does_not_break_requests(client, tmp_path, monkeypatch):
    readonly_dir = tmp_path / "readonly"
    readonly_dir.mkdir()
    readonly_dir.chmod(0o555)
    monkeypatch.setattr(main_module, "REQUEST_LOG_PATH", readonly_dir / "data" / "requests.log")
    monkeypatch.setattr(main_module, "_request_log_write_failed", False)

    resp = await client.get("/v1/sessions/latest?mode=user")
    assert resp.status_code == 200


async def test_auth_required_when_api_key_set(client):
    with patch("src.main.API_KEY", "secret"):
        resp = await client.post("/v1/responses", json={"prompt": "hello"})
    assert resp.status_code == 401


async def test_auth_passes_with_correct_key(client):
    with mock_llm(), patch("src.main.API_KEY", "secret"):
        resp = await client.post(
            "/v1/responses",
            json={"prompt": "hello"},
            headers={"Authorization": "Bearer secret"},
        )
    assert resp.status_code == 200


async def test_auth_fails_with_wrong_key(client):
    with patch("src.main.API_KEY", "secret"):
        resp = await client.post(
            "/v1/responses",
            json={"prompt": "hello"},
            headers={"Authorization": "Bearer wrong"},
        )
    assert resp.status_code == 401


# ─── Input validation ─────────────────────────────────────────────────────────

async def test_empty_prompt_returns_400(client):
    resp = await client.post("/v1/responses", json={"prompt": "  "})
    assert resp.status_code == 400


async def test_missing_prompt_returns_400(client):
    resp = await client.post("/v1/responses", json={})
    assert resp.status_code == 400


async def test_mode_param_is_ignored(client):
    # `mode` is no longer a concept; any value is accepted and ignored.
    with mock_llm():
        resp = await client.post("/v1/responses", json={"prompt": "hello", "mode": "other"})
    assert resp.status_code == 200


# ─── Session management ───────────────────────────────────────────────────────

async def test_new_session_created(client):
    with mock_llm():
        resp = await client.post("/v1/responses", json={"prompt": "hello"})
    assert resp.status_code == 200
    assert resp.headers.get("X-Session-Id") is not None
    assert len(sessions) == 1


async def test_session_id_returned_in_header(client):
    with mock_llm():
        resp = await client.post("/v1/responses", json={"prompt": "hello"})
    sid = resp.headers.get("X-Session-Id")
    assert sid is not None
    assert sid in sessions


async def test_existing_session_reused(client):
    with mock_llm():
        resp1 = await client.post("/v1/responses", json={"prompt": "hello"})
    sid = resp1.headers.get("X-Session-Id")

    with mock_llm():
        resp2 = await client.post("/v1/responses", json={"prompt": "world", "session_id": sid})
    assert resp2.headers.get("X-Session-Id") == sid
    assert len(sessions) == 1


async def test_unknown_session_id_creates_new_session(client):
    with mock_llm():
        resp = await client.post("/v1/responses", json={"prompt": "hello", "session_id": "nonexistent"})
    assert resp.status_code == 200
    # The provided ID is accepted and a new session is created under it
    assert "nonexistent" in sessions


# ─── Message history ──────────────────────────────────────────────────────────

async def test_system_prompt_injected_for_new_session(client):
    with mock_llm(), patch("src.main._load_system_prompt", return_value="prompt:sys"):
        resp = await client.post("/v1/responses", json={"prompt": "hello"})
    sid = resp.headers.get("X-Session-Id")
    assert sessions[sid][0] == {"role": "system", "content": "prompt:sys"}


def test_load_system_prompt_expands_docs():
    prompt = _load_system_prompt()
    assert "{{docs}}" not in prompt
    assert "publish_document" in prompt


async def test_user_message_appended_to_history(client):
    with mock_llm():
        resp = await client.post("/v1/responses", json={"prompt": "hello"})
    sid = resp.headers.get("X-Session-Id")
    user_msgs = [m for m in sessions[sid] if m["role"] == "user"]
    assert user_msgs == [{"role": "user", "content": "hello"}]


async def test_assistant_reply_saved_to_history(client):
    with mock_llm(_sse("Hi", " there")):
        resp = await client.post("/v1/responses", json={"prompt": "hello"})
        await resp.get_data()  # consume stream so generator runs to completion
    sid = resp.headers.get("X-Session-Id")
    last = sessions[sid][-1]
    assert last == {"role": "assistant", "content": "Hi there"}


async def test_multi_turn_history_grows(client):
    with mock_llm():
        resp1 = await client.post("/v1/responses", json={"prompt": "first"})
        await resp1.get_data()
    sid = resp1.headers.get("X-Session-Id")

    with mock_llm():
        resp2 = await client.post("/v1/responses", json={"prompt": "second", "session_id": sid})
        await resp2.get_data()

    # system + user1 + assistant1 + user2 + assistant2 = 5
    assert len(sessions[sid]) == 5


# ─── GET /v1/responses/<session_id> ──────────────────────────────────────────

async def test_get_session_returns_messages(client):
    with mock_llm(_sse("Hello")):
        resp = await client.post("/v1/responses", json={"prompt": "hi"})
        await resp.get_data()
    sid = resp.headers.get("X-Session-Id")

    resp = await client.get(f"/v1/responses/{sid}")
    assert resp.status_code == 200
    data = await resp.get_json()
    assert data["messages"] == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Hello"},
    ]


async def test_get_session_excludes_system_prompt(client):
    with mock_llm(), patch("src.main._load_system_prompt", return_value="sys"):
        resp = await client.post("/v1/responses", json={"prompt": "hi"})
    sid = resp.headers.get("X-Session-Id")

    data = await (await client.get(f"/v1/responses/{sid}")).get_json()
    assert all(m["role"] != "system" for m in data["messages"])


async def test_get_session_404_for_unknown(client):
    resp = await client.get("/v1/responses/no-such-session")
    assert resp.status_code == 404


async def test_get_session_auth(client):
    with patch("src.main.API_KEY", "secret"):
        resp = await client.get("/v1/responses/anything")
    assert resp.status_code == 401


async def test_tool_call_round_executes_backend_tool_and_continues_stream(client):
    rounds = [
        [
            b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n',
            (
                b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function",'
                b'"function":{"name":"python","arguments":"{\\"code\\": \\\"print(1)\\\"}"}}]},'
                b'"finish_reason":"tool_calls"}]}\n\n'
            ),
            b'data: [DONE]\n\n',
        ],
        _sse("The result is 1."),
    ]

    execute_tool = AsyncMock(return_value={"stdout": "1\n", "stderr": "", "exit_code": 0})
    request_bodies: list[dict] = []
    with mock_llm_rounds(rounds, capture_bodies=request_bodies), patch("src.streaming.execute_tool_call", execute_tool):
        resp = await client.post("/v1/responses", json={"prompt": "run python"})
        body = await resp.get_data()

    assert resp.status_code == 200
    text = body.decode()
    assert text.count("data: [DONE]") == 1
    assert '"content":"The"' in text
    assert '"content":" re"' in text
    assert '"content":"sul"' in text
    assert '"content":"t i"' in text
    execute_tool.assert_awaited_once()

    sid = resp.headers.get("X-Session-Id")
    assert sid is not None
    assert len(request_bodies) == 2
    assert request_bodies[0]["tools"]
    assert request_bodies[1]["messages"][-1]["role"] == "tool"
    assert json.loads(request_bodies[1]["messages"][-1]["content"]) == {"stdout": "1\n", "stderr": "", "exit_code": 0}
    assert sessions[sid][-1] == {"role": "assistant", "content": "The result is 1."}

    # The stream announces the running tool, then clears it.
    assert '"tool_status":{"name":"python"' in text
    assert '"tool_status":null' in text


async def test_web_search_tool_status_includes_arguments(client):
    rounds = [
        [
            b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n',
            (
                b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_s","type":"function",'
                b'"function":{"name":"web_search","arguments":"{\\"query\\": \\\"weather\\\"}"}}]},'
                b'"finish_reason":"tool_calls"}]}\n\n'
            ),
            b'data: [DONE]\n\n',
        ],
        _sse("done"),
    ]
    execute_tool = AsyncMock(return_value={"results": []})
    with mock_llm_rounds(rounds), patch("src.streaming.execute_tool_call", execute_tool):
        resp = await client.post("/v1/responses", json={"prompt": "weather?", "web_search": True})
        text = (await resp.get_data()).decode()

    # The active web_search call (with its query) is surfaced to the client.
    assert '"tool_status":{"name":"web_search"' in text
    assert '"query":"weather"' in text
    assert '"tool_status":null' in text


async def test_all_tools_available(client):
    request_bodies: list[dict] = []
    with mock_llm_rounds([_sse("ok")], capture_bodies=request_bodies), patch("src.main._load_system_prompt", return_value="sys"):
        resp = await client.post("/v1/responses", json={"prompt": "hello"})
        await resp.get_data()

    tool_names = {tool["function"]["name"] for tool in request_bodies[0]["tools"]}
    assert {"bash", "python", "web_search", "fetch_url", "get_logs", "publish_document"} <= tool_names


async def test_frontend_get_logs_tool_request_is_forwarded(client):
    rounds = [
        [
            b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n',
            (
                b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_logs","type":"function",'
                b'"function":{"name":"get_logs","arguments":"{\\"system\\": \\\"frontend\\\", \\\"limit\\\": 2}"}}]},'
                b'"finish_reason":"tool_calls"}]}\n\n'
            ),
            b'data: [DONE]\n\n',
        ],
        _sse("done"),
    ]
    request_bodies: list[dict] = []

    frontend_result = {"system": "frontend", "lines": ["a", "b"], "line_count": 2}
    with mock_llm_rounds(rounds, capture_bodies=request_bodies):
        resp = await client.post("/v1/responses", json={"prompt": "logs", "mode": "dev"})
        text = (await resp.get_data()).decode()
        sid = resp.headers.get("X-Session-Id")

        continuation = await client.post(
            "/v1/responses",
            json={"session_id": sid, "mode": "dev", "tool_results": [{"tool_call_id": "call_logs", "result": frontend_result}]},
        )
        await continuation.get_data()

    assert '"tool_request":{"session_id":"' in text
    assert '"tool_call_id":"call_logs"' in text
    assert text.count("data: [DONE]") == 1
    assert len(request_bodies) == 2
    assert request_bodies[1]["messages"][-1]["role"] == "tool"
    assert json.loads(request_bodies[1]["messages"][-1]["content"]) == frontend_result
    assert sessions[sid][-1] == {"role": "assistant", "content": "done"}


async def test_tool_results_can_continue_an_existing_session(client):
    request_bodies: list[dict] = []
    with mock_llm_rounds([_sse("first reply"), _sse("continued")], capture_bodies=request_bodies):
        first = await client.post("/v1/responses", json={"prompt": "hello", "mode": "dev"})
        await first.get_data()
        sid = first.headers.get("X-Session-Id")

        continuation = await client.post(
            "/v1/responses",
            json={
                "session_id": sid,
                "mode": "dev",
                "tool_results": [
                    {"tool_call_id": "call_logs", "result": {"system": "frontend", "lines": ["entry"], "line_count": 1}},
                ],
            },
        )
        await continuation.get_data()

    assert len(request_bodies) == 2
    assert request_bodies[1]["messages"][-1]["role"] == "tool"
    assert json.loads(request_bodies[1]["messages"][-1]["content"]) == {"system": "frontend", "lines": ["entry"], "line_count": 1}


async def test_tool_results_require_existing_session(client):
    resp = await client.post(
        "/v1/responses",
        json={
            "session_id": "missing",
            "mode": "dev",
            "tool_results": [{"tool_call_id": "call_logs", "result": {"ok": True}}],
        },
    )

    assert resp.status_code == 400


async def test_backend_get_logs_tool_is_executed_server_side(client):
    rounds = [
        [
            b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n',
            (
                b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_backend_logs","type":"function",'
                b'"function":{"name":"get_logs","arguments":"{\\"system\\": \\\"backend\\\", \\\"limit\\\": 3}"}}]},'
                b'"finish_reason":"tool_calls"}]}\n\n'
            ),
            b'data: [DONE]\n\n',
        ],
        _sse("done"),
    ]

    with mock_llm_rounds(rounds), patch("src.tool_executor.get_backend_logs", return_value={"system": "backend", "lines": ["x"], "line_count": 1}):
        resp = await client.post("/v1/responses", json={"prompt": "logs", "mode": "dev"})
        body = await resp.get_data()

    assert '"tool_request"' not in body.decode()


async def test_latest_session_returns_most_recent(client):
    with mock_llm(_sse("first reply")), patch("src.main._load_system_prompt", return_value="sys"):
        first = await client.post("/v1/responses", json={"prompt": "first hello"})
        await first.get_data()

    with mock_llm(_sse("second reply")), patch("src.main._load_system_prompt", return_value="sys"):
        second = await client.post("/v1/responses", json={"prompt": "second hello"})
        await second.get_data()

    second_sid = second.headers.get("X-Session-Id")
    latest = await (await client.get("/v1/sessions/latest")).get_json()

    assert latest == {
        "session_id": second_sid,
        "messages": [
            {"role": "user", "content": "second hello"},
            {"role": "assistant", "content": "second reply"},
        ],
    }


async def test_latest_session_returns_empty_without_history(client):
    data = await (await client.get("/v1/sessions/latest")).get_json()
    assert data == {"session_id": None, "messages": []}


async def test_tool_messages_are_hidden_from_history_endpoints(client):
    rounds = [
        [
            b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n',
            (
                b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function",'
                b'"function":{"name":"bash","arguments":"{\\"command\\": \\\"echo hi\\\"}"}}]},'
                b'"finish_reason":"tool_calls"}]}\n\n'
            ),
            b'data: [DONE]\n\n',
        ],
        _sse("hello"),
    ]

    with mock_llm_rounds(rounds), patch("src.streaming.execute_tool_call", AsyncMock(return_value={"stdout": "hi\n"})):
        resp = await client.post("/v1/responses", json={"prompt": "say hi"})
        await resp.get_data()

    sid = resp.headers.get("X-Session-Id")
    history = await (await client.get(f"/v1/responses/{sid}")).get_json()
    latest = await (await client.get("/v1/sessions/latest")).get_json()

    assert history["messages"] == [
        {"role": "user", "content": "say hi"},
        {"role": "assistant", "content": "hello"},
    ]
    assert latest["messages"] == history["messages"]


# ─── Images / web search / deep research ──────────────────────────────────────

async def test_images_build_multimodal_user_content(client):
    request_bodies: list[dict] = []
    data_url = "data:image/png;base64,AAAA"
    with mock_llm_rounds([_sse("ok")], capture_bodies=request_bodies):
        resp = await client.post("/v1/responses", json={"prompt": "what is this?", "images": [data_url]})
        await resp.get_data()

    user_msg = request_bodies[0]["messages"][-1]
    assert user_msg["role"] == "user"
    assert user_msg["content"] == [
        {"type": "text", "text": "what is this?"},
        {"type": "image_url", "image_url": {"url": data_url}},
    ]

    # History flattens multimodal content to text with an [image] marker.
    sid = resp.headers.get("X-Session-Id")
    history = await (await client.get(f"/v1/responses/{sid}")).get_json()
    assert history["messages"][0] == {"role": "user", "content": "what is this? [image]"}


async def test_non_image_data_urls_are_ignored(client):
    request_bodies: list[dict] = []
    with mock_llm_rounds([_sse("ok")], capture_bodies=request_bodies):
        resp = await client.post("/v1/responses", json={"prompt": "hi", "images": ["javascript:alert(1)"]})
        await resp.get_data()

    # No valid image → content stays a plain string.
    assert request_bodies[0]["messages"][-1]["content"] == "hi"


async def test_web_search_injects_system_instruction(client):
    request_bodies: list[dict] = []
    with mock_llm_rounds([_sse("ok")], capture_bodies=request_bodies):
        resp = await client.post("/v1/responses", json={"prompt": "latest news", "web_search": True})
        await resp.get_data()

    system_msgs = [m for m in request_bodies[0]["messages"] if m["role"] == "system"]
    assert any("web_search tool" in m["content"] for m in system_msgs)


async def test_deep_research_injects_system_instruction(client):
    request_bodies: list[dict] = []
    with mock_llm_rounds([_sse("ok")], capture_bodies=request_bodies):
        resp = await client.post("/v1/responses", json={"prompt": "research X", "deep_research": True})
        await resp.get_data()

    system_msgs = [m for m in request_bodies[0]["messages"] if m["role"] == "system"]
    assert any("publish_document tool" in m["content"] for m in system_msgs)


# ─── Dictation transcription endpoint ─────────────────────────────────────────

async def test_transcribe_endpoint_returns_text(client):
    with patch("src.main.transcribe", AsyncMock(return_value="hello world")):
        resp = await client.post("/v1/transcribe", data=b"\x00\x01" * 100,
                                 headers={"Content-Type": "application/octet-stream"})
    assert resp.status_code == 200
    assert (await resp.get_json())["text"] == "hello world"


async def test_transcribe_endpoint_rejects_empty_audio(client):
    resp = await client.post("/v1/transcribe", data=b"")
    assert resp.status_code == 400


# ─── Document download ────────────────────────────────────────────────────────

async def test_download_rejects_path_traversal(client):
    resp = await client.get("/download/..%2f..%2fetc%2fpasswd")
    assert resp.status_code == 404


async def test_published_document_is_downloadable(client, tmp_path, monkeypatch):
    import src.documents as documents
    monkeypatch.setattr(documents, "DOWNLOADS_DIR", tmp_path / "downloads")

    result = documents.publish_markdown("# Report\n\nSome **content**.", title="My Report")
    assert result["download_url"].startswith("/download/")
    assert result["filename"].endswith(".pdf")

    resp = await client.get(result["download_url"])
    assert resp.status_code == 200
    assert resp.headers["Content-Type"] == "application/pdf"
    assert (await resp.get_data())[:5] == b"%PDF-"


# ─── reverse-proxy subpath (APP_ROOT) ────────────────────────────────────────

async def test_no_app_root_emits_root_paths(client):
    resp = await client.get("/")
    body = await resp.get_data(as_text=True)
    assert '/static/chat/chat.js' in body
    assert 'window.APP_ROOT = ""' in body


async def test_app_root_prefixes_emitted_urls(client, monkeypatch):
    monkeypatch.setattr(main_module, "APP_ROOT", "/agent")
    resp = await client.get("/")
    body = await resp.get_data(as_text=True)
    assert 'window.APP_ROOT = "/agent"' in body
    assert '/agent/static/chat/chat.js' in body


async def test_app_root_redirects_keep_prefix(monkeypatch):
    monkeypatch.setattr(main_module, "APP_ROOT", "/agent")
    monkeypatch.setattr(main_module, "AUTH_MODE", "password")
    monkeypatch.setattr(main_module, "AUTH_PASSWORD", "pw")
    main_module.app.secret_key = "test"
    resp = await main_module.app.test_client().get("/")  # unauthenticated
    assert resp.status_code == 302
    assert resp.headers["Location"].endswith("/agent/login")
