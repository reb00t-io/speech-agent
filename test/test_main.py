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


def mock_llm(chunks: list[bytes] | None = None):
    """Patch httpx.AsyncClient to stream the given raw SSE byte chunks."""
    if chunks is None:
        chunks = _sse("Hi", " there")

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

    return patch("src.main.httpx.AsyncClient", _client)


def mock_llm_rounds(rounds: list[list[bytes]], *, capture_bodies: list[dict] | None = None):
    """Patch httpx.AsyncClient for multiple streamed completion rounds."""
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

    return patch("src.main.httpx.AsyncClient", _client)


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
    assert '"mode":"user"' in entries[1]["body"]


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


async def test_invalid_mode_returns_400(client):
    resp = await client.post("/v1/responses", json={"prompt": "hello", "mode": "other"})
    assert resp.status_code == 400


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
    with mock_llm(), patch("src.main._load_system_prompt", side_effect=lambda mode: f"prompt:{mode}"):
        resp = await client.post("/v1/responses", json={"prompt": "hello"})
    sid = resp.headers.get("X-Session-Id")
    assert sessions[sid][0] == {"role": "system", "content": "prompt:user"}


def test_load_system_prompt_uses_mode_specific_docs():
    user_prompt = _load_system_prompt("user")
    dev_prompt = _load_system_prompt("dev")

    assert "User Guide" in user_prompt
    assert "Developer Guide" in dev_prompt
    assert user_prompt != dev_prompt


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


async def test_user_mode_excludes_bash_tool(client):
    request_bodies: list[dict] = []
    with mock_llm_rounds([_sse("ok")], capture_bodies=request_bodies), patch("src.main._load_system_prompt", side_effect=lambda mode: f"prompt:{mode}"):
        resp = await client.post("/v1/responses", json={"prompt": "hello", "mode": "user"})
        await resp.get_data()

    tool_names = [tool["function"]["name"] for tool in request_bodies[0]["tools"]]
    assert "bash" not in tool_names
    assert session_modes[resp.headers.get("X-Session-Id")] == "user"


async def test_dev_mode_includes_bash_tool(client):
    request_bodies: list[dict] = []
    with mock_llm_rounds([_sse("ok")], capture_bodies=request_bodies), patch("src.main._load_system_prompt", side_effect=lambda mode: f"prompt:{mode}"):
        resp = await client.post("/v1/responses", json={"prompt": "hello", "mode": "dev"})
        await resp.get_data()

    tool_names = [tool["function"]["name"] for tool in request_bodies[0]["tools"]]
    assert "bash" in tool_names
    assert session_modes[resp.headers.get("X-Session-Id")] == "dev"


async def test_dev_mode_includes_get_logs_tool(client):
    request_bodies: list[dict] = []
    with mock_llm_rounds([_sse("ok")], capture_bodies=request_bodies), patch("src.main._load_system_prompt", side_effect=lambda mode: f"prompt:{mode}"):
        resp = await client.post("/v1/responses", json={"prompt": "hello", "mode": "dev"})
        await resp.get_data()

    tool_names = [tool["function"]["name"] for tool in request_bodies[0]["tools"]]
    assert "get_logs" in tool_names


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


async def test_latest_session_is_mode_specific(client):
    with mock_llm(_sse("user reply")), patch("src.main._load_system_prompt", side_effect=lambda mode: f"prompt:{mode}"):
        user_resp = await client.post("/v1/responses", json={"prompt": "user hello", "mode": "user"})
        await user_resp.get_data()

    with mock_llm(_sse("dev reply")), patch("src.main._load_system_prompt", side_effect=lambda mode: f"prompt:{mode}"):
        dev_resp = await client.post("/v1/responses", json={"prompt": "dev hello", "mode": "dev"})
        await dev_resp.get_data()

    user_sid = user_resp.headers.get("X-Session-Id")
    dev_sid = dev_resp.headers.get("X-Session-Id")

    user_latest = await (await client.get("/v1/sessions/latest?mode=user")).get_json()
    dev_latest = await (await client.get("/v1/sessions/latest?mode=dev")).get_json()

    assert user_latest == {
        "session_id": user_sid,
        "mode": "user",
        "messages": [
            {"role": "user", "content": "user hello"},
            {"role": "assistant", "content": "user reply"},
        ],
    }
    assert dev_latest == {
        "session_id": dev_sid,
        "mode": "dev",
        "messages": [
            {"role": "user", "content": "dev hello"},
            {"role": "assistant", "content": "dev reply"},
        ],
    }
    assert last_session_ids == {"user": user_sid, "dev": dev_sid}


async def test_latest_session_returns_empty_for_mode_without_history(client):
    with mock_llm(_sse("user reply")), patch("src.main._load_system_prompt", side_effect=lambda mode: f"prompt:{mode}"):
        resp = await client.post("/v1/responses", json={"prompt": "user hello", "mode": "user"})
        await resp.get_data()

    data = await (await client.get("/v1/sessions/latest?mode=dev")).get_json()
    assert data == {"session_id": None, "mode": "dev", "messages": []}


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
