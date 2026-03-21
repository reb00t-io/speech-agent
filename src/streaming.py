import asyncio
import codecs
import copy
import json
import logging
import secrets
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import aiohttp
from quart import Response, jsonify

try:
    from .tool_executor import execute_tool_call
except ImportError:
    from tool_executor import execute_tool_call

MAX_TOOL_CALL_ROUNDS = 10

logger = logging.getLogger(__name__)


def _split_stream_text(text: str, size: int = 3) -> list[str]:
    if len(text) <= size:
        return [text]
    return [text[i : i + size] for i in range(0, len(text), size)]


def _is_unauthorized(api_key: str, authorization: str) -> bool:
    return bool(api_key) and authorization != f"Bearer {api_key}"


@dataclass
class StreamState:
    stream_pace_seconds: float
    reply_parts: list[str] = field(default_factory=list)
    text_buf: str = ""
    tool_calls: dict[int, dict] = field(default_factory=dict)
    finish_reason: str | None = None


def emit_event(payload: str) -> bytes:
    return f"data: {payload}\n\n".encode("utf-8")


def _parse_tool_arguments(tool_call: dict[str, Any]) -> dict[str, Any] | None:
    raw_arguments = ((tool_call.get("function") or {}).get("arguments") or "{}").strip()
    try:
        parsed = json.loads(raw_arguments)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _is_frontend_log_request(tool_call: dict[str, Any]) -> bool:
    fn = tool_call.get("function") or {}
    if fn.get("name") != "get_logs":
        return False
    arguments = _parse_tool_arguments(tool_call) or {}
    return arguments.get("system") == "frontend"


def build_frontend_tool_request(session_id: str, tool_call: dict[str, Any]) -> dict[str, Any]:
    arguments = _parse_tool_arguments(tool_call) or {}
    return {
        "session_id": session_id,
        "tool_call_id": tool_call.get("id", ""),
        "name": (tool_call.get("function") or {}).get("name", ""),
        "arguments": arguments,
    }


def visible_messages(messages: list[dict]) -> list[dict]:
    visible_roles = {"user", "assistant"}
    return [message for message in messages if message.get("role") in visible_roles and message.get("content")]


def _merge_tool_call_delta(state: StreamState, tool_call_delta: dict) -> None:
    index = tool_call_delta.get("index", 0)
    builder = state.tool_calls.setdefault(index, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
    if tool_call_delta.get("id"):
        builder["id"] = tool_call_delta["id"]
    if tool_call_delta.get("type"):
        builder["type"] = tool_call_delta["type"]

    fn_delta = tool_call_delta.get("function") or {}
    if fn_delta.get("name"):
        builder["function"]["name"] += fn_delta["name"]
    if fn_delta.get("arguments"):
        builder["function"]["arguments"] += fn_delta["arguments"]


def finalize_tool_calls(state: StreamState) -> list[dict]:
    return [state.tool_calls[index] for index in sorted(state.tool_calls)]


async def handle_event(event: str, state: StreamState):
    for line in event.splitlines():
        if not line.startswith("data: "):
            continue

        payload = line[6:]
        if payload == "[DONE]":
            return

        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            continue

        choices = chunk.get("choices") or []
        if not choices:
            continue

        choice = choices[0]
        delta = choice.get("delta") or {}
        role = delta.get("role")
        content = delta.get("content") or ""
        for tool_call_delta in delta.get("tool_calls") or []:
            if isinstance(tool_call_delta, dict):
                _merge_tool_call_delta(state, tool_call_delta)
        if choice.get("finish_reason"):
            state.finish_reason = choice.get("finish_reason")

        if role and not content:
            role_chunk = copy.deepcopy(chunk)
            role_chunk["choices"][0]["delta"] = {"role": role}
            yield emit_event(json.dumps(role_chunk, separators=(",", ":")))
            continue

        if not content:
            continue

        for piece in _split_stream_text(content):
            content_chunk = copy.deepcopy(chunk)
            content_chunk["choices"][0]["delta"] = {"content": piece}
            state.reply_parts.append(piece)
            yield emit_event(json.dumps(content_chunk, separators=(",", ":")))
            if state.stream_pace_seconds > 0:
                await asyncio.sleep(state.stream_pace_seconds)


async def flush_events(events_text: str, state: StreamState):
    state.text_buf += events_text

    while True:
        boundary = state.text_buf.find("\n\n")
        separator_len = 2

        if boundary == -1:
            boundary = state.text_buf.find("\r\n\r\n")
            separator_len = 4

        if boundary == -1:
            break

        event = state.text_buf[:boundary]
        state.text_buf = state.text_buf[boundary + separator_len :]
        async for outbound in handle_event(event, state):
            yield outbound


def split_frontend_tool_calls(tool_calls: list[dict]) -> tuple[list[dict], list[dict]]:
    frontend_tool_calls: list[dict] = []
    backend_tool_calls: list[dict] = []
    for tool_call in tool_calls:
        if _is_frontend_log_request(tool_call):
            frontend_tool_calls.append(tool_call)
        else:
            backend_tool_calls.append(tool_call)
    return frontend_tool_calls, backend_tool_calls


def append_tool_result_messages(messages: list[dict], tool_results: list[dict]) -> None:
    for tool_result in tool_results:
        tool_call_id = str(tool_result.get("tool_call_id") or "").strip()
        if not tool_call_id:
            continue
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": json.dumps(tool_result.get("result"), ensure_ascii=False),
            }
        )


async def execute_backend_tool_round(messages: list[dict], tool_calls: list[dict]):
    async with aiohttp.ClientSession() as session:
        for tool_call in tool_calls:
            tool_result = await execute_tool_call(session, tool_call)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.get("id", ""),
                    "content": json.dumps(tool_result, ensure_ascii=False),
                }
            )


async def generate_stream(
    *,
    messages: list[dict],
    save_sessions: Callable[[], None],
    client_factory,
    llm_base_url: str,
    llm_api_key: str,
    llm_body: dict,
    stream_pace_seconds: float,
    tools: list[dict],
    session_id: str,
):
    try:
        async with client_factory(timeout=120) as client:
            for round_index in range(MAX_TOOL_CALL_ROUNDS):
                state = StreamState(stream_pace_seconds=stream_pace_seconds)
                decoder = codecs.getincrementaldecoder("utf-8")()
                request_body = dict(llm_body)
                request_body["messages"] = messages
                request_body["tools"] = tools

                async with client.stream(
                    "POST",
                    f"{llm_base_url}/chat/completions",
                    headers={
                        "Accept": "text/event-stream",
                        "Accept-Encoding": "identity",
                        "Authorization": f"Bearer {llm_api_key}",
                        "Content-Type": "application/json",
                    },
                    json=request_body,
                ) as resp:
                    resp.raise_for_status()
                    async for chunk in resp.aiter_raw():
                        if not chunk:
                            continue
                        async for outbound in flush_events(decoder.decode(chunk), state):
                            yield outbound

                    async for outbound in flush_events(decoder.decode(b"", final=True), state):
                        yield outbound

                    if state.text_buf.strip():
                        async for outbound in handle_event(state.text_buf, state):
                            yield outbound

                assistant_message: dict = {"role": "assistant", "content": "".join(state.reply_parts)}
                tool_calls = finalize_tool_calls(state)
                if tool_calls:
                    assistant_message["tool_calls"] = tool_calls
                messages.append(assistant_message)

                if not tool_calls:
                    yield emit_event("[DONE]")
                    return

                frontend_tool_calls, backend_tool_calls = split_frontend_tool_calls(tool_calls)

                if backend_tool_calls:
                    await execute_backend_tool_round(messages, backend_tool_calls)

                if frontend_tool_calls:
                    for tool_call in frontend_tool_calls:
                        tool_call_id = tool_call.get("id", "")
                        tool_name = (tool_call.get("function") or {}).get("name", "")
                        logger.info(
                            "Forwarding tool request to frontend and ending stream: session_id=%s tool_call_id=%s tool=%s",
                            session_id,
                            tool_call_id,
                            tool_name,
                        )
                        yield emit_event(json.dumps({"tool_request": build_frontend_tool_request(session_id, tool_call)}, separators=(",", ":")))
                    yield emit_event("[DONE]")
                    return

                if round_index == MAX_TOOL_CALL_ROUNDS - 1:
                    error_text = (
                        f"Tool-calling stopped after {MAX_TOOL_CALL_ROUNDS} rounds to prevent infinite loops. "
                        "Please answer with the information already gathered."
                    )
                    messages.append({"role": "assistant", "content": error_text})
                    yield emit_event(json.dumps({"choices": [{"delta": {"content": error_text}}]}, separators=(",", ":")))
                    yield emit_event("[DONE]")
                    return
    finally:
        save_sessions()


async def get_session_response(*, session_id: str, sessions: dict[str, list[dict]], api_key: str, authorization: str):
    if _is_unauthorized(api_key, authorization):
        return jsonify({"error": "Unauthorized"}), 401

    if session_id not in sessions:
        return jsonify({"error": "Session not found"}), 404

    return jsonify({"messages": visible_messages(sessions[session_id])})


def _normalize_tool_results(body: dict) -> list[dict]:
    tool_results = body.get("tool_results")
    if isinstance(tool_results, list):
        return [item for item in tool_results if isinstance(item, dict)]

    tool_result = body.get("tool_result")
    if isinstance(tool_result, dict):
        return [tool_result]

    return []


async def post_chat_response(
    *,
    body: dict,
    sessions: dict[str, list[dict]],
    session_modes: dict[str, str],
    api_key: str,
    authorization: str,
    load_system_prompt: Callable[[str], str],
    save_sessions: Callable[[], None],
    on_session_start: Callable[[str, str], None] | None = None,
    tools: list[dict] | None = None,
    client_factory,
    llm_base_url: str,
    llm_api_key: str,
    llm_model: str,
    stream_pace_seconds: float,
):
    if _is_unauthorized(api_key, authorization):
        return jsonify({"error": "Unauthorized"}), 401

    prompt = (body.get("prompt") or "").strip()
    tool_results = _normalize_tool_results(body)
    if not prompt and not tool_results:
        return jsonify({"error": "prompt or tool_results is required"}), 400

    mode = body.get("mode") or "user"
    session_id = body.get("session_id") or secrets.token_urlsafe(16)
    if prompt and on_session_start:
        on_session_start(session_id, mode)

    if session_id not in sessions:
        if tool_results:
            return jsonify({"error": "session_id is required for tool_results and must refer to an existing session"}), 400
        sessions[session_id] = [{"role": "system", "content": load_system_prompt(mode)}]
        session_modes[session_id] = mode
    else:
        session_modes.setdefault(session_id, mode)
    messages = sessions[session_id]
    if prompt:
        messages.append({"role": "user", "content": prompt})
    if tool_results:
        append_tool_result_messages(messages, tool_results)

    llm_body = {"model": llm_model, "stream": True, "messages": messages}

    return Response(
        generate_stream(
            messages=messages,
            save_sessions=save_sessions,
            client_factory=client_factory,
            llm_base_url=llm_base_url,
            llm_api_key=llm_api_key,
            llm_body=llm_body,
            stream_pace_seconds=stream_pace_seconds,
            tools=tools or [],
            session_id=session_id,
        ),
        content_type="text/event-stream",
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache",
            "X-Session-Id": session_id,
        },
    )
