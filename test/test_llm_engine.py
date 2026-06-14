"""Tests for src/llm_engine.py — the memorizer-backed request engine bridge."""
import os
from unittest.mock import patch

import pytest

os.environ.setdefault("LLM_BASE_URL", "http://fake-llm")
os.environ.setdefault("LLM_API_KEY", "test-key")

import src.llm_engine as llm_engine


class _FakeResponse:
    """Stand-in for a streaming requests.Response."""

    def __init__(self, lines):
        self._lines = lines
        self.closed = False

    def iter_lines(self):
        for line in self._lines:
            yield line

    def close(self):
        self.closed = True


class _FakeModel:
    def __init__(self, lines):
        self._lines = lines
        self.calls = []
        self.last_response = None

    def stream(self, messages, *, tools=None, reasoning_effort=None):
        self.calls.append({"messages": messages, "tools": tools, "reasoning_effort": reasoning_effort})
        self.last_response = _FakeResponse(self._lines)
        return self.last_response


def _data(obj_json: str) -> bytes:
    return f"data: {obj_json}".encode()


async def test_stream_chat_bridges_sse_to_chunk_dicts():
    lines = [
        _data('{"choices":[{"delta":{"content":"Hi"}}]}'),
        _data('{"choices":[{"delta":{"content":" there"}}]}'),
        _data("[DONE]"),
    ]
    fake = _FakeModel(lines)
    with patch("src.llm_engine.get_model", return_value=fake):
        chunks = [c async for c in llm_engine.stream_chat([{"role": "user", "content": "x"}])]

    text = "".join(c["choices"][0]["delta"]["content"] for c in chunks)
    assert text == "Hi there"
    assert fake.last_response.closed  # response is always released


async def test_stream_chat_passes_tools_and_reasoning_effort():
    fake = _FakeModel([_data("[DONE]")])
    tools = [{"type": "function", "function": {"name": "x"}}]
    with patch("src.llm_engine.get_model", return_value=fake):
        _ = [c async for c in llm_engine.stream_chat([{"role": "user", "content": "x"}], tools=tools, reasoning_effort="low")]

    assert fake.calls[0]["tools"] == tools
    assert fake.calls[0]["reasoning_effort"] == "low"


async def test_stream_chat_default_reasoning_effort_when_unset():
    fake = _FakeModel([_data("[DONE]")])
    with patch("src.llm_engine.get_model", return_value=fake), \
         patch("src.llm_engine.LLM_DEFAULT_REASONING_EFFORT", "high"):
        _ = [c async for c in llm_engine.stream_chat([{"role": "user", "content": "x"}])]
    assert fake.calls[0]["reasoning_effort"] == "high"


async def test_stream_chat_populates_usage_out():
    lines = [
        _data('{"choices":[{"delta":{"content":"a"}}]}'),
        _data('{"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2}}'),
        _data("[DONE]"),
    ]
    fake = _FakeModel(lines)
    usage: dict = {}
    with patch("src.llm_engine.get_model", return_value=fake):
        _ = [c async for c in llm_engine.stream_chat([{"role": "user", "content": "x"}], usage_out=usage)]
    assert usage.get("prompt_tokens") == 5
    assert usage.get("completion_tokens") == 2


async def test_stream_content_tokens_yields_text_only():
    lines = [
        _data('{"choices":[{"delta":{"role":"assistant"}}]}'),
        _data('{"choices":[{"delta":{"content":"Hello"}}]}'),
        _data('{"choices":[{"delta":{"content":" world"}}]}'),
        _data("[DONE]"),
    ]
    fake = _FakeModel(lines)
    with patch("src.llm_engine.get_model", return_value=fake):
        tokens = [t async for t in llm_engine.stream_content_tokens([{"role": "user", "content": "x"}])]
    assert tokens == ["Hello", " world"]


async def test_stream_chat_surfaces_worker_exception():
    class _BoomModel:
        def stream(self, *a, **k):
            raise RuntimeError("backend down")

    with patch("src.llm_engine.get_model", return_value=_BoomModel()):
        with pytest.raises(RuntimeError, match="backend down"):
            _ = [c async for c in llm_engine.stream_chat([{"role": "user", "content": "x"}])]
