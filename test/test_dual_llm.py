"""Tests for src/dual_llm.py — Dual-LLM 'Thinking Fast and Slow' system."""
import asyncio
import json
import os
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

os.environ.setdefault("LLM_BASE_URL", "http://fake-llm")
os.environ.setdefault("LLM_API_KEY", "test-key")

from src.dual_llm import (
    COMPLEXITY_THRESHOLD,
    _apply_reasoning_effort,
    _count_sentences,
    _extract_first_sentence,
    _route,
    _System2,
    dual_stream,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _sse_bytes(*tokens: str, done: bool = True) -> list[bytes]:
    """Build raw SSE byte chunks."""
    chunks = [
        f'data: {json.dumps({"choices": [{"delta": {"content": t}}]})}\n\n'.encode()
        for t in tokens
    ]
    if done:
        chunks.append(b"data: [DONE]\n\n")
    return chunks


def _make_mock_client(responses: list[list[bytes]]):
    """Create a mock httpx.AsyncClient that returns different responses per call.

    Each entry in `responses` is a list of SSE byte chunks for one stream() call.
    """
    call_index = 0

    @asynccontextmanager
    async def _stream(*args, **kwargs):
        nonlocal call_index
        idx = call_index
        call_index += 1
        chunks = responses[idx] if idx < len(responses) else _sse_bytes("fallback")

        async def aiter_raw():
            for c in chunks:
                yield c

        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        resp.aiter_raw = aiter_raw
        yield resp

    client = MagicMock()
    client.stream = _stream
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return client


# ─── Sentence utilities ─────────────────────────────────────────────────────

def test_count_sentences_empty():
    assert _count_sentences("") == 0


def test_count_sentences_one():
    assert _count_sentences("Hello world.") == 1


def test_count_sentences_multiple():
    assert _count_sentences("One. Two. Three.") == 3


def test_count_sentences_no_punctuation():
    assert _count_sentences("Hello world") == 1


def test_count_sentences_mixed():
    assert _count_sentences("Hello! How are you? Fine.") == 3


def test_extract_first_sentence():
    assert _extract_first_sentence("Hello world. More text.") == "Hello world."


def test_extract_first_sentence_question():
    assert _extract_first_sentence("How are you? Fine.") == "How are you?"


def test_extract_first_sentence_no_punct():
    assert _extract_first_sentence("Hello world") == "Hello world"


# ─── Reasoning-effort mapping ────────────────────────────────────────────────

def test_reasoning_effort_openai_low():
    body = {}
    _apply_reasoning_effort(body, "gpt-oss-120b", "low")
    assert body == {"reasoning_effort": "low"}


def test_reasoning_effort_openai_default_sets_nothing():
    body = {}
    _apply_reasoning_effort(body, "gpt-oss-120b", None)
    assert body == {}


def test_reasoning_effort_kimi_low_disables_thinking():
    body = {}
    _apply_reasoning_effort(body, "kimi-k2", "low")
    assert body == {"chat_template_kwargs": {"thinking": False}}
    assert "reasoning_effort" not in body


def test_reasoning_effort_kimi_deep_enables_thinking():
    body = {}
    _apply_reasoning_effort(body, "Kimi-K2-Instruct", None)
    assert body == {"chat_template_kwargs": {"thinking": True}}
    assert "reasoning_effort" not in body


def test_reasoning_effort_kimi_preserves_existing_template_kwargs():
    body = {"chat_template_kwargs": {"foo": "bar"}}
    _apply_reasoning_effort(body, "kimi-k2", "low")
    assert body == {"chat_template_kwargs": {"foo": "bar", "thinking": False}}


# ─── Router ──────────────────────────────────────────────────────────────────

async def test_route_parses_complexity():
    router_response = _sse_bytes('{"complexity": 3}')
    client = _make_mock_client([router_response])

    score = await _route(
        client=client, base_url="http://fake", api_key="key",
        model="test", messages=[{"role": "user", "content": "hello"}],
    )
    assert score == 3


async def test_route_clamps_to_range():
    client = _make_mock_client([_sse_bytes('{"complexity": 15}')])
    score = await _route(
        client=client, base_url="http://fake", api_key="key",
        model="test", messages=[{"role": "user", "content": "hello"}],
    )
    assert score == 10


async def test_route_defaults_on_error():
    client = _make_mock_client([_sse_bytes("not json")])
    score = await _route(
        client=client, base_url="http://fake", api_key="key",
        model="test", messages=[{"role": "user", "content": "hello"}],
    )
    assert score == 5  # default


# ─── System 2 ───────────────────────────────────────────────────────────────

async def test_system2_collects_text():
    s2 = _System2()
    chunks = _sse_bytes("Hello. ", "World. ", "Done.")
    client = _make_mock_client([chunks])

    await s2.run(
        client=client, base_url="http://fake", api_key="key",
        model="test", messages=[],
    )
    assert s2.done
    assert "Hello" in s2.text
    assert "Done" in s2.text


async def test_system2_wait_for_sentences():
    s2 = _System2()

    # Simulate slow streaming
    async def slow_stream():
        chunks = [
            _sse_bytes("First sentence. ", done=False)[0],
            _sse_bytes("Second sentence. ", done=False)[0],
            _sse_bytes("Third sentence.", done=True)[0],
            b"data: [DONE]\n\n",
        ]

        @asynccontextmanager
        async def _stream(*args, **kwargs):
            async def aiter_raw():
                for c in chunks:
                    yield c
                    await asyncio.sleep(0.05)
            resp = MagicMock()
            resp.status_code = 200
            resp.raise_for_status = MagicMock()
            resp.aiter_raw = aiter_raw
            yield resp

        client = MagicMock()
        client.stream = _stream
        return client

    client = await slow_stream()
    task = asyncio.create_task(s2.run(
        client=client, base_url="http://fake", api_key="key",
        model="test", messages=[],
    ))

    result = await s2.wait_for_sentences(1, timeout=5.0)
    assert result
    assert _count_sentences(s2.text) >= 1

    await task


# ─── dual_stream — trivial path ─────────────────────────────────────────────

async def test_dual_stream_trivial():
    """Score ≤ 2 → System 1's response is streamed directly."""
    # Responses: [router, system1, system2]
    responses = [
        _sse_bytes('{"complexity": 1}'),                    # router
        _sse_bytes("Hi", " there", "!"),                    # system 1
        _sse_bytes("Deep", " analysis", " here."),          # system 2 (will be cancelled)
    ]

    with patch("src.dual_llm.httpx.AsyncClient", return_value=_make_mock_client(responses)):
        tokens = []
        async for token in dual_stream(
            messages=[{"role": "user", "content": "hello"}],
            model="test", base_url="http://fake", api_key="key",
        ):
            tokens.append(token)

    text = "".join(tokens)
    assert "Hi" in text
    assert "there" in text


async def test_dual_stream_trivial_score_2():
    """Score = 2 is still trivial."""
    responses = [
        _sse_bytes('{"complexity": 2}'),
        _sse_bytes("Quick", " answer."),
        _sse_bytes("Deep."),
    ]

    with patch("src.dual_llm.httpx.AsyncClient", return_value=_make_mock_client(responses)):
        tokens = []
        async for token in dual_stream(
            messages=[{"role": "user", "content": "yes"}],
            model="test", base_url="http://fake", api_key="key",
        ):
            tokens.append(token)

    assert "Quick" in "".join(tokens)


# ─── dual_stream — complex path ─────────────────────────────────────────────

async def test_dual_stream_complex():
    """Score > 2 → System 2 thinks, System 1 presents progressively."""
    responses = [
        _sse_bytes('{"complexity": 7}'),                                  # router
        _sse_bytes("Fast", " but", " wrong."),                            # system 1 (discarded)
        # system 2 (deep thinking)
        _sse_bytes(
            "Step one analysis. ",
            "Step two analysis. ",
            "Step three analysis. ",
            "Step four analysis. ",
            "Step five analysis. ",
            "Final conclusion.",
        ),
        # system 1 present first sentence
        _sse_bytes("Here's what I found."),
        # system 1 present next sentence
        _sse_bytes(" The analysis shows interesting results."),
        # system 1 complete response
        _sse_bytes(" In conclusion, everything checks out."),
    ]

    with patch("src.dual_llm.httpx.AsyncClient", return_value=_make_mock_client(responses)):
        tokens = []
        async for token in dual_stream(
            messages=[{"role": "user", "content": "explain quantum computing"}],
            model="test", base_url="http://fake", api_key="key",
        ):
            tokens.append(token)

    text = "".join(tokens)
    # Should contain System 1 presenter output, NOT System 2 raw output
    assert "Here's what I found" in text
    # Should NOT contain System 1's discarded fast response
    assert "Fast" not in text
    assert "wrong" not in text


async def test_dual_stream_complex_has_final():
    """Complex path should include the final completion.

    S2 produces only 2 sentences, so 'present next' is skipped (needs 4 more).
    The flow is: router → s1(discard) → s2 → present-first → present-final.
    """
    responses = [
        _sse_bytes('{"complexity": 8}'),
        _sse_bytes("Quick."),               # s1 discarded
        _sse_bytes("Deep analysis. Done."), # s2 (2 sentences — not enough for "next")
        _sse_bytes("Opening."),             # s1 present first
        # "present next" is skipped since s2 didn't produce 4 more sentences
        _sse_bytes(" Final answer."),       # s1 complete
    ]

    with patch("src.dual_llm.httpx.AsyncClient", return_value=_make_mock_client(responses)):
        tokens = []
        async for token in dual_stream(
            messages=[{"role": "user", "content": "complex question"}],
            model="test", base_url="http://fake", api_key="key",
        ):
            tokens.append(token)

    text = "".join(tokens)
    assert "Final answer" in text


# ─── dual_stream — preserves message format ──────────────────────────────────

async def test_dual_stream_with_system_prompt():
    """Messages with system prompt should pass through correctly."""
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "hi"},
    ]
    responses = [
        _sse_bytes('{"complexity": 1}'),
        _sse_bytes("Hello!"),
        _sse_bytes("Deep."),
    ]

    captured_bodies = []
    original_make = _make_mock_client

    def capturing_client(resps):
        client = original_make(resps)
        original_stream = client.stream

        @asynccontextmanager
        async def capturing_stream(*args, **kwargs):
            body = kwargs.get("json") or (args[2] if len(args) > 2 else None)
            if body:
                captured_bodies.append(body)
            async with original_stream(*args, **kwargs) as resp:
                yield resp

        client.stream = capturing_stream
        return client

    with patch("src.dual_llm.httpx.AsyncClient", return_value=capturing_client(responses)):
        tokens = []
        async for token in dual_stream(
            messages=messages, model="test",
            base_url="http://fake", api_key="key",
        ):
            tokens.append(token)

    # All LLM calls should include the system prompt
    for body in captured_bodies:
        msgs = body.get("messages", [])
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "You are helpful."


# ─── Integration with text chat (via mock) ──────────────────────────────────

async def test_dual_stream_used_for_text_chat():
    """Verify dual_stream can be called the same way for text input."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
    ]
    responses = [
        _sse_bytes('{"complexity": 1}'),
        _sse_bytes("4", "."),
        _sse_bytes("The answer is 4."),
    ]

    with patch("src.dual_llm.httpx.AsyncClient", return_value=_make_mock_client(responses)):
        result = []
        async for token in dual_stream(
            messages=messages, model="test",
            base_url="http://fake", api_key="key",
        ):
            result.append(token)

    assert "4" in "".join(result)
