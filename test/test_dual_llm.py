"""Tests for src/dual_llm.py — Dual-LLM 'Thinking Fast and Slow' system."""
import asyncio
import os
from unittest.mock import patch

import pytest

os.environ.setdefault("LLM_BASE_URL", "http://fake-llm")
os.environ.setdefault("LLM_API_KEY", "test-key")

from src.dual_llm import (
    COMPLEXITY_THRESHOLD,
    _count_sentences,
    _extract_first_sentence,
    _route,
    _System2,
    dual_stream,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _seq_stream_chat(token_lists: list[list[str]], captured: list | None = None):
    """Patch for ``llm_engine.stream_chat``: each call yields the next token
    list as OpenAI streaming chunk dicts. Calls are sequenced in invocation
    order (Router, System 1, System 2, ...) the same way the old httpx mock was.
    """
    calls = iter(token_lists)

    async def _stream(messages, *, reasoning_effort=None, tools=None, usage_out=None):
        if captured is not None:
            captured.append(list(messages))
        try:
            tokens = next(calls)
        except StopIteration:
            tokens = ["fallback"]
        for token in tokens:
            yield {"choices": [{"delta": {"content": token}}]}

    return _stream


def _patch_engine(token_lists, captured=None):
    return patch("src.llm_engine.stream_chat", _seq_stream_chat(token_lists, captured))


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


# ─── Router ──────────────────────────────────────────────────────────────────

async def test_route_parses_complexity():
    with _patch_engine([['{"complexity": 3}']]):
        score = await _route(messages=[{"role": "user", "content": "hello"}])
    assert score == 3


async def test_route_clamps_to_range():
    with _patch_engine([['{"complexity": 15}']]):
        score = await _route(messages=[{"role": "user", "content": "hello"}])
    assert score == 10


async def test_route_defaults_on_error():
    with _patch_engine([["not json"]]):
        score = await _route(messages=[{"role": "user", "content": "hello"}])
    assert score == 5  # default


# ─── System 2 ───────────────────────────────────────────────────────────────

async def test_system2_collects_text():
    s2 = _System2()
    with _patch_engine([["Hello. ", "World. ", "Done."]]):
        await s2.run(messages=[])
    assert s2.done
    assert "Hello" in s2.text
    assert "Done" in s2.text


async def test_system2_wait_for_sentences():
    s2 = _System2()

    async def slow(messages, *, reasoning_effort=None, tools=None, usage_out=None):
        for token in ["First sentence. ", "Second sentence. ", "Third sentence."]:
            yield {"choices": [{"delta": {"content": token}}]}
            await asyncio.sleep(0.05)

    with patch("src.llm_engine.stream_chat", slow):
        task = asyncio.create_task(s2.run(messages=[]))
        result = await s2.wait_for_sentences(1, timeout=5.0)
        assert result
        assert _count_sentences(s2.text) >= 1
        await task


# ─── dual_stream — trivial path ─────────────────────────────────────────────

async def test_dual_stream_trivial():
    """Score ≤ 2 → System 1's response is streamed directly."""
    # Responses sequenced as: [router, system1, system2]
    responses = [
        ['{"complexity": 1}'],          # router
        ["Hi", " there", "!"],          # system 1
        ["Deep", " analysis", " here."],  # system 2 (will be cancelled)
    ]
    with _patch_engine(responses):
        tokens = [t async for t in dual_stream(messages=[{"role": "user", "content": "hello"}])]

    text = "".join(tokens)
    assert "Hi" in text
    assert "there" in text


async def test_dual_stream_trivial_score_2():
    """Score = 2 is still trivial."""
    responses = [
        ['{"complexity": 2}'],
        ["Quick", " answer."],
        ["Deep."],
    ]
    with _patch_engine(responses):
        tokens = [t async for t in dual_stream(messages=[{"role": "user", "content": "yes"}])]
    assert "Quick" in "".join(tokens)


# ─── dual_stream — complex path ─────────────────────────────────────────────

async def test_dual_stream_complex():
    """Score > 2 → System 2 thinks, System 1 presents progressively."""
    responses = [
        ['{"complexity": 7}'],                       # router
        ["Fast", " but", " wrong."],                 # system 1 (discarded)
        [                                            # system 2 (deep thinking)
            "Step one analysis. ",
            "Step two analysis. ",
            "Step three analysis. ",
            "Step four analysis. ",
            "Step five analysis. ",
            "Final conclusion.",
        ],
        ["Here's what I found."],                    # present first
        [" The analysis shows interesting results."],  # present next
        [" In conclusion, everything checks out."],  # present final
    ]
    with _patch_engine(responses):
        tokens = [t async for t in dual_stream(messages=[{"role": "user", "content": "explain quantum computing"}])]

    text = "".join(tokens)
    assert "Here's what I found" in text
    assert "Fast" not in text
    assert "wrong" not in text


async def test_dual_stream_complex_has_final():
    """S2 produces only 2 sentences, so 'present next' is skipped."""
    responses = [
        ['{"complexity": 8}'],
        ["Quick."],                  # s1 discarded
        ["Deep analysis. Done."],    # s2 (2 sentences)
        ["Opening."],                # present first
        [" Final answer."],          # present final
    ]
    with _patch_engine(responses):
        tokens = [t async for t in dual_stream(messages=[{"role": "user", "content": "complex question"}])]
    assert "Final answer" in "".join(tokens)


# ─── dual_stream — preserves message format ──────────────────────────────────

async def test_dual_stream_with_system_prompt():
    """Every LLM call should carry the system prompt prefix (shared context)."""
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "hi"},
    ]
    responses = [
        ['{"complexity": 1}'],
        ["Hello!"],
        ["Deep."],
    ]
    captured: list = []
    with _patch_engine(responses, captured=captured):
        _ = [t async for t in dual_stream(messages=messages)]

    assert captured  # at least one LLM call was made
    for msgs in captured:
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "You are helpful."
