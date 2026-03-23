"""Dual-LLM system — 'Thinking Fast and Slow' orchestration.

Every user query is processed by three concurrent subsystems:
  - Router:   rates complexity 1-10 (reasoning_effort=low, JSON output)
  - System 1: fast response for the user (reasoning_effort=low)
  - System 2: deep response (default reasoning_effort)

The user only sees System 1 output. For trivial queries (score ≤ 2),
System 1's direct answer is used. For complex queries, System 2 thinks
deeply and System 1 progressively presents that thinking to the user.
"""

from __future__ import annotations

import asyncio
import codecs
import json
import logging
import re
from collections.abc import AsyncGenerator
from typing import Any

import httpx

logger = logging.getLogger(__name__)

COMPLEXITY_THRESHOLD = 2
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")

ROUTER_INSTRUCTION = (
    "Rate the complexity of the user's latest message on a scale of 1 to 10. "
    "1-2 = trivial (greetings, yes/no, simple facts). "
    "3-5 = moderate (needs some reasoning). "
    "6-10 = complex (multi-step reasoning, analysis, coding). "
    'Respond with JSON only: {"complexity": N}'
)

S1_PRESENT_FIRST = (
    "Based on the analysis above, provide an opening response to the user. "
    "Write exactly ONE sentence. Be concise and natural. Do not repeat the analysis."
)

S1_PRESENT_NEXT = (
    "Continue your response to the user with ONE more sentence based on the "
    "updated analysis above. Be concise and natural."
)

S1_PRESENT_FINAL = (
    "Complete your response to the user based on the full analysis above. "
    "Be concise and natural. Do not repeat what you already said."
)


# ─── Low-level LLM helpers ──────────────────────────────────────────────────


async def _llm_stream_tokens(
    *,
    client: httpx.AsyncClient,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict],
    reasoning_effort: str | None = None,
    response_format: dict | None = None,
) -> AsyncGenerator[str, None]:
    """Stream tokens from the LLM. Yields content strings."""
    body: dict[str, Any] = {"model": model, "stream": True, "messages": messages}
    if reasoning_effort:
        body["reasoning_effort"] = reasoning_effort
    if response_format:
        body["response_format"] = response_format

    decoder = codecs.getincrementaldecoder("utf-8")()
    text_buf = ""

    async with client.stream(
        "POST",
        f"{base_url}/chat/completions",
        headers={
            "Accept": "text/event-stream",
            "Accept-Encoding": "identity",
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=body,
    ) as resp:
        resp.raise_for_status()
        async for raw_chunk in resp.aiter_raw():
            if not raw_chunk:
                continue
            text_buf += decoder.decode(raw_chunk)
            while "\n\n" in text_buf:
                event, text_buf = text_buf.split("\n\n", 1)
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
                    delta = choices[0].get("delta") or {}
                    content = delta.get("content") or ""
                    if content:
                        yield content


async def _llm_collect(
    *,
    client: httpx.AsyncClient,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict],
    reasoning_effort: str | None = None,
    response_format: dict | None = None,
) -> str:
    """Collect the full response from the LLM as a string."""
    parts = []
    async for token in _llm_stream_tokens(
        client=client,
        base_url=base_url,
        api_key=api_key,
        model=model,
        messages=messages,
        reasoning_effort=reasoning_effort,
        response_format=response_format,
    ):
        parts.append(token)
    return "".join(parts)


# ─── Sentence utilities ─────────────────────────────────────────────────────


def _count_sentences(text: str) -> int:
    """Count sentences in text (split on .!? followed by space)."""
    parts = _SENTENCE_END_RE.split(text.strip())
    return len([p for p in parts if p.strip()])


def _extract_first_sentence(text: str) -> str:
    """Extract the first sentence from text."""
    match = re.search(r"[.!?](?:\s|$)", text)
    if match:
        return text[: match.end()].strip()
    return text.strip()


# ─── Router ──────────────────────────────────────────────────────────────────


async def _route(
    *,
    client: httpx.AsyncClient,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict],
) -> int:
    """Rate query complexity 1-10. Returns the score."""
    router_messages = list(messages) + [
        {"role": "user", "content": ROUTER_INSTRUCTION},
    ]
    try:
        result = await _llm_collect(
            client=client,
            base_url=base_url,
            api_key=api_key,
            model=model,
            messages=router_messages,
            reasoning_effort="low",
            response_format={"type": "json_object"},
        )
        data = json.loads(result)
        score = int(data.get("complexity", 5))
        logger.info("Router: complexity=%d raw=%r", score, result.strip())
        return max(1, min(10, score))
    except Exception:
        logger.exception("Router failed, defaulting to complexity=5")
        return 5


# ─── System 2 collector (runs in background) ────────────────────────────────


class _System2:
    """Collects System 2's deep response progressively."""

    def __init__(self):
        self.text = ""
        self.done = False
        self._event = asyncio.Event()
        self._sentence_count = 0

    async def run(
        self,
        *,
        client: httpx.AsyncClient,
        base_url: str,
        api_key: str,
        model: str,
        messages: list[dict],
    ) -> None:
        try:
            async for token in _llm_stream_tokens(
                client=client,
                base_url=base_url,
                api_key=api_key,
                model=model,
                messages=messages,
            ):
                self.text += token
                new_count = _count_sentences(self.text)
                if new_count > self._sentence_count:
                    self._sentence_count = new_count
                    self._event.set()
                    self._event = asyncio.Event()
        except asyncio.CancelledError:
            logger.info("System 2 cancelled")
            raise
        except Exception:
            logger.exception("System 2 error")
        finally:
            self.done = True
            self._event.set()

    async def wait_for_sentences(self, n: int, timeout: float = 30.0) -> bool:
        """Wait until System 2 has at least n sentences or is done.
        Returns True if the condition was met."""
        while self._sentence_count < n and not self.done:
            try:
                await asyncio.wait_for(self._event.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                return False
        return True


# ─── Main orchestrator ───────────────────────────────────────────────────────


async def dual_stream(
    *,
    messages: list[dict],
    model: str,
    base_url: str,
    api_key: str,
) -> AsyncGenerator[str, None]:
    """Orchestrate Router + System 1 + System 2 and yield user-facing tokens.

    Args:
        messages: Full message history including system prompt and current
                  user message (already appended by the caller).
        model: LLM model name.
        base_url: LLM API base URL.
        api_key: LLM API key.

    Yields:
        Content tokens to show to the user.
    """
    async with httpx.AsyncClient(timeout=120) as client:
        kwargs = dict(client=client, base_url=base_url, api_key=api_key, model=model)

        # Start all three subsystems concurrently
        router_task = asyncio.create_task(_route(messages=messages, **kwargs))

        s1_tokens: list[str] = []
        s1_done = asyncio.Event()

        async def _collect_s1():
            async for token in _llm_stream_tokens(
                messages=messages,
                reasoning_effort="low",
                **kwargs,
            ):
                s1_tokens.append(token)
            s1_done.set()

        s1_task = asyncio.create_task(_collect_s1())

        s2 = _System2()
        s2_task = asyncio.create_task(s2.run(messages=messages, **kwargs))

        # Wait for router
        try:
            score = await router_task
        except Exception:
            logger.exception("Router failed")
            score = 5

        # ── Trivial path ─────────────────────────────────────────────
        if score <= COMPLEXITY_THRESHOLD:
            logger.info("Trivial query (score=%d), using System 1 directly", score)
            # Cancel System 2 as it is not needed
            if not s2_task.done():
                s2_task.cancel()
                try:
                    await s2_task
                except asyncio.CancelledError:
                    pass

            # Yield any tokens that System 1 has already produced
            for token in s1_tokens:
                yield token

            # If System 1 is still streaming, continue yielding new tokens as they appear
            if not s1_done.is_set():
                last_idx = len(s1_tokens)
                while not s1_done.is_set():
                    await asyncio.sleep(0.01)
                    while last_idx < len(s1_tokens):
                        yield s1_tokens[last_idx]
                        last_idx += 1
                while last_idx < len(s1_tokens):
                    yield s1_tokens[last_idx]
                    last_idx += 1

            return

        # ── Complex path ─────────────────────────────────────────────
        logger.info("Complex query (score=%d), using System 2 thinking", score)
        s1_task.cancel()  # discard System 1's initial response

        # Wait for System 2 to have initial content (~1 sentence)
        await s2.wait_for_sentences(1, timeout=30.0)
        if not s2.text.strip():
            # System 2 produced nothing — fall back
            logger.warning("System 2 produced no content, falling back")
            return

        # ── Present sentence 1 ───────────────────────────────────────
        s1_first_messages = list(messages) + [
            {"role": "assistant", "content": s2.text},
            {"role": "user", "content": S1_PRESENT_FIRST},
        ]
        first_response = ""
        async for token in _llm_stream_tokens(
            messages=s1_first_messages,
            reasoning_effort="low",
            **kwargs,
        ):
            first_response += token
            yield token
            # Stop after first sentence
            if re.search(r"[.!?]\s*$", first_response):
                break

        sentences_at_first = _count_sentences(s2.text)

        # ── Wait for 4 more sentences from System 2, then present ────
        target = sentences_at_first + 4
        await s2.wait_for_sentences(target, timeout=30.0)

        if _count_sentences(s2.text) > sentences_at_first:
            s1_next_messages = list(messages) + [
                {"role": "assistant", "content": s2.text},
                {"role": "user", "content": S1_PRESENT_NEXT},
            ]
            next_response = ""
            async for token in _llm_stream_tokens(
                messages=s1_next_messages,
                reasoning_effort="low",
                **kwargs,
            ):
                next_response += token
                yield token
                if re.search(r"[.!?]\s*$", next_response):
                    break

        # ── Wait for System 2 to finish, then complete ───────────────
        if not s2.done:
            await s2.wait_for_sentences(999, timeout=60.0)

        s1_final_messages = list(messages) + [
            {"role": "assistant", "content": s2.text},
            {"role": "user", "content": S1_PRESENT_FINAL},
        ]
        async for token in _llm_stream_tokens(
            messages=s1_final_messages,
            reasoning_effort="low",
            **kwargs,
        ):
            yield token

        # Cancel System 2 if still running
        if not s2_task.done():
            s2_task.cancel()
            try:
                await s2_task
            except asyncio.CancelledError:
                pass
