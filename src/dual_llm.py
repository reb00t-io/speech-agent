"""Dual-LLM system — 'Thinking Fast and Slow' orchestration.

Every user query is processed by three concurrent subsystems:
  - Router:   rates complexity 1-10 (reasoning_effort=low, JSON output)
  - System 1: fast response for the user (reasoning_effort=low)
  - System 2: deep response (default reasoning_effort)

The user only sees System 1 output. For trivial queries (score ≤ 2),
System 1's direct answer is used. For complex queries, System 2 thinks
deeply and System 1 progressively presents that thinking to the user.

Reasoning depth is requested via the OpenAI ``reasoning_effort`` parameter,
except for Kimi models which instead toggle ``chat_template_kwargs.thinking``
(see ``_apply_reasoning_effort``).
"""

from __future__ import annotations

import asyncio
import codecs
import json
import logging
import re
import time
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
    "The analysis above is your internal draft — the user cannot see it. "
    "Now write the opening of your response to the user: exactly ONE sentence. "
    "Be concise and natural. Do not quote or reference 'the analysis'."
)

S1_PRESENT_NEXT = (
    "The analysis above is your updated internal draft — the user cannot see it. "
    "You already sent the user:\n{prior}\n\n"
    "Continue with ONE more sentence. Be concise and natural. "
    "Do not repeat what you already sent."
)

S1_PRESENT_FINAL = (
    "The analysis above is your complete internal draft — the user cannot see it. "
    "You already sent the user:\n{prior}\n\n"
    "Now finish your response. Be concise and natural. "
    "Do not repeat what you already sent."
)


# ─── Low-level LLM helpers ──────────────────────────────────────────────────


def _is_kimi_model(model: str) -> bool:
    """Whether *model* is a Kimi model (which toggles reasoning differently)."""
    return "kimi" in model.lower()


def _apply_reasoning_effort(body: dict[str, Any], model: str, reasoning_effort: str | None) -> None:
    """Set the request field(s) controlling reasoning depth for *model*.

    Most OpenAI-compatible models accept ``reasoning_effort``. Kimi rejects it
    and instead toggles thinking via ``chat_template_kwargs={"thinking": bool}``.
    We map a "low" effort to thinking disabled (the fast path) and anything else
    to thinking enabled, so the fast/deep contrast the dual-LLM design relies on
    is preserved on Kimi regardless of its server-side default.
    """
    if _is_kimi_model(model):
        body.setdefault("chat_template_kwargs", {})["thinking"] = reasoning_effort != "low"
        return
    if reasoning_effort:
        body["reasoning_effort"] = reasoning_effort


async def _iter_sse_tokens(
    resp: httpx.Response,
    usage_out: dict[str, int] | None,
) -> AsyncGenerator[str, None]:
    """Parse SSE chunks from an httpx streaming response, yielding content tokens."""
    decoder = codecs.getincrementaldecoder("utf-8")()
    text_buf = ""
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
                if usage_out is not None:
                    usage = chunk.get("usage")
                    if usage and isinstance(usage, dict):
                        usage_out.update(usage)
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                content = delta.get("content") or ""
                if content:
                    yield content


async def _llm_stream_tokens(
    *,
    client: httpx.AsyncClient,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict],
    reasoning_effort: str | None = None,
    response_format: dict | None = None,
    usage_out: dict[str, int] | None = None,
) -> AsyncGenerator[str, None]:
    """Stream tokens from the LLM. Yields content strings.

    If *usage_out* is provided (a mutable dict), it will be populated with
    ``prompt_tokens``, ``completion_tokens``, and ``total_tokens`` from the
    final streaming chunk.
    """
    body: dict[str, Any] = {"model": model, "stream": True, "messages": messages}
    _apply_reasoning_effort(body, model, reasoning_effort)
    if response_format:
        body["response_format"] = response_format
    if usage_out is not None:
        body["stream_options"] = {"include_usage": True}

    headers = {
        "Accept": "text/event-stream",
        "Accept-Encoding": "identity",
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    url = f"{base_url}/chat/completions"

    async with client.stream("POST", url, headers=headers, json=body) as resp:
        if resp.status_code >= 400 and "stream_options" in body:
            # API may not support stream_options — retry without it
            logger.warning("LLM returned HTTP %d with stream_options, retrying without", resp.status_code)
        else:
            resp.raise_for_status()
            async for token in _iter_sse_tokens(resp, usage_out):
                yield token
            return

    # Retry without stream_options
    body.pop("stream_options", None)
    async with client.stream("POST", url, headers=headers, json=body) as resp:
        resp.raise_for_status()
        async for token in _iter_sse_tokens(resp, usage_out):
            yield token


async def _llm_collect(
    *,
    client: httpx.AsyncClient,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict],
    reasoning_effort: str | None = None,
    response_format: dict | None = None,
    usage_out: dict[str, int] | None = None,
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
        usage_out=usage_out,
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
    t0 = time.monotonic()
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
        elapsed_ms = (time.monotonic() - t0) * 1000
        result = result.strip()
        # Try JSON first, then fall back to extracting a number
        try:
            data = json.loads(result)
            score = int(data.get("complexity", 5))
        except (json.JSONDecodeError, ValueError, TypeError):
            # LLM may return plain text or markdown — extract first number
            match = re.search(r'\d+', result)
            score = int(match.group()) if match else 5
            logger.warning("Router returned non-JSON, extracted score=%d from: %r", score, result[:100])
        score = max(1, min(10, score))
        decision = "trivial → S1" if score <= COMPLEXITY_THRESHOLD else "complex → S2+S1"
        logger.info("Router: score=%d decision=%r e2e=%.0fms", score, decision, elapsed_ms)
        return score
    except Exception:
        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.exception("Router failed after %.0fms, defaulting to complexity=5", elapsed_ms)
        return 5


# ─── System 2 collector (runs in background) ────────────────────────────────


class _System2:
    """Collects System 2's deep response progressively."""

    def __init__(self):
        self.text = ""
        self.done = False
        self.usage: dict[str, int] = {}
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
        t0 = time.monotonic()
        logger.info("S2: started")
        try:
            async for token in _llm_stream_tokens(
                client=client,
                base_url=base_url,
                api_key=api_key,
                model=model,
                messages=messages,
                usage_out=self.usage,
            ):
                self.text += token
                new_count = _count_sentences(self.text)
                if new_count > self._sentence_count:
                    self._sentence_count = new_count
                    self._event.set()
                    self._event = asyncio.Event()
            elapsed_ms = (time.monotonic() - t0) * 1000
            logger.info("S2: done e2e=%.0fms len=%d sentences=%d", elapsed_ms, len(self.text), self._sentence_count)
        except asyncio.CancelledError:
            elapsed_ms = (time.monotonic() - t0) * 1000
            logger.info("S2: cancelled after %.0fms len=%d", elapsed_ms, len(self.text))
            raise
        except Exception:
            elapsed_ms = (time.monotonic() - t0) * 1000
            logger.exception("S2: error after %.0fms", elapsed_ms)
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
        t_start = time.monotonic()
        router_task = asyncio.create_task(_route(messages=messages, **kwargs))

        s1_tokens: list[str] = []
        s1_done = asyncio.Event()
        s1_initial_usage: dict[str, int] = {}

        async def _collect_s1():
            t0 = time.monotonic()
            logger.info("S1: started (initial)")
            async for token in _llm_stream_tokens(
                messages=messages,
                reasoning_effort="low",
                usage_out=s1_initial_usage,
                **kwargs,
            ):
                s1_tokens.append(token)
            elapsed_ms = (time.monotonic() - t0) * 1000
            logger.info("S1: initial done e2e=%.0fms len=%d", elapsed_ms, sum(len(t) for t in s1_tokens))
            s1_done.set()

        s1_task = asyncio.create_task(_collect_s1())

        s2 = _System2()
        s2_task = asyncio.create_task(s2.run(messages=messages, **kwargs))

        # Ensure all background tasks are cancelled before the httpx client
        # closes (e.g. when the caller cancels this generator).
        try:
            # Wait for router
            try:
                score = await router_task
            except Exception:
                logger.exception("Router failed")
                score = 5

            # ── Trivial path ─────────────────────────────────────────
            if score <= COMPLEXITY_THRESHOLD:
                logger.info("Trivial path: using S1 directly")
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

                # If System 1 is still streaming, continue yielding new tokens
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

                response_text = "".join(s1_tokens)
                elapsed_ms = (time.monotonic() - t_start) * 1000
                s1_comp = s1_initial_usage.get("completion_tokens", 0)
                s1_prom = s1_initial_usage.get("prompt_tokens", 0)
                logger.info(
                    "Response sent: source=S1 len=%d e2e=%.0fms | "
                    "S1: %d completion tokens (prompt: %d)",
                    len(response_text), elapsed_ms, s1_comp, s1_prom,
                )
                return

            # ── Complex path ─────────────────────────────────────────
            s1_discarded_len = len(s1_tokens)
            logger.info("Complex path: discarding S1 initial (%d tokens), waiting for S2", s1_discarded_len)
            s1_task.cancel()  # discard System 1's initial response
            full_response = ""
            s1_usage_list: list[dict[str, int]] = []

            # Wait for System 2 to have initial content (~1 sentence)
            await s2.wait_for_sentences(1, timeout=30.0)
            if not s2.text.strip():
                logger.warning("S2 produced no content, falling back")
                return

            # ── Present sentence 1 ───────────────────────────────────
            logger.info("S1: present-first (S2 has %d chars, %d sentences)", len(s2.text), _count_sentences(s2.text))
            t_s1 = time.monotonic()
            s1_first_usage: dict[str, int] = {}
            s1_first_messages = list(messages) + [
                {"role": "assistant", "content": s2.text},
                {"role": "user", "content": S1_PRESENT_FIRST},
            ]
            first_response = ""
            async for token in _llm_stream_tokens(
                messages=s1_first_messages,
                reasoning_effort="low",
                usage_out=s1_first_usage,
                **kwargs,
            ):
                first_response += token
                full_response += token
                yield token
                if re.search(r"[.!?]\s*$", first_response):
                    break
            s1_usage_list.append(s1_first_usage)
            logger.info("S1: present-first done e2e=%.0fms len=%d", (time.monotonic() - t_s1) * 1000, len(first_response))

            sentences_at_first = _count_sentences(s2.text)

            # ── Wait for 4 more sentences from S2, then present ──────
            target = sentences_at_first + 4
            await s2.wait_for_sentences(target, timeout=30.0)

            if _count_sentences(s2.text) > sentences_at_first:
                logger.info("S1: present-next (S2 has %d chars, %d sentences)", len(s2.text), _count_sentences(s2.text))
                t_s1 = time.monotonic()
                s1_next_usage: dict[str, int] = {}
                s1_next_messages = list(messages) + [
                    {"role": "assistant", "content": s2.text},
                    {"role": "user", "content": S1_PRESENT_NEXT.format(prior=full_response)},
                ]
                next_response = ""
                # Separator between S1 segments
                yield "\n\n"
                full_response += "\n\n"
                async for token in _llm_stream_tokens(
                    messages=s1_next_messages,
                    reasoning_effort="low",
                    usage_out=s1_next_usage,
                    **kwargs,
                ):
                    next_response += token
                    full_response += token
                    yield token
                    if re.search(r"[.!?]\s*$", next_response):
                        break
                s1_usage_list.append(s1_next_usage)
                logger.info("S1: present-next done e2e=%.0fms len=%d", (time.monotonic() - t_s1) * 1000, len(next_response))

            # ── Wait for System 2 to finish, then complete ───────────
            if not s2.done:
                await s2.wait_for_sentences(999, timeout=60.0)

            logger.info("S1: present-final (S2 has %d chars, %d sentences)", len(s2.text), _count_sentences(s2.text))
            t_s1 = time.monotonic()
            s1_final_usage: dict[str, int] = {}
            s1_final_messages = list(messages) + [
                {"role": "assistant", "content": s2.text},
                {"role": "user", "content": S1_PRESENT_FINAL.format(prior=full_response)},
            ]
            final_response = ""
            # Separator between S1 segments
            yield "\n\n"
            full_response += "\n\n"
            async for token in _llm_stream_tokens(
                messages=s1_final_messages,
                reasoning_effort="low",
                usage_out=s1_final_usage,
                **kwargs,
            ):
                final_response += token
                full_response += token
                yield token
            s1_usage_list.append(s1_final_usage)
            logger.info("S1: present-final done e2e=%.0fms len=%d", (time.monotonic() - t_s1) * 1000, len(final_response))

            elapsed_ms = (time.monotonic() - t_start) * 1000

            # ── Token usage summary ──────────────────────────────────
            s1_completion = sum(u.get("completion_tokens", 0) for u in s1_usage_list)
            s1_prompt = sum(u.get("prompt_tokens", 0) for u in s1_usage_list)
            s2_completion = s2.usage.get("completion_tokens", 0)
            s2_prompt = s2.usage.get("prompt_tokens", 0)
            logger.info(
                "Response sent: source=S2+S1 len=%d e2e=%.0fms | "
                "S1 forwarded: %d completion tokens (prompt: %d) across %d calls | "
                "S1 discarded: %d tokens | "
                "S2: %d completion tokens (prompt: %d)",
                len(full_response), elapsed_ms,
                s1_completion, s1_prompt, len(s1_usage_list),
                s1_discarded_len,
                s2_completion, s2_prompt,
            )

        finally:
            # Cancel all background tasks before the httpx client closes
            for task in (router_task, s1_task, s2_task):
                if not task.done():
                    task.cancel()
            for task in (router_task, s1_task, s2_task):
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
