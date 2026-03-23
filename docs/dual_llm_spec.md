# Dual-LLM System — "Thinking Fast and Slow" Spec

## Overview

Every user query is processed by three concurrent LLM subsystems sharing the
same model, system prompt, and message history (enabling KV-cache reuse):

| Subsystem | Role | `reasoning_effort` |
|-----------|------|-------------------|
| **Router** | Rates query complexity 1–10 | `low` |
| **System 1** (Speaking) | Produces the user-facing response | `low` |
| **System 2** (Thinking) | Produces a deep, thorough answer | _(default)_ |

The user only ever sees output from System 1. System 2's output is never
shown directly — it serves as internal context that System 1 draws on.

## Flow

```
User message
    │
    ├──▶ Router  (complexity 1–10, structured JSON, reasoning_effort=low)
    ├──▶ System 1 (quick answer, reasoning_effort=low)
    └──▶ System 2 (deep answer, default reasoning)
         │
    ┌────┘
    ▼
┌─────────────────────────────────────────────────────┐
│  Router returns score                                │
│                                                      │
│  score ≤ 2 (trivial):                                │
│    → Stream System 1's response to user              │
│    → Cancel System 2                                 │
│                                                      │
│  score > 2 (needs thinking):                         │
│    → Discard System 1's initial response             │
│    → Wait for System 2 to accumulate content         │
│    → System 1 formulates response from System 2      │
│      in progressive chunks (see below)               │
└─────────────────────────────────────────────────────┘
```

## Complex Query Flow (score > 2)

System 1 acts as a "presenter" that translates System 2's deep reasoning
into user-facing speech, delivered progressively:

```
System 2 streaming ────────────────────────────────▶ done
                 ▲                    ▲              ▲
                 │                    │              │
          has initial content   +4 sentences    complete
                 │                    │              │
                 ▼                    ▼              ▼
System 1:   sentence 1 ──stop──  sentence 2 ──  finish response
                 │                    │              │
                 ▼                    ▼              ▼
User sees:  "sentence 1"        "sentence 2"    "full rest"
```

**Step by step:**

1. System 2 starts streaming its deep response.
2. Once System 2 has produced initial content (~1 sentence), System 1 is
   called with System 2's partial output and asked to formulate an opening
   sentence for the user. That single sentence is streamed to the user,
   then System 1's stream is stopped.
3. System 2 continues. After 4 more sentences from System 2, System 1
   is called again to continue its response with the next sentence.
   Again, one sentence is streamed and then stopped.
4. When System 2 finishes, System 1 is called one final time to complete
   its response, incorporating all of System 2's output.

## Router

**Input:** Same messages as System 1/2 (system prompt + conversation history
+ current user message).

**Additional system instruction** (appended): Rate the complexity of the
user's latest message on a scale of 1–10.

**Output format:** `{"complexity": N}` via `response_format={"type": "json_object"}`.

**Interpretation:**
- 1–2: Trivial (greetings, yes/no, simple factual)
- 3–5: Moderate (requires some reasoning)
- 6–10: Complex (multi-step reasoning, analysis, coding)

## System 1 Prompts

### Trivial mode (score ≤ 2)
System 1 runs with `reasoning_effort=low`. Its response is streamed directly
to the user. No special prompt additions.

### Presenter mode (score > 2)
System 1 is called multiple times with System 2's partial/full output
injected as context:

```
[...original messages...]
{"role": "user", "content": "<original user message>"}
{"role": "assistant", "content": "<System 2 partial output>"}
{"role": "user", "content": "Based on the above analysis, provide a
 <first/next/complete> response to the user. Be concise and natural.
 <constraint: one sentence only / complete the response>"}
```

## Message History

- Only user-visible messages are stored in history: `user` and `assistant`
  (System 1's final output).
- System 2 output and router results are ephemeral — not persisted.
- All three subsystems receive the same message history on each call.

## Integration

The dual-LLM system is modality-agnostic. It provides:

```python
async def dual_stream(
    messages: list[dict],
    model: str,
    base_url: str,
    api_key: str,
) -> AsyncGenerator[str, None]:
    """Yields content tokens to show to the user."""
```

Both the text chat endpoint (`POST /v1/responses`) and the speech WebSocket
handler call this same function. The caller is responsible for:
- Appending the user message to history before calling
- Collecting the yielded tokens into the assistant message after

## Configuration

No new env vars — uses existing `LLM_MODEL`, `LLM_BASE_URL`, `LLM_API_KEY`.

## File Structure

```
src/
  dual_llm.py    # NEW: Router + System 1 + System 2 orchestration
  speech.py      # Modified: use dual_stream instead of direct LLM call
  streaming.py   # Modified: use dual_stream instead of direct LLM call
```
