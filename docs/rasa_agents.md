# Rasa Agents — How the System Works

A digest of Rasa's architecture, focused on the parts that matter when deciding
whether to borrow ideas for this speech-agent codebase. Sources are linked at
the bottom.

## 1. What Rasa is today

Rasa is a platform for building **task-oriented conversational agents**. Their
modern stack is called **CALM** ("Conversational AI with Language Models"),
introduced around v3.7. CALM replaces the older intent/entity classifier era of
Rasa and is the thing they actively market.

Two product surfaces:
- **Rasa (pro-code)** — YAML + Python, what most engineers touch.
- **Studio (no-code)** — graphical builder on top of the same runtime.

Voice/speech is a separately-positioned product ("Rasa Voice") that plugs ASR
and TTS into the same CALM core.

## 2. Speech: pipeline, not speech-to-speech

Rasa is **not** a speech-to-speech model. It is a classic 3-stage pipeline:

```
mic ──► ASR (STT) ──► CALM (text) ──► TTS ──► speaker
```

Two channel flavors:
- **Voice-Ready channels** — telephony/SIP-style integrations (Twilio,
  AudioCodes, Jambonz, Genesys). The channel hands Rasa text and takes back text.
- **Voice Stream channels** — Rasa itself owns the audio stream (e.g. the
  `browser_audio` channel). Rasa drives the ASR/TTS providers directly.

Common abstractions:
- **`RasaAudioBytes`** — a single intermediate audio format so channels, ASR,
  and TTS engines don't have to know about each other. Supported wire formats
  are 8 kHz μ-law, 24 kHz PCM, and 48 kHz PCM (mono).
- **`ASREngine` / `TTSEngine`** base classes — subclass these to add a custom
  provider. ASR emits `UserIsSpeaking` (interim) and `NewTranscript` (final)
  events; TTS implements either streaming (`send_text_chunk` →
  `signal_text_done` → `stream_audio`) or one-shot synthesis.
- **Built-in providers**: ASR — Deepgram, Azure. TTS — Deepgram, Cartesia,
  Azure, Rime. Most TTS engines stream input text so synthesis can begin
  before the LLM has finished generating.
- **Turn detection** is handed off to the ASR provider (e.g. Deepgram's VAD +
  utterance-end timing). Rasa does not run its own VAD.

So architecturally, Rasa's voice story is exactly the same shape as this
codebase: ASR → text agent → TTS, with streaming on both ends.

## 3. CALM: how the "thinking" actually works

CALM is the differentiator. It is a deliberate split between what an LLM is
allowed to do and what is hard-coded as business logic.

### 3.1 Three layers

1. **Dialogue Understanding** (LLM) — reads the conversation transcript +
   currently-collected slots and emits a small list of **commands**.
2. **Dialogue Manager** (deterministic) — consumes those commands and runs the
   appropriate **Flow**.
3. **Response generation** — by default, sends only **human-authored** template
   responses to the user. An *optional* "contextual response rephraser" can
   rewrite a template through an LLM for fluency, but you opt in per message.

The LLM never directly speaks to the user (unless you explicitly enable
rephrasing or the chitchat fallback). That's the whole point.

### 3.2 Commands the LLM is allowed to emit

A small, fixed vocabulary:

- `start flow <name>`
- `cancel flow`
- `set slot <name> <value>`
- `correct slot <name> <value>`
- `clarify flows` (disambiguate when multiple flows match)
- `chitchat`
- `knowledge answer` (RAG handoff)
- `human handoff`

Output looks like this (literally one command per line):

```
start flow transfer_money
set slot recipient John
set slot amount 500
```

This is essentially constrained tool calling, but the "tools" are *dialogue
acts*, not business APIs. It maps cleanly to function calling on any modern
model (GPT-4o, GPT-5.x, Claude 3.5+ are all listed as supported), but the
output protocol is text-based, not provider-specific.

### 3.3 Command generators

The LLM call is encapsulated by a **Command Generator** component:

- `CompactLLMCommandGenerator` — general-purpose default.
- `SearchReadyLLMCommandGenerator` — variant tuned for RAG/Enterprise Search.
- Templates are Jinja2, vary per model, and stuff the prompt with: task
  description, available actions, the live conversation state, currently
  active flow, and the list of available flows with their slot metadata.

You can also fall back to the older non-LLM NLU classifier, or run hybrid.

### 3.4 Flows (the business logic)

A **Flow** is a YAML file describing one task the agent can complete. Steps
are typed:

```yaml
flows:
  transfer_money:
    description: Send money to another account
    steps:
      - collect: recipient
      - collect: amount
      - action: validate_account
      - action: utter_confirm_transfer
```

Step types include `collect` (slot fill), `action` (template response or
custom code), `if`/branch, `link`/`call` (jump into another flow), and a few
control-flow primitives.

Crucial design choice: **flows do not enumerate every conversation path**.
They are the "happy path + the data I need." Off-path behavior (interruptions,
corrections, digressions, "wait what?") is handled by built-in **conversation
pattern flows** that the dialogue manager invokes automatically. This is what
lets a small, hand-written flow survive a real, messy user.

The dialogue manager keeps a **dialogue stack**, so a user can interrupt
`transfer_money` with `check_balance`, get an answer, and pop back to where
they were — without the flow author writing a single line for that case.

### 3.5 Tool calling

Rasa exposes three increasingly direct ways to call into code:

1. **Custom Actions** — the classic path. You implement a Python class; when a
   flow step says `action: my_action`, Rasa POSTs to an Action Server with the
   tracker + domain, your code runs, and you return events (typically
   `SlotSet`s) plus any responses. New in recent versions: you can run actions
   in-process via `actions_module` instead of a separate server.
2. **MCP tools** — flow steps can call a **Model Context Protocol** server
   directly. Rasa positions this as the modern, lower-boilerplate replacement
   for custom actions when you're just wrapping an external API.
3. **Templated `utter_*` responses** — for pure text replies, no code needed.

The important thing: **the LLM does not pick the tool**. The flow author does.
The LLM only decides "we are now in `transfer_money` and the `amount` slot is
500." The deterministic flow then decides "OK, call `validate_account` next."
This is the inverse of the typical ReAct agent.

## 4. What's actually their moat

Stripping the marketing:

1. **Constrained command vocabulary instead of free-form tool use.** The LLM
   is forced to emit a fixed set of dialogue acts. There is no surface area for
   it to invent a new tool, hallucinate an argument format, or improvise a
   response. This is the single most important design decision in CALM.

2. **Hand-authored responses by default.** The generated text the user hears
   is, by default, a template you wrote. The LLM is a *router*, not a
   *speaker*. Hallucinations are eliminated by construction, not by prompting.
   This is the reason regulated industries (banks, telcos, insurance — Rasa's
   actual customer base) buy Rasa instead of LangChain.

3. **Flows + automatic conversation patterns.** You write the happy path; Rasa
   ships the digression/correction/clarification/cancellation logic as
   conversation patterns. Most "agent frameworks" make you handle these
   yourself or hope the model figures it out.

4. **The dialogue stack.** First-class support for nested/interrupted tasks
   without flow authors having to think about it.

5. **Pluggable, model-agnostic core that runs on small models.** Because the
   LLM only emits a tiny constrained output, an 8B model can drive it. That
   means low latency, on-prem deployment, no per-token bill, and a
   defensible enterprise story.

6. **Operational maturity.** Tracing, evaluation harness, conversation
   analytics, Studio for non-engineers, voice channel integrations, Action
   Server, MCP support, deployment tooling. Less sexy than the architecture,
   but it's what an enterprise actually pays for.

The moat is **not** any single LLM trick — it's the *discipline* of separating
"understand the user" from "do the thing" and "speak to the user," and then
shipping the surrounding 80% of plumbing that makes that discipline workable
in production.

## 5. Sources

- [Rasa Docs — landing](https://rasa.com/docs/)
- [CALM concept page](https://rasa.com/docs/learn/concepts/calm/)
- [Dialogue management](https://rasa.com/docs/learn/concepts/dialogue-management/)
- [LLM Command Generators](https://rasa.com/docs/reference/config/components/llm-command-generators/)
- [Flows reference](https://rasa.com/docs/reference/primitives/flows/)
- [Custom Actions](https://rasa.com/docs/reference/primitives/custom-actions/)
- [MCP Servers in Rasa](https://rasa.com/docs/reference/integrations/mcp-servers/)
- [Speech Integrations](https://rasa.com/docs/reference/integrations/speech-integrations/)
- [Voice Assistants (Pro)](https://rasa.com/docs/pro/build/voice-assistants/)
- [Rasa Voice product page](https://rasa.com/solutions/voice/)
- [Rasa CALM marketing page](https://rasa.com/calm)
- [Building a Voice Bot with Rasa and Cartesia (blog)](https://rasa.com/blog/building-a-voice-bot-with-rasa-and-cartesia-a-technical-tutorial)
