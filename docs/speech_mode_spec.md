# Speech Mode — High-Level Spec

## Overview

Add a **speech mode** to the existing chat agent. In speech mode, the user speaks into the microphone; audio streams to the backend in real time, gets transcribed (ASR), and triggers LLM responses on speech pauses. The existing text chat continues to work as before.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Browser                             │
│                                                         │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────┐  │
│  │ Mic Input │───▶│ AudioWorklet │───▶│  WebSocket    │  │
│  └──────────┘    │ (PCM capture)│    │  (binary +    │  │
│                  └──────────────┘    │   JSON msgs)  │  │
│                                     └───────┬───────┘  │
│  ┌──────────────────────────────────────────┘          │
│  │  Receives: transcript chunks, LLM tokens            │
│  │  Displays: live transcript + LLM response            │
│  └──────────────────────────────────────────────────────┘
└─────────────────────────────────────────────────────────┘
                          │ WebSocket
                          ▼
┌─────────────────────────────────────────────────────────┐
│                   Python Backend (Quart)                 │
│                                                         │
│  ┌─────────────┐   ┌──────────┐   ┌──────────────────┐  │
│  │ Audio Buffer │──▶│ Chunker  │──▶│ ASR (OpenAI API) │  │
│  │ (per conn)  │   │ ~2s +    │   │ via LLM_BASE_URL │  │
│  │             │   │ silence  │   └────────┬─────────┘  │
│  └─────────────┘   │ detect   │            │            │
│                    └──────────┘   transcript text       │
│                                            │            │
│                                   ┌────────▼─────────┐  │
│                                   │ Pause Detector   │  │
│                                   │ (0.4s silence)   │  │
│                                   └────────┬─────────┘  │
│                                            │            │
│                              ┌─────────────▼──────────┐ │
│                              │ LLM (OpenAI Chat API)  │ │
│                              │ Streaming completions   │ │
│                              │ Cancel on new speech    │ │
│                              │ Continue on next pause  │ │
│                              └────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

## Key Behaviors

### Audio Chunking
- Audio streams from browser as raw PCM (16kHz, 16-bit, mono) via WebSocket binary frames.
- Backend buffers audio and produces chunks of ~2 seconds.
- Once the 2s threshold is reached, the chunker waits for a low-volume window (silence) before cutting. A chunk can be shorter than 2s if silence is detected earlier.
- Each chunk is sent to the ASR model immediately.

### Transcription
- ASR uses the OpenAI-compatible `/v1/audio/transcriptions` endpoint at `LLM_BASE_URL` with model `ASR_MODEL`.
- Transcribed text is streamed back to the frontend in real time (one message per chunk).
- The frontend appends transcript text to the current user message bubble.

### LLM Interaction
- On detecting a **speech pause** (0.4s of silence after speech), accumulated transcript text is sent to the LLM.
- The LLM response streams back token-by-token to the frontend.
- If the user **resumes speaking** while the LLM is responding:
  - The LLM request is **cancelled**.
  - The partial LLM response is preserved.
- On the **next pause**, the backend:
  - Includes the previous partial response in context.
  - Asks the LLM to **continue from exactly where it stopped**.
  - New tokens stream from the continuation point.

### Modes
- **Text mode** (existing): Type messages, get streamed LLM responses. No changes.
- **Speech mode** (new): Toggle via mic button. While active, audio streams to backend. Text input is disabled. Clicking mic again stops speech mode.

## Environment Variables

| Variable | Purpose |
|---|---|
| `LLM_BASE_URL` | Base URL for both ASR and LLM APIs |
| `LLM_API_KEY` | API key for both services |
| `LLM_MODEL` | Model for chat completions |
| `ASR_MODEL` | Model for speech recognition |

## Transport

- **WebSocket** at `/ws/speech` for the speech session.
- Binary frames: raw PCM audio from browser → backend.
- Text frames (JSON): control messages and responses from backend → browser.

## UI Changes

- Add a **microphone toggle button** next to the send button.
- While in speech mode:
  - Show a pulsing indicator that mic is active.
  - Display live transcript in a user bubble (grows as chunks arrive).
  - Display LLM response in an assistant bubble (streams in).
  - If LLM is interrupted and resumed, the assistant bubble continues seamlessly.
- Text input area is disabled during speech mode.
