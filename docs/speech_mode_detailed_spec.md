# Speech Mode — Detailed Spec

## 1. WebSocket Protocol (`/ws/speech`)

### Connection
- Client opens WebSocket to `/ws/speech?session_id=<id>&mode=<user|dev>`.
- If `session_id` is omitted, a new session is created.
- The server sends an initial `session_start` message with the assigned session ID.

### Client → Server Messages

#### Binary Frames (Audio)
- Raw PCM: 16-bit signed little-endian, mono, 16000 Hz.
- The browser captures audio via `AudioWorklet` and sends it in small frames (~128 or 256 samples per worklet callback, batched to ~4096 samples per WebSocket send for efficiency).

#### Text Frames (JSON Control)
```json
{"type": "stop"}
```
Signals the user has stopped speech mode. Backend finalizes any pending audio chunk and processes remaining transcript.

### Server → Client Messages (JSON Text Frames)

| `type` | Fields | Description |
|---|---|---|
| `session_start` | `session_id` | Sent once on connection |
| `transcript` | `text`, `is_final` | ASR result for a chunk. `is_final=false` for interim, `true` for chunk-final |
| `llm_token` | `token` | A single token from the LLM streaming response |
| `llm_done` | — | LLM finished its response |
| `llm_cancelled` | `partial_response` | LLM was cancelled because user resumed speaking |
| `error` | `message` | Error message |

## 2. Backend Components

### 2.1 `src/speech.py` — WebSocket Handler

```python
@app.websocket("/ws/speech")
async def ws_speech():
    ...
```

**State per connection:**
- `audio_buffer: bytearray` — accumulates raw PCM
- `transcript_parts: list[str]` — accumulated transcript for current utterance
- `llm_task: asyncio.Task | None` — current LLM streaming task (cancellable)
- `partial_llm_response: str` — text generated before cancellation
- `is_speaking: bool` — whether user is currently speaking
- `last_speech_time: float` — timestamp of last detected speech
- `session_id: str`
- `messages: list[dict]` — chat history (shared with text mode sessions)

**Main loop:**
1. Receive WebSocket message.
2. If binary: append to `audio_buffer`, run chunking logic.
3. If text JSON `{"type": "stop"}`: finalize.

### 2.2 `src/audio_chunking.py` — Audio Chunker

#### Constants
```python
SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2  # 16-bit
CHUNK_MIN_SECONDS = 2.0
CHUNK_MIN_BYTES = int(SAMPLE_RATE * BYTES_PER_SAMPLE * CHUNK_MIN_SECONDS)
SILENCE_THRESHOLD_RMS = 500  # RMS threshold for silence detection
SILENCE_WINDOW_MS = 50  # Window size for RMS calculation
PAUSE_SECONDS = 0.4  # Silence duration to trigger LLM
```

#### `AudioChunker` Class
```python
class AudioChunker:
    def __init__(self):
        self.buffer = bytearray()
        self.last_speech_time = 0.0
        self.has_speech = False

    def feed(self, pcm_data: bytes, current_time: float) -> list[ChunkEvent]
```

`feed()` returns a list of events:
- `ChunkEvent(type="chunk", audio=bytes)` — a complete audio chunk ready for ASR
- `ChunkEvent(type="pause")` — 0.4s of silence detected after speech

**Chunking algorithm:**
1. Append `pcm_data` to internal buffer.
2. Compute RMS of the new data in windows of `SILENCE_WINDOW_MS`.
3. If any window exceeds `SILENCE_THRESHOLD_RMS`, mark `has_speech=True` and update `last_speech_time`.
4. If `len(buffer) >= CHUNK_MIN_BYTES` and current window is silent:
   - Emit `ChunkEvent(type="chunk", audio=bytes(buffer))`.
   - Clear buffer.
5. If `has_speech` and `current_time - last_speech_time >= PAUSE_SECONDS`:
   - If buffer has any remaining audio, emit a chunk first.
   - Emit `ChunkEvent(type="pause")`.
   - Reset `has_speech`.

### 2.3 `src/asr.py` — ASR Client

```python
async def transcribe(audio_pcm: bytes, *, base_url: str, api_key: str, model: str) -> str:
```

- Wraps the audio as a WAV in-memory (adds WAV header to raw PCM).
- POST to `{base_url}/v1/audio/transcriptions` with:
  - `file`: the WAV bytes as multipart upload
  - `model`: `ASR_MODEL`
  - `response_format`: `text`
- Returns the transcribed text string.

### 2.4 LLM Interaction (in `src/speech.py`)

#### On Pause Detected
```python
async def handle_pause(state: SpeechState):
    user_text = " ".join(state.transcript_parts)
    if not user_text.strip():
        return

    # Build messages
    if state.partial_llm_response:
        # Continue from where we left off
        state.messages.append({"role": "user", "content": user_text})
        state.messages.append({
            "role": "assistant",
            "content": state.partial_llm_response
        })
        state.messages.append({
            "role": "user",
            "content": "Continue your response from exactly where you stopped. Do not repeat anything."
        })
    else:
        state.messages.append({"role": "user", "content": user_text})

    state.transcript_parts.clear()
    state.llm_task = asyncio.create_task(stream_llm(state))
```

#### On Speech Resumed (while LLM is streaming)
```python
async def handle_speech_resumed(state: SpeechState):
    if state.llm_task and not state.llm_task.done():
        state.llm_task.cancel()
        # partial_llm_response was accumulated during streaming
        await ws.send_json({"type": "llm_cancelled", "partial_response": state.partial_llm_response})
```

#### LLM Streaming
```python
async def stream_llm(state: SpeechState):
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream("POST", f"{base_url}/chat/completions", ...) as resp:
                async for chunk in resp.aiter_raw():
                    # parse SSE, extract tokens
                    for token in extract_tokens(chunk):
                        state.partial_llm_response += token
                        await ws.send_json({"type": "llm_token", "token": token})
        await ws.send_json({"type": "llm_done"})
        state.partial_llm_response = ""  # reset after complete response
    except asyncio.CancelledError:
        pass  # partial_llm_response is preserved
```

## 3. Frontend Components

### 3.1 `src/static/chat/speech.js` — Speech Mode Module

#### Audio Capture
- Uses `navigator.mediaDevices.getUserMedia({ audio: true })` with constraints:
  - `sampleRate: 16000`, `channelCount: 1`, `echoCancellation: true`
- Connects to an `AudioWorkletNode` that captures PCM samples.
- The worklet resamples to 16kHz if needed and sends Float32 → Int16 conversion.
- Batches samples and sends as binary WebSocket frames.

#### WebSocket Management
```javascript
class SpeechSession {
    constructor(sessionId, mode) { ... }
    start()   // opens WS, starts mic
    stop()    // sends stop message, closes mic
    onTranscript(callback)   // transcript chunks
    onLLMToken(callback)     // LLM tokens
    onLLMDone(callback)      // LLM finished
    onLLMCancelled(callback) // LLM interrupted
}
```

#### Audio Worklet (`src/static/chat/pcm-processor.js`)
```javascript
class PCMProcessor extends AudioWorkletProcessor {
    process(inputs) {
        const input = inputs[0][0]; // mono
        if (!input) return true;
        // Convert Float32 [-1,1] to Int16
        const pcm = new Int16Array(input.length);
        for (let i = 0; i < input.length; i++) {
            pcm[i] = Math.max(-32768, Math.min(32767, input[i] * 32768));
        }
        this.port.postMessage(pcm.buffer, [pcm.buffer]);
        return true;
    }
}
```

### 3.2 UI Integration (changes to `chat.js` and `index.html`)

#### New Elements
- **Mic button**: SVG microphone icon, placed left of the send button in the input row.
- **Active indicator**: When speech mode is on, mic button gets a pulsing red ring + the input row shows "Listening..." placeholder.

#### State Machine
```
IDLE ──[click mic]──▶ LISTENING ──[audio chunks]──▶ LISTENING
  ▲                      │                              │
  │                      │ [0.4s pause]                  │
  │                      ▼                              │
  │                 PROCESSING ◄─────────────────────────┘
  │                      │ [LLM streaming]
  │                      ▼
  │                 RESPONDING ──[user speaks]──▶ LISTENING
  │                      │
  │                      │ [LLM done]
  │                      ▼
  │                    IDLE
  │                      │
  └──[click mic]─────────┘
```

#### Bubble Management
- When speech starts: create a new user bubble (empty). Append transcript text as it arrives.
- When LLM starts: create a new assistant bubble. Append tokens as they arrive.
- On interruption: keep the assistant bubble as-is. The user bubble may get more text.
- On continuation: the assistant bubble continues receiving tokens (seamless).

## 4. File Structure

```
src/
├── main.py              # Add WebSocket route registration
├── speech.py            # NEW: WebSocket handler, speech state machine
├── audio_chunking.py    # NEW: Audio chunker with silence detection
├── asr.py               # NEW: ASR client
├── streaming.py         # Existing (unchanged)
├── static/chat/
│   ├── chat.js          # Modified: add mic button, speech mode toggle
│   ├── speech.js        # NEW: SpeechSession class, WebSocket client
│   └── pcm-processor.js # NEW: AudioWorklet for PCM capture
└── templates/
    └── index.html       # Modified: add mic button markup, load speech.js

test/
├── test_main.py              # Existing (unchanged)
├── test_audio_chunking.py    # NEW: chunking + silence detection tests
├── test_asr.py               # NEW: ASR client tests with mock server
├── test_speech.py            # NEW: WebSocket integration tests
└── test_e2e_speech.py        # NEW: E2E test with real audio samples
    audio_fixtures/
    ├── hello_world.wav        # ~2s spoken "hello world"
    ├── pause_resume.wav       # Speech with a pause in the middle
    └── silence.wav            # Pure silence
```

## 5. Testing Strategy

### Unit Tests (`test_audio_chunking.py`)
- Feed synthetic PCM (sine wave + silence) and verify chunk boundaries.
- Verify silence detection at various RMS thresholds.
- Verify pause detection timing.

### ASR Tests (`test_asr.py`)
- Mock the HTTP endpoint, verify WAV header construction and multipart upload format.
- Test error handling (timeout, 500, empty response).

### WebSocket Tests (`test_speech.py`)
- Use Quart test client's WebSocket support.
- Mock ASR + LLM backends.
- Test: audio → transcript → pause → LLM response flow.
- Test: interruption (send audio during LLM response) → cancellation → continuation.

### E2E Tests (`test_e2e_speech.py`)
- Generate real audio samples programmatically (sine wave for speech, silence for pauses).
- Connect via WebSocket, send real PCM data.
- Mock ASR and LLM backends to verify the full pipeline.
- Test the interruption/continuation flow with timed audio.

## 6. Configuration

Add to `docker-compose.yml`:
```yaml
- ASR_MODEL=${ASR_MODEL}
```

Add to `main.py`:
```python
ASR_MODEL = os.environ.get("ASR_MODEL", "whisper-1")
```
