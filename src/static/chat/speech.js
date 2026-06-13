/**
 * SpeechSession — manages mic capture and WebSocket communication for speech mode.
 */
export class SpeechSession {
    constructor({ sessionId = null, mode = 'user', dictation = false } = {}) {
        this.sessionId = sessionId;
        this.mode = mode;
        this.dictation = dictation;
        this.ws = null;
        this.stream = null;
        this.audioCtx = null;
        this.workletNode = null;
        this.analyser = null;
        this._levelRAF = null;
        this.active = false;

        // Callbacks
        this.onSessionStart = null;
        this.onTranscript = null;
        this.onLLMToken = null;
        this.onLLMDone = null;
        this.onLLMCancelled = null;
        this.onAudioLevel = null;  // (rms: number 0..1) called per worklet batch
        this.onTranscriptReplace = null; // (replaceLast: number, text: string)
        this.onTranscriptDone = null;
        this.onError = null;
        this.onClose = null;

        // TTS audio playback queue
        this._ttsQueue = [];
        this._ttsPlaying = false;
        this._ttsSource = null;
        this._ttsCtx = null;
        this.onTTSPlaybackChange = null; // (playing: boolean) called when TTS starts/stops
    }

    async start() {
        if (this.active) return;
        this.active = true;

        // Open WebSocket
        const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
        const params = new URLSearchParams({ mode: this.mode });
        if (this.sessionId) params.set('session_id', this.sessionId);
        if (this.dictation) params.set('dictation', '1');
        const root = window.APP_ROOT || '';
        const url = `${proto}//${location.host}${root}/ws/speech?${params}`;

        this.ws = new WebSocket(url);
        this.ws.binaryType = 'arraybuffer';

        this.ws.onmessage = (event) => this._handleMessage(event);
        this.ws.onerror = () => this.onError?.('WebSocket error');
        this.ws.onclose = () => {
            this.active = false;
            this.onClose?.();
        };

        // Wait for WebSocket to open
        await new Promise((resolve, reject) => {
            this.ws.onopen = resolve;
            this.ws.onerror = reject;
        });

        // Start mic capture
        try {
            this.stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    channelCount: 1,
                    sampleRate: 16000,
                    echoCancellation: true,
                    noiseSuppression: true,
                },
            });
        } catch (err) {
            this.onError?.(`Microphone access denied: ${err.message}`);
            this.ws.close();
            this.active = false;
            return;
        }

        this.audioCtx = new AudioContext({ sampleRate: 16000 });
        await this.audioCtx.audioWorklet.addModule((window.APP_ROOT || '') + '/static/chat/pcm-processor.js');

        const source = this.audioCtx.createMediaStreamSource(this.stream);
        this.workletNode = new AudioWorkletNode(this.audioCtx, 'pcm-processor');

        this.workletNode.port.onmessage = (event) => {
            if (this.ws?.readyState === WebSocket.OPEN) {
                this.ws.send(event.data);
            }
        };

        source.connect(this.workletNode);
        this.workletNode.connect(this.audioCtx.destination);

        // Use AnalyserNode for reliable audio level metering
        this.analyser = this.audioCtx.createAnalyser();
        this.analyser.fftSize = 256;
        source.connect(this.analyser);
        this._startLevelMeter();
    }

    _startLevelMeter() {
        const buf = new Uint8Array(this.analyser.frequencyBinCount);
        const tick = () => {
            if (!this.active || !this.analyser) return;
            this.analyser.getByteTimeDomainData(buf);
            // Compute RMS from time-domain data (byte values centered at 128)
            let sumSq = 0;
            for (let i = 0; i < buf.length; i++) {
                const v = (buf[i] - 128) / 128;
                sumSq += v * v;
            }
            const rms = Math.sqrt(sumSq / buf.length);
            this.onAudioLevel?.(Math.min(1, rms));
            this._levelRAF = requestAnimationFrame(tick);
        };
        this._levelRAF = requestAnimationFrame(tick);
    }

    _stopLevelMeter() {
        if (this._levelRAF != null) {
            cancelAnimationFrame(this._levelRAF);
            this._levelRAF = null;
        }
    }

    stop() {
        if (!this.active) return;
        this.active = false;
        this._stopLevelMeter();
        this._stopTTS();

        // Send stop signal
        if (this.ws?.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ type: 'stop' }));
        }

        // Cleanup mic
        this.analyser?.disconnect();
        this.workletNode?.disconnect();
        this.stream?.getTracks().forEach((t) => t.stop());
        this.audioCtx?.close();
        this.analyser = null;
        this.workletNode = null;
        this.stream = null;
        this.audioCtx = null;

        // Close WS after a short delay to let the stop message through
        setTimeout(() => {
            if (this.ws?.readyState === WebSocket.OPEN) {
                this.ws.close();
            }
            this.ws = null;
        }, 500);
    }

    _handleMessage(event) {
        if (typeof event.data !== 'string') return;
        let msg;
        try {
            msg = JSON.parse(event.data);
        } catch {
            return;
        }

        switch (msg.type) {
            case 'session_start':
                this.sessionId = msg.session_id;
                this.onSessionStart?.(msg.session_id);
                break;
            case 'transcript':
                this.onTranscript?.(msg.text);
                break;
            case 'transcript_replace':
                this.onTranscriptReplace?.(msg.replace_last, msg.text);
                break;
            case 'transcript_done':
                this.onTranscriptDone?.();
                break;
            case 'llm_token':
                this.onLLMToken?.(msg.token);
                break;
            case 'llm_done':
                this.onLLMDone?.();
                break;
            case 'llm_cancelled':
                this._stopTTS();
                this.onLLMCancelled?.(msg.partial_response);
                break;
            case 'stop_audio':
                // Server detected barge-in after the LLM finished; stop playback.
                this._stopTTS();
                break;
            case 'tts_audio':
                this._enqueueTTS(msg.audio_base64);
                break;
            case 'error':
                this.onError?.(msg.message);
                break;
        }
    }

    _enqueueTTS(base64Audio) {
        this._ttsQueue.push(base64Audio);
        if (!this._ttsPlaying) {
            this._playNextTTS();
        }
    }

    /** Notify UI and server when TTS playback starts or stops. */
    _notifyPlayback(playing) {
        this.onTTSPlaybackChange?.(playing);
        if (this.ws?.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ type: 'playback_state', playing }));
        }
    }

    async _playNextTTS() {
        if (!this._ttsQueue.length) {
            this._ttsPlaying = false;
            this._ttsSource = null;
            this._ttsCtx = null;
            this._notifyPlayback(false);
            return;
        }
        const wasPlaying = this._ttsPlaying;
        this._ttsPlaying = true;
        if (!wasPlaying) this._notifyPlayback(true);

        const base64 = this._ttsQueue.shift();
        try {
            const binary = atob(base64);
            const bytes = new Uint8Array(binary.length);
            for (let i = 0; i < binary.length; i++) {
                bytes[i] = binary.charCodeAt(i);
            }
            // Use a dedicated AudioContext for playback (not the mic one)
            const ctx = new AudioContext();
            const audioBuffer = await ctx.decodeAudioData(bytes.buffer);
            const source = ctx.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(ctx.destination);
            this._ttsCtx = ctx;
            this._ttsSource = source;
            source.onended = () => {
                ctx.close();
                this._ttsSource = null;
                this._ttsCtx = null;
                this._playNextTTS();
            };
            source.start(0);
        } catch (err) {
            console.error('TTS playback error:', err);
            this._ttsSource = null;
            this._ttsCtx = null;
            this._playNextTTS();
        }
    }

    _stopTTS() {
        this._ttsQueue.length = 0;
        const wasPlaying = this._ttsPlaying;
        this._ttsPlaying = false;
        try { this._ttsSource?.stop(); } catch { /* ignore */ }
        try { this._ttsCtx?.close(); } catch { /* ignore */ }
        this._ttsSource = null;
        this._ttsCtx = null;
        if (wasPlaying) this._notifyPlayback(false);
    }
}
