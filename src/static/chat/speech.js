/**
 * SpeechSession — manages mic capture and WebSocket communication for speech mode.
 */
export class SpeechSession {
    constructor({ sessionId = null, mode = 'user' } = {}) {
        this.sessionId = sessionId;
        this.mode = mode;
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
        this.onTranscriptDone = null;
        this.onError = null;
        this.onClose = null;
    }

    async start() {
        if (this.active) return;
        this.active = true;

        // Open WebSocket
        const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
        const params = new URLSearchParams({ mode: this.mode });
        if (this.sessionId) params.set('session_id', this.sessionId);
        const url = `${proto}//${location.host}/ws/speech?${params}`;

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
        await this.audioCtx.audioWorklet.addModule('/static/chat/pcm-processor.js');

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
                this.onLLMCancelled?.(msg.partial_response);
                break;
            case 'error':
                this.onError?.(msg.message);
                break;
        }
    }
}
