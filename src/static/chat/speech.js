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
        this.active = false;

        // Callbacks
        this.onSessionStart = null;
        this.onTranscript = null;
        this.onLLMToken = null;
        this.onLLMDone = null;
        this.onLLMCancelled = null;
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
    }

    stop() {
        if (!this.active) return;
        this.active = false;

        // Send stop signal
        if (this.ws?.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ type: 'stop' }));
        }

        // Cleanup mic
        this.workletNode?.disconnect();
        this.stream?.getTracks().forEach((t) => t.stop());
        this.audioCtx?.close();
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
                this.onTranscript?.(msg.text, msg.is_final);
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
