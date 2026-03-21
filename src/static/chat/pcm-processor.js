/**
 * AudioWorklet processor that captures PCM samples and posts them to the main thread.
 * Converts Float32 [-1, 1] to Int16 [-32768, 32767].
 * Batches samples to reduce message overhead (~4096 samples per post).
 */
class PCMProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this._buffer = new Float32Array(4096);
        this._offset = 0;
    }

    process(inputs) {
        const input = inputs[0]?.[0];
        if (!input) return true;

        for (let i = 0; i < input.length; i++) {
            this._buffer[this._offset++] = input[i];
            if (this._offset >= this._buffer.length) {
                this._flush();
            }
        }
        return true;
    }

    _flush() {
        const pcm = new Int16Array(this._offset);
        for (let i = 0; i < this._offset; i++) {
            const s = this._buffer[i];
            pcm[i] = s < 0 ? Math.max(-32768, s * 32768) : Math.min(32767, s * 32767);
        }
        this.port.postMessage(pcm.buffer, [pcm.buffer]);
        this._offset = 0;
        this._buffer = new Float32Array(4096);
    }
}

registerProcessor('pcm-processor', PCMProcessor);
