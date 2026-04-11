import MarkdownIt from 'markdown-it';
import createDOMPurify from 'dompurify';
import { SpeechSession } from './speech.js';

// ─── Setup ──────────────────────────────────────────────────────────────────
const md = new MarkdownIt({ breaks: true, linkify: true });
const DOMPurify = createDOMPurify(window);

const CHAT_API_KEY = window.__CHAT_API_KEY__ ?? '';

// ─── DOM refs ────────────────────────────────────────────────────────────────
const shell = document.getElementById('app-shell');
const chatAside = document.getElementById('chat-aside');
const toggleBtn = document.getElementById('chat-toggle-btn');
const closeBtn = document.getElementById('chat-close-btn');
const clearBtn = document.getElementById('chat-clear-btn');
const resizeHandle = document.getElementById('chat-resize-handle');
const msgList = document.getElementById('chat-messages');
const welcome = document.getElementById('chat-welcome');
const input = document.getElementById('chat-input');
const sendBtn = document.getElementById('chat-send-btn');
const modeBtn = document.getElementById('chat-mode-btn');

const MODE_STORAGE_KEY = 'chat-mode';
const USER_MODE = 'user';
const DEV_MODE = 'dev';
const FRONTEND_LOG_LIMIT = 500;
const frontendLogs = [];
const seenFrontendNetworkEvents = new Set();

function pushFrontendLog(level, args) {
    const text = args.map((value) => {
        if (typeof value === 'string') return value;
        try {
            return JSON.stringify(value);
        } catch {
            return String(value);
        }
    }).join(' ');
    frontendLogs.push(`${new Date().toISOString()} ${level} ${text}`);
    if (frontendLogs.length > FRONTEND_LOG_LIMIT) {
        frontendLogs.splice(0, frontendLogs.length - FRONTEND_LOG_LIMIT);
    }
}

for (const level of ['log', 'info', 'warn', 'error', 'debug', 'trace']) {
    const original = console[level].bind(console);
    console[level] = (...args) => {
        pushFrontendLog(level, args);
        original(...args);
    };
}

console.info('chat frontend initialized');

function recordNetworkEvent(kind, details) {
    const key = `${kind}:${details}`;
    if (seenFrontendNetworkEvents.has(key)) return;
    seenFrontendNetworkEvents.add(key);
    pushFrontendLog(kind, [details]);
}

function describeErrorEvent(event) {
    if (event.message) return event.message;

    const target = event.target;
    if (!target || target === window) return 'Unknown error event';

    const tagName = typeof target.tagName === 'string' ? target.tagName.toLowerCase() : 'resource';
    const source = target.currentSrc || target.src || target.href || target.action || '';
    return source ? `Failed to load ${tagName}: ${source}` : `Failed to load ${tagName}`;
}

window.addEventListener('error', (event) => {
    pushFrontendLog('error', [describeErrorEvent(event)]);
}, true);

window.addEventListener('unhandledrejection', (event) => {
    pushFrontendLog('unhandledrejection', [event.reason ?? 'unknown rejection']);
});

const originalFetch = window.fetch.bind(window);
window.fetch = async (...args) => {
    const resource = typeof args[0] === 'string' ? args[0] : args[0]?.url;
    try {
        const response = await originalFetch(...args);
        if (!response.ok) {
            recordNetworkEvent('network', `fetch ${response.status} ${response.url || resource || 'unknown url'}`);
        }
        return response;
    } catch (error) {
        recordNetworkEvent('network', `fetch failed ${resource || 'unknown url'} ${error?.message || error}`);
        throw error;
    }
};

const originalXhrOpen = XMLHttpRequest.prototype.open;
const originalXhrSend = XMLHttpRequest.prototype.send;

XMLHttpRequest.prototype.open = function (method, url, ...rest) {
    this.__frontendLogMethod = method;
    this.__frontendLogUrl = url;
    return originalXhrOpen.call(this, method, url, ...rest);
};

XMLHttpRequest.prototype.send = function (...args) {
    this.addEventListener('loadend', () => {
        if (this.status >= 400) {
            recordNetworkEvent('network', `xhr ${this.status} ${this.__frontendLogMethod || 'GET'} ${this.responseURL || this.__frontendLogUrl || 'unknown url'}`);
        }
    });
    this.addEventListener('error', () => {
        recordNetworkEvent('network', `xhr failed ${this.__frontendLogMethod || 'GET'} ${this.__frontendLogUrl || 'unknown url'}`);
    });
    return originalXhrSend.call(this, ...args);
};

function inspectResourceEntry(entry) {
    const status = typeof entry.responseStatus === 'number' ? entry.responseStatus : null;
    if (status && status >= 400) {
        recordNetworkEvent('resource', `${status} ${entry.initiatorType || 'resource'} ${entry.name}`);
    }
}

if (typeof PerformanceObserver !== 'undefined') {
    try {
        const resourceObserver = new PerformanceObserver((list) => {
            for (const entry of list.getEntries()) {
                inspectResourceEntry(entry);
            }
        });
        resourceObserver.observe({ type: 'resource', buffered: true });
    } catch {
        /* ignore unsupported performance observer modes */
    }
}

for (const entry of performance.getEntriesByType?.('resource') ?? []) {
    inspectResourceEntry(entry);
}

function getStoredMode() {
    try {
        return localStorage.getItem(MODE_STORAGE_KEY);
    } catch {
        return null;
    }
}

// ─── Session state ───────────────────────────────────────────────────────────
// Session ID comes from the server — no client-side storage needed.
let sessionId = null;
let currentMode = getStoredMode() || USER_MODE;
let hasExplicitModeSelection = Boolean(getStoredMode());

function updateInputPlaceholder() {
    input.placeholder = currentMode === DEV_MODE ? 'Ask about implementation or debugging…' : 'Ask about using the app…';
}

function applyMode(mode, { persist = true } = {}) {
    currentMode = mode === DEV_MODE ? DEV_MODE : USER_MODE;
    modeBtn.dataset.mode = currentMode;
    modeBtn.textContent = currentMode === DEV_MODE ? 'Dev mode' : 'User mode';
    const nextMode = currentMode === DEV_MODE ? USER_MODE : DEV_MODE;
    const nextLabel = nextMode === DEV_MODE ? 'dev' : 'user';
    modeBtn.setAttribute('aria-label', `Switch to ${nextLabel} mode`);
    modeBtn.title = `Switch to ${nextLabel} mode`;
    if (persist) {
        try { localStorage.setItem(MODE_STORAGE_KEY, currentMode); } catch { /* ignore */ }
        hasExplicitModeSelection = true;
    }
    updateInputPlaceholder();
}

applyMode(currentMode, { persist: false });

function isMobileLayout() {
    return window.matchMedia('(max-width: 768px)').matches;
}

// ─── Load history from backend ───────────────────────────────────────────────
async function loadHistory() {
    clearHistory({ keepSessionId: false });
    try {
        const res = await fetch(`/v1/sessions/latest?mode=${encodeURIComponent(currentMode)}`, {
            headers: CHAT_API_KEY ? { 'Authorization': `Bearer ${CHAT_API_KEY}` } : {},
        });
        if (!res.ok) return;
        const { session_id, mode, messages } = await res.json();
        if (mode && !hasExplicitModeSelection) {
            applyMode(mode, { persist: false });
        }
        if (mode && currentMode !== mode) return;
        if (!session_id || !messages.length) return;
        sessionId = session_id;
        for (const msg of messages) {
            appendMessage(msg.role, msg.content);
        }
    } catch { /* ignore — chat works fine without history */ }
}

loadHistory();

// ─── Panel toggle ────────────────────────────────────────────────────────────
async function openChat() {
    if (!isMobileLayout()) {
        const saved = parseInt(localStorage.getItem(RESIZE_STORAGE_KEY), 10);
        if (saved >= RESIZE_MIN && saved <= RESIZE_MAX) chatAside.style.width = `${saved}px`;
    } else {
        chatAside.style.width = '';
    }
    shell.classList.add('chat-open');
    input.focus();
}

function closeChat() {
    chatAside.style.width = '';
    shell.classList.remove('chat-open');
}

toggleBtn.addEventListener('click', openChat);
closeBtn.addEventListener('click', closeChat);
clearBtn.addEventListener('click', clearHistory);
modeBtn.addEventListener('click', () => {
    applyMode(currentMode === DEV_MODE ? USER_MODE : DEV_MODE);
    loadHistory();
    input.focus();
});

function clearHistory({ keepSessionId = false } = {}) {
    if (!keepSessionId) sessionId = null;
    msgList.querySelectorAll('.msg').forEach(el => el.remove());
    if (welcome) welcome.style.display = '';
}

// ─── Resize handle ────────────────────────────────────────────────────────────
const RESIZE_MIN = 280;
const RESIZE_MAX = 720;
const RESIZE_STORAGE_KEY = 'chat-width';

(function initResize() {
    resizeHandle.addEventListener('pointerdown', (e) => {
        if (isMobileLayout()) return;
        e.preventDefault();
        resizeHandle.setPointerCapture(e.pointerId);
        chatAside.classList.add('is-resizing');
        document.body.style.userSelect = 'none';
        document.body.style.cursor = 'col-resize';

        const onMove = (ev) => {
            // Aside is on the right; moving left (smaller x) = wider panel
            const rect = chatAside.getBoundingClientRect();
            const width = Math.min(RESIZE_MAX, Math.max(RESIZE_MIN, rect.right - ev.clientX));
            chatAside.style.width = `${width}px`;
        };

        const onUp = () => {
            resizeHandle.releasePointerCapture(e.pointerId);
            chatAside.classList.remove('is-resizing');
            document.body.style.userSelect = '';
            document.body.style.cursor = '';
            try { localStorage.setItem(RESIZE_STORAGE_KEY, parseInt(chatAside.style.width, 10)); } catch { /* ignore */ }
            resizeHandle.removeEventListener('pointermove', onMove);
            resizeHandle.removeEventListener('pointerup', onUp);
        };

        resizeHandle.addEventListener('pointermove', onMove);
        resizeHandle.addEventListener('pointerup', onUp);
    });
})();

// ─── Input behaviour ─────────────────────────────────────────────────────────
input.addEventListener('input', () => {
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 120) + 'px';
    sendBtn.disabled = input.value.trim() === '';
});

input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        if (!sendBtn.disabled) send();
    }
});

sendBtn.addEventListener('click', send);

// ─── Markdown render helper ───────────────────────────────────────────────────
function renderMarkdown(text) {
    return DOMPurify.sanitize(md.render(text));
}

// ─── Append a message bubble ──────────────────────────────────────────────────
// Returns { bubble, body } where body is the content div inside the bubble.
function appendMessage(role, text = '') {
    if (welcome) welcome.style.display = 'none';

    const wrap = document.createElement('div');
    wrap.className = `msg ${role}`;

    const bubble = document.createElement('div');
    bubble.className = 'msg-bubble';

    const body = document.createElement('div');
    body.className = 'msg-bubble-body';

    if (role === 'user') {
        // User text: plain (no markdown injection risk)
        body.textContent = text;
    } else {
        body.innerHTML = renderMarkdown(text || '\u200b');
    }

    bubble.appendChild(body);
    wrap.appendChild(bubble);
    msgList.appendChild(wrap);
    scrollToBottom();

    return { bubble, body };
}

function scrollToBottom() {
    msgList.scrollTop = msgList.scrollHeight;
}

function normalizeLogLimit(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) return 50;
    return Math.max(1, Math.min(500, Math.trunc(number)));
}

function getFrontendLogs(limit) {
    const normalizedLimit = normalizeLogLimit(limit);
    const lines = frontendLogs.slice(-normalizedLimit);
    return {
        system: 'frontend',
        limit: normalizedLimit,
        lines,
        line_count: lines.length,
    };
}

async function handleToolRequest(toolRequest) {
    if (!toolRequest || !toolRequest.tool_call_id) return null;
    const args = toolRequest.arguments ?? {};
    let result;

    if (toolRequest.name === 'get_logs' && args.system === 'frontend') {
        result = getFrontendLogs(args.limit);
    } else {
        result = { error: `Unsupported frontend tool request: ${toolRequest.name}` };
    }

    return {
        tool_call_id: toolRequest.tool_call_id,
        result,
    };
}

async function streamResponse(requestBody, body) {
    let reply = '';
    let sseBuffer = '';
    let pendingToolResults = [];

    while (true) {
        const res = await fetch('/v1/responses', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                ...(CHAT_API_KEY && { 'Authorization': `Bearer ${CHAT_API_KEY}` }),
            },
            body: JSON.stringify(requestBody),
        });

        if (!res.ok) throw new Error(`HTTP ${res.status}`);

        const sid = res.headers.get('X-Session-Id');
        if (sid) sessionId = sid;

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        pendingToolResults = [];

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            sseBuffer += decoder.decode(value, { stream: true });

            while (true) {
                let boundary = sseBuffer.indexOf('\n\n');
                let separatorLength = 2;
                if (boundary === -1) {
                    boundary = sseBuffer.indexOf('\r\n\r\n');
                    separatorLength = 4;
                }
                if (boundary === -1) break;

                const eventText = sseBuffer.slice(0, boundary);
                sseBuffer = sseBuffer.slice(boundary + separatorLength);

                const payload = eventText
                    .split(/\r?\n/)
                    .filter((line) => line.startsWith('data: '))
                    .map((line) => line.slice(6))
                    .join('\n');

                if (!payload || payload === '[DONE]') continue;

                try {
                    const message = JSON.parse(payload);
                    if (message.tool_request) {
                        const toolResult = await handleToolRequest(message.tool_request);
                        if (toolResult) pendingToolResults.push(toolResult);
                        continue;
                    }

                    const delta = message.choices?.[0]?.delta?.content ?? '';
                    if (!delta) continue;
                    reply += delta;
                    body.innerHTML = renderMarkdown(reply);
                    scrollToBottom();
                } catch {
                    /* ignore malformed or incomplete payloads */
                }
            }
        }

        if (!pendingToolResults.length) {
            return reply;
        }

        requestBody = {
            session_id: sessionId,
            mode: currentMode,
            tool_results: pendingToolResults,
        };
    }
}

// ─── Send & stream ────────────────────────────────────────────────────────────
async function send() {
    const text = input.value.trim();
    if (!text) return;

    input.value = '';
    input.style.height = 'auto';
    sendBtn.disabled = true;

    appendMessage('user', text);

    const { bubble, body } = appendMessage('assistant', '');

    // Blinking cursor appended inside the bubble (sibling to body)
    const cursor = document.createElement('span');
    cursor.className = 'chat-cursor';
    bubble.appendChild(cursor);

    let reply = '';

    try {
        reply = await streamResponse({ prompt: text, session_id: sessionId, mode: currentMode }, body);
    } catch (err) {
        body.innerHTML = '';
        body.textContent = `Error: ${err.message}`;
        body.style.color = '#ef4444';
    } finally {
        cursor.remove();
        // Final clean render (ensures consistent output after stream ends)
        if (reply) body.innerHTML = renderMarkdown(reply);
        input.focus();
    }
}

// ─── Speech mode ──────────────────────────────────────────────────────────────
const micBtn = document.getElementById('chat-mic-btn');
const audioViz = document.getElementById('audio-viz');
const NUM_VIZ_BARS = 24;
let speechSession = null;
let speechUserBody = null;
let speechUserText = '';
let speechUserParts = [];
let speechAssistantBody = null;
let speechAssistantBubble = null;
let speechAssistantText = '';
let speechCursor = null;

// Build visualizer bars
(function initVizBars() {
    for (let i = 0; i < NUM_VIZ_BARS; i++) {
        const bar = document.createElement('div');
        bar.className = 'audio-viz-bar';
        audioViz.appendChild(bar);
    }
})();

const vizBars = audioViz.querySelectorAll('.audio-viz-bar');

function updateViz(rms) {
    // rms is 0..1 float; map to bar heights with some randomness for liveliness
    const level = Math.min(1, rms * 5); // amplify — raw mic RMS is usually low
    for (let i = 0; i < vizBars.length; i++) {
        const bar = vizBars[i];
        // Each bar gets a slightly different height for organic look
        const variance = 0.3 + 0.7 * Math.random();
        const h = Math.max(3, Math.round(level * variance * 22));
        bar.style.height = `${h}px`;
        bar.classList.toggle('has-signal', level > 0.05);
        bar.classList.toggle('is-loud', level > 0.5 && variance > 0.7);
    }
}

function resetViz() {
    for (const bar of vizBars) {
        bar.style.height = '3px';
        bar.classList.remove('has-signal', 'is-loud');
    }
}

function setSpeechActive(active) {
    micBtn.classList.toggle('is-active', active);
    input.disabled = active;
    sendBtn.disabled = active;
    input.style.display = active ? 'none' : '';
    audioViz.classList.toggle('is-active', active);
    if (active) {
        resetViz();
    } else {
        resetViz();
        updateInputPlaceholder();
    }
}

async function toggleSpeech() {
    if (speechSession?.active) {
        speechSession.stop();
        setSpeechActive(false);
        return;
    }

    speechSession = new SpeechSession({ sessionId, mode: currentMode });
    speechUserText = '';
    speechUserBody = null;
    speechAssistantText = '';

    speechSession.onSessionStart = (sid) => {
        sessionId = sid;
    };

    speechSession.onTranscript = (text) => {
        // Create a new user bubble if needed (first transcript or after a completed utterance)
        if (!speechUserBody) {
            speechUserText = '';
            speechUserParts = [];
            const userMsg = appendMessage('user', '');
            speechUserBody = userMsg.body;
            speechUserBody.textContent = '';
        }
        speechUserParts.push(text);
        speechUserText = speechUserParts.join(' ');
        speechUserBody.textContent = speechUserText;
        scrollToBottom();
    };

    speechSession.onTranscriptReplace = (replaceLast, text) => {
        if (!speechUserBody || !speechUserParts.length) return;
        // Remove the last N parts and replace with the merged text
        speechUserParts.splice(-replaceLast, replaceLast, text);
        speechUserText = speechUserParts.join(' ');
        speechUserBody.textContent = speechUserText;
        scrollToBottom();
    };

    speechSession.onTranscriptDone = () => {
        // Current utterance is finalized — next transcript will create a new bubble
        speechUserBody = null;
        speechUserParts = [];
    };

    speechSession.onLLMToken = (token) => {
        if (!speechAssistantBody) {
            const msg = appendMessage('assistant', '');
            speechAssistantBubble = msg.bubble;
            speechAssistantBody = msg.body;
            speechAssistantText = '';
            speechCursor = document.createElement('span');
            speechCursor.className = 'chat-cursor';
            speechAssistantBubble.appendChild(speechCursor);
        }
        speechAssistantText += token;
        speechAssistantBody.innerHTML = renderMarkdown(speechAssistantText);
        scrollToBottom();
    };

    speechSession.onLLMDone = () => {
        speechCursor?.remove();
        speechCursor = null;
        if (speechAssistantText && speechAssistantBody) {
            speechAssistantBody.innerHTML = renderMarkdown(speechAssistantText);
        }
        // is-speaking will be removed by onTTSPlaybackChange(false) when audio finishes
        // Reset for next utterance within the same speech session
        speechAssistantBody = null;
        speechAssistantBubble = null;
        speechAssistantText = '';
    };

    speechSession.onLLMCancelled = (partialResponse) => {
        // Keep the partial response visible — it will be continued
        speechCursor?.remove();
        speechCursor = null;
        // Don't reset speechAssistantBody — continuation will append to it
        speechAssistantText = partialResponse || speechAssistantText;
    };

    let ttsBubble = null;
    speechSession.onTTSPlaybackChange = (playing) => {
        if (playing) {
            ttsBubble = speechAssistantBubble;
        }
        ttsBubble?.classList.toggle('is-speaking', playing);
        if (!playing) ttsBubble = null;
    };

    speechSession.onAudioLevel = (rms) => {
        updateViz(rms);
    };

    speechSession.onError = (msg) => {
        console.error('Speech error:', msg);
    };

    speechSession.onClose = () => {
        setSpeechActive(false);
        speechCursor?.remove();
        speechCursor = null;
        speechAssistantBody = null;
        speechAssistantBubble = null;
        // Remove empty trailing user bubble
        if (speechUserBody && !speechUserBody.textContent.trim()) {
            speechUserBody.closest('.msg')?.remove();
        }
    };

    try {
        await speechSession.start();
        setSpeechActive(true);
    } catch (err) {
        console.error('Failed to start speech:', err);
    }
}

if (micBtn) {
    micBtn.addEventListener('click', toggleSpeech);
}
