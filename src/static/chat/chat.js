import MarkdownIt from 'markdown-it';
import createDOMPurify from 'dompurify';
import { SpeechSession } from './speech.js';

// ─── Setup ──────────────────────────────────────────────────────────────────
const md = new MarkdownIt({ breaks: true, linkify: true });
const DOMPurify = createDOMPurify(window);
const CHAT_API_KEY = window.__CHAT_API_KEY__ ?? '';

const authHeaders = (extra = {}) => (CHAT_API_KEY ? { Authorization: `Bearer ${CHAT_API_KEY}`, ...extra } : extra);

// ─── Frontend log capture (bridge for the get_logs tool) ──────────────────────
const FRONTEND_LOG_LIMIT = 500;
const frontendLogs = [];
const seenNetworkEvents = new Set();

function pushFrontendLog(level, args) {
    const text = args.map((v) => {
        if (typeof v === 'string') return v;
        try { return JSON.stringify(v); } catch { return String(v); }
    }).join(' ');
    frontendLogs.push(`${new Date().toISOString()} ${level} ${text}`);
    if (frontendLogs.length > FRONTEND_LOG_LIMIT) frontendLogs.splice(0, frontendLogs.length - FRONTEND_LOG_LIMIT);
}

for (const level of ['log', 'info', 'warn', 'error', 'debug']) {
    const original = console[level].bind(console);
    console[level] = (...args) => { pushFrontendLog(level, args); original(...args); };
}
console.info('chat frontend initialized');

function recordNetworkEvent(kind, details) {
    const key = `${kind}:${details}`;
    if (seenNetworkEvents.has(key)) return;
    seenNetworkEvents.add(key);
    pushFrontendLog(kind, [details]);
}

window.addEventListener('error', (e) => pushFrontendLog('error', [e.message || 'error']), true);
window.addEventListener('unhandledrejection', (e) => pushFrontendLog('unhandledrejection', [e.reason ?? 'unknown']));

const originalFetch = window.fetch.bind(window);
window.fetch = async (...args) => {
    const resource = typeof args[0] === 'string' ? args[0] : args[0]?.url;
    try {
        const res = await originalFetch(...args);
        if (!res.ok) recordNetworkEvent('network', `fetch ${res.status} ${res.url || resource || 'unknown'}`);
        return res;
    } catch (err) {
        recordNetworkEvent('network', `fetch failed ${resource || 'unknown'} ${err?.message || err}`);
        throw err;
    }
};

function normalizeLogLimit(value) {
    const n = Number(value);
    if (!Number.isFinite(n)) return 50;
    return Math.max(1, Math.min(500, Math.trunc(n)));
}

function getFrontendLogs(limit) {
    const n = normalizeLogLimit(limit);
    const lines = frontendLogs.slice(-n);
    return { system: 'frontend', limit: n, lines, line_count: lines.length };
}

async function handleToolRequest(req) {
    if (!req || !req.tool_call_id) return null;
    const args = req.arguments ?? {};
    let result;
    if (req.name === 'get_logs' && args.system === 'frontend') result = getFrontendLogs(args.limit);
    else result = { error: `Unsupported frontend tool request: ${req.name}` };
    return { tool_call_id: req.tool_call_id, result };
}

// ─── DOM refs ─────────────────────────────────────────────────────────────────
const body = document.body;
const conversation = document.getElementById('conversation');
const messagesEl = document.getElementById('messages');
const input = document.getElementById('input');
const sendBtn = document.getElementById('send-btn');
const micBtn = document.getElementById('mic-btn');
const stopBtn = document.getElementById('stop-btn');
const voiceBtn = document.getElementById('voice-btn');
const attachBtn = document.getElementById('attach-btn');
const fileInput = document.getElementById('file-input');
const imagePreviews = document.getElementById('image-previews');
const viz = document.getElementById('viz');
const chipWeb = document.getElementById('chip-web');
const chipResearch = document.getElementById('chip-research');
const newChatBtn = document.getElementById('new-chat-btn');
const hint = document.getElementById('hint');

// ─── State ────────────────────────────────────────────────────────────────────
let sessionId = null;
let pendingImages = [];     // array of data URLs
let webSearch = false;
let deepResearch = false;
let busy = false;
let dictationSession = null;
let speech = null;

// ─── Helpers ──────────────────────────────────────────────────────────────────
function renderMarkdown(text) { return DOMPurify.sanitize(md.render(text)); }
function scrollToBottom() { conversation.scrollTop = conversation.scrollHeight; }

function markNotEmpty() { body.classList.remove('empty'); }

function appendMessage(role, text = '', images = []) {
    markNotEmpty();
    const wrap = document.createElement('div');
    wrap.className = `msg ${role}`;

    if (images.length) {
        const imgRow = document.createElement('div');
        imgRow.className = 'images';
        for (const url of images) {
            const img = document.createElement('img');
            img.src = url;
            imgRow.appendChild(img);
        }
        wrap.appendChild(imgRow);
    }

    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    if (role === 'user') bubble.textContent = text;
    else bubble.innerHTML = renderMarkdown(text || '​');

    wrap.appendChild(bubble);
    messagesEl.appendChild(wrap);
    return { wrap, bubble };
}

// Scroll so the just-sent user message sits at the top of the viewport, with
// room reserved below for the streaming reply. The user can then read top-down
// and scroll freely — the reply does not auto-scroll.
function anchorUserMessageToTop(userWrap, assistantWrap) {
    const reserve = conversation.clientHeight - userWrap.getBoundingClientRect().height - 32;
    assistantWrap.style.minHeight = `${Math.max(0, reserve)}px`;
    const delta = userWrap.getBoundingClientRect().top - conversation.getBoundingClientRect().top;
    conversation.scrollTop += delta - 8;
}

// ─── Load history ──────────────────────────────────────────────────────────────
async function loadHistory() {
    try {
        const res = await fetch('/v1/sessions/latest', { headers: authHeaders() });
        if (!res.ok) return;
        const { session_id, messages } = await res.json();
        if (!session_id || !messages?.length) return;
        sessionId = session_id;
        for (const m of messages) appendMessage(m.role, m.content);
        scrollToBottom();
    } catch { /* history is best-effort */ }
}
loadHistory();

// ─── Input behaviour ────────────────────────────────────────────────────────────
function refreshComposerButtons() {
    const hasText = input.value.trim() !== '';
    const dictating = !!dictationSession?.active;
    // While dictating, keep the send button visible (disabled until there is
    // text) so the button layout stays stable and the user doesn't misclick as
    // buttons appear/disappear. The stop button replaces the mic button.
    sendBtn.style.display = (hasText || dictating) ? '' : 'none';
    sendBtn.disabled = !hasText || busy;
    micBtn.style.display = (hasText || dictating) ? 'none' : '';
}

input.addEventListener('input', () => {
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 200) + 'px';
    refreshComposerButtons();
});

input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        if (!sendBtn.disabled) send();
    }
});

sendBtn.addEventListener('click', () => { if (!sendBtn.disabled) send(); });

// ─── Tool-call status ───────────────────────────────────────────────────────────
const TOOL_LABELS = {
    web_search: 'Searching the web',
    fetch_url: 'Reading a page',
    python: 'Running Python',
    bash: 'Running a command',
    get_logs: 'Reading logs',
    publish_document: 'Creating the PDF',
};

function updateToolStatus(el, status) {
    if (!status || !status.name) { el.style.display = 'none'; el.innerHTML = ''; return; }
    const label = TOOL_LABELS[status.name] || status.name;
    const args = status.arguments || {};
    let detail = '';
    if (status.name === 'web_search' && args.query) detail = `: “${args.query}”`;
    else if (status.name === 'fetch_url' && args.url) {
        try { detail = `: ${new URL(args.url).host}`; } catch { /* ignore */ }
    }
    el.innerHTML = '';
    const spinner = document.createElement('span');
    spinner.className = 'spinner';
    const text = document.createElement('span');
    text.textContent = `${label}${detail}…`;
    el.append(spinner, text);
    el.style.display = '';
}

// ─── Streaming ──────────────────────────────────────────────────────────────────
async function streamResponse(requestBody, bubble, statusEl) {
    let reply = '';
    let sseBuffer = '';
    let pendingToolResults = [];

    while (true) {
        const res = await fetch('/v1/responses', {
            method: 'POST',
            headers: authHeaders({ 'Content-Type': 'application/json' }),
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
                let sep = 2;
                if (boundary === -1) { boundary = sseBuffer.indexOf('\r\n\r\n'); sep = 4; }
                if (boundary === -1) break;
                const eventText = sseBuffer.slice(0, boundary);
                sseBuffer = sseBuffer.slice(boundary + sep);

                const payload = eventText.split(/\r?\n/).filter((l) => l.startsWith('data: ')).map((l) => l.slice(6)).join('\n');
                if (!payload || payload === '[DONE]') continue;

                try {
                    const message = JSON.parse(payload);
                    if ('tool_status' in message) {
                        updateToolStatus(statusEl, message.tool_status);
                        continue;
                    }
                    if (message.tool_request) {
                        const toolResult = await handleToolRequest(message.tool_request);
                        if (toolResult) pendingToolResults.push(toolResult);
                        continue;
                    }
                    const delta = message.choices?.[0]?.delta?.content ?? '';
                    if (!delta) continue;
                    reply += delta;
                    bubble.innerHTML = renderMarkdown(reply);
                    // No auto-scroll while streaming — the user reads/scrolls freely.
                } catch { /* ignore partial payloads */ }
            }
        }

        if (!pendingToolResults.length) return reply;
        requestBody = { session_id: sessionId, tool_results: pendingToolResults };
    }
}

async function send() {
    // Pressing send ends dictation; the streamed text is what gets sent.
    if (dictationSession?.active) endDictation({ keepHandlers: false });
    const text = input.value.trim();
    if (!text || busy) return;
    busy = true;

    const images = pendingImages.slice();
    input.value = '';
    input.style.height = 'auto';
    clearImages();
    refreshComposerButtons();

    // Drop any space reserved by a previous turn so messages sit flush.
    conversation.querySelectorAll('.msg').forEach((m) => { m.style.minHeight = ''; });

    const { wrap: userWrap } = appendMessage('user', text, images);
    const { wrap, bubble } = appendMessage('assistant', '');
    const cursor = document.createElement('span');
    cursor.className = 'cursor';
    bubble.appendChild(cursor);

    // Pin the user message to the top; the reply streams in below without
    // auto-scrolling so the user stays in control.
    anchorUserMessageToTop(userWrap, wrap);

    // One-line "currently running tool" indicator below the response.
    const statusEl = document.createElement('div');
    statusEl.className = 'tool-status';
    statusEl.style.display = 'none';
    wrap.appendChild(statusEl);

    const requestBody = { prompt: text, session_id: sessionId };
    if (images.length) requestBody.images = images;
    if (deepResearch) requestBody.deep_research = true;
    else if (webSearch) requestBody.web_search = true;

    let reply = '';
    try {
        reply = await streamResponse(requestBody, bubble, statusEl);
    } catch (err) {
        bubble.innerHTML = '';
        bubble.textContent = `Error: ${err.message}`;
        bubble.style.color = 'var(--danger)';
    } finally {
        cursor.remove();
        statusEl.remove();
        if (reply) bubble.innerHTML = renderMarkdown(reply);
        busy = false;
        refreshComposerButtons();
        input.focus();
    }
}

// ─── Image upload ───────────────────────────────────────────────────────────────
const MAX_IMAGES = 8;

attachBtn.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', async () => {
    for (const file of fileInput.files) {
        if (pendingImages.length >= MAX_IMAGES) break;
        if (!file.type.startsWith('image/')) continue;
        const dataUrl = await readFileAsDataURL(file);
        pendingImages.push(dataUrl);
    }
    fileInput.value = '';
    renderImagePreviews();
});

function readFileAsDataURL(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
}

function renderImagePreviews() {
    imagePreviews.innerHTML = '';
    pendingImages.forEach((url, i) => {
        const thumb = document.createElement('div');
        thumb.className = 'thumb';
        const img = document.createElement('img');
        img.src = url;
        const rm = document.createElement('button');
        rm.textContent = '×';
        rm.title = 'Remove';
        rm.addEventListener('click', () => { pendingImages.splice(i, 1); renderImagePreviews(); });
        thumb.append(img, rm);
        imagePreviews.appendChild(thumb);
    });
}

function clearImages() { pendingImages = []; renderImagePreviews(); }

// ─── Capability chips ───────────────────────────────────────────────────────────
chipWeb.addEventListener('click', () => {
    webSearch = !webSearch;
    if (webSearch) deepResearch = false;
    syncChips();
});
chipResearch.addEventListener('click', () => {
    deepResearch = !deepResearch;
    if (deepResearch) webSearch = false;
    syncChips();
});
function syncChips() {
    chipWeb.classList.toggle('active', webSearch);
    chipResearch.classList.toggle('active', deepResearch);
}

// ─── New chat ───────────────────────────────────────────────────────────────────
newChatBtn.addEventListener('click', () => {
    if (speech?.active) toggleVoice();
    if (dictationSession?.active) endDictation({ keepHandlers: false });
    sessionId = null;
    conversation.querySelectorAll('.msg').forEach((el) => el.remove());
    clearImages();
    webSearch = false; deepResearch = false; syncChips();
    body.classList.add('empty');
    input.focus();
});

// ─── Audio visualizer bars ──────────────────────────────────────────────────────
const NUM_BARS = 48;
for (let i = 0; i < NUM_BARS; i++) {
    const bar = document.createElement('div');
    bar.className = 'viz-bar';
    viz.appendChild(bar);
}
const vizBars = viz.querySelectorAll('.viz-bar');
function updateViz(rms) {
    const level = Math.min(1, rms * 5);
    for (const bar of vizBars) {
        const variance = 0.3 + 0.7 * Math.random();
        bar.style.height = `${Math.max(3, Math.round(level * variance * 22))}px`;
        bar.classList.toggle('signal', level > 0.05);
    }
}
function resetViz() { for (const bar of vizBars) { bar.style.height = '3px'; bar.classList.remove('signal'); } }

// ─── Dictation (mic button) ─────────────────────────────────────────────────────
// Streams chunk transcripts straight into the text box via the speech WebSocket
// in "dictation" mode (no LLM / no TTS on the backend). While dictating, a stop
// button replaces the mic button and pulses with the input loudness.
let dictationBaseline = '';   // text already in the box before dictation started
let dictationParts = [];      // streamed transcript chunks

function renderDictation() {
    const dictated = dictationParts.join(' ').trim();
    const sep = dictationBaseline && dictated ? ' ' : '';
    input.value = dictationBaseline + sep + dictated;
    input.dispatchEvent(new Event('input'));
}

function setDictationUI(active) {
    stopBtn.style.display = active ? '' : 'none';
    attachBtn.style.display = active ? 'none' : '';
    voiceBtn.style.display = active ? 'none' : '';
    if (!active) stopBtn.style.transform = '';
    refreshComposerButtons();
}

// Scale the stop button with loudness — a simple volume indicator.
function dictationLoudness(rms) {
    const scale = 1 + Math.min(0.45, rms * 4.5);
    stopBtn.style.transform = `scale(${scale.toFixed(3)})`;
}

// Tear down the dictation session. keepHandlers=true lets a final in-flight
// transcript still land in the box (used by the stop button); false detaches
// them so nothing arrives after send.
function endDictation({ keepHandlers } = { keepHandlers: true }) {
    const sess = dictationSession;
    if (!sess) return;
    if (!keepHandlers) { sess.onTranscript = null; sess.onTranscriptReplace = null; }
    if (sess.active) sess.stop();
    setDictationUI(false);
}

async function startDictation() {
    if (speech?.active) toggleVoice();
    dictationBaseline = input.value.trim();
    dictationParts = [];

    dictationSession = new SpeechSession({ sessionId, dictation: true });
    dictationSession.onSessionStart = (sid) => { sessionId = sid; };
    dictationSession.onTranscript = (text) => { dictationParts.push(text); renderDictation(); };
    dictationSession.onTranscriptReplace = (replaceLast, text) => {
        if (!dictationParts.length) return;
        dictationParts.splice(-replaceLast, replaceLast, text);
        renderDictation();
    };
    dictationSession.onAudioLevel = (rms) => dictationLoudness(rms);
    dictationSession.onError = (m) => console.error('Dictation error:', m);
    dictationSession.onClose = () => setDictationUI(false);

    try {
        await dictationSession.start();
        setDictationUI(true);
    } catch (e) {
        console.error('Microphone access failed', e);
        dictationSession = null;
        setDictationUI(false);
    }
}

micBtn.addEventListener('click', startDictation);
// Pressing stop ends dictation; the streamed text stays in the box.
stopBtn.addEventListener('click', () => endDictation({ keepHandlers: true }));

// ─── Voice conversation (voice button) ──────────────────────────────────────────
let vUserBubble = null, vUserParts = [], vAssistantBubble = null, vAssistantText = '', vCursor = null;

function setVoiceActive(active) {
    voiceBtn.classList.toggle('active', active);
    input.style.display = active ? 'none' : '';
    viz.classList.toggle('active', active);
    attachBtn.style.display = active ? 'none' : '';
    micBtn.style.display = active ? 'none' : (input.value.trim() ? 'none' : '');
    sendBtn.style.display = active ? 'none' : (input.value.trim() ? '' : 'none');
    if (active) resetViz(); else { resetViz(); refreshComposerButtons(); }
}

async function toggleVoice() {
    if (speech?.active) { speech.stop(); setVoiceActive(false); return; }
    if (dictationSession?.active) endDictation({ keepHandlers: true });

    speech = new SpeechSession({ sessionId });
    vAssistantText = ''; vAssistantBubble = null; vUserBubble = null; vUserParts = [];

    speech.onSessionStart = (sid) => { sessionId = sid; };

    speech.onTranscript = (text) => {
        if (!vUserBubble) { vUserParts = []; vUserBubble = appendMessage('user', '').bubble; }
        vUserParts.push(text);
        vUserBubble.textContent = vUserParts.join(' ');
        scrollToBottom();
    };
    speech.onTranscriptReplace = (replaceLast, text) => {
        if (!vUserBubble || !vUserParts.length) return;
        vUserParts.splice(-replaceLast, replaceLast, text);
        vUserBubble.textContent = vUserParts.join(' ');
        scrollToBottom();
    };
    speech.onTranscriptDone = () => { vUserBubble = null; vUserParts = []; };

    speech.onLLMToken = (token) => {
        if (!vAssistantBubble) {
            vAssistantBubble = appendMessage('assistant', '').bubble;
            vAssistantText = '';
            vCursor = document.createElement('span');
            vCursor.className = 'cursor';
            vAssistantBubble.appendChild(vCursor);
        }
        vAssistantText += token;
        vAssistantBubble.innerHTML = renderMarkdown(vAssistantText);
        scrollToBottom();
    };
    speech.onLLMDone = () => {
        vCursor?.remove(); vCursor = null;
        if (vAssistantText && vAssistantBubble) vAssistantBubble.innerHTML = renderMarkdown(vAssistantText);
        vAssistantBubble = null; vAssistantText = '';
    };
    speech.onLLMCancelled = (partial) => {
        vCursor?.remove(); vCursor = null;
        vAssistantText = partial || vAssistantText;
    };

    let ttsBubble = null;
    speech.onTTSPlaybackChange = (playing) => {
        if (playing) ttsBubble = vAssistantBubble;
        ttsBubble?.classList.toggle('is-speaking', playing);
        if (!playing) ttsBubble = null;
    };
    speech.onAudioLevel = (rms) => updateViz(rms);
    speech.onError = (m) => console.error('Speech error:', m);
    speech.onClose = () => {
        setVoiceActive(false);
        vCursor?.remove(); vCursor = null;
        vAssistantBubble = null;
        if (vUserBubble && !vUserBubble.textContent.trim()) vUserBubble.closest('.msg')?.remove();
    };

    try {
        await speech.start();
        setVoiceActive(true);
    } catch (e) {
        console.error('Failed to start voice:', e);
    }
}

voiceBtn.addEventListener('click', toggleVoice);
