// ========== Firefly LM Shared API ==========
const API = (() => {
    if (location.hostname === 'localhost' || location.hostname === '127.0.0.1') {
        return 'http://106.14.220.169:8000';
    }
    return 'https://api.firefly-lm.com';
})();

// ========== Auth State ==========
let authToken = localStorage.getItem('firefly_token') || '';
let currentUser = localStorage.getItem('firefly_user') || '';

function getToken() { return authToken; }
function getUser() { return currentUser; }
function setAuth(token, user) {
    authToken = token;
    currentUser = user;
    localStorage.setItem('firefly_token', token);
    localStorage.setItem('firefly_user', user);
}
function clearAuth() {
    authToken = '';
    currentUser = '';
    localStorage.removeItem('firefly_token');
    localStorage.removeItem('firefly_user');
}

// ========== API Call Helper ==========
async function api(path, opts = {}) {
    const headers = { 'Content-Type': 'application/json', ...(opts.headers || {}) };
    if (authToken) headers['Authorization'] = `Bearer ${authToken}`;
    if (opts.body && !(opts.body instanceof FormData)) {
        opts.body = JSON.stringify(opts.body);
    }
    if (opts.body instanceof FormData) {
        delete headers['Content-Type']; // let browser set multipart boundary
    }
    const resp = await fetch(API + path, { ...opts, headers });
    const data = resp.headers.get('content-type')?.includes('json')
        ? await resp.json()
        : await resp.text();
    if (!resp.ok) throw new Error((data && data.detail) || `HTTP ${resp.status}`);
    return data;
}

// ========== Auth ==========
async function login(username, password) {
    const data = await api('/api/v1/auth/login', {
        method: 'POST',
        body: { username, password }
    });
    const token = data.access_token || data.token;
    if (!token) throw new Error('服务器未返回 token');
    setAuth(token, username);
    return data;
}

async function register(username, password, email = '') {
    const data = await api('/api/v1/auth/register', {
        method: 'POST',
        body: { username, password, email }
    });
    return data;
}

function logout() { clearAuth(); }

// ========== Training ==========
async function submitTrain(file, domain) {
    const fd = new FormData();
    fd.append('file', file);
    fd.append('domain', domain);
    fd.append('node_id', currentUser || 'web_user');
    return api('/api/v1/train/submit', { method: 'POST', body: fd });
}

async function submitCPU(domain) {
    return api('/api/v1/train/submit-cpu', {
        method: 'POST',
        body: { domain, node_id: currentUser || 'web_user' }
    });
}

async function submitPaid(file, domain) {
    const fd = new FormData();
    if (file) fd.append('file', file);
    fd.append('domain', domain);
    fd.append('node_id', currentUser || 'web_user');
    return api('/api/v1/train/submit-paid', { method: 'POST', body: fd });
}

async function getStatus(taskId) {
    return api(`/api/v1/train/status/${taskId}`);
}

async function getHistory(limit = 50) {
    return api(`/api/v1/train/history?user=${currentUser}&limit=${limit}`);
}

async function generateData(domain, count = 20) {
    return api(`/api/v1/train/generate?domain=${domain}&count=${count}`, { method: 'POST' });
}

// ========== Aggregation ==========
async function getRounds(domain = '', limit = 10) {
    const q = domain ? `?domain=${domain}&limit=${limit}` : `?limit=${limit}`;
    return api(`/api/v1/aggregation/rounds${q}`);
}

// ========== Inference ==========
async function chat(messages, adapterId = '') {
    return api('/api/v1/inference/v1/chat', {
        method: 'POST',
        body: { messages, adapter_id: adapterId }
    });
}

// ========== Points ==========
async function getPointsBalance() {
    return api(`/api/v1/points/balance?user_id=${currentUser}`);
}

async function getPointsHistory() {
    return api(`/api/v1/points/transactions?user_id=${currentUser}`);
}

// ========== Version ==========
async function getVersion() {
    try { return await api('/api/v1/client/version'); } catch(e) { return null; }
}

// ========== Download Helpers ==========
function downloadResult(taskId) {
    location.href = `${API}/api/v1/train/result/${taskId}?token=${authToken}`;
}
function downloadRound(roundId) {
    location.href = `${API}/api/v1/aggregation/download?round_id=${roundId}&token=${authToken}`;
}

// ========== Toast ==========
function showToast(text, type = 'success', duration = 3000) {
    let el = document.getElementById('toast');
    if (!el) {
        el = document.createElement('div');
        el.id = 'toast';
        el.style.cssText = `
            position:fixed;bottom:24px;left:50%;transform:translateX(-50%) translateY(80px);
            background:#1A1A1A;border:1px solid rgba(255,255,255,0.1);
            padding:12px 24px;border-radius:8px;font-size:0.92em;
            z-index:200;opacity:0;transition:opacity .3s,transform .3s;
            pointer-events:none;color:#E8E8E8;
        `;
        document.body.appendChild(el);
    }
    el.textContent = text;
    el.style.borderColor = type === 'error' ? 'rgba(239,83,80,0.4)' : 'rgba(76,175,80,0.4)';
    el.style.color = type === 'error' ? '#EF5350' : '#4CAF50';
    // Trigger reflow then animate in
    void el.offsetWidth;
    el.style.opacity = '1';
    el.style.transform = 'translateX(-50%) translateY(0)';
    setTimeout(() => {
        el.style.opacity = '0';
        el.style.transform = 'translateX(-50%) translateY(80px)';
    }, duration);
}

// ========== Formatters ==========
function formatSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / 1048576).toFixed(1) + ' MB';
}

function formatDuration(seconds) {
    if (seconds < 60) return Math.floor(seconds) + 's';
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    if (m < 60) return `${m}m ${s}s`;
    const h = Math.floor(m / 60);
    return `${h}h ${m % 60}m`;
}

function escapeHtml(s) {
    const d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
}

function statusBadge(s) {
    const map = {
        pending: ['pending', '排队中'],
        running: ['running', '训练中'],
        completed: ['completed', '✅ 完成'],
        failed: ['failed', '❌ 失败'],
        rejected: ['rejected', '⚠️ 已拒绝']
    };
    const [cls, text] = map[s] || ['pending', s];
    return `<span class="badge ${cls}">${text}</span>`;
}

// ========== Export ==========
window.Firefly = {
    API, login, register, logout,
    submitTrain, submitCPU, submitPaid,
    getStatus, getHistory, generateData,
    getRounds, chat,
    getPointsBalance, getPointsHistory,
    getVersion,
    downloadResult, downloadRound,
    showToast,
    formatSize, formatDuration, escapeHtml, statusBadge,
    getToken, getUser, clearAuth
};
