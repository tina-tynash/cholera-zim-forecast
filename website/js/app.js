'use strict';

// ── API config ─────────────────────────────────────────────────────────────────
const API_BASE = (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')
  ? 'http://localhost:8000'
  : 'https://api.cholsurv-zim.org';

// ── Auth store ─────────────────────────────────────────────────────────────────
const Auth = {
  getToken()    { return sessionStorage.getItem('access_token'); },
  getRefresh()  { return sessionStorage.getItem('refresh_token'); },
  getUser()     { try { return JSON.parse(sessionStorage.getItem('user') || 'null'); } catch { return null; } },
  save(data)    {
    if (data.access_token)  sessionStorage.setItem('access_token',  data.access_token);
    if (data.refresh_token) sessionStorage.setItem('refresh_token', data.refresh_token);
    if (data.user)          sessionStorage.setItem('user', JSON.stringify(data.user));
  },
  clear()       { ['access_token','refresh_token','user'].forEach(k => sessionStorage.removeItem(k)); },
  isLoggedIn()  { return !!this.getToken(); },
  hasRole(r)    {
    const u = this.getUser();
    const rank = { viewer:0, researcher:1, admin:2 };
    return u && (rank[u.role] ?? -1) >= (rank[r] ?? 99);
  }
};

// ── API client ─────────────────────────────────────────────────────────────────
const API = {
  async _fetch(path, opts = {}) {
    const headers = { 'Content-Type': 'application/json', ...(opts.headers || {}) };
    const token = Auth.getToken();
    if (token) headers['Authorization'] = `Bearer ${token}`;
    const res = await fetch(API_BASE + path, { ...opts, headers });

    if (res.status === 401 && Auth.getRefresh()) {
      const ok = await this._refresh();
      if (ok) return this._fetch(path, opts);
      Auth.clear();
      window.location.href = '/website/login.html';
      return;
    }
    return res;
  },

  async _refresh() {
    try {
      const res = await fetch(API_BASE + '/auth/refresh', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ refresh_token: Auth.getRefresh() }),
      });
      if (!res.ok) return false;
      const data = await res.json();
      Auth.save(data);
      return true;
    } catch { return false; }
  },

  async login(username, password) {
    const form = new URLSearchParams({ username, password });
    const res  = await fetch(API_BASE + '/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: form,
    });
    return { ok: res.ok, status: res.status, data: await res.json() };
  },

  async verifyTotp(code, challenge_token) {
    const res = await fetch(API_BASE + '/auth/totp/verify', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ code, challenge_token }),
    });
    return { ok: res.ok, data: await res.json() };
  },

  async logout() {
    await this._fetch('/auth/logout', { method: 'POST' });
    Auth.clear();
  },

  async get(path)         { const r = await this._fetch(path); return r?.json(); },
  async post(path, body)  {
    const r = await this._fetch(path, { method:'POST', body: JSON.stringify(body) });
    return r?.json();
  },
  async del(path)         { const r = await this._fetch(path, { method:'DELETE' }); return r?.json(); },
};

// ── UI helpers ─────────────────────────────────────────────────────────────────
const UI = {
  show(el, display = 'block') { if (el) el.style.display = display; },
  hide(el)                    { if (el) el.style.display = 'none'; },
  setText(sel, txt)           { const e = document.querySelector(sel); if (e) e.textContent = txt; },
  setHTML(sel, html)          { const e = document.querySelector(sel); if (e) e.innerHTML = html; },
  showError(sel, msg)         {
    const e = document.querySelector(sel);
    if (!e) return;
    e.textContent = msg;
    e.style.display = msg ? 'flex' : 'none';
  },
  showSuccess(sel, msg) {
    const e = document.querySelector(sel);
    if (!e) return;
    e.textContent = msg;
    e.style.display = msg ? 'flex' : 'none';
  },
  loading(btn, yes) {
    if (!btn) return;
    if (yes) { btn.dataset.orig = btn.textContent; btn.textContent = 'Loading…'; btn.disabled = true; }
    else     { btn.textContent = btn.dataset.orig || 'Submit'; btn.disabled = false; }
  },
  riskBadge(level) {
    const map = { HIGH: '#dc2626', MEDIUM: '#d97706', LOW: '#059669' };
    const bg  = map[level] || '#6b7280';
    return `<span style="display:inline-block;padding:2px 10px;border-radius:20px;font-size:11px;font-weight:600;background:${bg}20;color:${bg};border:1px solid ${bg}40">${level}</span>`;
  },
  formatDate(d) { return new Date(d).toLocaleDateString('en-ZW', { day:'numeric', month:'short', year:'numeric' }); },
};

// Guard: redirect to login if not authenticated
function requireAuth(minRole) {
  if (!Auth.isLoggedIn()) { window.location.href = 'login.html'; return false; }
  if (minRole && !Auth.hasRole(minRole)) {
    document.body.innerHTML = '<div style="padding:3rem;text-align:center"><h2>Access denied</h2><p>You need the <strong>' + minRole + '</strong> role for this page.</p></div>';
    return false;
  }
  return true;
}
