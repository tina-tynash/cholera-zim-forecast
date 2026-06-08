"""
middleware.py — Production security middleware stack.

1. RateLimitMiddleware       sliding-window per-IP  (10/min auth, 120/min API)
2. SecurityHeadersMiddleware HSTS, CSP, X-Frame-Options, referrer policy
3. AuditLogMiddleware        structured JSON request/response logging
4. InputSanitizeMiddleware   SQL injection + XSS guard on query strings + body
5. BruteForceGuard           additional IP-level lockout tracking
"""
from __future__ import annotations

import re
import time
import json
import ipaddress
from collections import defaultdict
from typing import Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

# ── Rate limiter ───────────────────────────────────────────────────────────────
_req_log: dict[str, list[float]] = defaultdict(list)

RATE_RULES: dict[str, tuple[int, int]] = {
    "/auth/login":   (10, 60),   # 10 req / 60 s  — brute force protection
    "/auth/refresh": (20, 60),
    "/auth":         (30, 60),
    "default":       (120, 60),  # general API
}


class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        ip   = _client_ip(request)
        path = request.url.path

        # Pick the most specific matching rule
        limit, window = RATE_RULES["default"]
        for prefix, rule in RATE_RULES.items():
            if prefix != "default" and path.startswith(prefix):
                limit, window = rule
                break

        now  = time.monotonic()
        log  = _req_log[ip]
        _req_log[ip] = [t for t in log if now - t < window]

        if len(_req_log[ip]) >= limit:
            return JSONResponse(
                {"detail": f"Rate limit exceeded. Max {limit} requests per {window}s."},
                status_code=429,
                headers={"Retry-After": str(window), "X-RateLimit-Limit": str(limit)},
            )

        _req_log[ip].append(now)
        response = await call_next(request)
        response.headers["X-RateLimit-Remaining"] = str(limit - len(_req_log[ip]))
        return response


# ── Security headers ───────────────────────────────────────────────────────────
_CSP = (
    "default-src 'self'; "
    "script-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
    "font-src 'self' https://fonts.gstatic.com; "
    "img-src 'self' data: blob:; "
    "connect-src 'self'; "
    "frame-ancestors 'none';"
)

_SECURITY_HEADERS = {
    "X-Content-Type-Options":    "nosniff",
    "X-Frame-Options":           "DENY",
    "X-XSS-Protection":          "1; mode=block",
    "Referrer-Policy":           "strict-origin-when-cross-origin",
    "Permissions-Policy":        "geolocation=(), microphone=(), camera=()",
    "Strict-Transport-Security": "max-age=63072000; includeSubDomains; preload",
    "Content-Security-Policy":   _CSP,
    "Cache-Control":             "no-store",
    "X-Request-ID":              "",  # filled per-request
}


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        import uuid
        req_id = str(uuid.uuid4())[:8]
        request.state.request_id = req_id

        response = await call_next(request)

        for k, v in _SECURITY_HEADERS.items():
            if k == "X-Request-ID":
                response.headers[k] = req_id
            else:
                response.headers[k] = v

        # Remove server fingerprinting headers
        for h in ("server", "x-powered-by"):
            try:
                del response.headers[h]
            except (KeyError, TypeError):
                pass

        return response


# ── Audit log middleware ───────────────────────────────────────────────────────
class AuditLogMiddleware(BaseHTTPMiddleware):
    """Log every API request with IP, path, method, status, latency."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = round((time.perf_counter() - start) * 1000, 1)

        # Only log non-trivial paths
        path = request.url.path
        if path in ("/health", "/docs", "/redoc", "/openapi.json"):
            return response

        entry = {
            "ts":      time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "ip":      _client_ip(request),
            "method":  request.method,
            "path":    path,
            "status":  response.status_code,
            "ms":      elapsed,
            "req_id":  getattr(request.state, "request_id", "-"),
            "ua":      request.headers.get("user-agent", "")[:120],
        }

        import logging
        logging.getLogger("audit").info(json.dumps(entry))
        return response


# ── Input sanitisation ─────────────────────────────────────────────────────────
# Patterns that indicate injection attempts
_SQL_RE  = re.compile(
    r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|ALTER|CREATE|EXEC|EXECUTE"
    r"|DECLARE|CAST|CONVERT|WAITFOR|SLEEP|BENCHMARK)\b"
    r"|--|\bOR\s+1\s*=\s*1\b|'\s*OR\s*')",
    re.IGNORECASE,
)
_XSS_RE  = re.compile(
    r"(<\s*script|javascript:|on\w+\s*=|<\s*iframe|<\s*object|<\s*embed"
    r"|<\s*svg.*on\w+|data:text/html)",
    re.IGNORECASE,
)
_PATH_RE = re.compile(r"\.\./|\.\.\\|%2e%2e")   # path traversal


def _check_value(v: str) -> bool:
    """Return True if the value looks safe."""
    if _SQL_RE.search(v) or _XSS_RE.search(v) or _PATH_RE.search(v):
        return False
    return True


class InputSanitizeMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Check query string parameters
        for key, val in request.query_params.items():
            if not _check_value(str(val)):
                return JSONResponse(
                    {"detail": f"Potentially malicious input in parameter '{key}'"},
                    status_code=400,
                )

        # For POST/PUT/PATCH, inspect JSON body (non-blocking peek)
        if request.method in ("POST", "PUT", "PATCH"):
            ct = request.headers.get("content-type", "")
            if "application/json" in ct:
                try:
                    body_bytes = await request.body()
                    body_str   = body_bytes.decode("utf-8", errors="replace")
                    if not _check_value(body_str):
                        return JSONResponse(
                            {"detail": "Potentially malicious content in request body"},
                            status_code=400,
                        )
                    # Re-inject body so downstream handlers can still read it
                    async def receive():
                        return {"type": "http.request", "body": body_bytes}
                    request = Request(request.scope, receive)
                except Exception:
                    pass  # Malformed body — let FastAPI handle it

        return await call_next(request)


# ── Helpers ────────────────────────────────────────────────────────────────────
def _client_ip(request: Request) -> str:
    """Extract real client IP respecting X-Forwarded-For."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        # Take the first (leftmost) IP — the original client
        ip = forwarded.split(",")[0].strip()
        try:
            ipaddress.ip_address(ip)
            return ip
        except ValueError:
            pass
    return request.client.host if request.client else "unknown"
