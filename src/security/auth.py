"""
auth.py — Full security layer: JWT + RBAC + TOTP 2FA + session management.

Roles:   admin > researcher > viewer
Tokens:  access (30-min HS256) + refresh (7-day) + revocation blacklist
2FA:     TOTP via pyotp (Google Authenticator compatible)
Storage: SQLite (data/auth.db) — swap DATABASE_URL for Postgres
"""
from __future__ import annotations

import os
import sqlite3
import secrets
import re
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

import pyotp
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, field_validator

# ── Config ─────────────────────────────────────────────────────────────────────
SECRET_KEY   = os.getenv("SECRET_KEY", secrets.token_urlsafe(64))
ALGORITHM    = "HS256"
ACCESS_EXP   = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 30))
REFRESH_EXP  = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", 7))
AUTH_DB      = Path(os.getenv("AUTH_DB_PATH", "data/auth.db"))
MAX_ATTEMPTS = int(os.getenv("MAX_LOGIN_ATTEMPTS", 5))
LOCKOUT_MIN  = int(os.getenv("LOCKOUT_MINUTES", 15))

_pwd  = CryptContext(schemes=["bcrypt"], deprecated="auto", bcrypt__rounds=12)
_http = HTTPBearer(auto_error=False)


# ── Enums & schemas ────────────────────────────────────────────────────────────
class RoleEnum(str, Enum):
    admin      = "admin"
    researcher = "researcher"
    viewer     = "viewer"


ROLE_RANK = {RoleEnum.viewer: 0, RoleEnum.researcher: 1, RoleEnum.admin: 2}


class TokenPair(BaseModel):
    access_token:  str
    refresh_token: str
    token_type:    str = "bearer"
    expires_in:    int = ACCESS_EXP * 60
    requires_2fa:  bool = False


class UserPublic(BaseModel):
    username:    str
    role:        RoleEnum
    email:       str
    full_name:   str
    is_active:   bool
    totp_enabled: bool


class UserCreate(BaseModel):
    username:  str
    password:  str
    role:      RoleEnum = RoleEnum.viewer
    email:     str
    full_name: str

    @field_validator("password")
    @classmethod
    def strong_password(cls, v: str) -> str:
        errs = []
        if len(v) < 12:
            errs.append("at least 12 characters")
        if not re.search(r"[A-Z]", v):
            errs.append("one uppercase letter")
        if not re.search(r"[a-z]", v):
            errs.append("one lowercase letter")
        if not re.search(r"\d", v):
            errs.append("one digit")
        if not re.search(r"[!@#$%^&*()_+\-=\[\]{}|;':\",./<>?]", v):
            errs.append("one special character")
        if errs:
            raise ValueError(f"Password requires: {', '.join(errs)}")
        return v

    @field_validator("username")
    @classmethod
    def safe_username(cls, v: str) -> str:
        if not re.match(r"^[a-zA-Z0-9_\-]{3,32}$", v):
            raise ValueError("Username must be 3-32 alphanumeric characters")
        return v.lower()


# ── DB helpers ─────────────────────────────────────────────────────────────────
def _conn() -> sqlite3.Connection:
    AUTH_DB.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(AUTH_DB), check_same_thread=False)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA foreign_keys=ON")
    return c


def init_auth_db() -> None:
    """Initialise auth database with all tables and seed accounts."""
    with _conn() as c:
        c.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                username        TEXT PRIMARY KEY,
                hashed_password TEXT NOT NULL,
                role            TEXT NOT NULL DEFAULT 'viewer',
                email           TEXT NOT NULL UNIQUE,
                full_name       TEXT NOT NULL,
                is_active       INTEGER NOT NULL DEFAULT 1,
                totp_secret     TEXT,
                totp_enabled    INTEGER NOT NULL DEFAULT 0,
                failed_attempts INTEGER NOT NULL DEFAULT 0,
                locked_until    TEXT,
                created_at      TEXT NOT NULL,
                last_login      TEXT
            );

            CREATE TABLE IF NOT EXISTS revoked_tokens (
                jti        TEXT PRIMARY KEY,
                revoked_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS audit_log (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                ts         TEXT NOT NULL,
                username   TEXT,
                ip         TEXT,
                action     TEXT NOT NULL,
                detail     TEXT,
                success    INTEGER NOT NULL DEFAULT 1
            );
        """)

        # Seed default accounts (only if absent)
        seeds = [
            ("admin",      os.getenv("ADMIN_PASSWORD", "Admin@Cholsurv1!"),
             "admin",      "admin@cholsurv.zw",       "System Admin"),
            ("researcher", "Research@2024!",
             "researcher", "researcher@cholsurv.zw",  "Demo Researcher"),
            ("viewer",     "Viewer@2024!",
             "viewer",     "viewer@cholsurv.zw",      "Demo Viewer"),
        ]
        for uname, pwd, role, email, name in seeds:
            if not c.execute(
                "SELECT 1 FROM users WHERE username=?", (uname,)
            ).fetchone():
                _insert_user(c, uname, pwd, role, email, name)
        c.commit()


def _insert_user(c, username, plain_pwd, role, email, full_name) -> None:
    c.execute(
        "INSERT INTO users (username,hashed_password,role,email,full_name,created_at) "
        "VALUES (?,?,?,?,?,?)",
        (username, _pwd.hash(plain_pwd), role, email, full_name,
         datetime.now(timezone.utc).isoformat()),
    )


# ── CRUD ───────────────────────────────────────────────────────────────────────
def authenticate_user(username: str, password: str) -> Optional[dict]:
    """Verify credentials with brute-force lockout protection."""
    with _conn() as c:
        row = c.execute(
            "SELECT * FROM users WHERE username=? AND is_active=1", (username,)
        ).fetchone()

    if not row:
        return None
    row = dict(row)

    # Check lockout
    if row.get("locked_until"):
        lock_time = datetime.fromisoformat(row["locked_until"])
        if datetime.now(timezone.utc) < lock_time:
            remaining = int((lock_time - datetime.now(timezone.utc)).total_seconds() / 60)
            raise HTTPException(
                status_code=429,
                detail=f"Account locked for {remaining} more minutes.",
            )
        else:
            # Lockout expired — reset counter
            with _conn() as c:
                c.execute(
                    "UPDATE users SET failed_attempts=0, locked_until=NULL WHERE username=?",
                    (username,),
                )
                c.commit()

    if not _pwd.verify(password, row["hashed_password"]):
        # Increment failure counter
        new_attempts = row["failed_attempts"] + 1
        locked_until = None
        if new_attempts >= MAX_ATTEMPTS:
            locked_until = (
                datetime.now(timezone.utc) + timedelta(minutes=LOCKOUT_MIN)
            ).isoformat()
        with _conn() as c:
            c.execute(
                "UPDATE users SET failed_attempts=?, locked_until=? WHERE username=?",
                (new_attempts, locked_until, username),
            )
            c.commit()
        return None

    # Success — reset counter and record login
    with _conn() as c:
        c.execute(
            "UPDATE users SET failed_attempts=0, locked_until=NULL, last_login=? WHERE username=?",
            (datetime.now(timezone.utc).isoformat(), username),
        )
        c.commit()
    return row


def get_user(username: str) -> Optional[dict]:
    with _conn() as c:
        row = c.execute(
            "SELECT * FROM users WHERE username=?", (username,)
        ).fetchone()
    return dict(row) if row else None


def create_user(username, password, role, email, full_name) -> dict:
    with _conn() as c:
        if c.execute(
            "SELECT 1 FROM users WHERE username=?", (username,)
        ).fetchone():
            raise ValueError("Username already exists")
        if c.execute(
            "SELECT 1 FROM users WHERE email=?", (email,)
        ).fetchone():
            raise ValueError("Email already registered")
        _insert_user(c, username, password, role, email, full_name)
        c.commit()
    return get_user(username)


def list_users() -> list[dict]:
    with _conn() as c:
        return [
            dict(r)
            for r in c.execute(
                "SELECT username,role,email,full_name,is_active,"
                "totp_enabled,created_at,last_login FROM users"
            )
        ]


def deactivate_user(username: str) -> None:
    with _conn() as c:
        c.execute(
            "UPDATE users SET is_active=0 WHERE username=?", (username,)
        )
        c.commit()


# ── TOTP 2FA ───────────────────────────────────────────────────────────────────
def generate_totp_secret(username: str) -> dict:
    """Generate a new TOTP secret and provisioning URI."""
    secret = pyotp.random_base32()
    totp   = pyotp.TOTP(secret)
    uri    = totp.provisioning_uri(
        name=username, issuer_name="CholSurv Zimbabwe"
    )
    # Store secret (not yet enabled — user must verify first)
    with _conn() as c:
        c.execute(
            "UPDATE users SET totp_secret=? WHERE username=?", (secret, username)
        )
        c.commit()
    return {"secret": secret, "uri": uri}


def verify_and_enable_totp(username: str, code: str) -> bool:
    """Verify a TOTP code and enable 2FA if correct."""
    user = get_user(username)
    if not user or not user.get("totp_secret"):
        return False
    totp = pyotp.TOTP(user["totp_secret"])
    if totp.verify(code, valid_window=1):
        with _conn() as c:
            c.execute(
                "UPDATE users SET totp_enabled=1 WHERE username=?", (username,)
            )
            c.commit()
        return True
    return False


def verify_totp(username: str, code: str) -> bool:
    """Verify a TOTP code for login."""
    user = get_user(username)
    if not user or not user.get("totp_secret"):
        return False
    return pyotp.TOTP(user["totp_secret"]).verify(code, valid_window=1)


def disable_totp(username: str) -> None:
    with _conn() as c:
        c.execute(
            "UPDATE users SET totp_enabled=0, totp_secret=NULL WHERE username=?",
            (username,),
        )
        c.commit()


# ── Token helpers ──────────────────────────────────────────────────────────────
def create_token_pair(username: str, role: str) -> TokenPair:
    now   = datetime.now(timezone.utc)
    jti_a = secrets.token_urlsafe(16)
    jti_r = secrets.token_urlsafe(16)
    access = jwt.encode(
        {
            "sub": username, "role": role, "type": "access",
            "jti": jti_a,
            "exp": now + timedelta(minutes=ACCESS_EXP),
            "iat": now,
        },
        SECRET_KEY, algorithm=ALGORITHM,
    )
    refresh = jwt.encode(
        {
            "sub": username, "role": role, "type": "refresh",
            "jti": jti_r,
            "exp": now + timedelta(days=REFRESH_EXP),
            "iat": now,
        },
        SECRET_KEY, algorithm=ALGORITHM,
    )
    return TokenPair(access_token=access, refresh_token=refresh)


def create_totp_challenge_token(username: str, role: str) -> str:
    """Short-lived token issued after password check, before 2FA."""
    now = datetime.now(timezone.utc)
    return jwt.encode(
        {
            "sub": username, "role": role, "type": "totp_challenge",
            "jti": secrets.token_urlsafe(16),
            "exp": now + timedelta(minutes=5),
            "iat": now,
        },
        SECRET_KEY, algorithm=ALGORITHM,
    )


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")


def revoke_token(jti: str) -> None:
    with _conn() as c:
        c.execute(
            "INSERT OR IGNORE INTO revoked_tokens VALUES (?,?)",
            (jti, datetime.now(timezone.utc).isoformat()),
        )
        c.commit()


def _is_revoked(jti: str) -> bool:
    with _conn() as c:
        return bool(
            c.execute(
                "SELECT 1 FROM revoked_tokens WHERE jti=?", (jti,)
            ).fetchone()
        )


# ── Audit log ──────────────────────────────────────────────────────────────────
def audit(username: str, ip: str, action: str, detail: str = "", success: bool = True) -> None:
    with _conn() as c:
        c.execute(
            "INSERT INTO audit_log (ts,username,ip,action,detail,success) VALUES (?,?,?,?,?,?)",
            (
                datetime.now(timezone.utc).isoformat(),
                username, ip, action, detail, int(success),
            ),
        )
        c.commit()


def get_audit_log(limit: int = 100) -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM audit_log ORDER BY ts DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


# ── FastAPI dependency injectors ───────────────────────────────────────────────
async def get_current_user(
    creds: HTTPAuthorizationCredentials = Security(_http),
) -> dict:
    if not creds:
        raise HTTPException(status_code=401, detail="Not authenticated")
    payload = decode_token(creds.credentials)
    if payload.get("type") != "access":
        raise HTTPException(status_code=401, detail="Not an access token")
    if _is_revoked(payload.get("jti", "")):
        raise HTTPException(status_code=401, detail="Token has been revoked")
    user = get_user(payload["sub"])
    if not user or not user["is_active"]:
        raise HTTPException(status_code=401, detail="User not found or inactive")
    return user


def require_role(minimum: RoleEnum):
    async def _check(user: dict = Depends(get_current_user)) -> dict:
        if ROLE_RANK.get(RoleEnum(user["role"]), -1) < ROLE_RANK[minimum]:
            raise HTTPException(
                status_code=403,
                detail=f"Requires '{minimum}' role or higher",
            )
        return user
    return _check
