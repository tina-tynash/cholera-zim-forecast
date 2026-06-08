"""
api.py — Secured FastAPI application: CholSurv Zimbabwe.

Public:          GET /health  GET /districts
Auth:            POST /auth/login  /auth/refresh  /auth/logout  /auth/me
                 POST /auth/totp/setup  /auth/totp/verify  /auth/totp/confirm
Viewer+:         GET /forecast/{district}  GET /risk-scores
Researcher+:     POST /simulate  GET /data/cases
Admin:           GET /admin/users  POST /admin/users  DELETE /admin/users/{u}
                 GET /admin/audit
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.security.auth import (
    RoleEnum, TokenPair, UserPublic, UserCreate,
    authenticate_user, create_token_pair, decode_token,
    get_current_user, require_role, init_auth_db,
    create_user, list_users, revoke_token, get_user,
    deactivate_user, generate_totp_secret, verify_and_enable_totp,
    verify_totp, disable_totp, create_totp_challenge_token, audit,
    get_audit_log,
)
from src.security.middleware import (
    RateLimitMiddleware, SecurityHeadersMiddleware,
    AuditLogMiddleware, InputSanitizeMiddleware,
)

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="CholSurv Zimbabwe API",
    description="Secured cholera forecasting REST API — JWT + RBAC + 2FA",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Middleware stack (outermost wraps innermost)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(AuditLogMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(InputSanitizeMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv(
        "CORS_ORIGINS",
        "http://localhost:3000,http://localhost:8501,http://localhost:8080",
    ).split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)


@app.on_event("startup")
async def startup():
    init_auth_db()
    _load_data()


# ── Data loading ───────────────────────────────────────────────────────────────
_cases_df:    pd.DataFrame = pd.DataFrame()
_features_df: pd.DataFrame = pd.DataFrame()
DISTRICTS: list[str] = []


def _load_data() -> None:
    global _cases_df, _features_df, DISTRICTS
    processed = ROOT / "data" / "processed"
    cases_path    = processed / "cholera_cases.csv"
    features_path = processed / "features.csv"

    if cases_path.exists():
        _cases_df = pd.read_csv(cases_path, parse_dates=["date"])
        DISTRICTS = sorted(_cases_df["district"].unique().tolist())

    if features_path.exists():
        _features_df = pd.read_csv(features_path, parse_dates=["date"])

    if not DISTRICTS:
        DISTRICTS = [
            "Harare", "Bulawayo", "Chitungwiza", "Mutare", "Gweru",
            "Kwekwe", "Kadoma", "Masvingo", "Chinhoyi", "Marondera",
            "Zvishavane", "Chegutu", "Bindura", "Beitbridge", "Hwange",
        ]


# ── Pydantic schemas ───────────────────────────────────────────────────────────
class ForecastPoint(BaseModel):
    date:        str
    yhat:        float
    yhat_lower:  float
    yhat_upper:  float


class ForecastResponse(BaseModel):
    district:       str
    horizon_weeks:  int
    generated_at:   str
    model:          str
    forecasts:      List[ForecastPoint]


class RiskScore(BaseModel):
    district:      str
    risk_level:    str
    risk_score:    float
    cases_8week:   int
    trend:         str
    lat:           float
    lon:           float


class SimulationRequest(BaseModel):
    district:              str
    wash_improvement_pct:  float = Field(0.0, ge=0, le=100)
    rainfall_change_pct:   float = Field(0.0, ge=-100, le=200)
    ocv_coverage_pct:      float = Field(0.0, ge=0, le=100)


class SimulationResult(BaseModel):
    district:               str
    baseline_weekly_avg:    float
    projected_weekly_avg:   float
    cases_averted_per_week: float
    reduction_pct:          float
    annual_cases_averted:   float


class TOTPSetupResponse(BaseModel):
    secret:        str
    uri:           str
    challenge_token: Optional[str] = None


class TOTPVerifyRequest(BaseModel):
    code:            str
    challenge_token: Optional[str] = None


class LoginResponse(BaseModel):
    access_token:   Optional[str] = None
    refresh_token:  Optional[str] = None
    token_type:     str = "bearer"
    expires_in:     int = 1800
    requires_2fa:   bool = False
    challenge_token: Optional[str] = None
    user:           Optional[dict] = None


# ── District coordinates ───────────────────────────────────────────────────────
_COORDS: dict[str, tuple[float, float]] = {
    "Harare": (-17.83, 31.05), "Bulawayo": (-20.15, 28.58),
    "Chitungwiza": (-18.01, 31.07), "Mutare": (-18.97, 32.65),
    "Gweru": (-19.45, 29.82), "Kwekwe": (-18.93, 29.82),
    "Kadoma": (-18.34, 29.91), "Masvingo": (-20.06, 30.83),
    "Chinhoyi": (-17.36, 30.20), "Marondera": (-18.18, 31.55),
    "Zvishavane": (-20.33, 30.03), "Chegutu": (-18.13, 30.15),
    "Bindura": (-17.30, 31.33), "Beitbridge": (-22.21, 30.00),
    "Hwange": (-18.37, 26.50),
}


# ── Helper ─────────────────────────────────────────────────────────────────────
def _district_recent_cases(district: str, weeks: int = 8) -> pd.Series:
    if _cases_df.empty:
        return pd.Series([], dtype=float)
    df = _cases_df[_cases_df["district"] == district].sort_values("date")
    return df["cases"].tail(weeks)


def _simple_forecast(district: str, horizon: int) -> list[ForecastPoint]:
    """Statistical baseline forecast (no model training needed at runtime)."""
    recent = _district_recent_cases(district, 16)
    if len(recent) < 4:
        base = 15.0
        std  = 5.0
    else:
        base = float(recent.mean())
        std  = float(recent.std(ddof=1)) if len(recent) > 1 else base * 0.3

    rng = np.random.default_rng(abs(hash(district)) % (2**31))
    points = []
    from datetime import date, timedelta
    today = date.today()

    for i in range(horizon):
        trend = 1 + 0.02 * np.sin(2 * np.pi * (today.month + i / 4.3) / 12)
        noise = rng.normal(0, std * 0.3)
        yhat  = max(0.0, round(base * trend + noise, 1))
        lower = max(0.0, round(yhat - 1.645 * std, 1))
        upper = round(yhat + 1.645 * std, 1)
        points.append(ForecastPoint(
            date=str(today + timedelta(weeks=i + 1)),
            yhat=yhat, yhat_lower=lower, yhat_upper=upper,
        ))
    return points


# ═══════════════════════════════════════════════════════════════════════════════
# AUTH ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/auth/login", response_model=LoginResponse, tags=["auth"])
async def login(request: Request, form: OAuth2PasswordRequestForm = Depends()):
    """
    Authenticate with username + password.
    If 2FA is enabled, returns `requires_2fa=true` and a short-lived
    `challenge_token` — pass it with the TOTP code to `/auth/totp/verify`.
    """
    ip = request.client.host if request.client else "unknown"
    try:
        user = authenticate_user(form.username, form.password)
    except HTTPException as e:
        audit(form.username, ip, "LOGIN_BLOCKED", str(e.detail), success=False)
        raise

    if not user:
        audit(form.username, ip, "LOGIN_FAILED", "Bad credentials", success=False)
        raise HTTPException(status_code=401, detail="Invalid username or password")

    if user.get("totp_enabled"):
        challenge = create_totp_challenge_token(user["username"], user["role"])
        audit(user["username"], ip, "LOGIN_2FA_REQUIRED")
        return LoginResponse(requires_2fa=True, challenge_token=challenge)

    tokens = create_token_pair(user["username"], user["role"])
    audit(user["username"], ip, "LOGIN_SUCCESS")
    return LoginResponse(
        access_token=tokens.access_token,
        refresh_token=tokens.refresh_token,
        expires_in=tokens.expires_in,
        user={"username": user["username"], "role": user["role"],
              "full_name": user["full_name"]},
    )


@app.post("/auth/totp/verify", response_model=LoginResponse, tags=["auth"])
async def totp_verify(request: Request, body: TOTPVerifyRequest):
    """Complete 2FA login: verify TOTP code against the challenge token."""
    ip = request.client.host if request.client else "unknown"
    if not body.challenge_token:
        raise HTTPException(status_code=400, detail="challenge_token required")
    payload = decode_token(body.challenge_token)
    if payload.get("type") != "totp_challenge":
        raise HTTPException(status_code=400, detail="Invalid challenge token type")

    username = payload["sub"]
    if not verify_totp(username, body.code):
        audit(username, ip, "2FA_FAILED", "Bad TOTP code", success=False)
        raise HTTPException(status_code=401, detail="Invalid 2FA code")

    user   = get_user(username)
    tokens = create_token_pair(username, user["role"])
    audit(username, ip, "LOGIN_2FA_SUCCESS")
    return LoginResponse(
        access_token=tokens.access_token,
        refresh_token=tokens.refresh_token,
        expires_in=tokens.expires_in,
        user={"username": username, "role": user["role"],
              "full_name": user["full_name"]},
    )


@app.post("/auth/totp/setup", response_model=TOTPSetupResponse, tags=["auth"])
async def totp_setup(user: dict = Depends(get_current_user)):
    """Generate TOTP secret + QR provisioning URI. Scan with Authenticator app."""
    result = generate_totp_secret(user["username"])
    return TOTPSetupResponse(secret=result["secret"], uri=result["uri"])


@app.post("/auth/totp/confirm", tags=["auth"])
async def totp_confirm(body: TOTPVerifyRequest, user: dict = Depends(get_current_user)):
    """Confirm the TOTP code to enable 2FA on the account."""
    if not body.code:
        raise HTTPException(status_code=400, detail="code required")
    if not verify_and_enable_totp(user["username"], body.code):
        raise HTTPException(status_code=400, detail="Invalid code — scan the QR again")
    return {"message": "2FA enabled successfully"}


@app.post("/auth/totp/disable", tags=["auth"])
async def totp_disable(user: dict = Depends(get_current_user)):
    """Disable TOTP 2FA for the current user."""
    disable_totp(user["username"])
    return {"message": "2FA disabled"}


@app.post("/auth/refresh", tags=["auth"])
async def refresh_token(request: Request):
    """Exchange a refresh token for a new access token."""
    body = await request.json()
    token = body.get("refresh_token", "")
    payload = decode_token(token)
    if payload.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Not a refresh token")
    revoke_token(payload["jti"])  # Rotate: revoke old refresh
    user = get_user(payload["sub"])
    if not user or not user["is_active"]:
        raise HTTPException(status_code=401, detail="User inactive")
    return create_token_pair(user["username"], user["role"])


@app.post("/auth/logout", tags=["auth"])
async def logout(request: Request, user: dict = Depends(get_current_user)):
    """Revoke the current access token."""
    auth_header = request.headers.get("Authorization", "")
    token = auth_header.replace("Bearer ", "")
    payload = decode_token(token)
    revoke_token(payload.get("jti", ""))
    ip = request.client.host if request.client else "unknown"
    audit(user["username"], ip, "LOGOUT")
    return {"message": "Logged out successfully"}


@app.get("/auth/me", response_model=UserPublic, tags=["auth"])
async def me(user: dict = Depends(get_current_user)):
    """Return the authenticated user's profile."""
    return UserPublic(**{
        "username":    user["username"],
        "role":        user["role"],
        "email":       user["email"],
        "full_name":   user["full_name"],
        "is_active":   bool(user["is_active"]),
        "totp_enabled": bool(user.get("totp_enabled", False)),
    })


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/health", tags=["system"])
async def health():
    return {
        "status":    "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version":   "2.0.0",
        "districts": len(DISTRICTS),
    }


@app.get("/districts", tags=["data"])
async def districts():
    return DISTRICTS


# ═══════════════════════════════════════════════════════════════════════════════
# VIEWER+ ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/forecast/{district}", response_model=ForecastResponse, tags=["forecast"])
async def forecast(
    district: str,
    horizon_weeks: int = Query(4, ge=1, le=13),
    _user: dict = Depends(require_role(RoleEnum.viewer)),
):
    if district not in DISTRICTS:
        raise HTTPException(status_code=404, detail=f"District '{district}' not found")
    return ForecastResponse(
        district=district,
        horizon_weeks=horizon_weeks,
        generated_at=datetime.now(timezone.utc).isoformat(),
        model="statistical_ensemble",
        forecasts=_simple_forecast(district, horizon_weeks),
    )


@app.get("/risk-scores", response_model=List[RiskScore], tags=["forecast"])
async def risk_scores(_user: dict = Depends(require_role(RoleEnum.viewer))):
    scores = []
    for district in DISTRICTS[:10]:
        recent = _district_recent_cases(district, 8)
        total  = int(recent.sum()) if len(recent) else 0
        mean   = float(recent.mean()) if len(recent) else 0
        trend_val = float(recent.iloc[-1] - recent.iloc[0]) if len(recent) >= 2 else 0

        if mean > 20:
            level, score = "HIGH",   round(min(mean / 50, 1.0), 3)
        elif mean > 8:
            level, score = "MEDIUM", round(mean / 50, 3)
        else:
            level, score = "LOW",    round(mean / 50, 3)

        lat, lon = _COORDS.get(district, (-19.0, 29.8))
        scores.append(RiskScore(
            district=district, risk_level=level, risk_score=score,
            cases_8week=total, trend="up" if trend_val > 0 else "down",
            lat=lat, lon=lon,
        ))
    return sorted(scores, key=lambda x: x.risk_score, reverse=True)


# ═══════════════════════════════════════════════════════════════════════════════
# RESEARCHER+ ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/simulate", response_model=SimulationResult, tags=["simulation"])
async def simulate(
    body: SimulationRequest,
    _user: dict = Depends(require_role(RoleEnum.researcher)),
):
    if body.district not in DISTRICTS:
        raise HTTPException(status_code=404, detail=f"District '{body.district}' not found")
    recent = _district_recent_cases(body.district, 12)
    baseline = float(recent.mean()) if len(recent) else 20.0

    wash_factor  = 1 - (body.wash_improvement_pct / 100) * 0.6
    rain_factor  = 1 + (body.rainfall_change_pct  / 100) * 0.15
    ocv_factor   = 1 - (body.ocv_coverage_pct     / 100) * 0.45

    projected   = max(0.0, round(baseline * wash_factor * rain_factor * ocv_factor, 1))
    averted_wk  = round(baseline - projected, 1)
    reduction   = round((averted_wk / baseline * 100) if baseline > 0 else 0.0, 1)
    return SimulationResult(
        district=body.district,
        baseline_weekly_avg=round(baseline, 1),
        projected_weekly_avg=projected,
        cases_averted_per_week=max(0.0, averted_wk),
        reduction_pct=max(0.0, reduction),
        annual_cases_averted=round(max(0.0, averted_wk) * 52, 0),
    )


@app.get("/data/cases", tags=["data"])
async def data_cases(
    district: Optional[str] = None,
    limit: int = Query(100, ge=1, le=5000),
    _user: dict = Depends(require_role(RoleEnum.researcher)),
):
    if _cases_df.empty:
        return {"count": 0, "data": []}
    df = _cases_df.copy()
    if district:
        if district not in DISTRICTS:
            raise HTTPException(status_code=404, detail="District not found")
        df = df[df["district"] == district]
    df = df.sort_values("date", ascending=False).head(limit)
    df["date"] = df["date"].astype(str)
    return {"count": len(df), "data": df.to_dict(orient="records")}


# ═══════════════════════════════════════════════════════════════════════════════
# ADMIN ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/admin/users", tags=["admin"])
async def admin_list_users(_user: dict = Depends(require_role(RoleEnum.admin))):
    return list_users()


@app.post("/admin/users", tags=["admin"])
async def admin_create_user(
    body: UserCreate,
    request: Request,
    _user: dict = Depends(require_role(RoleEnum.admin)),
):
    try:
        new_user = create_user(
            body.username, body.password, body.role, body.email, body.full_name
        )
        ip = request.client.host if request.client else "unknown"
        audit(_user["username"], ip, "CREATE_USER", f"created:{body.username}")
        return new_user
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/admin/users/{username}", tags=["admin"])
async def admin_deactivate_user(
    username: str,
    request: Request,
    _user: dict = Depends(require_role(RoleEnum.admin)),
):
    if username == _user["username"]:
        raise HTTPException(status_code=400, detail="Cannot deactivate yourself")
    target = get_user(username)
    if not target:
        raise HTTPException(status_code=404, detail="User not found")
    deactivate_user(username)
    ip = request.client.host if request.client else "unknown"
    audit(_user["username"], ip, "DEACTIVATE_USER", f"deactivated:{username}")
    return {"message": f"User '{username}' deactivated"}


@app.get("/admin/audit", tags=["admin"])
async def admin_audit_log(
    limit: int = Query(100, ge=1, le=1000),
    _user: dict = Depends(require_role(RoleEnum.admin)),
):
    return get_audit_log(limit)
