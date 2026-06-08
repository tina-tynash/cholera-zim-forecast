"""
tests/test_api.py
-----------------
FastAPI endpoint tests: public, authenticated RBAC, injection guards,
security headers, and token lifecycle (logout must run last).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from fastapi.testclient import TestClient
from src.app.api import app
from src.security.middleware import _req_log

client = TestClient(app, raise_server_exceptions=False)


def _clear_rl():
    """Reset sliding-window rate-limit so tests never hit 429."""
    _req_log.clear()


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def admin_token():
    _clear_rl()
    r = client.post("/auth/login", data={"username": "admin", "password": "Admin@Cholsurv1!"})
    assert r.status_code == 200, f"Admin login failed: {r.text}"
    return r.json()["access_token"]


@pytest.fixture(scope="module")
def researcher_token():
    _clear_rl()
    r = client.post("/auth/login", data={"username": "researcher", "password": "Research@2024!"})
    assert r.status_code == 200, f"Researcher login failed: {r.text}"
    return r.json()["access_token"]


@pytest.fixture(scope="module")
def viewer_token():
    _clear_rl()
    r = client.post("/auth/login", data={"username": "viewer", "password": "Viewer@2024!"})
    assert r.status_code == 200, f"Viewer login failed: {r.text}"
    return r.json()["access_token"]


def auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# ── Public endpoints ──────────────────────────────────────────────────────────

def test_health_public():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


def test_districts_public():
    r = client.get("/districts")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list) and len(data) > 0
    assert "Harare" in data


def test_docs_accessible():
    r = client.get("/docs")
    assert r.status_code == 200


# ── Authentication ────────────────────────────────────────────────────────────

def test_login_valid(admin_token):
    assert len(admin_token) > 20


def test_login_bad_credentials():
    _clear_rl()
    r = client.post("/auth/login", data={"username": "admin", "password": "wrongpassword"})
    assert r.status_code == 401


def test_login_unknown_user():
    _clear_rl()
    r = client.post("/auth/login", data={"username": "ghost", "password": "anything"})
    assert r.status_code == 401


def test_me_endpoint(admin_token):
    r = client.get("/auth/me", headers=auth(admin_token))
    assert r.status_code == 200
    assert r.json()["username"] == "admin"
    assert r.json()["role"] == "admin"


def test_invalid_jwt():
    r = client.get("/forecast/Harare", headers={"Authorization": "Bearer invalid.jwt.token"})
    assert r.status_code == 401


def test_missing_auth_header():
    r = client.get("/forecast/Harare?horizon_weeks=2")
    assert r.status_code == 401


# ── Forecast (viewer+) ────────────────────────────────────────────────────────

def test_forecast_requires_auth():
    r = client.get("/forecast/Harare?horizon_weeks=2")
    assert r.status_code == 401


def test_forecast_valid(viewer_token):
    r = client.get("/forecast/Harare?horizon_weeks=4", headers=auth(viewer_token))
    assert r.status_code == 200
    d = r.json()
    assert d["district"] == "Harare"
    assert len(d["forecasts"]) == 4
    pt = d["forecasts"][0]
    assert pt["yhat"] >= 0
    assert pt["yhat_lower"] <= pt["yhat"] <= pt["yhat_upper"]


def test_forecast_invalid_district(viewer_token):
    r = client.get("/forecast/NoSuchPlace?horizon_weeks=2", headers=auth(viewer_token))
    assert r.status_code == 404


def test_forecast_horizon_clamp(viewer_token):
    r = client.get("/forecast/Harare?horizon_weeks=9999", headers=auth(viewer_token))
    assert r.status_code in (200, 422)


# ── Risk scores (viewer+) ─────────────────────────────────────────────────────

def test_risk_scores(viewer_token):
    r = client.get("/risk-scores", headers=auth(viewer_token))
    assert r.status_code == 200
    data = r.json()
    assert len(data) > 0
    for item in data:
        assert item["risk_level"] in ("HIGH", "MEDIUM", "LOW")
        assert 0 <= item["risk_score"] <= 1


# ── Simulate (researcher+) ────────────────────────────────────────────────────

def test_simulate_viewer_forbidden(viewer_token):
    r = client.post("/simulate",
        json={"district": "Harare", "wash_improvement_pct": 10,
              "rainfall_change_pct": 0, "ocv_coverage_pct": 50},
        headers=auth(viewer_token))
    assert r.status_code == 403


def test_simulate_researcher_allowed(researcher_token):
    r = client.post("/simulate",
        json={"district": "Harare", "wash_improvement_pct": 10,
              "rainfall_change_pct": 5, "ocv_coverage_pct": 70},
        headers=auth(researcher_token))
    assert r.status_code == 200
    d = r.json()
    assert d["baseline_weekly_avg"] >= 0
    assert d["annual_cases_averted"] >= 0


def test_simulate_invalid_district(researcher_token):
    r = client.post("/simulate",
        json={"district": "FakeDistrict", "wash_improvement_pct": 10,
              "rainfall_change_pct": 0, "ocv_coverage_pct": 50},
        headers=auth(researcher_token))
    assert r.status_code == 404


# ── Data (researcher+) ────────────────────────────────────────────────────────

def test_data_viewer_forbidden(viewer_token):
    r = client.get("/data/cases?limit=10", headers=auth(viewer_token))
    assert r.status_code == 403


def test_data_researcher_allowed(researcher_token):
    r = client.get("/data/cases?limit=10", headers=auth(researcher_token))
    assert r.status_code == 200
    d = r.json()
    assert "count" in d and "data" in d
    assert d["count"] <= 10


def test_data_district_filter(researcher_token):
    r = client.get("/data/cases?district=Harare&limit=20", headers=auth(researcher_token))
    assert r.status_code == 200
    rows = r.json()["data"]
    assert all(row["district"] == "Harare" for row in rows)


# ── Admin (admin only) ────────────────────────────────────────────────────────

def test_admin_users_viewer_forbidden(viewer_token):
    r = client.get("/admin/users", headers=auth(viewer_token))
    assert r.status_code == 403


def test_admin_users_admin_allowed(admin_token):
    r = client.get("/admin/users", headers=auth(admin_token))
    assert r.status_code == 200
    users = r.json()
    assert any(u["username"] == "admin" for u in users)
    # Passwords must never be exposed
    assert all("hashed_password" not in u and "password" not in u for u in users)


def test_admin_audit_log(admin_token):
    r = client.get("/admin/audit", headers=auth(admin_token))
    assert r.status_code == 200
    assert isinstance(r.json(), list)


# ── Security headers ──────────────────────────────────────────────────────────

def test_security_header_x_frame_options():
    r = client.get("/health")
    assert r.headers.get("x-frame-options") == "DENY"


def test_security_header_content_type_nosniff():
    r = client.get("/health")
    assert r.headers.get("x-content-type-options") == "nosniff"


def test_security_header_csp_present():
    r = client.get("/health")
    csp = r.headers.get("content-security-policy", "")
    assert "default-src" in csp


def test_security_header_cache_control():
    r = client.get("/health")
    assert r.headers.get("cache-control") == "no-store"


def test_security_header_referrer_policy():
    r = client.get("/health")
    assert "referrer-policy" in r.headers


# ── Injection guards ──────────────────────────────────────────────────────────

def test_sql_injection_blocked():
    r = client.get("/data/cases?district=' OR 1=1; DROP TABLE users--")
    assert r.status_code == 400


def test_xss_injection_blocked():
    r = client.get("/data/cases?district=<script>alert(1)</script>")
    assert r.status_code == 400


def test_path_traversal_blocked():
    r = client.get("/data/cases?district=../../etc/passwd")
    assert r.status_code == 400


# ── Logout ────────────────────────────────────────────────────────────────────

def test_logout_revokes_token():
    """Verify that after logout, the same token returns 401.
    Uses its own fresh login so the module-scoped admin_token stays valid
    for all other tests regardless of execution order.
    """
    _clear_rl()
    r = client.post("/auth/login", data={"username": "admin", "password": "Admin@Cholsurv1!"})
    assert r.status_code == 200, f"Fresh login failed: {r.text}"
    tmp_token = r.json()["access_token"]
    _clear_rl()
    r2 = client.post("/auth/logout", headers=auth(tmp_token))
    assert r2.status_code == 200
    r3 = client.get("/auth/me", headers=auth(tmp_token))
    assert r3.status_code == 401
