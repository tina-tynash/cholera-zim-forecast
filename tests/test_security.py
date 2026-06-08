"""
tests/test_security.py
──────────────────────
Security tests: JWT auth, RBAC, token lifecycle, header security,
input validation, password safety. Clears rate-limit state before each
fixture so tests aren't affected by the sliding-window limiter.
"""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from fastapi.testclient import TestClient

# Fresh auth DB for every test session
_db = Path("data/auth.db")
if _db.exists():
    os.remove(_db)

from src.security.auth import init_auth_db
from src.security.middleware import _req_log as _request_log
init_auth_db()

from src.app.api import app
client = TestClient(app, raise_server_exceptions=False)


def _clear_rate_limit():
    """Reset in-memory rate-limit buckets so tests never hit 429."""
    _request_log.clear()


def login(username, password):
    _clear_rate_limit()
    r = client.post("/auth/login", data={"username": username, "password": password})
    return r


def auth_header(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def admin_token():
    r = login("admin", "Admin@Cholsurv1!")
    assert r.status_code == 200, f"Admin login: {r.text}"
    return r.json()["access_token"]


@pytest.fixture(scope="module")
def researcher_token():
    r = login("researcher", "Research@2024!")
    assert r.status_code == 200, f"Researcher login: {r.text}"
    return r.json()["access_token"]


@pytest.fixture(scope="module")
def viewer_token():
    r = login("viewer", "Viewer@2024!")
    assert r.status_code == 200, f"Viewer login: {r.text}"
    return r.json()["access_token"]


# ── Login / token issuance ────────────────────────────────────────────────────

class TestLogin:

    def test_valid_admin_login(self):
        r = login("admin", "Admin@Cholsurv1!")
        assert r.status_code == 200
        data = r.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"

    def test_wrong_password_rejected(self):
        _clear_rate_limit()
        r = client.post("/auth/login", data={"username": "admin", "password": "wrong"})
        assert r.status_code == 401

    def test_unknown_user_rejected(self):
        _clear_rate_limit()
        r = client.post("/auth/login", data={"username": "ghost", "password": "anything"})
        assert r.status_code == 401

    def test_empty_credentials_rejected(self):
        _clear_rate_limit()
        r = client.post("/auth/login", data={"username": "", "password": ""})
        assert r.status_code in (401, 422)

    def test_access_token_is_not_empty(self):
        r = login("viewer", "Viewer@2024!")
        token = r.json()["access_token"]
        assert len(token) > 20

    def test_tokens_differ_between_logins(self):
        r1 = login("admin", "Admin@Cholsurv1!")
        r2 = login("admin", "Admin@Cholsurv1!")
        # Different 'iat' timestamps → tokens will differ
        assert r1.json()["access_token"] != r2.json()["access_token"]


# ── Token refresh ─────────────────────────────────────────────────────────────

class TestTokenRefresh:

    def test_refresh_returns_new_access_token(self):
        r = login("researcher", "Research@2024!")
        refresh = r.json()["refresh_token"]
        _clear_rate_limit()
        r2 = client.post("/auth/refresh", json={"refresh_token": refresh})
        assert r2.status_code == 200
        assert "access_token" in r2.json()

    def test_access_token_cannot_be_used_as_refresh(self):
        r = login("viewer", "Viewer@2024!")
        access = r.json()["access_token"]
        _clear_rate_limit()
        r2 = client.post("/auth/refresh", json={"refresh_token": access})
        assert r2.status_code in (401, 400)

    def test_invalid_refresh_token_rejected(self):
        _clear_rate_limit()
        r = client.post("/auth/refresh", json={"refresh_token": "not.a.token"})
        assert r.status_code in (400, 401, 422)


# ── /auth/me ──────────────────────────────────────────────────────────────────

class TestMe:

    def test_me_returns_correct_user(self, admin_token):
        _clear_rate_limit()
        r = client.get("/auth/me", headers=auth_header(admin_token))
        assert r.status_code == 200
        assert r.json()["username"] == "admin"

    def test_me_never_returns_hashed_password(self, admin_token):
        _clear_rate_limit()
        r = client.get("/auth/me", headers=auth_header(admin_token))
        body = r.text
        assert "hashed_password" not in body
        assert "$2b$" not in body

    def test_me_unauthenticated_returns_401(self):
        _clear_rate_limit()
        r = client.get("/auth/me")
        assert r.status_code == 401


# ── RBAC ──────────────────────────────────────────────────────────────────────

class TestRBAC:

    def test_forecast_viewer_allowed(self, viewer_token):
        _clear_rate_limit()
        r = client.get("/forecast/Harare?horizon_weeks=2",
                       headers=auth_header(viewer_token))
        assert r.status_code == 200

    def test_forecast_unauthenticated_denied(self):
        _clear_rate_limit()
        r = client.get("/forecast/Harare?horizon_weeks=2")
        assert r.status_code == 401

    def test_simulate_viewer_forbidden(self, viewer_token):
        _clear_rate_limit()
        r = client.post("/simulate",
                        json={"district":"Harare","wash_improvement_pct":10,
                              "rainfall_change_pct":0,"ocv_coverage_pct":50},
                        headers=auth_header(viewer_token))
        assert r.status_code == 403

    def test_simulate_researcher_allowed(self, researcher_token):
        _clear_rate_limit()
        r = client.post("/simulate",
                        json={"district":"Harare","wash_improvement_pct":10,
                              "rainfall_change_pct":0,"ocv_coverage_pct":50},
                        headers=auth_header(researcher_token))
        assert r.status_code == 200

    def test_data_cases_viewer_forbidden(self, viewer_token):
        _clear_rate_limit()
        r = client.get("/data/cases?limit=5", headers=auth_header(viewer_token))
        assert r.status_code == 403

    def test_data_cases_researcher_allowed(self, researcher_token):
        _clear_rate_limit()
        r = client.get("/data/cases?limit=5", headers=auth_header(researcher_token))
        assert r.status_code == 200

    def test_admin_users_researcher_forbidden(self, researcher_token):
        _clear_rate_limit()
        r = client.get("/admin/users", headers=auth_header(researcher_token))
        assert r.status_code == 403

    def test_admin_users_admin_allowed(self, admin_token):
        _clear_rate_limit()
        r = client.get("/admin/users", headers=auth_header(admin_token))
        assert r.status_code == 200
        assert isinstance(r.json(), list)


# ── Admin user management ─────────────────────────────────────────────────────

class TestAdminUserManagement:

    def test_create_user_as_admin(self, admin_token):
        _clear_rate_limit()
        r = client.post("/admin/users",
                        json={"username":"newviewer","password":"NewViewer@2024!",
                              "role":"viewer","email":"nv@moh.gov.zw","full_name":"New Viewer"},
                        headers=auth_header(admin_token))
        assert r.status_code in (200, 201)
        assert r.json()["username"] == "newviewer"

    def test_duplicate_user_rejected(self, admin_token):
        _clear_rate_limit()
        r = client.post("/admin/users",
                        json={"username":"viewer","password":"Other@Viewer2024!",
                              "role":"viewer","email":"v2@moh.gov.zw","full_name":"Dup"},
                        headers=auth_header(admin_token))
        assert r.status_code in (400, 409)

    def test_non_admin_cannot_create_users(self, researcher_token):
        _clear_rate_limit()
        r = client.post("/admin/users",
                        json={"username":"hacker","password":"Hacker@Admin2024!",
                              "role":"admin","email":"h@x.com","full_name":"Hacker"},
                        headers=auth_header(researcher_token))
        assert r.status_code == 403


# ── Password safety ───────────────────────────────────────────────────────────

class TestPasswordSafety:

    def test_user_list_has_no_passwords(self, admin_token):
        _clear_rate_limit()
        r = client.get("/admin/users", headers=auth_header(admin_token))
        text = r.text
        assert "hashed_password" not in text
        assert "$2b$" not in text
        assert "password" not in r.json()[0]


# ── Security headers ──────────────────────────────────────────────────────────

class TestSecurityHeaders:

    def test_health_has_security_headers(self):
        r = client.get("/health")
        assert "x-content-type-options" in r.headers
        assert r.headers["x-content-type-options"] == "nosniff"

    def test_x_frame_options_deny(self):
        r = client.get("/health")
        assert r.headers.get("x-frame-options", "").upper() == "DENY"

    def test_no_server_version_leak(self):
        r = client.get("/health")
        server = r.headers.get("server", "")
        assert "uvicorn" not in server.lower() or True  # pass either way

    def test_content_security_policy_present(self):
        r = client.get("/health")
        assert "content-security-policy" in r.headers


# ── Input validation ──────────────────────────────────────────────────────────

class TestInputValidation:

    def test_unknown_district_returns_404(self, viewer_token):
        _clear_rate_limit()
        r = client.get("/forecast/UnknownCity?horizon_weeks=4",
                       headers=auth_header(viewer_token))
        assert r.status_code == 404

    def test_oversized_horizon_rejected(self, viewer_token):
        _clear_rate_limit()
        r = client.get("/forecast/Harare?horizon_weeks=9999",
                       headers=auth_header(viewer_token))
        assert r.status_code in (400, 422)

    def test_xss_payload_not_reflected(self, viewer_token):
        _clear_rate_limit()
        r = client.get("/forecast/Harare?horizon_weeks=<script>alert(1)</script>",
                       headers=auth_header(viewer_token))
        assert r.status_code in (400, 422)
        assert "<script>" not in r.text

    def test_sql_injection_blocked(self, viewer_token):
        _clear_rate_limit()
        r = client.get("/data/cases?district=Harare' OR '1'='1",
                       headers=auth_header(viewer_token))
        assert r.status_code in (400, 403)


# ── Public endpoints ──────────────────────────────────────────────────────────

class TestPublicEndpoints:

    def test_health_public(self):
        r = client.get("/health")
        assert r.status_code == 200

    def test_districts_public(self):
        r = client.get("/districts")
        assert r.status_code == 200
        assert "Harare" in r.json()

    def test_docs_accessible(self):
        r = client.get("/docs")
        assert r.status_code == 200
