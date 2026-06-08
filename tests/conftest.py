"""
conftest.py — shared test setup for all test modules.
Ensures auth DB is initialised and API data is loaded before any test runs.
"""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ── Clean auth DB each session ────────────────────────────────────────────────
_db = Path("data/auth.db")
if _db.exists():
    os.remove(_db)

from src.security.auth import init_auth_db
init_auth_db()

# ── Pre-load API data so TestClient doesn't need startup event ────────────────
import src.app.api as _api
_api._load_data()
