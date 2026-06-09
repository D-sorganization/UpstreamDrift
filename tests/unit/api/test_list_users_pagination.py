"""Pagination-bound enforcement for admin list_users (issue #7140).

The previous ``@precondition`` lambda bound its own defaults rather than the
real call args, so it never capped ``limit`` — an unbounded ``limit`` loaded the
whole table (memory-exhaustion DoS). Bounds are now enforced via FastAPI
``Query(ge=..., le=...)``; these tests assert 422 for out-of-range values.
"""

from __future__ import annotations

import inspect

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

pytestmark = pytest.mark.unit


def _build_client() -> TestClient:
    from src.api.auth.dependencies import get_current_user_flexible
    from src.api.database import get_db
    from src.api.routes import auth as auth_routes

    app = FastAPI()
    # auth_routes.router already carries a "/auth" prefix.
    app.include_router(auth_routes.router)

    # Override DB to return an empty user list and admin auth to a stub admin.
    class _Result(list):
        pass

    class _Query:
        def offset(self, *_a, **_k):
            return self

        def limit(self, *_a, **_k):
            return self

        def all(self):
            return _Result()

    class _DB:
        def query(self, *_a, **_k):
            return _Query()

    def _fake_db():
        yield _DB()

    def _fake_admin():
        class _Admin:
            role = "admin"
            is_active = True

        return _Admin()

    app.dependency_overrides[get_db] = _fake_db
    # role_dependency calls get_current_user_flexible; override it to a stub
    # admin so RoleChecker(ADMIN) passes without real auth.
    app.dependency_overrides[get_current_user_flexible] = _fake_admin
    return TestClient(app, raise_server_exceptions=False)


def test_list_users_signature_has_query_bounds() -> None:
    from src.api.routes.auth import list_users

    sig = inspect.signature(list_users)
    limit_default = sig.parameters["limit"].default
    skip_default = sig.parameters["skip"].default

    def _bounds(query_obj) -> dict[str, float]:
        out: dict[str, float] = {}
        for meta in getattr(query_obj, "metadata", []):
            for attr in ("ge", "le"):
                val = getattr(meta, attr, None)
                if val is not None:
                    out[attr] = val
        return out

    limit_bounds = _bounds(limit_default)
    skip_bounds = _bounds(skip_default)
    assert limit_bounds.get("le") == 1000
    assert limit_bounds.get("ge") == 1
    assert skip_bounds.get("ge") == 0


def test_list_users_rejects_oversized_limit() -> None:
    client = _build_client()
    resp = client.get("/auth/users", params={"limit": 100000})
    assert resp.status_code == 422


def test_list_users_rejects_negative_skip() -> None:
    client = _build_client()
    resp = client.get("/auth/users", params={"skip": -1})
    assert resp.status_code == 422


def test_list_users_accepts_in_range() -> None:
    client = _build_client()
    resp = client.get("/auth/users", params={"skip": 0, "limit": 100})
    assert resp.status_code == 200
