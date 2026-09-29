"""Tests for API hardening fixes from issue #6643.

Covers:
- F1: Last-admin / self-deactivation guard
- F2: Narrow exception in _lookup_api_key_by_prefix
- F3: Exhaustive _handle_common_exceptions (no None return)
- F6: Narrowed import exception in api/__init__
- F7: LOD accessor get_agent_state_store
- F8: _unauthorized and _assert_type helpers
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Shared mock infrastructure
# ---------------------------------------------------------------------------


class _User:
    """Minimal mock user for route tests."""

    def __init__(self, **kw: object) -> None:
        self.id = 1
        self.email = "admin@example.com"
        self.full_name = "Admin"
        self.organization = None
        self.role = "admin"
        self.is_active = True
        self.is_verified = True
        self.hashed_password = "hashed"
        self.last_login = None
        self.api_calls_this_month = 0
        self.video_analyses_this_month = 0
        self.simulations_this_month = 0
        self.subscription_status = "active"
        import datetime

        self.created_at = datetime.datetime.now()
        for k, v in kw.items():
            setattr(self, k, v)


class _Query:
    """Minimal SQLAlchemy query mock supporting filter/first/count/all."""

    def __init__(self, result: object = None, count: int = 1) -> None:
        self._result = result
        self._count = count

    def filter(self, *_: object) -> _Query:
        return self

    def first(self) -> object:
        return self._result

    def count(self) -> int:
        return self._count

    def offset(self, _: int) -> _Query:
        return self

    def limit(self, _: int) -> _Query:
        return self

    def all(self) -> list:
        return [self._result] if self._result is not None else []


class _DB:
    def __init__(self, user: object = None, admin_count: int = 1) -> None:
        self._user = user
        self._admin_count = admin_count

    def query(self, model: type) -> _Query:
        return _Query(result=self._user, count=self._admin_count)

    def commit(self) -> None:
        pass

    def add(self, _: object) -> None:
        pass

    def refresh(self, obj: object) -> None:
        pass


def _make_app_with_db_and_user(
    db: _DB, current_user: _User
) -> tuple[FastAPI, TestClient]:
    from src.api.routes.auth import router
    from src.api.database import get_db
    from src.api.auth.dependencies import get_current_user_flexible

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_db] = lambda: db
    # Override the base auth dep so role check sees our mock user
    app.dependency_overrides[get_current_user_flexible] = lambda: current_user
    return app, TestClient(app)


# ---------------------------------------------------------------------------
# F1 — Last-admin / self-deactivation guard
# ---------------------------------------------------------------------------


def test_update_role_demote_last_admin_returns_409() -> None:
    """Demoting the only active admin must return 409."""
    admin_user = _User(id=1, role="admin")
    # admin_count=1 → only one admin
    db = _DB(user=admin_user, admin_count=1)
    current_admin = _User(id=99, role="admin")
    _, client = _make_app_with_db_and_user(db, current_admin)

    resp = client.put("/auth/users/1/role", params={"new_role": "free"})
    assert resp.status_code == 409, resp.text
    assert "last" in resp.json()["detail"].lower()


def test_update_role_demote_non_last_admin_succeeds() -> None:
    """Demoting an admin when other admins exist must succeed (200)."""
    admin_user = _User(id=1, role="admin")
    db = _DB(user=admin_user, admin_count=2)
    current_admin = _User(id=99, role="admin")
    _, client = _make_app_with_db_and_user(db, current_admin)

    resp = client.put("/auth/users/1/role", params={"new_role": "free"})
    assert resp.status_code == 200


def test_update_role_promote_to_admin_always_succeeds() -> None:
    """Promoting any user to admin must never be blocked."""
    free_user = _User(id=2, role="free")
    db = _DB(user=free_user, admin_count=1)
    current_admin = _User(id=99, role="admin")
    _, client = _make_app_with_db_and_user(db, current_admin)

    resp = client.put("/auth/users/2/role", params={"new_role": "admin"})
    assert resp.status_code == 200


def test_update_status_self_deactivate_returns_409() -> None:
    """An admin deactivating themselves must return 409."""
    self_admin = _User(id=99, role="admin")
    db = _DB(user=self_admin, admin_count=2)
    _, client = _make_app_with_db_and_user(db, self_admin)

    resp = client.put("/auth/users/99/status", params={"is_active": "false"})
    assert resp.status_code == 409, resp.text
    assert "own account" in resp.json()["detail"].lower()


def test_update_status_deactivate_last_admin_returns_409() -> None:
    """Deactivating the last admin (not self) must return 409."""
    other_admin = _User(id=5, role="admin")
    db = _DB(user=other_admin, admin_count=1)
    current_admin = _User(id=99, role="admin")
    _, client = _make_app_with_db_and_user(db, current_admin)

    resp = client.put("/auth/users/5/status", params={"is_active": "false"})
    assert resp.status_code == 409, resp.text
    assert "last" in resp.json()["detail"].lower()


def test_update_status_deactivate_non_admin_succeeds() -> None:
    """Deactivating a non-admin user must succeed (200)."""
    free_user = _User(id=3, role="free")
    db = _DB(user=free_user, admin_count=1)
    current_admin = _User(id=99, role="admin")
    _, client = _make_app_with_db_and_user(db, current_admin)

    resp = client.put("/auth/users/3/status", params={"is_active": "false"})
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# F2 — Narrow exception in _lookup_api_key_by_prefix
# ---------------------------------------------------------------------------


def test_lookup_api_key_only_narrows_to_db_schema_errors() -> None:
    """Verify OperationalError/ProgrammingError triggers the fallback scan (not RuntimeError)."""
    from sqlalchemy.exc import OperationalError
    from src.api.auth import dependencies

    class _FakeQuery:
        """Query that fails on the first (indexed) call, then returns [] on the fallback."""

        def __init__(self) -> None:
            self._call_count = 0

        def filter(self, *_: object) -> _FakeQuery:
            return self

        def all(self) -> list:
            self._call_count += 1
            if self._call_count == 1:
                raise OperationalError(
                    "no such column key_prefix",
                    None,
                    Exception("no such column: key_prefix"),
                )
            # Fallback call: no keys found → should yield 401
            return []

    _q = _FakeQuery()

    class _FakeDB:
        def query(self, _: type) -> _FakeQuery:
            return _q

    # OperationalError triggers fallback → no matching keys → 401, not 500
    with pytest.raises(HTTPException) as exc_info:
        dependencies._lookup_api_key_by_prefix("gms_12345678extra", _FakeDB())  # type: ignore[arg-type]
    assert exc_info.value.status_code == 401


def test_lookup_api_key_non_db_errors_propagate() -> None:
    """RuntimeError must NOT be swallowed — it must propagate to prevent silent DoS."""
    from src.api.auth import dependencies

    class _FakeQuery:
        def filter(self, *_: object) -> _FakeQuery:
            return self

        def all(self) -> list:
            raise RuntimeError("unexpected failure")

    class _FakeDB:
        def query(self, _: type) -> _FakeQuery:
            return _FakeQuery()

    with pytest.raises(RuntimeError):
        dependencies._lookup_api_key_by_prefix("gms_12345678extra", _FakeDB())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# F3 — Exhaustive _handle_common_exceptions
# ---------------------------------------------------------------------------


def test_handle_common_exceptions_never_returns_none() -> None:
    """_handle_common_exceptions must always raise for any exception type."""
    from src.api.middleware.error_handler import _handle_common_exceptions

    class _Novel(Exception):
        pass

    with pytest.raises(HTTPException) as exc_info:
        _handle_common_exceptions(_Novel("boom"), "test_func")
    assert exc_info.value.status_code == 500


def test_handle_common_exceptions_value_error_is_400() -> None:
    from src.api.middleware.error_handler import _handle_common_exceptions

    with pytest.raises(HTTPException) as exc_info:
        _handle_common_exceptions(ValueError("bad input"), "test_func")
    assert exc_info.value.status_code == 400


def test_handle_common_exceptions_reraises_http() -> None:
    from src.api.middleware.error_handler import _handle_common_exceptions

    orig = HTTPException(status_code=422, detail="unprocessable")
    with pytest.raises(HTTPException) as exc_info:
        _handle_common_exceptions(orig, "test_func")
    assert exc_info.value.status_code == 422


# ---------------------------------------------------------------------------
# F7 — LOD accessor get_agent_state_store
# ---------------------------------------------------------------------------


def test_get_agent_state_store_returns_history_store() -> None:
    from src.shared.python.app_state import get_agent_state_store, get_state_logger
    from src.shared.python.app_state._history_store import HistoryStore

    store = get_agent_state_store()
    assert isinstance(store, HistoryStore)
    # Same object as the singleton's store
    assert store is get_state_logger().store


# ---------------------------------------------------------------------------
# F8 — _unauthorized and _assert_type helpers
# ---------------------------------------------------------------------------


def test_unauthorized_helper_returns_401_with_bearer() -> None:
    from src.api.auth.dependencies import _unauthorized

    exc = _unauthorized("test detail")
    assert exc.status_code == 401
    assert exc.detail == "test detail"
    assert exc.headers == {"WWW-Authenticate": "Bearer"}


def test_assert_type_passes_for_correct_type() -> None:
    from src.api.auth.dependencies import _assert_type

    _assert_type("hello", str, "greeting")  # should not raise


def test_assert_type_raises_for_wrong_type() -> None:
    from src.api.auth.dependencies import _assert_type

    with pytest.raises(ValueError, match="Expected str"):
        _assert_type(42, str, "greeting")
