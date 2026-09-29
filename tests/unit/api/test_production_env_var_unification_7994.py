"""Regression tests for issue #7994 — two rival "am I in production?" gates.

``src/api/database.py`` and ``src/api/debug_guard.py`` used to read
``UPSTREAM_DRIFT_ENV`` while every deployment document (``SECURITY.md``, the
user manual) instructs operators to set ``ENVIRONMENT``. A deployment that
followed the documentation therefore skipped the Alembic-head assertion, ran
``Base.metadata.create_all()`` against the production database, and seeded a
default admin account.

Both spellings now resolve through the UpstreamDrift-owned API resolver and the
gate fails closed: if *either* variable names production, production wins.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from src.api import database, debug_guard, environment as api_env

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every test from a known, unset environment."""
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    monkeypatch.delenv("UPSTREAM_DRIFT_ENV", raising=False)
    monkeypatch.delenv("UPSTREAM_DRIFT_DEBUG_ENDPOINTS", raising=False)
    yield


@pytest.mark.parametrize(
    ("env_vars", "expected"),
    [
        ({"ENVIRONMENT": "production"}, True),
        ({"UPSTREAM_DRIFT_ENV": "production"}, True),
        ({"ENVIRONMENT": "prod"}, True),
        ({"ENVIRONMENT": "PRODUCTION"}, True),
        ({"ENVIRONMENT": " production "}, True),
        # Fail closed: a half-configured deploy must not lose the gate.
        ({"ENVIRONMENT": "development", "UPSTREAM_DRIFT_ENV": "production"}, True),
        ({"ENVIRONMENT": "production", "UPSTREAM_DRIFT_ENV": "development"}, True),
        ({"ENVIRONMENT": "development"}, False),
        ({"ENVIRONMENT": "staging"}, False),
        ({}, False),
    ],
)
def test_production_gate_honours_both_spellings(
    monkeypatch: pytest.MonkeyPatch, env_vars: dict[str, str], expected: bool
) -> None:
    """The DB gate must agree with the canonical predicate for every spelling."""
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)

    assert api_env.is_production_environment() is expected
    assert database._is_production_environment() is expected


def test_documented_production_deploy_does_not_run_create_all(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``ENVIRONMENT=production`` alone (per SECURITY.md) must take the prod path.

    This is the exact configuration from the documented production checklist.
    Before the fix, ``create_tables()`` ran and a default admin was seeded.
    """
    create_tables = Mock()
    verify_head = Mock()
    session_local = Mock()

    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setattr(database, "create_tables", create_tables)
    monkeypatch.setattr(database, "_assert_alembic_head_applied", verify_head)
    monkeypatch.setattr(database, "SessionLocal", session_local)

    database.init_db()

    create_tables.assert_not_called()
    verify_head.assert_called_once_with()
    assert session_local.call_count == 0, "no default admin may be seeded in production"


def test_debug_endpoints_disabled_under_documented_production_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Debug endpoints must be off when only ``ENVIRONMENT`` is set."""
    monkeypatch.setenv("ENVIRONMENT", "production")

    assert debug_guard.debug_endpoints_enabled() is False


def test_debug_endpoints_still_opt_in_via_explicit_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The explicit escape hatch keeps working."""
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("UPSTREAM_DRIFT_DEBUG_ENDPOINTS", "true")

    assert debug_guard.debug_endpoints_enabled() is True


def test_resolve_environment_is_not_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """Startup gates must observe env changes without a cache_clear()."""
    monkeypatch.setenv("ENVIRONMENT", "development")
    assert api_env.resolve_environment() == "development"

    monkeypatch.setenv("ENVIRONMENT", "production")
    assert api_env.resolve_environment() == "production"
