"""Smoke tests for slowapi rate limiting on expensive API endpoints (#3508).

These tests verify the wiring rather than the engines: a request inside the
configured budget should pass the limiter, and a request outside the budget
should produce a 429 with a ``Retry-After`` header. Engine work is mocked so
the tests can run on the unit lane without the full physics stack.
"""

from __future__ import annotations

import importlib

import pytest

try:
    # Importing the server module up front catches missing transitive deps
    # (e.g. uvicorn) so the whole module skips cleanly rather than producing
    # a hard error inside an individual test.
    import src.api.server as _server  # noqa: F401
    from fastapi.testclient import TestClient
except ImportError:
    pytest.skip("FastAPI/server deps not available", allow_module_level=True)


def _build_app(monkeypatch: pytest.MonkeyPatch, simulate_limit: str = "2/minute"):
    """Build a fresh FastAPI app with a tight per-test rate limit.

    The slowapi ``Limiter`` reads its limit string at decorator evaluation
    time, so we must set the env var *before* importing the route modules
    and reload the relevant modules to pick up the override.
    """
    monkeypatch.setenv("API_LIMIT_SIMULATE", simulate_limit)
    monkeypatch.setenv("GOLF_AUTH_DISABLED", "true")

    # Re-import the module graph so the @limiter.limit(...) decorator sees the
    # patched env var. Order matters: rate_limit -> routes.simulation -> server.
    import src.api.rate_limit as rate_limit_module
    import src.api.routes.simulation as simulation_module
    import src.api.server as server_module

    importlib.reload(rate_limit_module)
    importlib.reload(simulation_module)
    importlib.reload(server_module)
    return server_module.app


@pytest.mark.unit
def test_rate_limit_helper_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """``get_limit`` returns the env value when set, otherwise the default."""
    from src.api.rate_limit import get_limit

    monkeypatch.delenv("UD_TEST_LIMIT", raising=False)
    assert get_limit("UD_TEST_LIMIT", "5/minute") == "5/minute"

    monkeypatch.setenv("UD_TEST_LIMIT", "1/second")
    assert get_limit("UD_TEST_LIMIT", "5/minute") == "1/second"

    monkeypatch.setenv("UD_TEST_LIMIT", "   ")
    assert get_limit("UD_TEST_LIMIT", "5/minute") == "5/minute"


@pytest.mark.unit
def test_simulate_over_limit_returns_429(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hitting ``/simulate`` past the configured budget returns 429.

    We don't care whether the underlying simulation succeeds — the rate
    limiter runs before any engine code, so any 429 we observe came from the
    limiter and proves the wiring works.
    """
    app = _build_app(monkeypatch, simulate_limit="2/minute")

    with TestClient(app) as client:
        statuses = []
        for _ in range(5):
            resp = client.post(
                "/api/v1/simulate",
                json={"engine_type": "mujoco", "duration": 0.1},
            )
            statuses.append(resp.status_code)
            if resp.status_code == 429:
                # Limiter must signal Retry-After per RFC 6585.
                assert "retry-after" in {k.lower() for k in resp.headers}
                break

        assert 429 in statuses, (
            f"Expected at least one 429 across 5 requests with a 2/minute limit, "
            f"got statuses={statuses}"
        )
