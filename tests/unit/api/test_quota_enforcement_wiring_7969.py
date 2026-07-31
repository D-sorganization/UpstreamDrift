"""Regression tests for issue #7969 — quota dependency never actually ran.

``check_usage_quota(resource)`` returns a *generator* dependency.
``route_registry._request_time_quota_dependency`` called it as a plain
function and discarded the result, so merely constructing the generator was
mistaken for running it: ``usage_tracker.consume_quota`` was never invoked and
the documented 429 response was unreachable for ``/simulation`` and ``/video``.

These tests exercise the registry wiring end to end (the pre-existing tests in
``test_quota_dependencies.py`` drive ``check_usage_quota`` by hand, which is
exactly why they stayed green while the wiring was broken).
"""

from __future__ import annotations

import inspect
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

import src.api.route_registry as route_registry
from src.api.database import get_db

pytestmark = pytest.mark.unit

_QUOTA_DEPENDENCIES = {
    "simulations": route_registry._SIMULATION_QUOTA_DEPENDENCY,
    "video_analyses": route_registry._VIDEO_QUOTA_DEPENDENCY,
}


@pytest.fixture
def quota_app(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Build a client factory for a route guarded by a quota dependency."""

    def _build(resource_type: str, *, consume_ok: bool) -> tuple[TestClient, Any]:
        tracker = MagicMock()
        tracker.consume_quota.return_value = consume_ok
        tracker.quota_limit.return_value = 10
        monkeypatch.setattr(
            "src.api.auth.dependencies.usage_tracker", tracker, raising=True
        )
        monkeypatch.setattr(route_registry, "is_auth_disabled", lambda: False)

        async def _fake_user(request: Any, db: Any) -> Any:
            user = MagicMock()
            user.email = "quota@example.com"
            return user

        monkeypatch.setattr(
            route_registry, "_current_user_from_bearer_header", _fake_user
        )

        app = FastAPI()
        app.dependency_overrides[get_db] = lambda: MagicMock()

        @app.post(
            "/guarded",
            dependencies=[Depends(_QUOTA_DEPENDENCIES[resource_type])],
        )
        def guarded() -> dict[str, bool]:
            return {"ok": True}

        return TestClient(app, raise_server_exceptions=False), tracker

    return _build


@pytest.mark.parametrize("resource_type", sorted(_QUOTA_DEPENDENCIES))
def test_quota_dependency_is_a_driven_generator(resource_type: str) -> None:
    """The registry dependency must be a generator FastAPI can actually drive.

    A plain ``async def`` that *returns* the inner generator is the bug: the
    generator body never executes.
    """
    dependency = _QUOTA_DEPENDENCIES[resource_type]
    assert inspect.isasyncgenfunction(dependency), (
        "the quota dependency must be an async generator so FastAPI steps it; "
        "a coroutine returning a generator object never runs the quota check"
    )


@pytest.mark.parametrize("resource_type", sorted(_QUOTA_DEPENDENCIES))
def test_quota_is_consumed_on_a_successful_request(
    quota_app: Any, resource_type: str
) -> None:
    """A request through the guarded route must consume exactly one unit."""
    client, tracker = quota_app(resource_type, consume_ok=True)

    response = client.post("/guarded")

    assert response.status_code == 200
    assert tracker.consume_quota.call_count == 1, (
        "consume_quota was not called — the quota generator was constructed "
        "but never iterated"
    )
    assert tracker.consume_quota.call_args.args[2] == resource_type


@pytest.mark.parametrize("resource_type", sorted(_QUOTA_DEPENDENCIES))
def test_exhausted_quota_returns_429(quota_app: Any, resource_type: str) -> None:
    """``consume_quota`` returning False must surface as HTTP 429."""
    client, tracker = quota_app(resource_type, consume_ok=False)

    response = client.post("/guarded")

    assert (
        response.status_code == 429
    ), "the documented 'Usage quota exceeded' response is unreachable"
    assert "quota exceeded" in response.json()["detail"].lower()


@pytest.mark.parametrize("resource_type", sorted(_QUOTA_DEPENDENCIES))
def test_quota_is_refunded_when_the_endpoint_fails(
    quota_app: Any, resource_type: str
) -> None:
    """The refund-on-failure path must be reachable, not dead code."""
    client, tracker = quota_app(resource_type, consume_ok=True)
    app = client.app

    @app.post("/boom", dependencies=[Depends(_QUOTA_DEPENDENCIES[resource_type])])
    def boom() -> dict[str, bool]:
        raise RuntimeError("endpoint blew up")

    response = client.post("/boom")

    assert response.status_code == 500
    assert tracker.consume_quota.call_count == 1
    assert (
        tracker.refund_quota.call_count == 1
    ), "a failed protected operation must refund the consumed quota unit"


def test_auth_disabled_mode_skips_quota_entirely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Local mode must not consume quota (and must not 401)."""
    tracker = MagicMock()
    monkeypatch.setattr(
        "src.api.auth.dependencies.usage_tracker", tracker, raising=True
    )
    monkeypatch.setattr(route_registry, "is_auth_disabled", lambda: True)

    app = FastAPI()
    app.dependency_overrides[get_db] = lambda: MagicMock()

    @app.post(
        "/guarded",
        dependencies=[Depends(route_registry._SIMULATION_QUOTA_DEPENDENCY)],
    )
    def guarded() -> dict[str, bool]:
        return {"ok": True}

    response = TestClient(app, raise_server_exceptions=False).post("/guarded")

    assert response.status_code == 200
    tracker.consume_quota.assert_not_called()
