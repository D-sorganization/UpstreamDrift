"""Tests for the canonical-core status routes (issue #8081).

`/tools/canonical-core/estimation` and `/tools/canonical-core/comparison`
rendered as static shells with no service call, so a user could not tell
whether the workspace was broken, loading, or unbuilt. These routes give the
React page a machine-readable availability report — including, while no compute
service exists, an explicit reason and next step.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes import canonical_core
from src.tools.canonical_core.registry import canonical_core_tools


@pytest.fixture
def client() -> TestClient:
    """Return a client for an app mounting only the canonical-core router."""
    app = FastAPI()
    app.include_router(canonical_core.router)
    return TestClient(app)


@pytest.mark.unit
def test_status_lists_every_registry_workspace(client: TestClient) -> None:
    """The list endpoint must cover the registry exactly, with no drift."""
    response = client.get("/tools/canonical-core/status")

    assert response.status_code == 200
    modes = {entry["mode"] for entry in response.json()["workspaces"]}
    assert modes == {tool.mode for tool in canonical_core_tools()}


@pytest.mark.unit
@pytest.mark.parametrize("mode", ["estimation", "comparison"])
def test_mode_status_returns_reason_and_next_step(
    client: TestClient, mode: str
) -> None:
    """An unavailable workspace must always be actionable (#8081)."""
    response = client.get(f"/tools/canonical-core/{mode}/status")

    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == mode
    assert body["available"] is False
    # The whole point of the issue: never a silent shell.
    assert body["reason"].strip(), "unavailable workspace must explain why"
    assert body["next_step"].strip(), "unavailable workspace must offer a next step"
    assert mode in body["reason"]


@pytest.mark.unit
@pytest.mark.parametrize("mode", ["estimation", "comparison"])
def test_mode_status_matches_registry_descriptor(client: TestClient, mode: str) -> None:
    """Payload fields are sourced from the registry, not duplicated by hand."""
    tool = next(t for t in canonical_core_tools() if t.mode == mode)
    body = client.get(f"/tools/canonical-core/{mode}/status").json()

    assert body["tool_id"] == tool.tool_id
    assert body["name"] == tool.name
    assert body["description"] == tool.description
    assert body["web_route"] == tool.web_route
    assert body["capabilities"] == list(tool.capabilities)


@pytest.mark.unit
def test_unknown_mode_returns_404_naming_the_known_modes(client: TestClient) -> None:
    """A typo'd workspace gets a useful 404, not a 500 or an empty 200."""
    response = client.get("/tools/canonical-core/nope/status")

    assert response.status_code == 404
    detail = response.json()["detail"]
    assert "estimation" in detail
    assert "comparison" in detail


@pytest.mark.unit
@pytest.mark.parametrize("method", ["post", "put", "delete", "patch"])
def test_status_routes_are_read_only(client: TestClient, method: str) -> None:
    """Status is a GET-only resource; other verbs must be 405, not 200.

    This is the inverse of the #8079 failure mode: a path that exists for one
    verb and silently 405s for the verb the UI actually uses. Pinning the
    allowed verb set here keeps that from being introduced by accident.
    """
    response = client.request(method, "/tools/canonical-core/estimation/status")

    assert response.status_code == 405


@pytest.mark.unit
def test_get_is_allowed_on_every_declared_path(client: TestClient) -> None:
    """Every declared route answers the verb it declares (guards #8079)."""
    for route in canonical_core.router.routes:
        methods = getattr(route, "methods", set())
        path = getattr(route, "path", "")
        if "GET" not in methods:
            continue
        concrete = path.replace("{mode}", "estimation")
        response = client.get(concrete)
        assert response.status_code != 405, f"GET {concrete} unexpectedly 405"
