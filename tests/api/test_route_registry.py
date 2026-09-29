"""Tests for the route registry."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def route_registry():
    """Import ``src.api.route_registry`` with ``src.api.database`` mocked.

    Uses ``patch.dict`` to scope the mocked ``src.api.database`` to the test;
    once the fixture exits, the original module entries (if any) are restored
    so other tests in the same pytest worker resolve real imports. Without
    this scoping, a module-level ``sys.modules["src.api.database"] = ...``
    permanently swaps the database module out for ``MagicMock`` for the rest
    of the worker, causing order-dependent failures in unrelated route tests.
    """
    db_mock = MagicMock()
    auth_dependencies_mock = MagicMock()
    auth_dependencies_mock.CheckSimulationQuota.dependency = object()
    auth_dependencies_mock.CheckVideoQuota.dependency = object()
    auth_dependencies_mock.check_usage_quota.return_value = lambda user, db: user
    auth_models_mock = MagicMock()
    auth_models_mock.User = object
    # Drop any cached import of route_registry so it re-imports against
    # the mocked database module each time.
    cached = sys.modules.pop("src.api.route_registry", None)
    with patch.dict(
        sys.modules,
        {
            "src.api.auth.dependencies": auth_dependencies_mock,
            "src.api.auth.models": auth_models_mock,
            "src.api.database": db_mock,
            "src.api.database.get_db": MagicMock(),
        },
    ):
        from src.api import route_registry as module

        yield module
    # Restore prior cached module, if any, so subsequent tests see a
    # fresh import that resolves the real database module.
    sys.modules.pop("src.api.route_registry", None)
    if cached is not None:
        sys.modules["src.api.route_registry"] = cached


def test_discover_routes_success(route_registry):
    """Test discovering routes."""
    from fastapi import APIRouter

    routes = route_registry.discover_routes("tests.api.dummy_routes")

    # We expect auth and core to be found, and in that order (based on _REGISTRATION_ORDER)
    assert len(routes) == 2
    assert routes[0][0] == "auth"
    assert isinstance(routes[0][1], APIRouter)
    assert routes[1][0] == "core"
    assert isinstance(routes[1][1], APIRouter)


def test_discover_routes_not_a_package(route_registry):
    """Test discover_routes raises an error if path is not a package."""
    with pytest.raises(ImportError, match="is not a package"):
        route_registry.discover_routes("tests.api.test_route_registry")


def test_discover_routes_fails_closed_on_mandatory_import_error(route_registry):
    """Issue #7128: a broken *mandatory* route module fails discovery closed.

    ``dummy_routes_broken.video`` raises ImportError at import time. When it is
    treated as mandatory (empty optional allowlist), route discovery must
    re-raise rather than silently dropping the endpoint and returning 404s.
    """
    with pytest.raises(ImportError):
        route_registry.discover_routes(
            "tests.api.dummy_routes_broken",
            optional=frozenset(),
        )


def test_discover_routes_skips_optional_import_error(route_registry):
    """Issue #7128: a broken *optional* route module is skipped, not fatal.

    With ``video`` declared optional, its ImportError is tolerated while the
    healthy ``auth`` and ``core`` routers are still discovered.
    """
    from fastapi import APIRouter

    routes = route_registry.discover_routes(
        "tests.api.dummy_routes_broken",
        optional=frozenset({"video"}),
    )

    discovered = {name for name, _ in routes}
    assert "auth" in discovered
    assert "core" in discovered
    assert "video" not in discovered  # skipped due to optional import failure
    assert all(isinstance(router, APIRouter) for _, router in routes)


def test_video_is_in_default_optional_modules(route_registry):
    """The video router is the canonical optional, dependency-gated module."""
    assert "video" in route_registry._OPTIONAL_MODULES


def test_default_discovery_is_fail_closed(route_registry):
    """By default (no ``optional`` override) only the allowlist may be skipped."""
    # The default optional set is small and explicit; mandatory routers like
    # physics/dataset/auth must never be in it.
    for mandatory in ("auth", "core", "physics", "dataset", "simulation"):
        assert mandatory not in route_registry._OPTIONAL_MODULES


def test_register_routes(route_registry):
    """Test registering routes to a FastAPI app."""
    from fastapi import APIRouter, FastAPI

    app = MagicMock(spec=FastAPI)

    # We will patch discover_routes to return some mock routes
    router1 = APIRouter()
    router2 = APIRouter()

    routes = [
        ("core", router1),
        ("simulation", router2),
    ]

    with patch.object(route_registry, "discover_routes", return_value=routes):
        count = route_registry.register_routes(app, prefix="/api")

        assert count == 2
        assert app.include_router.call_count == 2

        calls = app.include_router.call_args_list
        assert calls[0][1]["prefix"] == "/api"
        assert len(calls[0][1]["dependencies"]) == 0

        assert len(calls[1][1]["dependencies"]) == 1


def test_dependencies_for_route(route_registry):
    """Test getting dependencies for a route module."""
    deps = route_registry._dependencies_for_route("simulation")
    assert len(deps) == 1

    deps = route_registry._dependencies_for_route("core")
    assert len(deps) == 0


def test_protected_routers_get_global_auth_dependency(route_registry):
    """Issue #6636 F1: non-public routers receive a global auth dependency.

    Previously only ``simulation``/``video`` carried auth (as a quota side
    effect) and every other router was registered with ``dependencies=[]`` —
    so an unauthenticated client could drive physics, enumerate datasets, and
    command actuators in cloud mode.
    """
    for module_name in (
        "physics",
        "data_explorer",
        "dataset",
        "models",
        "actuator_controls",
        "launch_monitor_analytics",
    ):
        deps = route_registry._dependencies_for_route(module_name)
        assert deps == (route_registry._global_auth_dependency,), module_name


def test_public_routers_have_no_auth_dependency(route_registry):
    """Issue #6636 F1: explicitly public routers stay reachable without auth."""
    for module_name in ("auth", "core", "observability", "capabilities"):
        assert route_registry._dependencies_for_route(module_name) == (), module_name


def test_database_module_not_leaked_after_tests():
    """Regression: importing this test module must not leak ``MagicMock``
    into ``sys.modules['src.api.database']``.

    If the scoped patch in the fixture works correctly, there should be no
    leftover MagicMock in ``sys.modules`` once the fixture has torn down.
    """
    db_mod = sys.modules.get("src.api.database")
    if db_mod is not None:
        assert not isinstance(db_mod, MagicMock), (
            "src.api.database leaked as MagicMock into sys.modules; "
            "scope the patch with patch.dict instead of assigning to sys.modules"
        )
