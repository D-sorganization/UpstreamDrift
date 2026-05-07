"""Tests for the route registry."""

import sys
from unittest.mock import MagicMock, patch

import pytest

sys.modules["src.api.database"] = MagicMock()
sys.modules["src.api.database.get_db"] = MagicMock()

from fastapi import APIRouter, FastAPI  # noqa: E402
from src.api.route_registry import (  # noqa: E402
    _dependencies_for_route,
    discover_routes,
    register_routes,
)


def test_discover_routes_success():
    """Test discovering routes."""
    routes = discover_routes("tests.api.dummy_routes")

    # We expect auth and core to be found, and in that order (based on _REGISTRATION_ORDER)
    assert len(routes) == 2
    assert routes[0][0] == "auth"
    assert isinstance(routes[0][1], APIRouter)
    assert routes[1][0] == "core"
    assert isinstance(routes[1][1], APIRouter)


def test_discover_routes_not_a_package():
    """Test discover_routes raises an error if path is not a package."""
    with pytest.raises(ImportError, match="is not a package"):
        discover_routes("tests.api.test_route_registry")


def test_register_routes():
    """Test registering routes to a FastAPI app."""
    app = MagicMock(spec=FastAPI)

    # We will patch discover_routes to return some mock routes
    router1 = APIRouter()
    router2 = APIRouter()

    routes = [
        ("core", router1),
        ("simulation", router2),
    ]

    with patch("src.api.route_registry.discover_routes", return_value=routes):
        count = register_routes(app, prefix="/api")

        assert count == 2
        assert app.include_router.call_count == 2

        calls = app.include_router.call_args_list
        assert calls[0][1]["prefix"] == "/api"
        assert len(calls[0][1]["dependencies"]) == 0

        assert len(calls[1][1]["dependencies"]) == 1


def test_dependencies_for_route():
    """Test getting dependencies for a route module."""
    deps = _dependencies_for_route("simulation")
    assert len(deps) == 1

    deps = _dependencies_for_route("core")
    assert len(deps) == 0
