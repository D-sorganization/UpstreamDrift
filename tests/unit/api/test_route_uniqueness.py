"""Route-uniqueness regression tests (issue #7998).

FastAPI resolves requests with first-match-wins semantics. When two route
modules declare the *same* method + path, the second handler becomes
permanently unreachable and clients silently receive a response shape they did
not ask for.

That is exactly what happened in issue #7998: ``physics.py`` declared
``GET/POST /simulation/actuators`` and ``GET /simulation/forces``, and — being
registered before ``actuator_controls.py`` and ``force_overlays.py`` in
``_REGISTRATION_ORDER`` — shadowed both. The React ``ActuatorPanel`` received an
``ActuatorStateResponse`` instead of the ``ActuatorPanelResponse`` it expects
and threw a render-time ``TypeError`` that tripped the root ``ErrorBoundary``,
taking the whole UI to the error screen. ``ForceOverlayPanel`` rendered no
arrows at all.

These tests assert the invariant directly against the assembled route table so
the failure mode cannot recur silently.
"""

from __future__ import annotations

import re
from collections import defaultdict

import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute

from src.api.route_registry import discover_routes, register_routes

pytestmark = pytest.mark.unit

_ROUTE_PARAMETER_RE = re.compile(r"\{[^}/]+\}")


def _normalize_route_path(path: str) -> str:
    """Treat routes that differ only by parameter names as the same path shape."""
    return _ROUTE_PARAMETER_RE.sub("{}", path)


def _route_table(app: FastAPI) -> dict[tuple[str, str], list[str]]:
    """Map (method, path) -> list of handler qualified names."""
    table: dict[tuple[str, str], list[str]] = defaultdict(list)
    for route in app.routes:
        if not isinstance(route, APIRoute):
            continue
        for method in sorted(route.methods or ()):
            if method in {"HEAD", "OPTIONS"}:
                continue
            endpoint = route.endpoint
            qualname = getattr(endpoint, "__qualname__", str(endpoint))
            table[(method, _normalize_route_path(route.path))].append(
                f"{endpoint.__module__}.{qualname}"
            )
    return table


def _reachable_endpoints(app: FastAPI) -> set[int]:
    """Endpoint function identities that win first-match-wins resolution."""
    seen: dict[tuple[str, str], int] = {}
    for route in app.routes:
        if not isinstance(route, APIRoute):
            continue
        for method in route.methods or ():
            seen.setdefault(
                (method, _normalize_route_path(route.path)), id(route.endpoint)
            )
    return set(seen.values())


@pytest.fixture(scope="module")
def registered_app() -> FastAPI:
    """A FastAPI app with every discovered router mounted once, no prefix."""
    app = FastAPI()
    register_routes(app, prefix="")
    return app


def test_no_duplicate_method_path_pairs(registered_app: FastAPI) -> None:
    """Every method+path maps to exactly ONE handler."""
    duplicates = {
        key: handlers
        for key, handlers in _route_table(registered_app).items()
        if len(handlers) > 1
    }
    assert not duplicates, (
        "Shadowed routes detected — the later handler is unreachable "
        "(issue #7998):\n"
        + "\n".join(
            f"  {method} {path}: {handlers}"
            for (method, path), handlers in sorted(duplicates.items())
        )
    )


def test_route_path_normalization_ignores_parameter_names() -> None:
    """Same-shape parameterized routes must be treated as duplicate paths."""
    assert _normalize_route_path(
        "/engines/{engine_name}/load"
    ) == _normalize_route_path("/engines/{engine_type}/load")


@pytest.mark.parametrize(
    ("method", "path", "expected_module"),
    [
        ("GET", "/simulation/actuators", "src.api.routes.actuator_controls"),
        ("POST", "/simulation/actuators", "src.api.routes.actuator_controls"),
        ("GET", "/simulation/forces", "src.api.routes.force_overlays"),
    ],
)
def test_frontend_paths_resolve_to_expected_module(
    registered_app: FastAPI, method: str, path: str, expected_module: str
) -> None:
    """The UI-facing paths must be served by the module the UI's types match."""
    handlers = _route_table(registered_app).get((method, path), [])
    assert handlers, f"{method} {path} is not registered"
    assert handlers[0].startswith(expected_module), (
        f"{method} {path} is served by {handlers[0]}, expected {expected_module}"
    )


def test_physics_control_paths_are_distinct(registered_app: FastAPI) -> None:
    """physics.py keeps its own endpoints under /simulation/control/*."""
    table = _route_table(registered_app)
    for method, path in (
        ("GET", "/simulation/control/actuators"),
        ("POST", "/simulation/control/actuators"),
        ("GET", "/simulation/control/forces"),
    ):
        handlers = table.get((method, path), [])
        assert handlers, f"{method} {path} is not registered"
        assert handlers[0].startswith("src.api.routes.physics")


def test_every_discovered_module_contributes_reachable_routes(
    registered_app: FastAPI,
) -> None:
    """No discovered route module is fully shadowed by an earlier one."""
    reachable = _reachable_endpoints(registered_app)
    for module_name, router in discover_routes():
        endpoints = {
            id(route.endpoint) for route in router.routes if isinstance(route, APIRoute)
        }
        if not endpoints:
            continue
        assert endpoints & reachable, (
            f"Every route in src.api.routes.{module_name} is shadowed by an "
            "earlier module (issue #7998)"
        )
