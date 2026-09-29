"""Shared Calculation Backend -- FastAPI application.

Provides a unified REST API wrapping all process calculators so React frontends
and other HTTP clients can call validated Python calculation engines.

Usage:
    uvicorn calc_backend.app:app --reload --port 8010

See issue #613.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, Request

from src.shared.python.cors import add_cors_middleware

from .health import CheckStatus, get_health_checker
from .routers import (
    acid_gas_dewpoint,
    baghouse,
    financial,
    flare,
    flow_rate,
    ode_solver,
    pressure_drop,
    rotation_converter,
    scrubber,
    symbolic_solver,
    syngas_water,
    thermal_profile,
    wgs_reactor,
)

logger = logging.getLogger(__name__)

CALCULATOR_ROUTERS: tuple[APIRouter, ...] = (
    flare.router,
    wgs_reactor.router,
    baghouse.router,
    scrubber.router,
    financial.router,
    acid_gas_dewpoint.router,
    pressure_drop.router,
    flow_rate.router,
    syngas_water.router,
    thermal_profile.router,
    ode_solver.router,
    rotation_converter.router,
    symbolic_solver.router,
)

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Shared Calculation Backend",
    description=(
        "Unified REST API for process engineering calculators.  "
        "Auto-generated docs at /docs (Swagger UI) and /redoc."
    ),
    version="1.0.0",
)
add_cors_middleware(app)

# ---------------------------------------------------------------------------
# Include routers
# ---------------------------------------------------------------------------

for router in CALCULATOR_ROUTERS:
    app.include_router(router)


# ---------------------------------------------------------------------------
# Health check endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict[str, str]:
    """Health-check endpoint (liveness probe).

    Used by Docker/Kubernetes to verify the application is running.
    Returns HTTP 200 if responsive.

    Returns:
        {"status": "ok"} if healthy
    """
    return {"status": "ok"}


@app.get("/api/health")
async def api_health() -> dict[str, str]:
    """API health endpoint (liveness probe).

    Alternative path for containerized deployment.
    Used by Docker HEALTHCHECK and Kubernetes liveness probes.

    Returns:
        {"status": "ok"} if healthy
    """
    return {"status": "ok"}


@app.get("/api/ready")
async def api_ready() -> dict[str, str | dict[str, object] | bool]:
    """Readiness probe endpoint.

    Verifies the application is ready to serve requests.
    Runs comprehensive health checks on:
    - Python runtime
    - Required dependencies
    - Application initialization state

    Used by Kubernetes readiness probes and load balancers to determine
    if traffic should be routed to this instance.

    Returns:
        {
            "status": "ok" | "degraded" | "unhealthy",
            "ready": true | false,
            "checks": {
                "python_runtime": {"status": "ok", "details": {...}},
                "dependencies": {"status": "ok", "details": {...}},
                ...
            }
        }

    Status codes:
        200: Ready to serve requests
        503: Not ready (degraded or unhealthy)
    """
    health_checker = get_health_checker()
    overall_status, checks = await health_checker.run_checks()

    ready = overall_status == CheckStatus.OK

    response = {
        "status": overall_status.value,
        "ready": ready,
        "checks": checks,
    }

    # Raise HTTP 503 if not ready
    if not ready:
        raise HTTPException(
            status_code=503,
            detail=response,
        )

    return response


@app.get("/api/calc/endpoints")
def list_endpoints(request: Request) -> dict[str, list[str]]:
    """List all available calculator endpoints."""
    active_app = request.app
    expected = _expected_calculator_route_signatures()
    _ensure_calculator_routes_registered(active_app, expected=expected)
    calculators = _registered_calculator_route_signatures(active_app) | expected
    return {"calculators": sorted(f"{method} {path}" for method, path in calculators)}


def _calculator_route_signatures(
    routes: Iterable[Any],
    *,
    prefix: str = "",
) -> set[tuple[str, str]]:
    signatures: set[tuple[str, str]] = set()
    for route in routes:
        raw_path = getattr(route, "path", None)
        if raw_path is None:
            raw_path = getattr(route, "path_format", None)
        methods = getattr(route, "methods", None)
        if raw_path is None or not methods:
            continue

        path = _join_route_prefix(prefix, str(raw_path))
        if not path.startswith("/api/calc/") or path == "/api/calc/endpoints":
            continue

        for method in methods:
            method = str(method)
            if method not in {"HEAD", "OPTIONS"}:
                signatures.add((method, path))
    return signatures


def _expected_calculator_route_signatures() -> set[tuple[str, str]]:
    signatures: set[tuple[str, str]] = set()
    for router in CALCULATOR_ROUTERS:
        signatures.update(
            _calculator_route_signatures(
                router.routes,
                prefix=str(getattr(router, "prefix", "")),
            )
        )
    return signatures


def _registered_calculator_route_signatures(
    active_app: FastAPI,
) -> set[tuple[str, str]]:
    signatures = _calculator_route_signatures(active_app.routes)
    if signatures:
        return signatures

    openapi = getattr(active_app, "openapi", None)
    if not callable(openapi):
        return signatures

    schema = openapi()
    for path, operations in schema.get("paths", {}).items():
        if not path.startswith("/api/calc/") or path == "/api/calc/endpoints":
            continue
        for method in operations:
            method = str(method).upper()
            if method not in {"HEAD", "OPTIONS", "PARAMETERS"}:
                signatures.add((method, path))
    return signatures


def _ensure_calculator_routes_registered(
    active_app: FastAPI,
    *,
    expected: set[tuple[str, str]] | None = None,
) -> None:
    registered = _calculator_route_signatures(active_app.routes)
    expected = expected or _expected_calculator_route_signatures()
    if expected.issubset(registered):
        return

    for router in CALCULATOR_ROUTERS:
        router_signatures = _calculator_route_signatures(
            router.routes,
            prefix=str(getattr(router, "prefix", "")),
        )
        if router_signatures and not router_signatures.issubset(registered):
            active_app.include_router(router)
            active_app.openapi_schema = None
            registered = _calculator_route_signatures(active_app.routes)


def _join_route_prefix(prefix: str, path: str) -> str:
    if not prefix:
        return path
    if path.startswith((prefix, "/api/calc/")):
        return path
    if not path or path == "/":
        return prefix
    return f"{prefix.rstrip('/')}/{path.lstrip('/')}"
