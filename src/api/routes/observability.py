"""Production observability endpoints for health, readiness, and metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, PlainTextResponse

PROMETHEUS_MEDIA_TYPE = "text/plain; version=0.0.4; charset=utf-8"
REQUIRED_READY_STATE = (
    "engine_manager",
    "simulation_service",
    "analysis_service",
    "task_manager",
)

router = APIRouter()


@dataclass(frozen=True)
class Readiness:
    """Snapshot of the API readiness contract."""

    missing: tuple[str, ...]
    engines_available: int

    @property
    def ready(self) -> bool:
        """Return True when all required warmup dependencies are available."""
        return not self.missing


def _count_available_engines(engine_manager: Any) -> int:
    """Return the number of available engines exposed by the manager."""
    if engine_manager is None:
        return 0
    engines = engine_manager.get_available_engines()
    return len(engines)


def _readiness_from_state(state: Any) -> Readiness:
    """Build a readiness snapshot from FastAPI app state.

    Preconditions:
        ``state`` is the ``request.app.state`` object.

    Postconditions:
        Missing warmup dependencies are reported by state attribute name.
    """
    if state is None:
        raise ValueError("state must be provided")

    missing = tuple(name for name in REQUIRED_READY_STATE if not hasattr(state, name))
    engine_manager = getattr(state, "engine_manager", None)
    return Readiness(
        missing=missing,
        engines_available=_count_available_engines(engine_manager),
    )


def _sidekick_identity(state: Any) -> dict[str, str]:
    """Return the public launcher identity, never its capability token."""
    instance_id = getattr(state, "sidekick_instance_id", "")
    if not isinstance(instance_id, str) or not instance_id:
        return {}
    return {"sidekick_instance_id": instance_id}


def _metric_help(name: str, description: str, metric_type: str) -> list[str]:
    """Return Prometheus HELP and TYPE lines for one metric."""
    return [f"# HELP {name} {description}", f"# TYPE {name} {metric_type}"]


def _bool_metric(value: bool) -> int:
    """Convert a boolean state to a Prometheus gauge value."""
    return 1 if value else 0


def _build_prometheus_metrics(request: Request) -> str:
    """Build the Prometheus text exposition for API observability."""
    state = request.app.state
    readiness = _readiness_from_state(state)
    started_at = float(getattr(state, "api_started_at", 0.0))
    static_files_mounted = bool(getattr(state, "static_files_mounted", False))
    route_count = len(request.app.routes)

    lines: list[str] = []
    lines.extend(_metric_help("upstreamdrift_api_info", "API build info.", "gauge"))
    lines.append('upstreamdrift_api_info{service="upstreamdrift"} 1')
    lines.extend(
        _metric_help("upstreamdrift_api_ready", "API readiness state.", "gauge")
    )
    lines.append(f"upstreamdrift_api_ready {_bool_metric(readiness.ready)}")
    lines.extend(
        _metric_help(
            "upstreamdrift_api_routes_total", "Registered route count.", "gauge"
        )
    )
    lines.append(f"upstreamdrift_api_routes_total {route_count}")
    lines.extend(
        _metric_help(
            "upstreamdrift_api_engines_available", "Available physics engines.", "gauge"
        )
    )
    lines.append(f"upstreamdrift_api_engines_available {readiness.engines_available}")
    lines.extend(
        _metric_help(
            "upstreamdrift_api_static_files_mounted", "Static UI mount state.", "gauge"
        )
    )
    lines.append(
        f"upstreamdrift_api_static_files_mounted {_bool_metric(static_files_mounted)}"
    )
    lines.extend(
        _metric_help(
            "upstreamdrift_api_startup_timestamp_seconds",
            "API startup Unix timestamp.",
            "gauge",
        )
    )
    lines.append(f"upstreamdrift_api_startup_timestamp_seconds {started_at:.3f}")
    return "\n".join(lines) + "\n"


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    """Return a shallow liveness response independent of warmup state."""
    return {"status": "alive"}


@router.get("/readyz", response_model=None)
async def readyz(request: Request) -> dict[str, Any] | JSONResponse:
    """Return readiness after required startup warmup dependencies exist."""
    state = request.app.state
    readiness = _readiness_from_state(state)
    sidekick_identity = _sidekick_identity(state)
    if readiness.ready:
        return {
            "status": "ready",
            "engines_available": readiness.engines_available,
            **sidekick_identity,
        }
    return JSONResponse(
        status_code=503,
        content={
            "status": "not_ready",
            "missing": list(readiness.missing),
            "engines_available": readiness.engines_available,
            **sidekick_identity,
        },
    )


@router.get("/metrics", response_class=PlainTextResponse)
async def metrics(request: Request) -> PlainTextResponse:
    """Return Prometheus text exposition for the API scrape endpoint."""
    return PlainTextResponse(
        _build_prometheus_metrics(request),
        media_type=PROMETHEUS_MEDIA_TYPE,
    )
