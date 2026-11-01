"""Putting analytics API routes (#14177).

REST access to the shared putting studies in
:mod:`src.shared.python.putting_analytics.service`, the same functions the
simulation MCP tools call (:mod:`src.sim_run.putting_tools`):

* ``POST /tools/putting-analytics/optimize`` -- best aim and pace for one
  putt, by strokes gained against the caller's expected-putts curve (``sg``)
  or by make rate (``must_make``).
* ``POST /tools/putting-analytics/misread-cost`` -- expected-strokes cost of a
  set of green reads (slope and speed misjudgements) against a perfect read.
* ``POST /tools/putting-analytics/library-study`` -- weighted report over a
  library of putt classes: expected putts per round, make rate by band,
  optimal pace and strokes gained.
* ``GET /tools/putting-analytics/limits`` -- the input bounds and work caps
  the studies enforce, so a client can build its form without guessing.

Request bodies are the service's JSON arguments, passed through unchanged:
the service validates every field and caps the work (an invalid or
oversized request is a 400 with the service's message). The service is
imported inside each handler so API start-up never loads the roll
integrator. Handlers are synchronous because a study is seconds of CPU work;
FastAPI runs them in its thread pool instead of on the event loop.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from fastapi import APIRouter, Body, Request

from src.api.middleware.error_handler import handle_api_errors
from src.api.rate_limit import get_limit, limiter

router = APIRouter(prefix="/tools/putting-analytics", tags=["putting-analytics"])

#: Studies run for seconds each; this keeps one client from queueing many.
_STUDY_LIMIT = get_limit("API_LIMIT_PUTTING_ANALYTICS", "10/minute")

_STUDY_BODY = Body(
    ...,
    description=(
        "Study arguments as documented by the matching putting MCP tool: "
        "distance_ft, grade_pct, aspect_deg, stimp, line_sd_deg, speed_sd, "
        "baseline {distances_ft, expected_putts}, aims_deg, paces_ft, "
        "objective ('sg' or 'must_make'), n and seed."
    ),
)


def _run(function_name: str, args: Mapping[str, Any]) -> dict[str, Any]:
    """Call one study in the shared putting service, imported on first use."""
    from src.shared.python.putting_analytics import service

    study: Callable[[Mapping[str, Any]], dict[str, Any]] = getattr(
        service, function_name
    )
    return study(args)


@router.post("/optimize")
@limiter.limit(_STUDY_LIMIT)
@handle_api_errors
def optimize_putt(
    request: Request, payload: dict[str, Any] = _STUDY_BODY
) -> dict[str, Any]:
    """Best aim and pace for one putt under the player's stroke noise."""
    del request
    return _run("optimize_putt", payload)


@router.post("/misread-cost")
@limiter.limit(_STUDY_LIMIT)
@handle_api_errors
def putting_misread_cost(
    request: Request, payload: dict[str, Any] = _STUDY_BODY
) -> dict[str, Any]:
    """Cost of the supplied ``reads`` (slope_scale, stimp_ratio) vs a true read."""
    del request
    return _run("misread_cost", payload)


@router.post("/library-study")
@limiter.limit(_STUDY_LIMIT)
@handle_api_errors
def putting_library_study(
    request: Request, payload: dict[str, Any] = _STUDY_BODY
) -> dict[str, Any]:
    """Weighted report over ``classes`` of {distance_ft, slope_pct, ...}."""
    del request
    return _run("library_study", payload)


@router.get("/limits")
@handle_api_errors
def putting_analytics_limits() -> dict[str, Any]:
    """Input bounds and work caps enforced by the putting studies."""
    from src.shared.python.putting_analytics import library, service

    return {
        "max_distance_ft": service.MAX_DISTANCE_FT,
        "max_grade_pct": service.MAX_GRADE_PCT,
        "stimp_range": list(service.STIMP_RANGE),
        "max_abs_aim_deg": service.MAX_ABS_AIM_DEG,
        "pace_range_ft": list(service.PACE_RANGE_FT),
        "default_aims_deg": list(service.DEFAULT_AIMS_DEG),
        "default_paces_ft": list(service.DEFAULT_PACES_FT),
        "objectives": list(service.OBJECTIVES),
        "max_rollouts": service.MAX_ROLLOUTS,
        "max_misread_rollouts": service.MAX_MISREAD_ROLLOUTS,
        "max_library_classes": service.MAX_LIBRARY_CLASSES,
        "max_library_rollouts": service.MAX_LIBRARY_ROLLOUTS,
        "class_distance_range_ft": list(library.DISTANCE_RANGE_FT),
        "class_slope_range_pct": list(library.SLOPE_RANGE_PCT),
        "orientations": list(library.ORIENTATIONS),
    }
