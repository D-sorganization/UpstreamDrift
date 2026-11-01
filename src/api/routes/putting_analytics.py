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
* ``POST /tools/putting-analytics/variance-study`` -- which error source
  (face, path, speed, read) costs the player the most strokes over a library
  of putt classes: Sobol indices and strokes gained by removing each source.
* ``POST /tools/putting-analytics/simulated-curve`` -- the player's make rate
  and miss-leave fit by distance, the table the strokes-gained optimizer's
  ``"simulation"`` putting provider reads.
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

import importlib
from collections.abc import Callable, Mapping
from typing import Any

from fastapi import APIRouter, Body, Request

from src.api.middleware.error_handler import handle_api_errors
from src.api.rate_limit import get_limit, limiter

router = APIRouter(prefix="/tools/putting-analytics", tags=["putting-analytics"])

_SERVICE_PACKAGE = "src.shared.python.putting_analytics"

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
_VARIANCE_BODY = Body(
    ...,
    description=(
        "baseline {distances_ft, expected_putts}, classes [{distance_ft, "
        "slope_pct, orientation, count}], the player's stroke dispersion "
        "(face_sd_deg, path_sd_deg, speed_cv, biases) and read dispersion "
        "(grade_rel_sd, aspect_sd_deg, stimp_rel_sd, grade_rel_bias), factors, "
        "base_samples, n_samples, n_bootstrap, reduction and seed."
    ),
)
_CURVE_BODY = Body(
    ...,
    description=(
        "The player's stroke and read dispersion (as for variance-study), "
        "distances_ft (defaults to the optimizer's grid), n putts per "
        "distance and seed."
    ),
)


def _run(
    function_name: str, args: Mapping[str, Any], module: str = "service"
) -> dict[str, Any]:
    """Call one study in a shared putting service module, imported on first use."""
    service = importlib.import_module(f"{_SERVICE_PACKAGE}.{module}")
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


@router.post("/variance-study")
@limiter.limit(_STUDY_LIMIT)
@handle_api_errors
def putting_variance_study(
    request: Request, payload: dict[str, Any] = _VARIANCE_BODY
) -> dict[str, Any]:
    """Strokes each error source costs over ``classes``, ranked per round."""
    del request
    return _run("variance_study", payload, "service_variance")


@router.post("/simulated-curve")
@limiter.limit(_STUDY_LIMIT)
@handle_api_errors
def putting_simulated_curve(
    request: Request, payload: dict[str, Any] = _CURVE_BODY
) -> dict[str, Any]:
    """Make rate and miss-leave fit by distance for the player's dispersion."""
    del request
    return _run("curve_study", payload, "service_curve")


@router.get("/limits")
@handle_api_errors
def putting_analytics_limits() -> dict[str, Any]:
    """Input bounds and work caps enforced by the putting studies."""
    from src.shared.python.putting_analytics import (
        library,
        service,
        service_curve,
        service_variance,
    )
    from src.shared.python.putting_analytics.optimize import OBJECTIVES

    return {
        "max_distance_ft": service.MAX_DISTANCE_FT,
        "max_grade_pct": service.MAX_GRADE_PCT,
        "stimp_range": list(service.STIMP_RANGE),
        "max_abs_aim_deg": service.MAX_ABS_AIM_DEG,
        "pace_range_ft": list(service.PACE_RANGE_FT),
        "default_aims_deg": list(service.DEFAULT_AIMS_DEG),
        "default_paces_ft": list(service.DEFAULT_PACES_FT),
        "objectives": list(OBJECTIVES),
        "max_rollouts": service.MAX_ROLLOUTS,
        "max_misread_rollouts": service.MAX_MISREAD_ROLLOUTS,
        "max_library_classes": service.MAX_LIBRARY_CLASSES,
        "max_library_rollouts": service.MAX_LIBRARY_ROLLOUTS,
        "class_distance_range_ft": list(library.DISTANCE_RANGE_FT),
        "class_slope_range_pct": list(library.SLOPE_RANGE_PCT),
        "orientations": list(library.ORIENTATIONS),
        "variance_factors": list(service_variance.MODELLED_FACTORS),
        "max_variance_rollouts": service_variance.MAX_VARIANCE_ROLLOUTS,
        "default_curve_distances_ft": list(service_curve.DEFAULT_CURVE_GRID_FT),
        "max_curve_distances": service_curve.MAX_CURVE_DISTANCES,
        "max_curve_n": service_curve.MAX_CURVE_N,
        "max_curve_rollouts": service_curve.MAX_CURVE_ROLLOUTS,
    }
