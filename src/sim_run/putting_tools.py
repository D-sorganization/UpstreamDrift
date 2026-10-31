"""Putting MCP tools for `SimulationMcpServer` (epic #14141, putting v4 step 9a/9b).

Design of record (``research/step9_plan.md`` rows 9a/9b): these tools join
the shared stdio server rather than vendoring the retired Gemini
``putting_mcp_server.py`` (different tool surface). Every
handler is a pure computation, so Private Mode's
``check_execution_allowed`` never applies — only ``check_auth`` does, as
for the read-only ``get_run``/``fetch_artifact`` tools. The step 9b
``putting_make_probability`` tool is a seeded Monte Carlo estimate,
not a calibrated player or tour figure.
"""

from __future__ import annotations

import json
import math
from typing import Any, Protocol

from src.shared.python.putting_dynamics import (
    HOLE_RADIUS_M,
    USGA_STIMP_ROLL_MODEL,
    PutterState,
    SurfaceSpec,
    capture_speed_mps,
    measure_stimp_ft,
    simulate_strike,
    solve_aim,
)
from src.shared.python.sg_optimizer.shot_model.putting_physics import (
    StrokeNoise,
    simulated_make_probability,
)


class _SecurityCheck(Protocol):
    """Structural type for the one `SecurityPolicy` method these tools use."""

    def check_auth(self, token: str | None) -> None: ...


def _envelope(payload: dict[str, Any]) -> dict[str, Any]:
    """The MCP tool-result content envelope, shared by all four handlers."""
    return {"content": [{"type": "text", "text": json.dumps(payload)}]}


def _require_number(
    arguments: dict[str, Any],
    key: str,
    default: float | None = None,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Validate one numeric MCP argument; raises ValueError if invalid."""
    raw = arguments.get(key, default)
    if raw is None:
        raise ValueError(f"Parameter '{key}' is required")
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        raise ValueError(f"Parameter '{key}' must be a number")
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(f"Parameter '{key}' must be finite")
    if minimum is not None and value < minimum:
        raise ValueError(f"Parameter '{key}' must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"Parameter '{key}' must be <= {maximum}")
    return value


def _surface_from_args(arguments: dict[str, Any]) -> SurfaceSpec:
    """Build the planar, uniform-friction surface shared by every tool (DRY)."""
    stimp_ft = _require_number(arguments, "stimp_ft", 10.0, minimum=3.0, maximum=16.0)
    slope_pct = _require_number(arguments, "slope_pct", 0.0, minimum=0.0, maximum=10.0)
    slope_aspect_deg = _require_number(
        arguments, "slope_aspect_deg", 0.0, minimum=-360.0, maximum=360.0
    )
    return SurfaceSpec.planar_uniform(stimp_ft, slope_pct, slope_aspect_deg)


def tool_specs() -> list[dict[str, Any]]:
    """Input schemas for the four putting tools (merged into `_tools()`)."""
    auth = {"auth_token": {"type": "string"}}
    surface = {
        "stimp_ft": {"type": "number"},
        "slope_pct": {"type": "number"},
        "slope_aspect_deg": {"type": "number"},
    }
    return [
        {
            "name": "putting_simulate_putt",
            "description": "Simulate one putt on the shared putting-dynamics core.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    **auth,
                    **surface,
                    "putter_speed_mps": {"type": "number"},
                    "heading_deg": {"type": "number"},
                    "distance_m": {"type": "number"},
                },
                "required": ["putter_speed_mps", "distance_m"],
            },
        },
        {
            "name": "putting_read_green",
            "description": "Solve the putter speed and heading that holes a putt.",
            "inputSchema": {
                "type": "object",
                "properties": {**auth, **surface, "distance_m": {"type": "number"}},
                "required": ["distance_m"],
            },
        },
        {
            "name": "putting_stimpmeter",
            "description": "Virtual USGA Stimpmeter reading of a green.",
            "inputSchema": {"type": "object", "properties": {**auth, **surface}},
        },
        {
            "name": "putting_capture_speed",
            "description": "Max ball speed the hole can still capture at an offset.",
            "inputSchema": {
                "type": "object",
                "properties": {**auth, "entry_offset_m": {"type": "number"}},
            },
        },
        {
            "name": "putting_make_probability",
            "description": (
                "Seeded Monte Carlo make-% estimate from stroke/read noise "
                "(not a calibrated player or tour figure)."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    **auth,
                    **surface,
                    "distance_ft": {"type": "number"},
                    "face_sd_deg": {"type": "number"},
                    "speed_cv": {"type": "number"},
                    "read_sd_deg": {"type": "number"},
                    "n_trials": {"type": "number"},
                    "seed": {"type": "number"},
                },
                "required": ["distance_ft"],
            },
        },
    ]


def tool_putting_simulate_putt(
    security: _SecurityCheck, arguments: dict[str, Any]
) -> dict[str, Any]:
    """Strike a putt and roll it to rest (or the hole) on the shared core."""
    security.check_auth(arguments.get("auth_token"))
    surface = _surface_from_args(arguments)
    putter_speed_mps = _require_number(
        arguments, "putter_speed_mps", minimum=0.01, maximum=10.0
    )
    heading_deg = _require_number(
        arguments, "heading_deg", 0.0, minimum=-360.0, maximum=360.0
    )
    distance_m = _require_number(arguments, "distance_m", minimum=0.1, maximum=30.0)
    result = simulate_strike(
        PutterState.typical(putter_speed_mps),
        surface,
        hole_x_m=distance_m,
        hole_y_m=0.0,
        heading_rad=math.radians(heading_deg),
    )
    return _envelope(
        {
            "rest_x_m": result.rest_x_m,
            "rest_y_m": result.rest_y_m,
            "holed": result.holed,
            "total_distance_m": result.total_distance_m,
            "time_s": result.time_s,
            "roll_model": USGA_STIMP_ROLL_MODEL,
        }
    )


def tool_putting_read_green(
    security: _SecurityCheck, arguments: dict[str, Any]
) -> dict[str, Any]:
    """Solve the aim and pace that holes a putt to a hole along +x."""
    security.check_auth(arguments.get("auth_token"))
    surface = _surface_from_args(arguments)
    distance_m = _require_number(arguments, "distance_m", minimum=0.1, maximum=30.0)
    solution = solve_aim(
        surface, (0.0, 0.0), (distance_m, 0.0), PutterState.typical(1.0)
    )
    return _envelope(
        {
            "putter_speed_mps": solution.putter_speed_mps,
            "heading_deg": math.degrees(solution.heading_rad),
            "break_m": solution.break_m,
            "holed": solution.holed,
            "converged": solution.converged,
            "roll_model": USGA_STIMP_ROLL_MODEL,
        }
    )


def tool_putting_stimpmeter(
    security: _SecurityCheck, arguments: dict[str, Any]
) -> dict[str, Any]:
    """Run the virtual USGA Stimpmeter two-direction reading on a green."""
    security.check_auth(arguments.get("auth_token"))
    surface = _surface_from_args(arguments)
    return _envelope({"measured_stimp_ft": measure_stimp_ft(surface)})


def tool_putting_capture_speed(
    security: _SecurityCheck, arguments: dict[str, Any]
) -> dict[str, Any]:
    """Report the hole's max capture speed at a lateral entry offset."""
    security.check_auth(arguments.get("auth_token"))
    entry_offset_m = _require_number(
        arguments, "entry_offset_m", 0.0, minimum=0.0, maximum=HOLE_RADIUS_M
    )
    return _envelope(
        {
            "max_capture_speed_mps": capture_speed_mps(entry_offset_m),
            "entry_offset_m": entry_offset_m,
        }
    )


def tool_putting_make_probability(
    security: _SecurityCheck, arguments: dict[str, Any]
) -> dict[str, Any]:
    """Seeded Monte Carlo make-% estimate; not a calibrated tour figure."""
    security.check_auth(arguments.get("auth_token"))
    stimp_ft = _require_number(arguments, "stimp_ft", 10.0, minimum=3.0, maximum=16.0)
    slope_pct = _require_number(arguments, "slope_pct", 0.0, minimum=0.0, maximum=10.0)
    slope_aspect_deg = _require_number(
        arguments, "slope_aspect_deg", 90.0, minimum=-360.0, maximum=360.0
    )
    distance_ft = _require_number(arguments, "distance_ft", minimum=0.3, maximum=98.0)
    noise = StrokeNoise(
        face_sd_deg=_require_number(
            arguments, "face_sd_deg", 0.0, minimum=0.0, maximum=45.0
        ),
        speed_cv=_require_number(arguments, "speed_cv", 0.0, minimum=0.0, maximum=2.0),
        read_sd_deg=_require_number(
            arguments, "read_sd_deg", 0.0, minimum=0.0, maximum=45.0
        ),
    )
    n_trials = int(
        _require_number(arguments, "n_trials", 32.0, minimum=1.0, maximum=200.0)
    )
    seed = int(_require_number(arguments, "seed", 0.0, minimum=0.0, maximum=2**31 - 1))
    make_probability = simulated_make_probability(
        distance_ft,
        noise,
        stimp_ft=stimp_ft,
        slope_pct=slope_pct,
        slope_aspect_deg=slope_aspect_deg,
        n_trials=n_trials,
        seed=seed,
    )
    return _envelope(
        {
            "make_probability": make_probability,
            "n_trials": n_trials,
            "standard_error": math.sqrt(
                make_probability * (1.0 - make_probability) / n_trials
            ),
        }
    )
