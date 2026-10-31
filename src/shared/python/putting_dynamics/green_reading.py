"""Simulation-based green reading: line profile and aim solving (#14141 7b).

Reused shared infrastructure (AGENTS.md section A discovery):

* DbC helpers from ``src.shared.python.contracts``.
* The strike + mode-machine solver in :mod:`.solver`
  (:func:`~.solver.simulate_strike`); this module reads a green by
  *simulation* rather than a separate closed-form heuristic, so a
  recommendation always agrees with the trajectory another caller of
  :mod:`.solver` would see for the same surface.

Model
-----
:func:`read_line` samples :class:`~.surfaces.HeightField` elevation and
gradient along the straight ``start`` -> ``end`` segment — a thin
wrapper with no physics of its own, used for the elevation/slope
profile a UI plots alongside the aim recommendation.

:func:`solve_aim` finds the putter-head speed and heading so a struck
ball's rest point lands ``past_hole_m`` beyond the hole along the aim
line, or is holed. Two nested 1-D root-finds, both bounded:

* **Speed** (inner, bisection): for a fixed heading, the along-heading
  rest distance is monotone non-decreasing in head speed except for a
  flat "captured" plateau at the hole distance (a holed putt never
  overruns); bisection against the target distance — treating a holed
  result as sitting on that plateau — finds the slowest speed that
  reaches the target without relying on a separate closed form.
* **Heading** (outer, secant): the lateral miss of the rest point from
  the straight ``start`` -> ``hole`` line, extended past the hole, is
  driven to zero. A holed putt has zero miss by definition. The first
  secant step is the small-angle estimate ``-miss / distance``; it
  converges in a handful of iterations because the miss is close to
  linear in heading for putting-sized slopes and distances.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from src.shared.python.contracts import ensure, require, require_finite

from .solver import simulate_strike
from .surfaces import SurfaceSpec
from .types import PutterState

__all__ = [
    "AimSolution",
    "LineReading",
    "read_line",
    "solve_aim",
]

#: Putter-head speed search bracket [m/s] (within `PutterState`'s own
#: ``(0, 10]`` contract; 8 m/s covers putts well beyond any realistic
#: green while keeping the exponential bracket search's worst case
#: cheap).
_SPEED_MIN_MPS = 0.05
_SPEED_MAX_MPS = 8.0

#: Bounded iteration caps (module docstring: deterministic, bounded).
_MAX_SPEED_ITERATIONS = 30
_MAX_HEADING_ITERATIONS = 12

#: Convergence tolerances [m] (sub-cm; a few cm is the documented
#: real-world precision this module targets, so these leave headroom
#: without burning extra RK4-step-bounded simulation iterations).
_SPEED_DISTANCE_TOL_M = 5e-3
_HEADING_MISS_TOL_M = 5e-3


@dataclass(frozen=True)
class LineReading:
    """Sampled elevation/slope profile along a straight putt line.

    Attributes:
        positions_m: Sampled ``(x, y)`` points from ``start`` to
            ``end``, inclusive of both ends.
        elevations_m: Surface elevation at each sampled point [m].
        slopes: Surface gradient ``(dh/dx, dh/dy)`` at each point.
        distance_m: Straight-line ``start`` -> ``end`` distance [m].
    """

    positions_m: tuple[tuple[float, float], ...]
    elevations_m: tuple[float, ...]
    slopes: tuple[tuple[float, float], ...]
    distance_m: float

    def __post_init__(self) -> None:
        """Cheap invariants on one line reading (characterization)."""
        require(
            len(self.positions_m) == len(self.elevations_m) == len(self.slopes),
            "positions, elevations and slopes must share a length",
            (len(self.positions_m), len(self.elevations_m), len(self.slopes)),
        )
        require(len(self.positions_m) >= 2, "a line needs >= 2 samples")
        for x, y in self.positions_m:
            require_finite(x, "positions_m.x")
            require_finite(y, "positions_m.y")
        for elevation in self.elevations_m:
            require_finite(elevation, "elevations_m")
        for gx, gy in self.slopes:
            require_finite(gx, "slopes.x")
            require_finite(gy, "slopes.y")
        require_finite(self.distance_m, "distance_m")
        require(self.distance_m >= 0.0, "distance_m must be >= 0", self.distance_m)


def read_line(
    surface: SurfaceSpec,
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    num_samples: int = 20,
) -> LineReading:
    """Sample elevation and slope along the segment ``start`` -> ``end``.

    Args:
        surface: Green description.
        start_xy: Line start ``(x, y)`` [m].
        end_xy: Line end ``(x, y)`` [m].
        num_samples: Number of evenly spaced samples; must be >= 2.

    Returns:
        A populated :class:`LineReading`.

    Raises:
        ValueError: If an input is non-finite or ``num_samples < 2``.
    """
    require_finite(start_xy[0], "start_xy[0]")
    require_finite(start_xy[1], "start_xy[1]")
    require_finite(end_xy[0], "end_xy[0]")
    require_finite(end_xy[1], "end_xy[1]")
    require(num_samples >= 2, "num_samples must be >= 2", num_samples)
    sx, sy = start_xy
    ex, ey = end_xy
    positions = tuple(
        (sx + t * (ex - sx), sy + t * (ey - sy))
        for t in (i / (num_samples - 1) for i in range(num_samples))
    )
    return LineReading(
        positions_m=positions,
        elevations_m=tuple(surface.height.elevation(x, y) for x, y in positions),
        slopes=tuple(surface.height.gradient(x, y) for x, y in positions),
        distance_m=math.hypot(ex - sx, ey - sy),
    )


@dataclass(frozen=True)
class AimSolution:
    """Solved aim line for one putt (module docstring algorithm).

    Attributes:
        putter_speed_mps: Putter-head speed at impact [m/s].
        heading_rad: Putter forward-axis heading, CCW from +x [rad].
        aim_point_xy: Point at the hole's distance along ``heading_rad``
            from the start — where the golfer should aim.
        break_m: Signed lateral offset of ``aim_point_xy`` from the
            hole, measured perpendicular to the straight start-hole
            line (sign flips with the slope's cross-component).
        rest_xy: Where the simulated ball actually came to rest (or the
            hole, when holed).
        converged: Whether the heading root-find met its tolerance
            within :data:`_MAX_HEADING_ITERATIONS`.
    """

    putter_speed_mps: float
    heading_rad: float
    aim_point_xy: tuple[float, float]
    break_m: float
    rest_xy: tuple[float, float]
    converged: bool

    def __post_init__(self) -> None:
        """Cheap invariants on one aim solution (characterization)."""
        require_finite(self.putter_speed_mps, "putter_speed_mps")
        require_finite(self.heading_rad, "heading_rad")
        require_finite(self.aim_point_xy[0], "aim_point_xy[0]")
        require_finite(self.aim_point_xy[1], "aim_point_xy[1]")
        require_finite(self.break_m, "break_m")
        require_finite(self.rest_xy[0], "rest_xy[0]")
        require_finite(self.rest_xy[1], "rest_xy[1]")
        require(
            isinstance(self.converged, bool), "converged must be a bool", self.converged
        )


def _strike_along_distance(
    surface: SurfaceSpec,
    putter_template: PutterState,
    start_xy: tuple[float, float],
    hole_xy: tuple[float, float],
    heading: float,
    speed_mps: float,
) -> tuple[float, float, bool, float]:
    """One strike at ``speed_mps``; returns rest, holed, and along-distance.

    A holed putt never overruns, so its along-heading distance is
    reported as the straight start-hole distance (module docstring: the
    speed bisection then treats it as sitting on the under-target side
    of the bracket, never as an overshoot).
    """
    sx, sy = start_xy
    hx, hy = hole_xy
    result = simulate_strike(
        replace(putter_template, speed_mps=speed_mps),
        surface,
        hole_x_m=hx,
        hole_y_m=hy,
        start_x_m=sx,
        start_y_m=sy,
        heading_rad=heading,
    )
    if result.holed:
        along = math.hypot(hx - sx, hy - sy)
    else:
        along = (result.rest_x_m - sx) * math.cos(heading) + (
            result.rest_y_m - sy
        ) * math.sin(heading)
    return result.rest_x_m, result.rest_y_m, result.holed, along


def _rest_for_heading(
    surface: SurfaceSpec,
    putter_template: PutterState,
    start_xy: tuple[float, float],
    hole_xy: tuple[float, float],
    heading: float,
    target_distance_m: float,
) -> tuple[float, float, float, bool]:
    """Bisect putter speed so the along-heading rest distance hits target.

    An exponential search first brackets the answer (most putts need a
    putter speed well under :data:`_SPEED_MAX_MPS`, so this avoids
    repeatedly simulating expensive, far-too-fast strikes), then
    bisection narrows the bracket to :data:`_SPEED_DISTANCE_TOL_M`.

    Returns:
        ``(rest_x_m, rest_y_m, speed_mps, holed)`` for the converged
        (or iteration-capped) speed.
    """
    rest_x, rest_y, holed, along = _strike_along_distance(
        surface, putter_template, start_xy, hole_xy, heading, _SPEED_MIN_MPS
    )
    diff = along - target_distance_m
    lo, hi = _SPEED_MIN_MPS, _SPEED_MIN_MPS
    while diff < 0.0 and hi < _SPEED_MAX_MPS:
        lo = hi
        hi = min(2.0 * hi, _SPEED_MAX_MPS)
        rest_x, rest_y, holed, along = _strike_along_distance(
            surface, putter_template, start_xy, hole_xy, heading, hi
        )
        diff = along - target_distance_m
    speed = hi
    for _ in range(_MAX_SPEED_ITERATIONS):
        if abs(diff) <= _SPEED_DISTANCE_TOL_M or hi <= lo:
            break
        speed = 0.5 * (lo + hi)
        rest_x, rest_y, holed, along = _strike_along_distance(
            surface, putter_template, start_xy, hole_xy, heading, speed
        )
        diff = along - target_distance_m
        if diff < 0.0:
            lo = speed
        else:
            hi = speed
    return rest_x, rest_y, speed, holed


def solve_aim(
    surface: SurfaceSpec,
    start_xy: tuple[float, float],
    hole_xy: tuple[float, float],
    putter_template: PutterState,
    past_hole_m: float = 0.3,
) -> AimSolution:
    """Solve the putter speed and heading that holes (or just passes) the hole.

    Args:
        surface: Green description.
        start_xy: Ball start ``(x, y)`` [m].
        hole_xy: Hole center ``(x, y)`` [m]; must differ from ``start_xy``.
        putter_template: Putter head whose ``speed_mps`` is replaced by
            the solved value; every other field (mass, MOI, loft, COR,
            hosel) is used as given.
        past_hole_m: Desired rest-point overrun past the hole along the
            aim line when the putt is not holed [m]; must be > 0.

    Returns:
        The solved :class:`AimSolution`.

    Raises:
        ValueError: If an input is non-finite, ``past_hole_m <= 0``, or
            ``start_xy == hole_xy``.
    """
    require_finite(start_xy[0], "start_xy[0]")
    require_finite(start_xy[1], "start_xy[1]")
    require_finite(hole_xy[0], "hole_xy[0]")
    require_finite(hole_xy[1], "hole_xy[1]")
    require_finite(past_hole_m, "past_hole_m")
    require(past_hole_m > 0.0, "past_hole_m must be > 0", past_hole_m)
    sx, sy = start_xy
    hx, hy = hole_xy
    distance = math.hypot(hx - sx, hy - sy)
    require(distance > 1e-6, "start_xy and hole_xy must differ", distance)
    heading0 = math.atan2(hy - sy, hx - sx)
    perp = (-math.sin(heading0), math.cos(heading0))
    target_distance = distance + past_hole_m

    def lateral_miss(heading: float) -> tuple[float, float, float, float, bool]:
        rest_x, rest_y, speed, holed = _rest_for_heading(
            surface, putter_template, start_xy, hole_xy, heading, target_distance
        )
        miss = 0.0 if holed else (rest_x - hx) * perp[0] + (rest_y - hy) * perp[1]
        return miss, rest_x, rest_y, speed, holed

    heading_prev = heading0
    miss_prev, rest_x, rest_y, speed, holed = lateral_miss(heading_prev)
    heading_curr = heading0 - miss_prev / distance
    miss_curr, rest_x, rest_y, speed, holed = lateral_miss(heading_curr)
    converged = abs(miss_curr) <= _HEADING_MISS_TOL_M
    for _ in range(_MAX_HEADING_ITERATIONS):
        if converged:
            break
        denom = miss_curr - miss_prev
        if abs(denom) < 1e-12:
            break
        heading_next = heading_curr - miss_curr * (heading_curr - heading_prev) / denom
        heading_prev, miss_prev = heading_curr, miss_curr
        heading_curr = heading_next
        miss_curr, rest_x, rest_y, speed, holed = lateral_miss(heading_curr)
        converged = abs(miss_curr) <= _HEADING_MISS_TOL_M

    aim_point = (
        sx + distance * math.cos(heading_curr),
        sy + distance * math.sin(heading_curr),
    )
    break_m = (aim_point[0] - hx) * perp[0] + (aim_point[1] - hy) * perp[1]
    solution = AimSolution(
        putter_speed_mps=speed,
        heading_rad=heading_curr,
        aim_point_xy=aim_point,
        break_m=break_m,
        rest_xy=(hx, hy) if holed else (rest_x, rest_y),
        converged=bool(converged),
    )
    ensure(
        _SPEED_MIN_MPS <= solution.putter_speed_mps <= _SPEED_MAX_MPS,
        "solved speed must stay within the putter-head search bracket",
        solution.putter_speed_mps,
    )
    return solution
