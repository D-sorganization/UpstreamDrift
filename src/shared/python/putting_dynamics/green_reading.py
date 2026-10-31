"""Green reading by simulation (epic #14141, putting v4 step 7b).

Replaces the legacy engine's cross-slope heuristic and stimp speed
estimate (``putting_green/_surface_analysis.py`` and
``putter_stroke.estimate_required_speed``, both uncited) with the shared
solver itself: a recommended line and pace are whatever
:func:`~.solver.simulate_strike` says rolls the ball to the hole, so a
reading always agrees with the trajectory any other caller would see.

* :func:`read_line` samples elevation and gradient along a straight
  segment (no physics; the profile a UI plots).
* :func:`solve_aim` finds the putter-head speed and heading that would
  stop the ball ``past_hole_m`` beyond the hole if there were no cup
  (the usual definition of a read's pace), then reports whether that
  putt drops with the cup in play. Speed is
  bracketed by doubling and then bisected on the along-line rest
  distance; heading is solved by secant on the lateral miss, starting
  from the small-angle step ``-miss / distance``. Both loops are
  bounded, so the solve is deterministic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from src.shared.python.contracts import ensure, require, require_finite

from .solver import simulate_strike
from .surfaces import SurfaceSpec
from .types import PutterState, PuttResult

__all__ = ["AimSolution", "LineReading", "read_line", "solve_aim"]

#: Putter-head speed search bracket [m/s], inside ``PutterState``'s bounds.
_SPEED_MIN_MPS = 0.05
_SPEED_MAX_MPS = 8.0
_MAX_SPEED_ITERATIONS = 30
_MAX_HEADING_ITERATIONS = 12
#: Convergence tolerances [m]; well inside the few-cm precision a read needs.
_SPEED_DISTANCE_TOL_M = 5e-3
_HEADING_MISS_TOL_M = 5e-3

_Point = tuple[float, float]


def _require_point(point: _Point, name: str) -> None:
    require_finite(point[0], f"{name}[0]")
    require_finite(point[1], f"{name}[1]")


@dataclass(frozen=True)
class LineReading:
    """Elevation and slope sampled along a straight putt line.

    Attributes:
        positions_m: Sampled ``(x, y)`` points, both ends included.
        elevations_m: Surface elevation at each point [m].
        slopes: Surface gradient ``(dh/dx, dh/dy)`` at each point.
        distance_m: Straight-line length of the segment [m].
    """

    positions_m: tuple[_Point, ...]
    elevations_m: tuple[float, ...]
    slopes: tuple[_Point, ...]
    distance_m: float


@dataclass(frozen=True)
class AimSolution:
    """Solved line and pace for one putt.

    Attributes:
        putter_speed_mps: Putter-head speed at impact [m/s].
        heading_rad: Putter forward-axis heading, CCW from +x [rad].
        aim_point_xy: Point at the hole's distance along ``heading_rad``.
        break_m: Signed offset of the aim point from the hole, positive
            to the left of the start-to-hole line [m].
        rest_xy: Where the solved putt stops with the cup in play.
        holed: Whether the solved putt drops.
        converged: Whether the heading solve met its tolerance.
    """

    putter_speed_mps: float
    heading_rad: float
    aim_point_xy: _Point
    break_m: float
    rest_xy: _Point
    holed: bool
    converged: bool


def read_line(
    surface: SurfaceSpec, start_xy: _Point, end_xy: _Point, num_samples: int = 20
) -> LineReading:
    """Sample elevation and slope along ``start`` -> ``end``.

    Raises:
        ValueError: If a point is not finite or ``num_samples < 2``.
    """
    _require_point(start_xy, "start_xy")
    _require_point(end_xy, "end_xy")
    require(num_samples >= 2, "num_samples must be >= 2", num_samples)
    (sx, sy), (ex, ey) = start_xy, end_xy
    steps = [i / (num_samples - 1) for i in range(num_samples)]
    positions = tuple((sx + t * (ex - sx), sy + t * (ey - sy)) for t in steps)
    return LineReading(
        positions_m=positions,
        elevations_m=tuple(surface.height.elevation(x, y) for x, y in positions),
        slopes=tuple(surface.height.gradient(x, y) for x, y in positions),
        distance_m=math.hypot(ex - sx, ey - sy),
    )


@dataclass(frozen=True)
class _AimProblem:
    """Fixed inputs shared by every trial strike of one solve."""

    surface: SurfaceSpec
    putter: PutterState
    start: _Point
    hole: _Point

    def strike(
        self, heading: float, speed: float, with_hole: bool = False
    ) -> PuttResult:
        """One trial strike from the start along ``heading``."""
        (sx, sy), (hx, hy) = self.start, self.hole
        return simulate_strike(
            replace(self.putter, speed_mps=speed),
            self.surface,
            hole_x_m=hx if with_hole else None,
            hole_y_m=hy if with_hole else None,
            start_x_m=sx,
            start_y_m=sy,
            heading_rad=heading,
        )

    def along(self, heading: float, speed: float) -> tuple[_Point, float]:
        """Cup-free rest point and its distance along ``heading``."""
        result = self.strike(heading, speed)
        dx, dy = result.rest_x_m - self.start[0], result.rest_y_m - self.start[1]
        rest = (result.rest_x_m, result.rest_y_m)
        return rest, dx * math.cos(heading) + dy * math.sin(heading)

    def pace(self, heading: float, target_m: float) -> tuple[_Point, float]:
        """Cup-free rest point and the speed whose roll reaches ``target_m``."""
        lo = hi = _SPEED_MIN_MPS
        rest, along = self.along(heading, hi)
        while along < target_m and hi < _SPEED_MAX_MPS:
            lo, hi = hi, min(2.0 * hi, _SPEED_MAX_MPS)
            rest, along = self.along(heading, hi)
        speed = hi
        for _ in range(_MAX_SPEED_ITERATIONS):
            if abs(along - target_m) <= _SPEED_DISTANCE_TOL_M or hi <= lo:
                break
            speed = 0.5 * (lo + hi)
            rest, along = self.along(heading, speed)
            lo, hi = (speed, hi) if along < target_m else (lo, speed)
        return rest, speed


def solve_aim(
    surface: SurfaceSpec,
    start_xy: _Point,
    hole_xy: _Point,
    putter_template: PutterState,
    past_hole_m: float = 0.3,
) -> AimSolution:
    """Solve the putter-head speed and heading for one putt.

    Args:
        surface: Green description.
        start_xy: Ball start [m].
        hole_xy: Hole centre [m]; must differ from ``start_xy``.
        putter_template: Head whose ``speed_mps`` is replaced by the
            solved value; mass, MOI, loft, COR and hosel are kept.
        past_hole_m: Target overrun beyond the hole when not holed [m].

    Returns:
        The :class:`AimSolution`.

    Raises:
        ValueError: On non-finite input, ``past_hole_m <= 0`` or a start
            on the hole.
    """
    _require_point(start_xy, "start_xy")
    _require_point(hole_xy, "hole_xy")
    require_finite(past_hole_m, "past_hole_m")
    require(past_hole_m > 0.0, "past_hole_m must be > 0", past_hole_m)
    (sx, sy), (hx, hy) = start_xy, hole_xy
    distance = math.hypot(hx - sx, hy - sy)
    require(distance > 1e-6, "start_xy and hole_xy must differ", distance)
    problem = _AimProblem(surface, putter_template, start_xy, hole_xy)
    direct = math.atan2(hy - sy, hx - sx)
    left = (-math.sin(direct), math.cos(direct))

    def miss(heading: float) -> tuple[float, float]:
        rest, speed = problem.pace(heading, distance + past_hole_m)
        return (rest[0] - hx) * left[0] + (rest[1] - hy) * left[1], speed

    prev_heading = direct
    prev_miss, speed = miss(prev_heading)
    heading = direct - prev_miss / distance
    cur_miss, speed = miss(heading)
    for _ in range(_MAX_HEADING_ITERATIONS):
        if abs(cur_miss) <= _HEADING_MISS_TOL_M or cur_miss == prev_miss:
            break
        step = cur_miss * (heading - prev_heading) / (cur_miss - prev_miss)
        prev_heading, prev_miss = heading, cur_miss
        heading -= step
        cur_miss, speed = miss(heading)

    final = problem.strike(heading, speed, with_hole=True)
    aim = (sx + distance * math.cos(heading), sy + distance * math.sin(heading))
    solution = AimSolution(
        putter_speed_mps=speed,
        heading_rad=heading,
        aim_point_xy=aim,
        break_m=(aim[0] - hx) * left[0] + (aim[1] - hy) * left[1],
        rest_xy=(final.rest_x_m, final.rest_y_m),
        holed=final.holed,
        converged=abs(cur_miss) <= _HEADING_MISS_TOL_M,
    )
    ensure(
        _SPEED_MIN_MPS <= solution.putter_speed_mps <= _SPEED_MAX_MPS
        and math.isfinite(solution.heading_rad),
        "solved speed and heading must be finite and inside the search bracket",
        solution,
    )
    return solution
