"""JSON service for player-specific Bellman strokes gained (#14467, epic #14452).

:func:`sg_study` answers "how many strokes did that putt gain against my own
game on this green?" for the putting surfaces (desktop simulator, web page,
studio):

* **Curve.** The player's own expected putts to hole out, E(d), on the
  caller's planar green, solved by policy iteration
  (:func:`.optimize.optimal_expected_putts`): at every grid distance the
  player picks the intended pace that minimises E, and every pace is played
  with the shared skill roll model (:func:`.sg_curve.skill_model`: the line
  is planned on the green as read and rolled on the true green with the
  player's stroke and read dispersion from
  :func:`.service_variance.profile_from_args`).
* **Strokes gained.** For the caller's putt (``holed``, ``leave_ft``),
  :func:`.optimize.strokes_gained` against that curve:
  ``E(start) - 1 - E(leave)``, ``E(start) - 1`` when holed. Zero is the
  player's average from that spot; positive beat it.

No reference tour curve ships with the tool (#14176), so only the
player-relative value is reported. Arguments are untrusted tool input,
checked with :func:`.service._check`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.shared.python.putting_analytics import optimize
from src.shared.python.putting_analytics.aggregate import DEFAULT_GRID_FT
from src.shared.python.putting_analytics.expected_putts import (
    PuttAction,
    PuttOutcomes,
)
from src.shared.python.putting_analytics.roll_model import FT_M
from src.shared.python.putting_analytics.sampling import sample_execution, sample_read
from src.shared.python.putting_analytics.service import (
    MAX_DISTANCE_FT,
    PACE_RANGE_FT,
    _check,
    _number,
)
from src.shared.python.putting_analytics.service_curve import (
    MAX_CURVE_ROLLOUTS,
    _seed,
    distances_from_args,
)
from src.shared.python.putting_analytics.service_dispersion import (
    _green_dict,
    _whole,
    green_from_args,
)
from src.shared.python.putting_analytics.service_variance import profile_from_args
from src.shared.python.putting_analytics.sg_curve import skill_model
from src.shared.python.putting_analytics.skill import PuttingSkillProfile
from src.shared.python.putting_analytics.skill_roll_model import PlaneGreen

DEFAULT_SG_PACES_FT = (0.5, 1.5, 3.0)
DEFAULT_SG_N = 200
MAX_SG_N = 2000
MAX_SG_PACES = 8


@dataclass(frozen=True)
class SkillPuttModel:
    """:class:`.PuttOutcomeModel` of one player on one green, paced by the action.

    The player plans the line (on the green as read), so an action carries
    only the intended pace; its aim offset must be zero.
    """

    profile: PuttingSkillProfile
    green: PlaneGreen

    def simulate(
        self, distance_ft: float, action: PuttAction, n: int, seed: int
    ) -> PuttOutcomes:
        """``n`` seeded putts from ``distance_ft`` at ``action``'s pace."""
        if action.aim_offset_deg != 0.0:
            raise ValueError("the player plans the line: aim_offset_deg must be 0")
        model = dataclasses.replace(
            skill_model(self.profile, self.green, distance_ft),
            pace_ft_past=action.pace_ft_past,
        )
        out = model(
            sample_execution(self.profile.stroke, n, seed),
            sample_read(self.profile.read, n, seed),
        )
        return PuttOutcomes(out.holed, out.leave_m / FT_M)


def _paces(args: Mapping[str, Any]) -> tuple[float, ...]:
    raw = args.get("paces_ft", list(DEFAULT_SG_PACES_FT))
    lo, hi = PACE_RANGE_FT
    _check(
        isinstance(raw, list) and 1 <= len(raw) <= MAX_SG_PACES,
        f"'paces_ft' needs 1 to {MAX_SG_PACES} values",
    )
    paces = tuple(_number({"pace": p}, "pace") for p in raw)
    _check(all(lo <= p <= hi for p in paces), f"'paces_ft' must be in [{lo}, {hi}]")
    return paces


def _putt(args: Mapping[str, Any]) -> PuttOutcomes | None:
    """The caller's putt, or ``None`` when only the curve is asked for."""
    if "holed" not in args and "leave_ft" not in args:
        return None
    holed = args.get("holed")
    _check(isinstance(holed, bool), "'holed' must be true or false")
    leave = 0.0 if holed else _number(args, "leave_ft")
    _check(
        0.0 <= leave <= MAX_DISTANCE_FT, f"'leave_ft' must be in [0, {MAX_DISTANCE_FT}]"
    )
    return PuttOutcomes(np.array([holed]), np.array([leave]))


def sg_study(args: Mapping[str, Any]) -> dict[str, Any]:
    """The player's E(d) on the caller's green and the putt's strokes gained.

    Returns ``curve`` (``distances_ft``, ``expected_putts``, the optimal
    ``pace_ft_past`` per distance), ``expected_putts`` from ``distance_ft``,
    ``sg_player`` (``None`` without a putt), the green, ``n`` and the paces.
    """
    profile = profile_from_args(args)
    green = green_from_args(args)
    grid = distances_from_args({"distances_ft": list(DEFAULT_GRID_FT), **args})
    distance = _number(args, "distance_ft", 10.0)
    _check(
        grid[0] <= distance <= grid[-1],
        f"'distance_ft' must lie on the grid [{grid[0]}, {grid[-1]}]",
    )
    paces = _paces(args)
    n = _whole(args, "n", DEFAULT_SG_N, MAX_SG_N)
    work = n * len(grid) * len(paces)
    _check(
        work <= MAX_CURVE_ROLLOUTS, f"work is {work} putts; cap is {MAX_CURVE_ROLLOUTS}"
    )
    putt = _putt(args)
    curve, policy = optimize.optimal_expected_putts(
        SkillPuttModel(profile, green),
        grid,
        [PuttAction(0.0, p) for p in paces],
        n,
        _seed(args),
    )
    sg = None if putt is None else optimize.strokes_gained(distance, putt, curve, curve)
    return {
        "schema_version": 1,
        "distance_ft": distance,
        "green": _green_dict(green),
        "n": n,
        "paces_ft": list(paces),
        "curve": {
            "distances_ft": list(grid),
            "expected_putts": curve.values.tolist(),
            "pace_ft_past": [a.pace_ft_past for a in policy],
        },
        "expected_putts": float(curve(distance)),
        "sg_player": None if sg is None else sg.sg_player,
    }


__all__ = [
    "DEFAULT_SG_N",
    "DEFAULT_SG_PACES_FT",
    "MAX_SG_N",
    "MAX_SG_PACES",
    "SkillPuttModel",
    "sg_study",
]
