"""Generate the putting studio's core expected-putts curves (#14650).

The standalone studio (``src/apps/putting_studio``) opens from ``file://`` and
cannot call the Python core, so its strokes gained used a Bellman formula with
a hand-tuned comeback fit. This script runs the shared core's
:func:`service_sg.sg_study` (policy iteration over the player's own E(d)) for
the fitted tour-average player on a side-slope grid and writes each solved
curve to ``js/core_sg.js``, with the lookups the studio calls.

Usage::

    python3 -m scripts.generate_studio_sg   # rewrites the fixture

Each skill-model putt plans its line on the green as read, so a cell costs
about a second per putt; the cells run in parallel and the whole fixture
takes about an hour. A side slope is mirror-symmetric, so one fall direction
covers both signs of the studio's cross slope.
"""

from __future__ import annotations

import dataclasses
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from src.shared.python.putting_analytics import optimize, service_sg
from src.shared.python.putting_analytics.aggregate import DEFAULT_GRID_FT
from src.shared.python.putting_analytics.calibration import TOUR_AVERAGE
from src.shared.python.putting_analytics.expected_putts import (
    ExpectedPuttsCurve,
    PuttAction,
    distance_seed,
)
from src.shared.python.putting_analytics.service_dispersion import green_from_args
from src.shared.python.putting_analytics.sampling import ReadSample
from src.shared.python.putting_analytics.service_variance import profile_from_args
from src.shared.python.putting_analytics.skill_roll_model import nominal_speeds_mps

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "apps"
    / "putting_studio"
    / "js"
    / "core_sg.js"
)
SLOPES_PCT: tuple[float, ...] = (0.0, 2.0, 4.0)
STIMP_FT = 10.0
SAMPLING = {"n": 100, "seed": 1}
# Pace sweep for the "E vs capture speed" panel. It includes the solve's own
# paces, and each distance reuses the solve's seed for that grid point, so the
# sweep's best of those paces is the solved E(d).
SWEEP_DISTANCES_FT: tuple[float, ...] = (3.0, 6.0, 10.0, 15.0, 20.0, 30.0)
SWEEP_PACES_FT: tuple[float, ...] = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0)
_PREFIX = "window.CORE_SG = "
_LOOKUP = """
// Nearest solved curve for the studio's cross slope (%).
function coreSgCurve(crossSlopePct) {
  const slope = Math.abs(crossSlopePct);
  return window.CORE_SG.curves.reduce((best, c) =>
    (Math.abs(c.slope_pct - slope) < Math.abs(best.slope_pct - slope) ? c : best));
}

// E(d): linear between grid distances, flat beyond the ends (numpy.interp).
function coreExpectedPutts(curve, distFt) {
  const xs = curve.distances_ft, ys = curve.expected_putts;
  if (distFt <= xs[0]) return ys[0];
  for (let i = 1; i < xs.length; i++) {
    if (distFt <= xs[i]) {
      return ys[i - 1] + ((distFt - xs[i - 1]) / (xs[i] - xs[i - 1])) * (ys[i] - ys[i - 1]);
    }
  }
  return ys[ys.length - 1];
}

// Core expected strokes per pace at the nearest swept distance.
function coreSweepRow(curve, distFt) {
  const ds = window.CORE_SG.meta.sweep_distances_ft;
  const k = ds.reduce((b, d, i) => (Math.abs(d - distFt) < Math.abs(ds[b] - distFt) ? i : b), 0);
  return { distance_ft: ds[k], paces_ft: window.CORE_SG.meta.sweep_paces_ft, expected: curve.pace_sweep[k], entry_speed_mps: curve.sweep_entry_speed_mps };
}

// Player strokes gained by one putt (optimize.strokes_gained):
// E(start) - 1 - E(leave), or E(start) - 1 when holed.
function coreStrokesGained(curve, distFt, holed, leaveFt) {
  const after = holed ? 0.0 : coreExpectedPutts(curve, leaveFt);
  return coreExpectedPutts(curve, distFt) - 1.0 - after;
}
"""


def cell_args(slope_pct: float, n: int = SAMPLING["n"]) -> dict[str, Any]:
    """``sg_study`` arguments for one side slope, tour-average player."""
    return {
        **dataclasses.asdict(TOUR_AVERAGE.stroke),
        **dataclasses.asdict(TOUR_AVERAGE.read),
        "face_weight": TOUR_AVERAGE.face_weight,
        "grade_pct": slope_pct,
        "aspect_deg": 90.0,
        "stimp": STIMP_FT,
        "distances_ft": list(DEFAULT_GRID_FT),
        "paces_ft": list(service_sg.DEFAULT_SG_PACES_FT),
        "n": n,
        "seed": SAMPLING["seed"],
    }


def cell_curve(report: dict[str, Any], slope_pct: float) -> dict[str, Any]:
    """One fixture curve from an ``sg_study`` report; E rounded to 4 dp."""
    curve = report["curve"]
    return {
        "slope_pct": slope_pct,
        "distances_ft": curve["distances_ft"],
        "expected_putts": [round(float(e), 4) for e in curve["expected_putts"]],
        "pace_ft_past": curve["pace_ft_past"],
    }


def pace_sweep(
    slope_pct: float,
    curve: dict[str, Any],
    n: int,
    distances_ft: tuple[float, ...] = SWEEP_DISTANCES_FT,
) -> list[list[float]]:
    """Expected strokes per sweep pace at each distance, on ``curve``.

    ``optimize.evaluate_actions`` with the solved curve as the comeback price:
    a miss costs 1 + E(leave). Rows follow ``distances_ft`` (grid distances),
    columns ``SWEEP_PACES_FT``; values rounded to 4 dp.
    """
    if not set(distances_ft) <= set(DEFAULT_GRID_FT):
        raise ValueError(f"sweep distances must be grid distances: {distances_ft}")
    args = cell_args(slope_pct, n)
    model = service_sg.SkillPuttModel(profile_from_args(args), green_from_args(args))
    baseline = ExpectedPuttsCurve(
        np.asarray(curve["distances_ft"], float),
        np.asarray(curve["expected_putts"], float),
    )
    actions = [PuttAction(0.0, p) for p in SWEEP_PACES_FT]
    rows = []
    for distance in distances_ft:
        seed = distance_seed(SAMPLING["seed"], DEFAULT_GRID_FT.index(distance))
        evals = optimize.evaluate_actions(model, distance, actions, baseline, n, seed)
        rows.append([round(e.expected_strokes, 4) for e in evals])
    return rows


def entry_speeds(slope_pct: float) -> list[float]:
    """Entry speed (m/s) at the hole for each sweep pace, 4 dp.

    The core roll law (``skill_roll_model.nominal_speeds_mps``) at zero
    distance with a perfect read: the speed that rolls each pace past the hole
    on the slope's green.
    """
    green = green_from_args(cell_args(slope_pct))
    one = np.ones(1)
    perfect = ReadSample(one, np.zeros(1), one)
    return [
        round(float(nominal_speeds_mps(green, perfect, 0.0, pace)[0]), 4)
        for pace in SWEEP_PACES_FT
    ]


def _solve(slope_pct: float) -> dict[str, Any]:
    curve = cell_curve(service_sg.sg_study(cell_args(slope_pct)), slope_pct)
    return {
        **curve,
        "pace_sweep": pace_sweep(slope_pct, curve, SAMPLING["n"]),
        "sweep_entry_speed_mps": entry_speeds(slope_pct),
    }


def build() -> dict[str, Any]:
    """Run the core for every slope (slow, in parallel) and return the data."""
    with ProcessPoolExecutor(max_workers=len(SLOPES_PCT)) as pool:
        curves = list(pool.map(_solve, SLOPES_PCT))
    meta = {
        "schema_version": 1,
        "source": "service_sg.sg_study (#14650)",
        "profile": TOUR_AVERAGE.name,
        "stimp_ft": STIMP_FT,
        "paces_ft": list(service_sg.DEFAULT_SG_PACES_FT),
        "sampling": SAMPLING,
        "sweep_distances_ft": list(SWEEP_DISTANCES_FT),
        "sweep_paces_ft": list(SWEEP_PACES_FT),
    }
    return {"meta": meta, "curves": curves}


def render(data: dict[str, Any]) -> str:
    """The fixture file text for ``data``: one line per curve."""
    curves = ",\n".join(f"  {json.dumps(c, sort_keys=True)}" for c in data["curves"])
    meta = json.dumps(data["meta"], sort_keys=True)
    body = f'{{\n "curves": [\n{curves}\n ],\n "meta": {meta}\n}}'
    return (
        "// Generated by scripts/generate_studio_sg.py -- do not edit.\n"
        f"{_PREFIX}{body};\n{_LOOKUP}"
    )


def parse(text: str) -> dict[str, Any]:
    """Inverse of :func:`render`: the data object in a fixture file."""
    start = text.index(_PREFIX) + len(_PREFIX)
    end = text.index(";\n", start)
    data: dict[str, Any] = json.loads(text[start:end])
    return data


def main() -> None:
    FIXTURE.write_text(render(build()), encoding="utf-8")


if __name__ == "__main__":
    main()
