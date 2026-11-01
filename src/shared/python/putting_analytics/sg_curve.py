"""Build the strokes-gained optimizer's putting curve from simulation (#14177).

:func:`simulate_putting_curve` plays a :class:`.PuttingSkillProfile` at each
distance of a grid with :class:`.SkillRollModel` (lines planned on the green as
read, rolled on the true green) and tabulates the make rate and a log-normal
fit to the leave after a miss. The result is a
:class:`.SimulatedPuttingCurve`, which
:class:`src.shared.python.sg_optimizer.shot_model.putting_provider.CurvePuttingProvider`
serves to the optimizer.

The curve is built at stimp 10 because the optimizer's green model applies its
green-speed modifiers relative to stimp 10. The same execution and read draws
are used at every distance (common random numbers), so the curve is smooth in
distance at modest sample sizes.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.shared.python.contracts import require
from src.shared.python.putting_analytics.expected_putts import (
    ExpectedPuttsCurve,
    validated_grid,
)
from src.shared.python.putting_analytics.roll_model import FT_M
from src.shared.python.putting_analytics.sampling import sample_execution, sample_read
from src.shared.python.putting_analytics.skill import PuttingSkillProfile
from src.shared.python.putting_analytics.skill_roll_model import (
    PlaneGreen,
    ReadLattice,
    SkillRollModel,
)
from src.shared.python.sg_optimizer.shot_model.putting_provider import (
    REFERENCE_STIMP,
    SimulatedPuttingCurve,
)

DEFAULT_CURVE_GRID_FT = (2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0)
MIN_MISSES = 20  # fewer misses than this give no usable leave fit at a distance
_MIN_LEAVE_FT = 0.01  # a lip-out can stop a hair from the hole; keeps log finite

# Strokes are not part of the curve, but the roll model prices a miss with a
# baseline; any valid curve will do.
_PRICING_BASELINE = ExpectedPuttsCurve(
    np.array([0.5, 3.0, 10.0, 30.0, 100.0]), np.array([1.0, 1.1, 1.5, 1.9, 2.5])
)


def _leave_fit(leave_ft: np.ndarray) -> tuple[float, float] | None:
    """(exp(mu), sigma) of a log-normal fitted to miss leaves, if enough misses."""
    if leave_ft.size < MIN_MISSES:
        return None
    logs = np.log(np.maximum(leave_ft, _MIN_LEAVE_FT))
    return float(np.exp(logs.mean())), float(logs.std(ddof=1))


def _fill(grid: np.ndarray, fits: list[tuple[float, float] | None]) -> np.ndarray:
    """Interpolate missing fits (in log distance) from the distances that have one."""
    have = [i for i, fit in enumerate(fits) if fit is not None]
    if not have:
        raise ValueError(f"no distance had {MIN_MISSES} misses; raise n")
    known = np.array([fits[i] for i in have])
    log_d = np.log(grid)
    columns = [np.interp(log_d, log_d[have], known[:, j]) for j in range(2)]
    return np.stack(columns, axis=1)


def simulate_putting_curve(
    profile: PuttingSkillProfile,
    distances_ft: Sequence[float] = DEFAULT_CURVE_GRID_FT,
    n: int = 2000,
    seed: int = 0,
) -> SimulatedPuttingCurve:
    """Make rate and miss-leave fit by distance for ``profile`` on a flat green.

    ``n`` putts are rolled at each distance. Distances where fewer than
    :data:`MIN_MISSES` putts miss take their leave fit by interpolation from the
    neighbouring distances.
    """
    require(n >= MIN_MISSES, f"n must be at least {MIN_MISSES}", n)
    grid = validated_grid(distances_ft)
    execution = sample_execution(profile.stroke, n, seed)
    read = sample_read(profile.read, n, seed)
    lattice = ReadLattice.from_dispersion(profile.read)
    green = PlaneGreen(0.0, 0.0, REFERENCE_STIMP)
    make, fits = [], []
    for distance in grid:
        model = SkillRollModel(
            green,
            float(distance),
            _PRICING_BASELINE,
            lattice,
            face_weight=profile.face_weight,
        )
        out = model(execution, read)
        make.append(float(out.holed.mean()))
        fits.append(_leave_fit(out.leave_m[~out.holed] / FT_M))
    leave = _fill(grid, fits)
    return SimulatedPuttingCurve(
        tuple(float(d) for d in grid),
        tuple(make),
        tuple(float(v) for v in leave[:, 0]),
        tuple(float(v) for v in leave[:, 1]),
        samples_per_distance=n,
    )
