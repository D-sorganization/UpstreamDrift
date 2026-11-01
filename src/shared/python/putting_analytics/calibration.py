"""Fit a putting skill profile to a published make-rate table (#14177).

The fit keeps the *shape* of a base profile (how its spreads compare with one
another) and scales two groups of channels:

* line: face, path, toe strike and slope-aspect read, which set start line and
  so dominate short putts;
* pace: speed, grade read and stimp read, which dominate long putts.

Each band is predicted as the mean make rate at the centres of its thirds.
The two log scales are found by Nelder-Mead on the RMS error in percentage
points; every evaluation reuses one seed, so the objective is deterministic.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize

from src.shared.python.contracts import require
from src.shared.python.putting_analytics.make_rate_reference import (
    MakeRatePoint,
    MakeRateTable,
)
from src.shared.python.putting_analytics.sg_curve import simulated_make_rates
from src.shared.python.putting_analytics.skill import (
    PuttingSkillProfile,
    ReadDispersion,
    StrokeDispersion,
)

LINE_CHANNELS: tuple[str, ...] = ("face", "path", "strike", "read_aspect")
PACE_CHANNELS: tuple[str, ...] = ("speed", "read_grade", "read_stimp")
_BAND_FRACTIONS = (1 / 6, 1 / 2, 5 / 6)  # centres of a band's thirds
# Start at the base profile and step about 50 % in each log scale: far enough
# that the common-random-number objective does not look flat.
_INITIAL_SIMPLEX = np.array([[0.0, 0.0], [0.4, 0.0], [0.0, 0.4]])

MakeRateFn = Callable[[PuttingSkillProfile, Sequence[float]], Sequence[float]]


@dataclass(frozen=True)
class FitSettings:
    """Monte Carlo size, seed and scale search limits for :func:`fit_profile`."""

    n: int = 1000
    seed: int = 0
    scale_bounds: tuple[float, float] = (0.2, 5.0)
    max_evaluations: int = 80

    def __post_init__(self) -> None:
        low, high = self.scale_bounds
        require(0.0 < low < high, "scale_bounds must be positive and ordered")
        require(self.max_evaluations >= 4, "max_evaluations must be at least 4")


@dataclass(frozen=True)
class TableComparison:
    """Predicted against published make % at every point of one table."""

    table: MakeRateTable
    predicted_pct: tuple[float, ...]

    @property
    def residuals_pct(self) -> tuple[float, ...]:
        """Predicted minus published, in percentage points."""
        return tuple(
            p - point.pct
            for p, point in zip(self.predicted_pct, self.table.points, strict=True)
        )

    @property
    def rms_error_pct(self) -> float:
        """Root-mean-square residual, in percentage points."""
        return math.sqrt(float(np.mean(np.square(self.residuals_pct))))


@dataclass(frozen=True)
class CalibrationFit:
    """The fitted profile, its two scales and how it compares with the target."""

    profile: PuttingSkillProfile
    line_scale: float
    pace_scale: float
    comparison: TableComparison


def scaled_profile(
    base: PuttingSkillProfile, line_scale: float, pace_scale: float, name: str = ""
) -> PuttingSkillProfile:
    """``base`` with its line and pace spreads multiplied by the two scales."""
    scaled = base.scaled(line_scale, LINE_CHANNELS).scaled(pace_scale, PACE_CHANNELS)
    return (
        scaled
        if not name
        else PuttingSkillProfile(name, scaled.stroke, scaled.read, scaled.face_weight)
    )


def evaluation_distances(point: MakeRatePoint) -> tuple[float, ...]:
    """Distances whose mean make rate stands for ``point``."""
    if not point.is_band:
        return (point.low_ft,)
    width = point.high_ft - point.low_ft
    return tuple(point.low_ft + f * width for f in _BAND_FRACTIONS)


def _seeded_rates(settings: FitSettings) -> MakeRateFn:
    def rates(profile: PuttingSkillProfile, grid: Sequence[float]) -> Sequence[float]:
        return simulated_make_rates(profile, grid, settings.n, settings.seed)

    return rates


def compare(
    profile: PuttingSkillProfile, table: MakeRateTable, rates: MakeRateFn
) -> TableComparison:
    """Predict ``table``'s points for ``profile`` with ``rates``."""
    per_point = [evaluation_distances(point) for point in table.points]
    grid = sorted({d for ds in per_point for d in ds})
    require(len(grid) >= 2, "a table needs at least two evaluation distances")
    by_distance = dict(zip(grid, rates(profile, grid), strict=True))
    predicted = tuple(
        100.0 * float(np.mean([by_distance[d] for d in ds])) for ds in per_point
    )
    return TableComparison(table, predicted)


def compare_seeded(
    profile: PuttingSkillProfile, table: MakeRateTable, settings: FitSettings
) -> TableComparison:
    """:func:`compare` with the simulated putting curve at ``settings``' size."""
    return compare(profile, table, _seeded_rates(settings))


def fit_profile(
    base: PuttingSkillProfile,
    target: MakeRateTable,
    settings: FitSettings = FitSettings(),
    rates: MakeRateFn | None = None,
) -> CalibrationFit:
    """Scale ``base``'s line and pace spreads to match ``target`` (RMS, points)."""
    rates = rates or _seeded_rates(settings)
    low, high = (math.log(b) for b in settings.scale_bounds)
    bounds: list[tuple[float, float]] = [(low, high)] * 2

    def error(log_scales: npt.NDArray[np.float64]) -> float:
        line, pace = (float(v) for v in np.exp(log_scales))
        return compare(scaled_profile(base, line, pace), target, rates).rms_error_pct

    result = minimize(
        error,
        x0=np.zeros(2),
        method="Nelder-Mead",
        bounds=bounds,
        options={
            "maxfev": settings.max_evaluations,
            "xatol": 0.01,
            "fatol": 0.05,
            "initial_simplex": _INITIAL_SIMPLEX,
        },
    )
    line, pace = (float(v) for v in np.exp(result.x))
    profile = scaled_profile(base, line, pace, name=f"fit:{target.source_id}")
    return CalibrationFit(profile, line, pace, compare(profile, target, rates))


# Base shape for the tour fit: the ratios between spreads are a modelling
# choice; only the two scales below come from data.
TOUR_SHAPE = PuttingSkillProfile(
    "tour-shape",
    StrokeDispersion(
        face_sd_deg=1.0, path_sd_deg=2.0, speed_cv=0.07, strike_toe_sd_mm=3.0
    ),
    ReadDispersion(grade_rel_sd=0.15, aspect_sd_deg=10.0, stimp_rel_sd=0.08),
)
# fit_profile(TOUR_SHAPE, reference_table("golfmagic_2022", "pga_tour"),
# FitSettings(n=2000)); RMS error 0.60 percentage points.
TOUR_LINE_SCALE = 1.7234
TOUR_PACE_SCALE = 0.8375
TOUR_AVERAGE = scaled_profile(
    TOUR_SHAPE, TOUR_LINE_SCALE, TOUR_PACE_SCALE, "tour-average"
)
