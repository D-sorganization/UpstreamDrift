"""Monte Carlo make-probability physics (epic #14141, putting v4 step 9b).

Design of record (``research/step9_plan.md`` row 9b): strokes gained
stays in the existing Bellman/value layer (:mod:`.putting`); this adds
a second, physics-grounded make-% *source* next to the Broadie
logistic, without changing that function's behaviour or tests. Both
functions below estimate *this model's* make-%, not a calibrated
player or tour figure — no "+2.42 SG" claim follows from either.
:class:`StrokeNoise` has no measured calibration yet; see the pending
``broadie2012_shotlink_50pct_putt_distance_ft`` fixture in
``src/config/putting_reference_fixtures.json`` and its xfail test.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src.shared.python.contracts import require, require_finite
from src.shared.python.core.physics_constants import FT_TO_M
from src.shared.python.putting_dynamics import (
    AimSolution,
    PutterState,
    SurfaceSpec,
    simulate_strike,
    solve_aim,
)

__all__ = ["StrokeNoise", "simulated_make_probability", "variance_attribution"]

#: Small on purpose; see `simulated_make_probability` for the std-error cost.
_DEFAULT_N_TRIALS = 32


@dataclass(frozen=True)
class StrokeNoise:
    """Three independent, uncalibrated error channels on a nominal putt.

    ``face_sd_deg``/``read_sd_deg``: std dev of face-angle / green-read
    heading error at impact [deg]. ``speed_cv``: coefficient of
    variation of putter-head speed, dimensionless. All must be >= 0
    and finite.
    """

    face_sd_deg: float
    speed_cv: float
    read_sd_deg: float

    def __post_init__(self) -> None:
        for name in ("face_sd_deg", "speed_cv", "read_sd_deg"):
            value = getattr(self, name)
            require_finite(value, name)
            require(value >= 0.0, f"{name} must be >= 0", value)


def _solve_nominal(
    distance_ft: float, stimp_ft: float, slope_pct: float, slope_aspect_deg: float
) -> tuple[SurfaceSpec, float, AimSolution]:
    """Surface build + nominal-aim solve, shared by both entry points (DRY)."""
    require(distance_ft > 0.0, "distance_ft must be > 0", distance_ft)
    surface = SurfaceSpec.planar_uniform(stimp_ft, slope_pct, slope_aspect_deg)
    distance_m = distance_ft * float(FT_TO_M)
    nominal = solve_aim(
        surface, (0.0, 0.0), (distance_m, 0.0), PutterState.typical(1.0)
    )
    return surface, distance_m, nominal


def _replay(
    nominal: AimSolution,
    surface: SurfaceSpec,
    distance_m: float,
    noise: StrokeNoise,
    n_trials: int,
    seed: int,
) -> float:
    """Seeded Monte Carlo replay of ``nominal`` under ``noise``; fraction holed."""
    rng = np.random.default_rng(seed)
    holed = 0
    for _ in range(n_trials):
        heading_error_deg = rng.normal(0.0, noise.face_sd_deg) + rng.normal(
            0.0, noise.read_sd_deg
        )
        heading = nominal.heading_rad + math.radians(heading_error_deg)
        speed_mult = max(1e-3, 1.0 + rng.normal(0.0, noise.speed_cv))
        result = simulate_strike(
            PutterState.typical(nominal.putter_speed_mps * speed_mult),
            surface,
            hole_x_m=distance_m,
            hole_y_m=0.0,
            heading_rad=heading,
        )
        holed += int(result.holed)
    return holed / n_trials


def simulated_make_probability(
    distance_ft: float,
    noise: StrokeNoise,
    stimp_ft: float = 10.0,
    slope_pct: float = 0.0,
    slope_aspect_deg: float = 90.0,
    n_trials: int = _DEFAULT_N_TRIALS,
    seed: int = 0,
) -> float:
    """Seeded Monte Carlo make probability (fraction holed) on the shared core.

    Solves the nominal aim/pace once, then replays ``n_trials`` seeded
    strikes with the heading perturbed by ``face_sd_deg + read_sd_deg``
    and the speed scaled by ``speed_cv``. Binomial standard error:
    ``sqrt(p * (1 - p) / n_trials)`` (peaks ~0.088 at the default
    ``n_trials=32`` near ``p=0.5``) — raise ``n_trials`` before reading
    small output differences as signal. Raises ``ValueError`` if
    ``distance_ft``/``n_trials`` is not positive, ``noise`` has a
    negative/non-finite channel, or a surface parameter is out of the
    shared core's valid range.
    """
    require(n_trials > 0, "n_trials must be > 0", n_trials)
    surface, distance_m, nominal = _solve_nominal(
        distance_ft, stimp_ft, slope_pct, slope_aspect_deg
    )
    return _replay(nominal, surface, distance_m, noise, n_trials, seed)


def variance_attribution(
    distance_ft: float,
    noise: StrokeNoise,
    stimp_ft: float = 10.0,
    slope_pct: float = 0.0,
    slope_aspect_deg: float = 90.0,
    n_trials: int = _DEFAULT_N_TRIALS,
    seed: int = 0,
) -> dict[str, float]:
    """One-at-a-time make-% loss attribution across the three noise channels.

    Approximate (ignores interactions, so will not generally match the
    all-three-active loss): each channel's share is the make-% drop
    from zero noise with *only that channel* active, normalized to sum
    to 1; all 0.0 when no channel shows a loss. All four replays share
    one seed (common random numbers). Arguments/errors are as
    :func:`simulated_make_probability`; ``noise`` is the full profile.
    """
    require(n_trials > 0, "n_trials must be > 0", n_trials)
    surface, distance_m, nominal = _solve_nominal(
        distance_ft, stimp_ft, slope_pct, slope_aspect_deg
    )
    zero_noise = StrokeNoise(0.0, 0.0, 0.0)
    p_zero = _replay(nominal, surface, distance_m, zero_noise, n_trials, seed)
    channel_noises = {
        "face": StrokeNoise(noise.face_sd_deg, 0.0, 0.0),
        "speed": StrokeNoise(0.0, noise.speed_cv, 0.0),
        "read": StrokeNoise(0.0, 0.0, noise.read_sd_deg),
    }
    losses = {
        name: max(
            p_zero - _replay(nominal, surface, distance_m, cn, n_trials, seed), 0.0
        )
        for name, cn in channel_noises.items()
    }
    total = sum(losses.values())
    if total <= 0.0:
        return dict.fromkeys(losses, 0.0)
    return {name: loss / total for name, loss in losses.items()}
