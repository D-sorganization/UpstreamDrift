"""Putting model — logistic make-% and leave-distance distribution.

Strictly separate from the through-the-bag swing model (spec pitfall #6).
Accepts a ``GreenModel`` so that green-speed conditions modulate both
make-% and leave-distance spread.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.shared.python.contracts import require
from src.shared.python.sg_optimizer.shot_model.player_profile import PuttingSkill

if TYPE_CHECKING:  # pragma: no cover - import cycle guard
    from src.shared.python.sg_optimizer.course.conditions import GreenModel


# Logistic baseline fitted to Broadie's table A.5 (tour make-% by distance).
# Form:  P_base(d) = sigmoid(alpha + beta * log(d_ft))
# Values give ≈99 % at 3 ft, 50 % at ~8 ft, 7 % at 30 ft.
_ALPHA = 6.0
_BETA = -2.85


def _baseline_make(distance_ft: float) -> float:
    require(distance_ft > 0, "distance_ft must be > 0", distance_ft)
    z = _ALPHA + _BETA * math.log(distance_ft)
    return 1.0 / (1.0 + math.exp(-z))


def make_probability(
    distance_ft: float,
    profile: PuttingSkill,
    greens: GreenModel,
) -> float:
    """Logistic baseline × player multiplier × green-speed modifier.

    Clamped to (0, 1).
    """
    base = _baseline_make(distance_ft)
    p = (
        base
        * profile.multiplier_at(distance_ft)
        * greens.make_pct_modifier(distance_ft)
    )
    return min(max(p, 0.0), 1.0)


@dataclass(frozen=True)
class LeaveDistribution:
    """Conditional leave-distance distribution given a missed putt.

    Log-normal with mean ≈ (1 - make_p) * f(distance), spread scaled by
    green speed and (1 / three_putt_avoidance).
    """

    mean_ft: float
    sigma_log: float

    def expected_three_putt_probability(self, near_hole_ft: float = 3.0) -> float:
        """P(leave distance > ``near_hole_ft``) — proxy for 3-putt rate."""
        require(near_hole_ft > 0, "near_hole_ft must be > 0")
        if self.mean_ft <= 0:
            return 0.0
        mu = math.log(max(self.mean_ft, 1e-3))
        # P(X > x) for log-normal = 1 - Phi((log x - mu)/sigma).
        z = (math.log(near_hole_ft) - mu) / max(self.sigma_log, 1e-6)
        return 1.0 - _phi(z)


def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def leave_distance_distribution(
    distance_ft: float,
    profile: PuttingSkill,
    greens: GreenModel,
) -> LeaveDistribution:
    """Conditional-on-miss leave distribution.

    Spec §1.3 / §1.4.3: faster greens widen the leave distribution from lag
    putts; stronger 3-putt avoidance tightens it.
    """
    require(distance_ft > 0, "distance_ft must be > 0", distance_ft)
    # Mean miss distance grows with original distance and shrinks with skill.
    # Tour proximity from 30 ft is ~3 ft median; we calibrate around that.
    base_mean = 1.0 + 0.06 * distance_ft  # feet  (30 ft lag → ~2.8 ft median)
    mean_ft = base_mean / max(profile.three_putt_avoidance, 1e-6)
    # Log-σ widens with stimp; ~0.4 baseline at stimp 10.
    sigma_log = 0.40 * greens.leave_distribution_modifier(distance_ft)
    return LeaveDistribution(mean_ft=mean_ft, sigma_log=sigma_log)
