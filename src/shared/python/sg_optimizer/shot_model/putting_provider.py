"""Selectable putting providers for the strokes-gained optimizer (#14177).

A provider answers the two putting questions the optimizer asks: the make
probability from a distance, and the leave-distance distribution after a miss.

* :class:`LogisticPuttingProvider` -- the existing logistic model in
  :mod:`.putting`. It stays the default until the putting validation issue
  closes.
* :class:`CurvePuttingProvider` -- reads both answers from a
  :class:`SimulatedPuttingCurve`, a per-distance table produced by the
  skill-conditioned putting simulation
  (:func:`src.shared.python.putting_analytics.sg_curve.simulate_putting_curve`).

Both apply the same condition layer: the green-speed modifiers of
:class:`.GreenModel` and the player's :class:`.PuttingSkill` adjustments
(make multipliers, three-putt avoidance). For the curve provider the curve is
the player, so leave the :class:`.PuttingSkill` at its neutral default unless
you want an extra adjustment on top of the simulated player.

This module stays free of ball physics: the curve is plain data.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

from src.shared.python.contracts import require
from src.shared.python.sg_optimizer.shot_model.player_profile import PuttingSkill
from src.shared.python.sg_optimizer.shot_model.putting import (
    LeaveDistribution,
    leave_distance_distribution,
    make_probability,
)

if TYPE_CHECKING:  # pragma: no cover - import cycle guard
    from src.shared.python.sg_optimizer.course.conditions import GreenModel

PROVIDER_NAMES: tuple[str, ...] = ("logistic", "simulation")
REFERENCE_STIMP = 10.0  # GreenModel modifiers are neutral at stimp 10


class PuttingProvider(Protocol):
    """Make probability and miss-leave distribution from a distance."""

    def make_probability(
        self, distance_ft: float, profile: PuttingSkill, greens: GreenModel
    ) -> float:
        """Probability of holing a putt of ``distance_ft``, in [0, 1]."""

    def leave_distribution(
        self, distance_ft: float, profile: PuttingSkill, greens: GreenModel
    ) -> LeaveDistribution:
        """Leave-distance distribution given a miss."""


class LogisticPuttingProvider:
    """The logistic make model and its leave law (:mod:`.putting`)."""

    def make_probability(
        self, distance_ft: float, profile: PuttingSkill, greens: GreenModel
    ) -> float:
        """Logistic baseline x player multiplier x green-speed modifier."""
        return make_probability(distance_ft, profile, greens)

    def leave_distribution(
        self, distance_ft: float, profile: PuttingSkill, greens: GreenModel
    ) -> LeaveDistribution:
        """Log-normal leave widened by stimp, tightened by 3-putt avoidance."""
        return leave_distance_distribution(distance_ft, profile, greens)


def _strictly_increasing(values: Sequence[float]) -> bool:
    return all(b > a for a, b in zip(values, values[1:], strict=False))


@dataclass(frozen=True)
class SimulatedPuttingCurve:
    """Per-distance make rate and log-normal miss leave, at stimp 10.

    ``leave_scale_ft`` is the log-normal scale ``exp(mu)`` (the median leave),
    which is how :class:`.LeaveDistribution` reads ``mean_ft``.
    ``samples_per_distance`` records the Monte Carlo size behind each row.
    """

    distances_ft: tuple[float, ...]
    make_rate: tuple[float, ...]
    leave_scale_ft: tuple[float, ...]
    leave_sigma_log: tuple[float, ...]
    samples_per_distance: int = 0

    def __post_init__(self) -> None:
        n = len(self.distances_ft)
        require(n >= 2, "a curve needs at least two distances", n)
        for name in ("make_rate", "leave_scale_ft", "leave_sigma_log"):
            require(len(getattr(self, name)) == n, f"{name} length must match", n)
        require(self.distances_ft[0] > 0.0, "distances must be positive")
        require(_strictly_increasing(self.distances_ft), "distances must increase")
        require(all(0.0 <= p <= 1.0 for p in self.make_rate), "make_rate in [0, 1]")
        require(all(s > 0.0 for s in self.leave_scale_ft), "leave_scale_ft > 0")
        require(all(s >= 0.0 for s in self.leave_sigma_log), "leave_sigma_log >= 0")

    def _at(self, values: Sequence[float], distance_ft: float) -> float:
        """Linear in log distance; flat beyond the tabulated range."""
        require(distance_ft > 0, "distance_ft must be > 0", distance_ft)
        log_d = np.log(np.asarray(self.distances_ft))
        return float(np.interp(math.log(distance_ft), log_d, np.asarray(values)))

    def make_rate_at(self, distance_ft: float) -> float:
        """Simulated make rate at ``distance_ft``."""
        return self._at(self.make_rate, distance_ft)

    def leave_at(self, distance_ft: float) -> tuple[float, float]:
        """Simulated (leave scale ft, log sigma) at ``distance_ft``."""
        scale = self._at(self.leave_scale_ft, distance_ft)
        return scale, self._at(self.leave_sigma_log, distance_ft)


@dataclass(frozen=True)
class CurvePuttingProvider:
    """Make probability and leave read from a simulated putting curve."""

    curve: SimulatedPuttingCurve

    def make_probability(
        self, distance_ft: float, profile: PuttingSkill, greens: GreenModel
    ) -> float:
        """Simulated make rate x player multiplier x green-speed modifier."""
        p = (
            self.curve.make_rate_at(distance_ft)
            * profile.multiplier_at(distance_ft)
            * greens.make_pct_modifier(distance_ft)
        )
        return min(max(p, 0.0), 1.0)

    def leave_distribution(
        self, distance_ft: float, profile: PuttingSkill, greens: GreenModel
    ) -> LeaveDistribution:
        """Simulated leave, with the same stimp and 3-putt layer as the logistic."""
        scale, sigma = self.curve.leave_at(distance_ft)
        return LeaveDistribution(
            mean_ft=scale / max(profile.three_putt_avoidance, 1e-6),
            sigma_log=sigma * greens.leave_distribution_modifier(distance_ft),
        )


def putting_provider(
    name: str = "logistic", curve: SimulatedPuttingCurve | None = None
) -> PuttingProvider:
    """Select a provider by name; ``"simulation"`` needs a ``curve``."""
    require(name in PROVIDER_NAMES, f"provider must be one of {PROVIDER_NAMES}", name)
    if name == "logistic":
        require(curve is None, "the logistic provider takes no curve")
        return LogisticPuttingProvider()
    if curve is None:
        raise ValueError("the simulation provider needs a curve")
    return CurvePuttingProvider(curve)
