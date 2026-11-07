"""Metric access for policy studies: the #14421 registry seam (NP-01, #14476).

The evaluator asks a :class:`MetricRegistry` for every metric by name.
:class:`StubMetricRegistry` stands in until #14421 (GM-1a #14422) lands.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar, Protocol

import numpy as np
from numpy.typing import NDArray

from src.learning.policy_experiments.spec import ExperimentSpecError

Array = NDArray[np.float64]

METRIC_NAMES = ("tracking_rmse", "clubhead_speed_error", "constraint_violation_max")


@dataclass(frozen=True)
class SwingTrial:
    """A reference swing and a policy rollout on the same time grid.

    ``reference_q``/``policy_q`` are ``(n_samples, n_coordinates)`` in rad,
    speeds are m/s, and ``constraint_residual`` holds one loop-closure
    residual per sample. Preconditions: shapes agree and values are finite.
    """

    time_s: Array
    reference_q: Array
    policy_q: Array
    reference_clubhead_speed: float
    policy_clubhead_speed: float
    constraint_residual: Array

    def __post_init__(self) -> None:
        n = len(self.time_s)
        shape = np.shape(self.reference_q)
        if np.shape(self.policy_q) != shape or len(shape) != 2 or shape[0] != n:
            raise ExperimentSpecError("trajectory shape mismatch")
        if np.shape(self.constraint_residual) != (n,):
            raise ExperimentSpecError("constraint_residual shape mismatch")
        arrays = (self.time_s, self.reference_q, self.policy_q)
        scalars = (self.reference_clubhead_speed, self.policy_clubhead_speed)
        if not all(np.all(np.isfinite(a)) for a in (*arrays, self.constraint_residual)):
            raise ExperimentSpecError("trajectories must be finite")
        if not all(np.isfinite(s) for s in scalars):
            raise ExperimentSpecError("clubhead speeds must be finite")


class MetricRegistry(Protocol):
    """Where metric definitions come from (the #14421 registry contract)."""

    version: str

    def compute(self, name: str, trial: SwingTrial) -> float:
        """Value of metric ``name`` on ``trial``; ``KeyError`` if unknown."""
        ...


def _tracking_rmse(trial: SwingTrial) -> float:
    return float(np.sqrt(np.mean((trial.policy_q - trial.reference_q) ** 2)))


def _clubhead_speed_error(trial: SwingTrial) -> float:
    return float(trial.policy_clubhead_speed - trial.reference_clubhead_speed)


def _constraint_violation_max(trial: SwingTrial) -> float:
    return float(np.max(np.abs(trial.constraint_residual)))


class StubMetricRegistry:
    """Stand-in for the #14421 registry until GM-1a (#14422) lands.

    ``tracking_rmse`` is the RMS joint-angle error over all samples and
    coordinates (rad); ``clubhead_speed_error`` is policy minus reference
    (m/s); ``constraint_violation_max`` is the largest absolute residual.
    """

    version: ClassVar[str] = "stub-14421-v0"
    _METRICS: ClassVar[dict[str, Callable[[SwingTrial], float]]] = {
        "tracking_rmse": _tracking_rmse,
        "clubhead_speed_error": _clubhead_speed_error,
        "constraint_violation_max": _constraint_violation_max,
    }

    def compute(self, name: str, trial: SwingTrial) -> float:
        """Value of metric ``name`` on ``trial`` (see class doc)."""
        if name not in self._METRICS:
            raise KeyError(f"unknown metric {name!r}")
        return self._METRICS[name](trial)
