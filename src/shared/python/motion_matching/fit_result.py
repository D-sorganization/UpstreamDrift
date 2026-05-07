from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .cost import CostBreakdown


@dataclass(frozen=True)
class CanonicalFitResult:
    """Canonical engine-agnostic fit result for motion matching.

    This replaces engine-specific FitResult dataclasses to guarantee
    cross-engine parity.
    """

    theta_optimal: NDArray[np.float64]
    final_cost: float
    final_rmse_m: float
    solver_status: str
    iterations: int
    n_evaluations: int
    wall_clock_s: float
    message: str
    history: tuple[float, ...]
    method: str
    git_commit: str
    engine_version: str
    target_hash: str
    timestamp_utc: str

    # Engine-specific extensions (optional)
    cost_breakdown: CostBreakdown | None = None
    final_total_work_J: float | None = None
    n_jac_eval: int = 0
    solver_options: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def coefficients(self) -> NDArray[np.float64]:
        warnings.warn(
            "coefficients is deprecated; use theta_optimal",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.theta_optimal

    @property
    def cost(self) -> float:
        warnings.warn(
            "cost is deprecated; use final_cost", DeprecationWarning, stacklevel=2
        )
        return self.final_cost

    @property
    def n_iter(self) -> int:
        warnings.warn(
            "n_iter is deprecated; use iterations", DeprecationWarning, stacklevel=2
        )
        return self.iterations

    @property
    def n_eval(self) -> int:
        warnings.warn(
            "n_eval is deprecated; use n_evaluations", DeprecationWarning, stacklevel=2
        )
        return self.n_evaluations

    @property
    def n_evals(self) -> int:
        warnings.warn(
            "n_evals is deprecated; use n_evaluations", DeprecationWarning, stacklevel=2
        )
        return self.n_evaluations

    @property
    def success(self) -> bool:
        warnings.warn(
            "success is deprecated; use solver_status == 'success'",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.solver_status == "success"

    @property
    def duration_s(self) -> float:
        warnings.warn(
            "duration_s is deprecated; use wall_clock_s",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.wall_clock_s

    @property
    def elapsed_s(self) -> float:
        warnings.warn(
            "elapsed_s is deprecated; use wall_clock_s",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.wall_clock_s

    @property
    def solver(self) -> str:
        warnings.warn(
            "solver is deprecated; use method", DeprecationWarning, stacklevel=2
        )
        return self.method
