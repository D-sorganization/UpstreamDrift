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

    @classmethod
    def from_api_contract(
        cls,
        result: Any,
        *,
        git_commit: str,
        engine_version: str,
        target_hash: str,
        timestamp_utc: str,
        solver_status: str = "success",
        iterations: int = 0,
        n_evaluations: int = 0,
        message: str = "",
    ) -> CanonicalFitResult:
        """Build the canonical result from the legacy API-contract payload."""
        coefficients = np.asarray(result.coefficients, dtype=np.float64)
        metadata = dict(result.metadata)
        time_s = float(metadata.get("time_s", 0.0))
        method = str(metadata.get("solver", metadata.get("method", "api_contract")))
        return cls(
            theta_optimal=coefficients,
            final_cost=float(result.final_loss),
            final_rmse_m=float(metadata.get("final_rmse_m", result.final_loss)),
            solver_status=solver_status,
            iterations=iterations,
            n_evaluations=n_evaluations,
            wall_clock_s=time_s,
            message=message,
            history=(),
            method=method,
            git_commit=git_commit,
            engine_version=engine_version,
            target_hash=target_hash,
            timestamp_utc=timestamp_utc,
            meta={"api_contract_metadata": metadata},
        )

    @property
    def coefficients(self) -> NDArray[np.float64]:
        warnings.warn(
            "FitResult.coefficients is deprecated; use theta_optimal "
            "(see issue #4250 for the cross-engine FitResult standardisation).",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.theta_optimal

    @property
    def theta(self) -> NDArray[np.float64]:
        """Legacy alias for ``theta_optimal``.

        Pinocchio, OpenSim, and Drake call sites historically read
        ``result.theta``. The canonical schema renamed the field to
        ``theta_optimal``; this shim preserves backward compatibility
        until those call sites are migrated. Emits a ``DeprecationWarning``
        on access.
        """
        warnings.warn(
            "theta is deprecated; use theta_optimal",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.theta_optimal

    @property
    def mujoco_version(self) -> str:
        """Legacy alias for ``engine_version`` on MuJoCo provenance.

        The previous engine-specific MuJoCo result exposed
        ``mujoco_version``. The canonical schema unified that field as
        ``engine_version``; this shim preserves backward compatibility
        for existing MuJoCo provenance contracts and integration tests.
        Emits a ``DeprecationWarning`` on access.
        """
        warnings.warn(
            "mujoco_version is deprecated; use engine_version",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.engine_version

    @property
    def cost(self) -> float:
        warnings.warn(
            "FitResult.cost is deprecated; use final_cost "
            "(see issue #4250 for the cross-engine FitResult standardisation).",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.final_cost

    @property
    def n_iter(self) -> int:
        warnings.warn(
            "FitResult.n_iter is deprecated; use iterations "
            "(see issue #4250 for the cross-engine FitResult standardisation).",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.iterations

    @property
    def n_eval(self) -> int:
        warnings.warn(
            "FitResult.n_eval is deprecated; use n_evaluations "
            "(see issue #4250 for the cross-engine FitResult standardisation).",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.n_evaluations

    @property
    def n_evals(self) -> int:
        warnings.warn(
            "FitResult.n_evals is deprecated; use n_evaluations "
            "(see issue #4250 for the cross-engine FitResult standardisation).",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.n_evaluations

    @property
    def success(self) -> bool:
        warnings.warn(
            "FitResult.success is deprecated; use solver_status == 'success' "
            "(see issue #4250 for the cross-engine FitResult standardisation).",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.solver_status == "success"

    @property
    def duration_s(self) -> float:
        warnings.warn(
            "FitResult.duration_s is deprecated; use wall_clock_s "
            "(see issue #4250 for the cross-engine FitResult standardisation).",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.wall_clock_s

    @property
    def elapsed_s(self) -> float:
        warnings.warn(
            "FitResult.elapsed_s is deprecated; use wall_clock_s "
            "(see issue #4250 for the cross-engine FitResult standardisation).",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.wall_clock_s

    @property
    def solver(self) -> str:
        warnings.warn(
            "FitResult.solver is deprecated; use method "
            "(see issue #4250 for the cross-engine FitResult standardisation).",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.method
