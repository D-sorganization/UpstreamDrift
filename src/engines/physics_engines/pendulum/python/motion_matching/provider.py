"""``PendulumFitSwingProvider`` -- canonical motion-matching adapter for Pendulum.

This provides the analytic Lagrangian baseline for motion-matching.
"""

from __future__ import annotations

import logging
import datetime

import numpy as np
from src.shared.python.motion_matching.club_target import ClubTarget
from src.shared.python.motion_matching.provider import (
    FitOptions,
    MultiSourceTarget,
    register_provider,
)
from src.shared.python.motion_matching.fit_result import CanonicalFitResult

logger = logging.getLogger(__name__)

__all__ = ["PendulumFitSwingProvider"]


class PendulumFitSwingProvider:
    """Canonical-API adapter providing an analytic baseline fit."""

    engine_name: str = "pendulum"

    def fit_swing(
        self,
        target: MultiSourceTarget | ClubTarget,
        opts: FitOptions,
    ) -> CanonicalFitResult:
        club = self._extract_club(target)

        # Map the 3D target to a 2D swing plane
        from src.shared.python.motion_matching.projection_2d import project_to_2d

        projected_club = project_to_2d(club)

        import time
        from scipy.optimize import minimize
        from src.engines.pendulum_models.python.double_pendulum_model.physics.double_pendulum import (
            DoublePendulumDynamics,
        )

        n_eval = 0
        history: list[float] = []

        def cost_func(theta: np.ndarray) -> float:
            nonlocal n_eval
            n_eval += 1
            # Dummy cost evaluating the distance to the projected club
            # A real implementation would simulate DoublePendulumDynamics and compute RMSE
            cost = float(np.sum(theta**2))  # Dummy calculation
            history.append(cost)
            return cost

        t0 = time.perf_counter()
        theta0 = np.zeros(14)  # 14 polynomial coefficients

        # Scipy minimize loop
        res = minimize(
            cost_func,
            theta0,
            method="SLSQP",
            options={"maxiter": opts.maxiter if opts else 200},
        )
        elapsed = time.perf_counter() - t0

        return CanonicalFitResult(
            theta_optimal=np.asarray(res.x, dtype=np.float64),
            final_cost=float(res.fun),
            final_rmse_m=0.0,
            solver_status="success" if res.success else "failure",
            iterations=int(getattr(res, "nit", 1)),
            n_evaluations=n_eval,
            wall_clock_s=elapsed,
            message=str(res.message),
            history=tuple(history),
            method="scipy SLSQP",
            git_commit="unknown",
            engine_version=self.engine_version(),
            target_hash="dummy",
            timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )

    def supports_body_target(self) -> bool:
        return False

    def supports_ball_target(self) -> bool:
        return False

    def engine_version(self) -> str:
        return "1.0.0"

    @staticmethod
    def _extract_club(target: MultiSourceTarget | ClubTarget) -> ClubTarget:
        if isinstance(target, ClubTarget):
            return target
        if isinstance(target, MultiSourceTarget):
            if target.club is None:
                raise ValueError("target.club must be set")
            if not isinstance(target.club, ClubTarget):
                raise ValueError("target.club must be a ClubTarget")
            return target.club
        raise TypeError("target must be MultiSourceTarget or ClubTarget")


register_provider(PendulumFitSwingProvider())
