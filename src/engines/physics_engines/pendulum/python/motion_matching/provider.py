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
            DoublePendulumState,
        )

        n_eval = 0
        history: list[float] = []

        # Polynomial forcing functions
        def make_forcing_func(coefs: np.ndarray):
            # Evaluate polynomial: c0 + c1*t + c2*t^2 + ...
            def forcing(t: float, state: DoublePendulumState) -> float:
                return float(np.polyval(coefs[::-1], t))

            return forcing

        def cost_func(theta: np.ndarray) -> float:
            nonlocal n_eval
            n_eval += 1

            # theta is 14 elements (7 for shoulder torque, 7 for wrist torque)
            shoulder_coefs = theta[:7]
            wrist_coefs = theta[7:]

            dynamics = DoublePendulumDynamics(
                forcing_functions=(
                    make_forcing_func(shoulder_coefs),
                    make_forcing_func(wrist_coefs),
                )
            )

            # Time grid from projected club. ``ClubTarget`` exposes only the
            # raw ``time`` array; derive dt and frame count locally rather than
            # rely on accessors that don't exist on the dataclass.
            n_frames = int(projected_club.time.shape[0])
            dt = (
                float(projected_club.time[1] - projected_club.time[0])
                if n_frames >= 2
                else 0.0
            )

            # Initial state
            # Assuming club target butt is shoulder, clubhead is end of club
            # For simplicity, we just use 0s for initial state or try to derive it.
            # In a real model, we would IK the initial frame. Here we use zero for simplicity.
            state = DoublePendulumState(theta1=0.0, theta2=0.0, omega1=0.0, omega2=0.0)

            total_sq_error = 0.0

            for i in range(n_frames):
                t = i * dt
                # Step physics
                state = dynamics.step(t, state, dt)

                # Compute forward kinematics for clubhead
                l1 = dynamics.parameters.upper_segment.length_m
                l2 = dynamics.parameters.lower_segment.length_m
                x_head = l1 * np.sin(state.theta1) + l2 * np.sin(
                    state.theta1 + state.theta2
                )
                y_head = -l1 * np.cos(state.theta1) - l2 * np.cos(
                    state.theta1 + state.theta2
                )

                # Target clubhead
                target_head = projected_club.clubhead[i]

                # Error (we assume target is translated such that shoulder is at 0,0)
                sq_err = (x_head - target_head[0]) ** 2 + (y_head - target_head[1]) ** 2
                total_sq_error += sq_err

            cost = float(total_sq_error / n_frames)
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
        )  # type: ignore[call-overload]
        elapsed = time.perf_counter() - t0

        return CanonicalFitResult(
            theta_optimal=np.asarray(res.x, dtype=np.float64),
            final_cost=float(res.fun),
            final_rmse_m=float(np.sqrt(res.fun)),
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
