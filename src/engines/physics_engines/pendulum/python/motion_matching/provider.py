"""``PendulumFitSwingProvider`` -- canonical motion-matching adapter for Pendulum.

This provides the analytic Lagrangian baseline for motion-matching using the
calibrated driven planar double pendulum infrastructure.
"""

from __future__ import annotations

import logging

from src.shared.python.motion_matching.club_target import ClubTarget
from src.shared.python.motion_matching.fit_result import CanonicalFitResult
from src.shared.python.motion_matching.provider import (
    FitOptions,
    MultiSourceTarget,
    publish_leaderboard_row,
    register_provider,
    resolve_club_target,
)
from src.shared.python.tour_baselines.pendulum_fit import (
    PendulumFitOptions,
    fit_driven_double_pendulum,
)

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
        """Fit continuous double-pendulum torques to match target motion."""
        club = resolve_club_target(target)
        fit_opts = PendulumFitOptions(maxiter=opts.maxiter if opts else 200)
        fit_result = fit_driven_double_pendulum(club, opts=fit_opts)
        result = fit_result.to_canonical_fit_result(
            engine_version=self.engine_version()
        )
        publish_leaderboard_row(self.engine_name, result, self.engine_version())
        return result

    def supports_body_target(self) -> bool:
        """Pendulum model only supports club kinematic targets."""
        return False

    def supports_ball_target(self) -> bool:
        """Pendulum model does not simulate ball flight."""
        return False

    def engine_version(self) -> str:
        """Engine release version."""
        return "1.0.0"

    @staticmethod
    def _extract_club(target: MultiSourceTarget | ClubTarget) -> ClubTarget:
        """Delegate to the shared :func:`resolve_club_target` (issue #6935).

        Retained for back-compat with direct callers/tests; unwrap behaviour
        is now identical across every engine.
        """
        return resolve_club_target(target)


register_provider(PendulumFitSwingProvider())
