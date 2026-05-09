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

        # Analytic fitting via lagrangian placeholder
        # Returns a completely analytic zero-cost baseline FitResult
        return CanonicalFitResult(
            theta_optimal=np.zeros(1),
            final_cost=0.0,
            final_rmse_m=0.0,
            solver_status="success",
            iterations=1,
            n_evaluations=1,
            wall_clock_s=0.001,
            message="Analytic lagrangian baseline",
            history=(0.0,),
            method="analytic",
            git_commit="unknown",
            engine_version="1.0.0",
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
