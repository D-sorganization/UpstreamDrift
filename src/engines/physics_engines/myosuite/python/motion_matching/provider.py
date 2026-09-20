"""MyoSuite motion-matching provider.

Fail-closed implementation satisfying the canonical discovery interface (MS-50, #10343).
Muscle activation optimization is deferred to Phase 2 surrogate modeling.
Never returns fabricated muscle activations or placeholder tensors. See AUDIT.md.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Final

import numpy as np

from src.shared.python.contracts import postcondition, precondition
from src.shared.python.motion_matching.club_ball_target import ClubBallTarget
from src.shared.python.motion_matching.club_target import ClubTarget
from src.shared.python.motion_matching.fit_result import CanonicalFitResult
from src.shared.python.motion_matching.provenance import engine_package_version
from src.shared.python.motion_matching.provider import (
    FitOptions,
    MultiSourceTarget,
    has_body_target,
    register_provider,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

__all__ = ["MyoSuiteFitSwingProvider", "UnsupportedTargetError"]

ENGINE_NAME: Final[str] = "myosuite"
UNSUPPORTED_REASON: Final[str] = (
    "MyoSuite muscle activation optimization is unsupported: "
    "direct polynomial torque matching is not compatible with muscle actuation, "
    "and Phase 2 inverse surrogate is not yet trained or integrated (see AUDIT.md)."
)


class UnsupportedTargetError(ValueError):
    """Raised when an unsupported target type is passed to the provider."""


class MyoSuiteFitSwingProvider:
    engine_name: str = ENGINE_NAME

    @precondition(
        lambda self, target, opts=None: target is not None, "target must not be None"
    )
    @postcondition(
        lambda r: (
            r.solver_status == "unsupported" and "muscle_activations" not in r.meta
        ),
        "fail-closed: never return fake muscle activations",
    )
    def fit_swing(
        self,
        target: MultiSourceTarget | ClubTarget | ClubBallTarget,
        opts: FitOptions | None = None,
    ) -> CanonicalFitResult:
        """Fail closed for MyoSuite swing fitting (MS-50).

        MyoSuite control inputs are 290 muscle activations, not joint torques.
        Until a validated inverse surrogate model is trained (Phase 2),
        returns a CanonicalFitResult with status 'unsupported' and never
        synthesizes fake zero-tensor activations.
        """
        if has_body_target(target):
            raise UnsupportedTargetError(
                f"{self.engine_name} does not support body targets."
            )

        logger.info(
            "MyoSuite fit_swing called: failing closed (%s)", UNSUPPORTED_REASON
        )
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        target_hash = getattr(target, "target_hash", "") or getattr(
            getattr(target, "source", None), "sha256", ""
        )

        return CanonicalFitResult(
            theta_optimal=np.empty((0,), dtype=np.float64),
            final_cost=float("inf"),
            final_rmse_m=float("inf"),
            solver_status="unsupported",
            iterations=0,
            n_evaluations=0,
            wall_clock_s=0.0,
            message=UNSUPPORTED_REASON,
            history=(),
            method="unsupported",
            git_commit="",
            engine_version=self.engine_version(),
            target_hash=str(target_hash),
            timestamp_utc=now_iso,
            meta={
                "status": "unsupported",
                "reason": UNSUPPORTED_REASON,
                "inverse_surrogate_applied": False,
            },
        )

    def supports_body_target(self) -> bool:
        """Return False until MS-53 (body-target marker IK + muscle inversion)."""
        return False

    def supports_ball_target(self) -> bool:
        """Ball impact constraints not yet implemented for MyoSuite."""
        return False

    def engine_version(self) -> str:
        try:
            import myosuite
        except ImportError:
            return "unknown"

        return engine_package_version(myosuite, "myosuite")


register_provider(MyoSuiteFitSwingProvider())  # type: ignore[arg-type]
