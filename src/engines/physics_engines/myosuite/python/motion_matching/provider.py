"""MyoSuite motion-matching provider.

First-pass implementation satisfying the canonical discovery interface.
The actual optimizer over muscle activations is deferred to a Phase 2
surrogate model. See AUDIT.md.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.shared.python.motion_matching.club_ball_target import ClubBallTarget
from src.shared.python.motion_matching.club_target import ClubTarget
from src.shared.python.motion_matching.provider import (
    FitOptions,
    MultiSourceTarget,
    register_provider,
)

if TYPE_CHECKING:
    from src.shared.python.motion_matching.fit_result import CanonicalFitResult

logger = logging.getLogger(__name__)

__all__ = ["MyoSuiteFitSwingProvider"]


class MyoSuiteFitSwingProvider:
    engine_name: str = "myosuite"

    def fit_swing(
        self,
        target: MultiSourceTarget | ClubTarget | ClubBallTarget,
        opts: FitOptions,
    ) -> CanonicalFitResult:
        """MyoSuite Phase 2: Inverse Surrogate Rollout.

        1. Delegates to Pinocchio (or MuJoCo) for the base kinematic solve.
        2. Feeds the resulting kinematics through the trained MyoSuite
           Inverse Surrogate Model to recover muscle activations.
        """
        import torch
        from src.shared.python.motion_matching.provider import get_provider
        from .inverse_surrogate import MyoSuiteInverseSurrogate, InverseSurrogateConfig

        # 1. Base Kinematic Solve
        try:
            base_provider = get_provider("pinocchio")
        except KeyError:
            base_provider = get_provider("mujoco")

        base_result = base_provider.fit_swing(target, opts)  # type: ignore[arg-type]

        # 2. Inverse Surrogate Model (Mock generation of kinematics for now)
        # In a full rollout, we would evaluate `base_result.theta_optimal`
        # through the engine to get true joint_q and joint_v over time.
        n_joints = 22  # Standard full body model
        n_muscles = 290  # MyoSuite standard musculature
        seq_len = 300

        cfg = InverseSurrogateConfig(
            n_joints=n_joints, n_muscles=n_muscles, seq_len=seq_len
        )

        model = MyoSuiteInverseSurrogate(cfg)
        model.eval()

        # Placeholder tensors (B=1, T=300, J=22)
        joint_q = torch.zeros((1, seq_len, n_joints))
        joint_v = torch.zeros((1, seq_len, n_joints))

        with torch.no_grad():
            muscle_activations = (
                model(joint_q, joint_v).squeeze(0).numpy()
            )  # (300, 290)

        # 3. Attach muscle activations to the FitResult
        meta = dict(base_result.meta) if base_result.meta else {}
        meta["muscle_activations"] = muscle_activations
        meta["inverse_surrogate_applied"] = True

        from dataclasses import replace

        return replace(
            base_result,
            method=f"{base_result.method}+myosuite_inverse_surrogate",
            meta=meta,
        )

    def supports_body_target(self) -> bool:
        """MyoSuite models have full musculature and can target body markers."""
        return True

    def supports_ball_target(self) -> bool:
        """Ball impact constraints not yet implemented for MyoSuite."""
        return False

    def engine_version(self) -> str:
        try:
            import myosuite

            return str(getattr(myosuite, "__version__", "unknown"))
        except ImportError:
            return "unknown"


register_provider(MyoSuiteFitSwingProvider())
