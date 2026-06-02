"""
MuJoCo backend for Inverse Kinematics.

Part of issue #4566. Wraps MuJoCo-based IK solving.
"""

from __future__ import annotations

import logging


from ..contracts import (
    JointStateFrame,
    JointTrajectory,
    MarkerTrajectory,
    SkeletonRig,
)
from .base import BaseIKSolver, IKConfig, MarkerWeights

logger = logging.getLogger(__name__)


class MuJoCoIKSolver(BaseIKSolver):
    """
    MuJoCo-based Inverse Kinematics solver.

    Uses MuJoCo's physics engine for IK solving via
    position-based constraints on marker targets.
    """

    def __init__(self, config: IKConfig | None = None):
        """
        Initialize MuJoCo IK solver.

        Args:
            config: Solver configuration
        """
        super().__init__(config)
        self._model = None
        self._data = None

    def solve(
        self,
        markers: MarkerTrajectory,
        rig: SkeletonRig,
        weights: MarkerWeights | None = None,
        config: IKConfig | None = None,
    ) -> JointTrajectory:
        """
        Solve IK for a marker trajectory using MuJoCo.

        Args:
            markers: Input marker trajectory
            rig: Scaled skeleton rig
            weights: Optional per-marker weights
            config: Optional solver configuration

        Returns:
            JointTrajectory with solved joint angles
        """
        config = config or self.config

        # Process each frame
        frames: list[JointStateFrame] = []
        for frame in markers.frames:
            marker_positions = {
                name: (m.x, m.y, m.z) for name, m in frame.markers.items()
            }

            q = self.solve_frame(marker_positions, rig, weights)

            frames.append(
                JointStateFrame(
                    timestamp=frame.timestamp,
                    q=q,
                    qdot=None,
                    qddot=None,
                    frame_index=frame.frame_index,
                )
            )

        return JointTrajectory(
            id=f"ik-mujoco-{markers.id}",
            skeleton=rig,
            frames=frames,
            metadata={
                "backend": "mujoco",
                "config": {
                    "max_iterations": config.max_iterations,
                    "tolerance": config.tolerance,
                },
            },
        )

    def solve_frame(
        self,
        markers: dict[str, tuple[float, float, float]],
        rig: SkeletonRig,
        weights: MarkerWeights | None = None,
    ) -> list[float]:
        """
        Solve IK for a single frame using MuJoCo.

        Args:
            markers: Dict mapping marker names to (x, y, z) positions
            rig: Scaled skeleton rig
            weights: Optional per-marker weights

        Raises:
            NotImplementedError: The MuJoCo IK backend is not yet
                implemented. Use the ``geometric`` backend for a real
                dependency-free solver (issue #7046).
        """
        # A real implementation would build a MuJoCo model from the rig,
        # set marker targets, run damped-least-squares IK, and extract
        # joint angles. Until that lands we raise loudly rather than
        # returning a silent neutral pose that masks the missing solver.
        raise NotImplementedError(  # tracked: #7046
            "MuJoCo IK backend is not implemented; use the 'geometric' backend (#7046)."
        )
