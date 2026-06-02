"""
OpenSim backend for Inverse Kinematics.

Part of issue #4566. Wraps OpenSim's InverseKinematicsTool.
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


class OpenSimIKSolver(BaseIKSolver):
    """
    OpenSim-based Inverse Kinematics solver.

    Uses OpenSim's InverseKinematicsTool which is the
    gold standard for biomechanics IK solving.
    """

    def __init__(self, config: IKConfig | None = None):
        """
        Initialize OpenSim IK solver.

        Args:
            config: Solver configuration
        """
        super().__init__(config)

    def solve(
        self,
        markers: MarkerTrajectory,
        rig: SkeletonRig,
        weights: MarkerWeights | None = None,
        config: IKConfig | None = None,
    ) -> JointTrajectory:
        """
        Solve IK for a marker trajectory using OpenSim.

        Args:
            markers: Input marker trajectory
            rig: Scaled skeleton rig
            weights: Optional per-marker weights
            config: Optional solver configuration

        Returns:
            JointTrajectory with solved joint angles
        """
        config = config or self.config

        # Check if OpenSim is available
        try:
            import opensim as osim  # noqa: F401
        except ImportError as err:
            raise ImportError(
                "OpenSim not installed. Install with: pip install opensim"
            ) from err

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
            id=f"ik-opensim-{markers.id}",
            skeleton=rig,
            frames=frames,
            metadata={
                "backend": "opensim",
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
        Solve IK for a single frame using OpenSim.

        Args:
            markers: Dict mapping marker names to (x, y, z) positions
            rig: Scaled skeleton rig
            weights: Optional per-marker weights

        Raises:
            NotImplementedError: The OpenSim IK backend is not yet
                implemented. Use the ``geometric`` backend for a real
                dependency-free solver (issue #7046).
        """
        # A real implementation would write a TRC file + setup XML, run
        # OpenSim's InverseKinematicsTool, and parse the output .mot file.
        # Until that lands we raise loudly rather than returning a silent
        # neutral pose.
        raise NotImplementedError(  # tracked: #7046
            "OpenSim IK backend is not implemented; use the 'geometric' "
            "backend (#7046)."
        )
