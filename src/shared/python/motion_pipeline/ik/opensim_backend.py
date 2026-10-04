"""
OpenSim backend for Inverse Kinematics.

Part of issue #4566. Retired per ADR-0051 (MS-12 #10331); use the unified
MatchingPlant pipeline (MS-10) or 'geometric' / 'pinocchio' backend.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..contracts import (
    JointStateFrame,
    JointTrajectory,
    Marker,
    MarkerFrame,
    MarkerTrajectory,
    SkeletonRig,
)
from .base import BaseIKSolver, IKConfig, MarkerWeights

logger = logging.getLogger(__name__)


class OpenSimIKBackend(BaseIKSolver):
    """
    OpenSim-based Inverse Kinematics solver.

    Uses OpenSim's InverseKinematicsTool which is the
    gold standard for biomechanics IK solving. Supports optional
    geometric fallback for headless or hermetic test environments.
    """

    def __init__(
        self,
        config: IKConfig | None = None,
        *,
        model_path: Path | str | None = None,
        allow_fallback: bool = False,
    ) -> None:
        """
        Initialize OpenSim IK solver / backend.

        Args:
            config: Solver configuration.
            model_path: Optional path to subject-scaled OpenSim .osim model.
            allow_fallback: If True, fall back to geometric solver when
                OpenSim is unavailable or unconfigured.
        """
        super().__init__(config)
        self.model_path = Path(model_path) if model_path is not None else None
        self.allow_fallback = bool(allow_fallback)

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
            if self.allow_fallback:
                logger.info("OpenSim not installed; falling back to GeometricIKSolver")
                return self._solve_geometric_fallback(markers, rig, weights, config)
            raise ImportError(
                "OpenSim not installed. Install with: pip install opensim"
            ) from err

        if self.allow_fallback and self.model_path is None:
            return self._solve_geometric_fallback(markers, rig, weights, config)

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

    @staticmethod
    def _adapt_markers_for_rig(
        marker_positions: dict[str, tuple[float, float, float]],
        rig: SkeletonRig,
    ) -> dict[str, tuple[float, float, float]]:
        """Ensure marker dictionary has targets matching rig joints."""
        label_to_joint = {
            jdef.semantic_label: jname
            for jname, jdef in rig.joints.items()
            if jdef.semantic_label
        }
        matches = any(m in rig.joints or m in label_to_joint for m in marker_positions)
        if matches:
            return marker_positions

        adapted = dict(marker_positions)
        _MAPPING: dict[str, list[str]] = {
            "pelvis": ["r.ASIS_study", "L.ASIS_study", "r.PSIS_study", "L.PSIS_study"],
            "torso": ["C7_study", "r_shoulder_study", "L_shoulder_study"],
            "neck": ["C7_study"],
            "right_shoulder": ["r_shoulder_study"],
            "left_shoulder": ["L_shoulder_study"],
            "right_elbow": ["r_lelbow_study", "r_melbow_study"],
            "left_elbow": ["L_lelbow_study", "L_melbow_study"],
            "right_wrist": ["r_lwrist_study", "r_mwrist_study"],
            "left_wrist": ["L_lwrist_study", "L_mwrist_study"],
            "right_thigh": ["RHJC_study", "r_trochanter_study"],
            "left_thigh": ["LHJC_study", "L_trochanter_study"],
            "right_knee": ["r_knee_study", "r_mknee_study"],
            "left_knee": ["L_knee_study", "L_mknee_study"],
            "right_ankle": ["r_ankle_study", "r_mankle_study"],
            "left_ankle": ["L_ankle_study", "L_mankle_study"],
            "right_foot": ["r_calc_study", "r_toe_study", "r_5meta_study"],
            "left_foot": ["L_calc_study", "L_toe_study", "L_5meta_study"],
        }
        for jname, candidates in _MAPPING.items():
            if jname in rig.joints:
                pts = [marker_positions[c] for c in candidates if c in marker_positions]
                if pts:
                    avg_pt = tuple(
                        float(sum(p[i] for p in pts) / len(pts)) for i in range(3)
                    )
                    adapted[jname] = (avg_pt[0], avg_pt[1], avg_pt[2])
        return adapted

    def _solve_geometric_fallback(
        self,
        markers: MarkerTrajectory,
        rig: SkeletonRig,
        weights: MarkerWeights | None,
        config: IKConfig,
    ) -> JointTrajectory:
        from .geometric_backend import GeometricIKSolver

        adapted_frames: list[MarkerFrame] = []
        for f in markers.frames:
            pos_dict = {name: (m.x, m.y, m.z) for name, m in f.markers.items()}
            adapted_pos = self._adapt_markers_for_rig(pos_dict, rig)
            new_markers = dict(f.markers)
            for jname, (x, y, z) in adapted_pos.items():
                if jname not in new_markers:
                    new_markers[jname] = Marker(name=jname, x=x, y=y, z=z)
            adapted_frames.append(
                MarkerFrame(
                    timestamp=f.timestamp,
                    markers=new_markers,
                    frame_index=f.frame_index,
                )
            )

        adapted_markers = MarkerTrajectory(
            id=markers.id,
            frames=adapted_frames,
            calibration=markers.calibration,
            subject_id=markers.subject_id,
            metadata=markers.metadata,
        )
        res = GeometricIKSolver(config).solve(adapted_markers, rig, weights)
        return JointTrajectory(
            id=f"ik-opensim-{markers.id}",
            skeleton=rig,
            frames=res.frames,
            metadata={
                **res.metadata,
                "backend": "opensim",
                "mode": "geometric_fallback",
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
            NotImplementedError: If allow_fallback is False and OpenSim Tool
                has no per-frame model setup.
        """
        if self.allow_fallback:
            from .geometric_backend import GeometricIKSolver

            adapted = self._adapt_markers_for_rig(markers, rig)
            return GeometricIKSolver(self.config).solve_frame(adapted, rig, weights)

        raise NotImplementedError(  # tracked: #7046, retired per ADR-0051
            "OpenSim IK backend is retired per ADR-0051; use the unified "
            "MatchingPlant pipeline or the 'geometric' backend."
        )


OpenSimIKSolver = OpenSimIKBackend

__all__ = ["OpenSimIKBackend", "OpenSimIKSolver"]
