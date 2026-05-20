from __future__ import annotations

import mujoco
import numpy as np

from ._mocap_data import MarkerSet, MotionCaptureFrame, MotionCaptureSequence
from .advanced_kinematics import AdvancedKinematicsAnalyzer


class MotionRetargeting:
    """Retarget motion capture data to MuJoCo model.

    This class maps motion capture markers to the model's body positions
    and solves inverse kinematics to generate joint trajectories.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        marker_set: MarkerSet,
    ) -> None:
        """Initialize motion retargeting.

        Args:
            model: MuJoCo model
            data: MuJoCo data
            marker_set: Marker set configuration
        """
        if model is None:
            raise ValueError("model must be provided")
        self.model = model
        self.data = data
        self.marker_set = marker_set

        # Initialize IK analyzer
        self.ik_analyzer = AdvancedKinematicsAnalyzer(model, data)

        # Build marker-to-body mapping
        self._build_body_mapping()

    def _build_body_mapping(self) -> None:
        """Build mapping from markers to MuJoCo body IDs."""
        self.marker_to_body_id: dict[str, int] = {}

        for marker_name, body_name in self.marker_set.markers.items():
            body_id = self.ik_analyzer._find_body_id(body_name)
            if body_id is not None:
                self.marker_to_body_id[marker_name] = body_id

    def retarget_sequence(
        self,
        mocap_sequence: MotionCaptureSequence,
        use_markers: list[str] | None = None,
        ik_iterations: int = 50,
    ) -> tuple[np.ndarray, np.ndarray, list[bool]]:
        """Retarget motion capture sequence to model joint trajectories.

        Args:
            mocap_sequence: Motion capture sequence
            use_markers: List of markers to use (default: all available)
            ik_iterations: Max IK iterations per frame

        Returns:
            Tuple of (times [N], joint_trajectories [N x nv], success_flags [N])
        """
        if mocap_sequence is None:
            raise ValueError("mocap_sequence must be provided")
        if use_markers is None:
            use_markers = list(self.marker_to_body_id.keys())

        times = []
        joint_trajectories = []
        success_flags = []

        # Initialize with current configuration
        q_prev = self.data.qpos.copy()

        for frame in mocap_sequence.frames:
            # Solve IK for this frame
            q_solution, success = self._solve_frame_ik(
                frame,
                use_markers,
                q_init=q_prev,
                max_iterations=ik_iterations,
            )

            times.append(frame.time)
            joint_trajectories.append(q_solution)
            success_flags.append(success)

            q_prev = q_solution

        return (np.array(times), np.array(joint_trajectories), success_flags)

    def _solve_frame_ik(
        self,
        frame: MotionCaptureFrame,
        use_markers: list[str],
        q_init: np.ndarray,
        max_iterations: int,
    ) -> tuple[np.ndarray, bool]:
        """Solve IK for a single frame.

        Args:
            frame: Motion capture frame
            use_markers: Markers to use for IK
            q_init: Initial joint configuration
            max_iterations: Max IK iterations

        Returns:
            Tuple of (joint_config, success)
        """
        # Multi-target IK: minimize error to all marker positions
        if frame is None:
            raise ValueError("frame must be provided")
        q = q_init.copy()

        for _iteration in range(max_iterations):
            # Compute error for all markers
            total_error = 0.0
            total_jacobian = None
            total_error_vector = None

            for marker_name in use_markers:
                if marker_name not in frame.marker_positions:
                    continue
                if marker_name not in self.marker_to_body_id:
                    continue

                body_id = self.marker_to_body_id[marker_name]
                target_pos = frame.marker_positions[marker_name]

                # Current body position
                self.data.qpos[:] = q
                mujoco.mj_forward(self.model, self.data)
                current_pos = self.data.xpos[body_id].copy()

                # Position error
                pos_error = target_pos - current_pos
                total_error += float(np.linalg.norm(pos_error))

                # Jacobian
                jacp, _ = self.ik_analyzer.compute_body_jacobian(body_id)

                # Accumulate
                if total_jacobian is None:
                    total_jacobian = jacp
                    total_error_vector = pos_error
                else:
                    total_jacobian = np.vstack([total_jacobian, jacp])
                    total_error_vector = np.concatenate([total_error_vector, pos_error])

            # Check convergence
            if total_error < 1e-3:  # 1mm threshold
                return q, True

            # Solve for joint update
            if total_jacobian is not None and total_error_vector is not None:
                # Damped least-squares
                damping = 0.01
                J = total_jacobian
                e = total_error_vector

                dq = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(J.shape[0]), e)

                # Update
                q = q + 0.5 * dq  # Step size 0.5 for stability

                # Clamp to limits
                q = self.ik_analyzer._clamp_to_joint_limits(q)

        # Did not converge
        return q, False

    def compute_marker_errors(
        self,
        frame: MotionCaptureFrame,
        q: np.ndarray,
    ) -> dict[str, float]:
        """Compute marker position errors for a configuration.

        Args:
            frame: Motion capture frame with target marker positions
            q: Joint configuration to evaluate

        Returns:
            Dictionary of marker_name -> error (m)
        """
        if frame is None:
            raise ValueError("frame must be provided")
        self.data.qpos[:] = q
        mujoco.mj_forward(self.model, self.data)

        errors = {}
        for marker_name, target_pos in frame.marker_positions.items():
            if marker_name in self.marker_to_body_id:
                body_id = self.marker_to_body_id[marker_name]
                current_pos = self.data.xpos[body_id].copy()
                error = float(np.linalg.norm(target_pos - current_pos))
                errors[marker_name] = error

        return errors
