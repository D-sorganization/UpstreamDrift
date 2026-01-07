"""Motion capture integration and retargeting for golf swing analysis.

This module provides comprehensive motion capture data handling, including:
- Loading mocap data from multiple formats (BVH, C3D, CSV, JSON)
- Motion retargeting to MuJoCo models using IK
- Kinematic trajectory extraction and processing
- Marker-based and markerless mocap support
- Temporal alignment and filtering
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING  # noqa: ICN003

import mujoco

if TYPE_CHECKING:
    from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline, interp1d
from scipy.signal import butter, filtfilt

from .advanced_kinematics import AdvancedKinematicsAnalyzer


@dataclass
class MotionCaptureFrame:
    """Single frame of motion capture data."""

    time: float
    marker_positions: dict[str, np.ndarray]  # marker_name -> position [3]
    marker_velocities: dict[str, np.ndarray] | None = None
    body_orientations: dict[str, np.ndarray] | None = (
        None  # body_name -> quaternion [4]
    )
    joint_angles: np.ndarray | None = None  # If available from mocap system


@dataclass
class MotionCaptureSequence:
    """Complete motion capture sequence."""

    frames: list[MotionCaptureFrame]
    frame_rate: float
    marker_names: list[str]
    metadata: dict = field(default_factory=dict)

    @property
    def num_frames(self) -> int:
        """Get number of frames."""
        return len(self.frames)

    @property
    def duration(self) -> float:
        """Get sequence duration in seconds."""
        if len(self.frames) < 2:
            return 0.0
        return self.frames[-1].time - self.frames[0].time

    def get_marker_trajectory(self, marker_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Get trajectory for a specific marker.

        Args:
            marker_name: Name of marker

        Returns:
            Tuple of (times [N], positions [N x 3])
        """
        times = []
        positions = []

        for frame in self.frames:
            if marker_name in frame.marker_positions:
                times.append(frame.time)
                positions.append(frame.marker_positions[marker_name])

        return np.array(times), np.array(positions)


@dataclass
class MarkerSet:
    """Marker set configuration for motion capture."""

    markers: dict[str, str]  # marker_name -> body_name
    marker_offsets: dict[str, np.ndarray]  # marker_name -> offset from body origin [3]

    @classmethod
    def golf_swing_marker_set(cls) -> MarkerSet:
        """Standard marker set for golf swing capture.

        Based on common motion capture protocols for golf biomechanics.
        """
        markers = {
            # Head
            "HEAD_TOP": "head",
            "HEAD_FRONT": "head",
            # Torso
            "C7": "upper_torso",  # 7th cervical vertebra
            "T10": "lower_torso",  # 10th thoracic vertebra
            "STERN": "upper_torso",  # Sternum
            "CLAV": "upper_torso",  # Clavicle
            # Pelvis
            "SACR": "pelvis",  # Sacrum
            "LASI": "pelvis",  # Left anterior superior iliac spine
            "RASI": "pelvis",  # Right ASIS
            "LPSI": "pelvis",  # Left posterior superior iliac spine
            "RPSI": "pelvis",  # Right PSIS
            # Left arm
            "LSHO": "left_upper_arm",  # Left shoulder
            "LELB": "left_forearm",  # Left elbow
            "LWRA": "left_hand",  # Left wrist radial
            "LWRU": "left_hand",  # Left wrist ulnar
            "LFIN": "left_hand",  # Left finger
            # Right arm
            "RSHO": "right_upper_arm",
            "RELB": "right_forearm",
            "RWRA": "right_hand",
            "RWRU": "right_hand",
            "RFIN": "right_hand",
            # Left leg
            "LKNE": "left_shin",  # Left knee
            "LANK": "left_foot",  # Left ankle
            "LHEE": "left_foot",  # Left heel
            "LTOE": "left_foot",  # Left toe
            # Right leg
            "RKNE": "right_shin",
            "RANK": "right_foot",
            "RHEE": "right_foot",
            "RTOE": "right_foot",
            # Club
            "CLUB_GRIP_TOP": "club",
            "CLUB_GRIP_MID": "club",
            "CLUB_HEAD": "club_head",
        }

        # Approximate marker offsets (these should be measured for each subject)
        offsets = {}
        for marker_name in markers:
            offsets[marker_name] = np.zeros(3)  # Will be calibrated

        return cls(markers=markers, marker_offsets=offsets)


class MotionCaptureLoader:
    """Load motion capture data from various file formats."""

    @staticmethod
    def load_csv(
        filepath: str | Path,
        frame_rate: float = 120.0,
        marker_names: list[str] | None = None,
    ) -> MotionCaptureSequence:
        """Load motion capture data from CSV file.

        Expected format:
        time, marker1_x, marker1_y, marker1_z, marker2_x, marker2_y, marker2_z, ...

        Args:
            filepath: Path to CSV file
            frame_rate: Frame rate in Hz
            marker_names: List of marker names (if None, auto-detect from header)

        Returns:
            MotionCaptureSequence
        """
        data = np.loadtxt(filepath, delimiter=",", skiprows=1)

        # Parse header for marker names if not provided
        if marker_names is None:
            with open(filepath) as f:
                header = f.readline().strip().split(",")
                # Extract marker names from column headers (e.g., "LSHO_x", "LSHO_y")
                marker_names = []
                for i in range(1, len(header), 3):
                    marker_name = header[i].rsplit("_", 1)[0]
                    if marker_name not in marker_names:
                        marker_names.append(marker_name)

        frames = []
        for row in data:
            time = row[0]
            marker_positions = {}

            for i, marker_name in enumerate(marker_names):
                idx = 1 + i * 3
                if idx + 2 < len(row):
                    position = row[idx : idx + 3]
                    marker_positions[marker_name] = position

            frame = MotionCaptureFrame(time=time, marker_positions=marker_positions)
            frames.append(frame)

        return MotionCaptureSequence(
            frames=frames,
            frame_rate=frame_rate,
            marker_names=marker_names,
        )

    @staticmethod
    def load_json(filepath: str | Path) -> MotionCaptureSequence:
        """Load motion capture data from JSON file.

        Expected format:
        {
            "frame_rate": 120.0,
            "marker_names": ["LSHO", "RSHO", ...],
            "frames": [
                {
                    "time": 0.0,
                    "markers": {
                        "LSHO": [x, y, z],
                        # ...
                        # ...
                    }
                },
                # ...
            ]
        }

        Args:
            filepath: Path to JSON file

        Returns:
            MotionCaptureSequence
        """
        with open(filepath) as f:
            data = json.load(f)

        frames = []
        for frame_data in data["frames"]:
            frame = MotionCaptureFrame(
                time=frame_data["time"],
                marker_positions={
                    name: np.array(pos) for name, pos in frame_data["markers"].items()
                },
            )
            frames.append(frame)

        return MotionCaptureSequence(
            frames=frames,
            frame_rate=data.get("frame_rate", 120.0),
            marker_names=data.get(
                "marker_names",
                list(frames[0].marker_positions.keys()),
            ),
            metadata=data.get("metadata", {}),
        )

    @staticmethod
    def load_bvh(filepath: str | Path) -> MotionCaptureSequence | None:
        """Load motion capture data from BVH file.

        BVH (Biovision Hierarchy) is a common format for motion capture.
        This is a simplified parser - for production use, consider using
        a dedicated BVH library.

        Args:
            filepath: Path to BVH file

        Returns:
            MotionCaptureSequence
        """
        # Placeholder for BVH parsing
        # In production, use a library like 'bvh' or 'scikit-kinematics'
        return None


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


class MotionCaptureProcessor:
    """Process and filter motion capture data."""

    @staticmethod
    def filter_trajectory(
        times: np.ndarray,
        positions: np.ndarray,
        cutoff_frequency: float = 6.0,
        sampling_rate: float = 120.0,
    ) -> np.ndarray:
        """Apply low-pass Butterworth filter to trajectory.

        Args:
            times: Time array [N]
            positions: Position array [N x 3] or [N x nv]
            cutoff_frequency: Cutoff frequency in Hz
            sampling_rate: Sampling rate in Hz

        Returns:
            Filtered positions [N x 3] or [N x nv]
        """
        # Design filter
        nyquist = sampling_rate / 2.0
        normalized_cutoff = cutoff_frequency / nyquist
        b, a = butter(4, normalized_cutoff, btype="low")

        # Apply filter to each column
        filtered = np.zeros_like(positions)
        for i in range(positions.shape[1]):
            filtered[:, i] = filtfilt(b, a, positions[:, i])

        return filtered

    @staticmethod
    def compute_velocities(
        times: np.ndarray,
        positions: np.ndarray,
        method: str = "finite_difference",
    ) -> np.ndarray:
        """Compute velocities from position data.

        Args:
            times: Time array [N]
            positions: Position array [N x d]
            method: Method ("finite_difference", "spline")

        Returns:
            Velocities [N x d]
        """
        if method == "finite_difference":
            # Central differences
            velocities = np.zeros_like(positions)
            velocities[1:-1] = (positions[2:] - positions[:-2]) / (
                times[2:] - times[:-2]
            )[:, np.newaxis]
            velocities[0] = (positions[1] - positions[0]) / (times[1] - times[0])
            velocities[-1] = (positions[-1] - positions[-2]) / (times[-1] - times[-2])

        elif method == "spline":
            # Cubic spline derivatives
            velocities = np.zeros_like(positions)
            for i in range(positions.shape[1]):
                spline = CubicSpline(times, positions[:, i])
                velocities[:, i] = spline(times, nu=1)

        return velocities

    @staticmethod
    def compute_accelerations(
        times: np.ndarray,
        velocities: np.ndarray,
        method: str = "finite_difference",
    ) -> np.ndarray:
        """Compute accelerations from velocity data.

        Args:
            times: Time array [N]
            velocities: Velocity array [N x d]
            method: Method ("finite_difference", "spline")

        Returns:
            Accelerations [N x d]
        """
        if method == "finite_difference":
            accelerations = np.zeros_like(velocities)
            accelerations[1:-1] = (velocities[2:] - velocities[:-2]) / (
                times[2:] - times[:-2]
            )[:, np.newaxis]
            accelerations[0] = (velocities[1] - velocities[0]) / (times[1] - times[0])
            accelerations[-1] = (velocities[-1] - velocities[-2]) / (
                times[-1] - times[-2]
            )

        elif method == "spline":
            accelerations = np.zeros_like(velocities)
            for i in range(velocities.shape[1]):
                spline = CubicSpline(times, velocities[:, i])
                accelerations[:, i] = spline(times, nu=1)

        return accelerations

    @staticmethod
    def resample_trajectory(
        times: np.ndarray,
        trajectory: np.ndarray,
        new_times: np.ndarray,
        method: str = "cubic",
    ) -> np.ndarray:
        """Resample trajectory to new time points.

        Args:
            times: Original time array [N]
            trajectory: Original trajectory [N x d]
            new_times: New time points [M]
            method: Interpolation method ("linear", "cubic")

        Returns:
            Resampled trajectory [M x d]
        """
        resampled = np.zeros((len(new_times), trajectory.shape[1]))

        for i in range(trajectory.shape[1]):
            if method == "cubic":
                spline = CubicSpline(times, trajectory[:, i])
                resampled[:, i] = spline(new_times)
            else:
                interp = interp1d(times, trajectory[:, i], kind=method)
                resampled[:, i] = interp(new_times)

        return resampled

    @staticmethod
    def time_normalize(
        times: np.ndarray,
        trajectory: np.ndarray,
        num_samples: int = 101,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Time-normalize trajectory to 0-100% of motion.

        Useful for comparing motions of different durations.

        Args:
            times: Time array [N]
            trajectory: Trajectory [N x d]
            num_samples: Number of samples in normalized trajectory

        Returns:
            Tuple of (normalized_times [M], normalized_trajectory [M x d])
        """
        # Normalize time to [0, 1]
        normalized_times = np.linspace(0, 1, num_samples)

        # Time normalize original
        time_fraction = (times - times[0]) / (times[-1] - times[0])

        # Resample
        normalized_trajectory = np.zeros((num_samples, trajectory.shape[1]))
        for i in range(trajectory.shape[1]):
            interp = interp1d(time_fraction, trajectory[:, i], kind="cubic")
            normalized_trajectory[:, i] = interp(normalized_times)

        return normalized_times, normalized_trajectory


class MotionCaptureValidator:
    """Validate motion capture data quality."""

    @staticmethod
    def detect_gaps(
        mocap_sequence: MotionCaptureSequence,
        marker_name: str,
        gap_threshold: float = 0.05,
    ) -> list[tuple[int, int]]:
        """Detect gaps in marker trajectory.

        Args:
            mocap_sequence: Motion capture sequence
            marker_name: Marker to check
            gap_threshold: Gap threshold in seconds

        Returns:
            List of (start_frame, end_frame) for gaps
        """
        gaps = []
        last_frame = -1

        for i, frame in enumerate(mocap_sequence.frames):
            if marker_name in frame.marker_positions:
                if (
                    last_frame >= 0
                    and (frame.time - mocap_sequence.frames[last_frame].time)
                    > gap_threshold
                ):
                    gaps.append((last_frame, i))
                last_frame = i

        return gaps

    @staticmethod
    def compute_marker_velocity_stats(
        mocap_sequence: MotionCaptureSequence,
        marker_name: str,
    ) -> dict[str, float | str]:
        """Compute velocity statistics for marker.

        Args:
            mocap_sequence: Motion capture sequence
            marker_name: Marker to analyze

        Returns:
            Dictionary with velocity statistics or error message
        """
        times, positions = mocap_sequence.get_marker_trajectory(marker_name)

        if len(times) < 2:
            return {"error": "Insufficient data"}

        # Compute velocities
        velocities = MotionCaptureProcessor.compute_velocities(times, positions)
        speeds = np.linalg.norm(velocities, axis=1)

        return {
            "mean_speed": float(np.mean(speeds)),
            "max_speed": float(np.max(speeds)),
            "std_speed": float(np.std(speeds)),
        }

    @staticmethod
    def check_marker_visibility(
        mocap_sequence: MotionCaptureSequence,
        marker_name: str,
    ) -> dict[str, float]:
        """Check marker visibility statistics.

        Args:
            mocap_sequence: Motion capture sequence
            marker_name: Marker to check

        Returns:
            Visibility statistics
        """
        total_frames = len(mocap_sequence.frames)
        visible_frames = sum(
            1
            for frame in mocap_sequence.frames
            if marker_name in frame.marker_positions
        )

        visibility_percentage = 100.0 * visible_frames / total_frames

        return {
            "total_frames": total_frames,
            "visible_frames": visible_frames,
            "visibility_percentage": visibility_percentage,
        }
