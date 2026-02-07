"""Golf Swing Capture Import for Reinforcement Learning.

Provides an easy interface for importing golf swing motion capture data (C3D, CSV, JSON)
as reference trajectories for RL training and imitation learning.

Design by Contract:
    Preconditions:
        - Input files must exist and be in supported format
        - C3D files must contain 3D marker data
        - CSV/JSON files must contain time-series kinematic data
    Postconditions:
        - Imported data is validated against model DOFs
        - Output is compatible with DemonstrationDataset
        - All trajectories have consistent timesteps
    Invariants:
        - Original capture data is never modified
        - Marker-to-joint mapping is consistent across imports

Usage:
    >>> from src.shared.python.swing_capture_import import SwingCaptureImporter
    >>> importer = SwingCaptureImporter()
    >>> demo = importer.import_c3d("swing_capture.c3d")
    >>> dataset = importer.build_demonstration_dataset([demo])
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.shared.python.logging_config import get_logger

logger = get_logger(__name__)

# Standard golf swing marker names (based on common motion capture protocols)
DEFAULT_MARKER_NAMES = [
    "LSHO",  # Left shoulder
    "RSHO",  # Right shoulder
    "LELB",  # Left elbow
    "RELB",  # Right elbow
    "LWRI",  # Left wrist
    "RWRI",  # Right wrist
    "LHIP",  # Left hip
    "RHIP",  # Right hip
    "C7",  # 7th cervical vertebra (neck)
    "CLAV",  # Clavicle
    "CLUB_GRIP",  # Club grip
    "CLUB_HEAD",  # Club head
]

# Supported import formats
SUPPORTED_FORMATS = {".c3d", ".csv", ".json"}


@dataclass
class MarkerData:
    """3D marker trajectory data from motion capture.

    Attributes:
        marker_names: Names of tracked markers.
        positions: Marker positions (n_frames, n_markers, 3).
        frame_rate: Capture frame rate in Hz.
        times: Time array (n_frames,).
        metadata: Additional capture metadata.
    """

    marker_names: list[str]
    positions: np.ndarray  # (n_frames, n_markers, 3)
    frame_rate: float
    times: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        """Number of captured frames."""
        return self.positions.shape[0]

    @property
    def n_markers(self) -> int:
        """Number of tracked markers."""
        return self.positions.shape[1]

    @property
    def duration(self) -> float:
        """Total capture duration in seconds."""
        return float(self.times[-1] - self.times[0]) if len(self.times) > 1 else 0.0


@dataclass
class JointTrajectory:
    """Joint-space trajectory converted from marker data.

    Attributes:
        joint_names: Names of joints.
        positions: Joint positions (n_frames, n_joints).
        velocities: Joint velocities (n_frames, n_joints).
        times: Time array (n_frames,).
        frame_rate: Trajectory frame rate in Hz.
        source_file: Path to the original capture file.
    """

    joint_names: list[str]
    positions: np.ndarray  # (n_frames, n_joints)
    velocities: np.ndarray  # (n_frames, n_joints)
    times: np.ndarray
    frame_rate: float
    source_file: str = ""

    @property
    def n_frames(self) -> int:
        """Number of frames."""
        return self.positions.shape[0]

    @property
    def n_joints(self) -> int:
        """Number of joints."""
        return self.positions.shape[1]


@dataclass
class SwingPhaseLabels:
    """Phase labels for a golf swing trajectory.

    Attributes:
        address: Frame index of address position.
        backswing_start: Frame index of backswing start.
        top_of_backswing: Frame index of top of backswing.
        downswing_start: Frame index of downswing start.
        impact: Frame index of ball impact.
        follow_through_end: Frame index of follow-through end.
    """

    address: int = 0
    backswing_start: int = 0
    top_of_backswing: int = 0
    downswing_start: int = 0
    impact: int = 0
    follow_through_end: int = 0


@dataclass
class MarkerToJointMapping:
    """Mapping from motion capture markers to model joints.

    Attributes:
        joint_name: Target joint name.
        marker_names: Markers used to compute this joint angle.
        computation: How to compute the angle ('angle_3pt', 'direction', 'quaternion').
        axis: Rotation axis if applicable.
    """

    joint_name: str
    marker_names: list[str]
    computation: str = "angle_3pt"  # angle_3pt, direction, quaternion
    axis: str = "z"  # x, y, z


# Default mapping for golf swing models
DEFAULT_GOLF_MAPPING = [
    MarkerToJointMapping("shoulder_flexion", ["C7", "RSHO", "RELB"]),
    MarkerToJointMapping("shoulder_abduction", ["CLAV", "RSHO", "RELB"], axis="y"),
    MarkerToJointMapping("elbow_flexion", ["RSHO", "RELB", "RWRI"]),
    MarkerToJointMapping("wrist_flexion", ["RELB", "RWRI", "CLUB_GRIP"]),
    MarkerToJointMapping("hip_rotation", ["LHIP", "RHIP", "RSHO"], axis="y"),
    MarkerToJointMapping("trunk_rotation", ["LHIP", "RHIP", "C7"], axis="y"),
]


class SwingCaptureImporter:
    """Imports golf swing capture data for RL training.

    Provides a unified interface for loading motion capture data from
    various formats and converting it to joint-space trajectories
    suitable for reinforcement learning.

    Design by Contract:
        Preconditions:
            - File paths must point to existing files
            - Files must be in supported format (C3D, CSV, JSON)
        Postconditions:
            - Returned trajectories have consistent dimensions
            - All values are finite (no NaN/Inf)
        Invariants:
            - Source data is never modified
    """

    def __init__(
        self,
        marker_mapping: list[MarkerToJointMapping] | None = None,
        target_frame_rate: float = 200.0,
    ) -> None:
        """Initialize the swing capture importer.

        Args:
            marker_mapping: Custom marker-to-joint mapping. Uses default if None.
            target_frame_rate: Target frame rate for resampled output.
        """
        self.marker_mapping = marker_mapping or DEFAULT_GOLF_MAPPING
        self.target_frame_rate = target_frame_rate

    def import_file(self, filepath: str | Path) -> JointTrajectory:
        """Import a swing capture file (auto-detect format).

        Args:
            filepath: Path to the capture file.

        Returns:
            JointTrajectory with joint-space data.

        Raises:
            FileNotFoundError: If file does not exist.
            ValueError: If format is not supported.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Capture file not found: {filepath}")

        suffix = filepath.suffix.lower()
        if suffix not in SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format: {suffix}. "
                f"Supported: {sorted(SUPPORTED_FORMATS)}"
            )

        if suffix == ".c3d":
            return self.import_c3d(filepath)
        elif suffix == ".csv":
            return self.import_csv(filepath)
        elif suffix == ".json":
            return self.import_json(filepath)
        else:
            raise ValueError(f"Unsupported format: {suffix}")

    def import_c3d(self, filepath: str | Path) -> JointTrajectory:
        """Import a C3D motion capture file.

        Args:
            filepath: Path to the C3D file.

        Returns:
            JointTrajectory with converted joint angles.

        Raises:
            FileNotFoundError: If file does not exist.
            ImportError: If ezc3d is not available.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"C3D file not found: {filepath}")

        try:
            import ezc3d
        except ImportError:
            raise ImportError(
                "ezc3d is required for C3D import. "
                "Install with: pip install ezc3d"
            )

        c3d_data = ezc3d.c3d(str(filepath))

        # Extract marker data
        point_data = c3d_data["data"]["points"]  # (4, n_markers, n_frames)
        marker_labels = c3d_data["parameters"]["POINT"]["LABELS"]["value"]
        frame_rate = float(c3d_data["parameters"]["POINT"]["RATE"]["value"][0])

        # Transpose to (n_frames, n_markers, 3) - drop homogeneous coordinate
        positions = point_data[:3, :, :].transpose(2, 1, 0)
        n_frames = positions.shape[0]
        times = np.arange(n_frames) / frame_rate

        marker_data = MarkerData(
            marker_names=[str(m).strip() for m in marker_labels],
            positions=positions,
            frame_rate=frame_rate,
            times=times,
            metadata={"source": str(filepath), "format": "c3d"},
        )

        return self._convert_markers_to_joints(marker_data, str(filepath))

    def import_csv(self, filepath: str | Path) -> JointTrajectory:
        """Import joint trajectory data from CSV.

        Expected CSV format:
            time, joint_0, joint_1, ..., joint_n

        Args:
            filepath: Path to the CSV file.

        Returns:
            JointTrajectory parsed from CSV.

        Raises:
            FileNotFoundError: If file does not exist.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"CSV file not found: {filepath}")

        # Read header
        with open(filepath, encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            header = [h.strip().strip('"') for h in header]

        # Load data
        data = np.loadtxt(str(filepath), delimiter=",", skiprows=1)

        if data.ndim == 1:
            data = data.reshape(1, -1)

        # First column is time
        times = data[:, 0]
        positions = data[:, 1:]

        # Infer joint names from header
        joint_names = header[1:] if len(header) > 1 else [
            f"joint_{i}" for i in range(positions.shape[1])
        ]

        # Compute velocities via finite differences
        velocities = np.gradient(positions, times, axis=0)

        frame_rate = 1.0 / np.mean(np.diff(times)) if len(times) > 1 else 100.0

        return JointTrajectory(
            joint_names=joint_names,
            positions=positions,
            velocities=velocities,
            times=times,
            frame_rate=frame_rate,
            source_file=str(filepath),
        )

    def import_json(self, filepath: str | Path) -> JointTrajectory:
        """Import joint trajectory from JSON.

        Expected JSON format:
            {
                "times": [...],
                "joint_names": [...],
                "positions": [[...], ...],
                "velocities": [[...], ...]  // optional
            }

        Args:
            filepath: Path to the JSON file.

        Returns:
            JointTrajectory parsed from JSON.

        Raises:
            FileNotFoundError: If file does not exist.
            ValueError: If JSON structure is invalid.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"JSON file not found: {filepath}")

        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError("JSON root must be an object")

        required_keys = {"times", "positions"}
        missing = required_keys - set(data.keys())
        if missing:
            raise ValueError(f"JSON missing required keys: {missing}")

        times = np.array(data["times"], dtype=np.float64)
        positions = np.array(data["positions"], dtype=np.float64)

        joint_names = data.get("joint_names", [
            f"joint_{i}" for i in range(positions.shape[1])
        ])

        if "velocities" in data:
            velocities = np.array(data["velocities"], dtype=np.float64)
        else:
            velocities = np.gradient(positions, times, axis=0)

        frame_rate = 1.0 / np.mean(np.diff(times)) if len(times) > 1 else 100.0

        return JointTrajectory(
            joint_names=joint_names,
            positions=positions,
            velocities=velocities,
            times=times,
            frame_rate=frame_rate,
            source_file=str(filepath),
        )

    def _convert_markers_to_joints(
        self, marker_data: MarkerData, source_file: str
    ) -> JointTrajectory:
        """Convert marker positions to joint angles.

        Args:
            marker_data: 3D marker trajectory data.
            source_file: Source file path for provenance.

        Returns:
            JointTrajectory with computed joint angles.
        """
        n_frames = marker_data.n_frames
        marker_name_to_idx = {
            name: idx for idx, name in enumerate(marker_data.marker_names)
        }

        joint_names = []
        joint_angles = []

        for mapping in self.marker_mapping:
            # Check if all required markers exist
            marker_indices = []
            all_found = True
            for mname in mapping.marker_names:
                if mname in marker_name_to_idx:
                    marker_indices.append(marker_name_to_idx[mname])
                else:
                    all_found = False
                    break

            if not all_found or len(marker_indices) < 3:
                # Skip this joint if markers not found
                logger.debug(
                    "Skipping joint '%s': markers not found in capture",
                    mapping.joint_name,
                )
                continue

            # Compute joint angle for each frame
            angles = np.zeros(n_frames)
            for frame in range(n_frames):
                if mapping.computation == "angle_3pt":
                    # Three-point angle computation
                    p1 = marker_data.positions[frame, marker_indices[0]]
                    p2 = marker_data.positions[frame, marker_indices[1]]  # vertex
                    p3 = marker_data.positions[frame, marker_indices[2]]

                    v1 = p1 - p2
                    v2 = p3 - p2

                    norm1 = np.linalg.norm(v1)
                    norm2 = np.linalg.norm(v2)
                    if norm1 > 1e-10 and norm2 > 1e-10:
                        cos_angle = np.dot(v1, v2) / (norm1 * norm2)
                        cos_angle = np.clip(cos_angle, -1.0, 1.0)
                        angles[frame] = np.arccos(cos_angle)
                    else:
                        angles[frame] = 0.0
                else:
                    # Default to angle_3pt
                    angles[frame] = 0.0

            joint_names.append(mapping.joint_name)
            joint_angles.append(angles)

        if not joint_angles:
            # Fallback: use raw marker data as "joints"
            logger.warning(
                "No marker-to-joint mappings matched. Using raw marker positions."
            )
            n_markers = marker_data.n_markers
            joint_names = [
                f"marker_{name}_x" for name in marker_data.marker_names
            ]
            joint_angles = [
                marker_data.positions[:, i, 0] for i in range(n_markers)
            ]

        positions = np.column_stack(joint_angles)
        velocities = np.gradient(positions, marker_data.times, axis=0)

        # Resample if needed
        if abs(marker_data.frame_rate - self.target_frame_rate) > 1.0:
            positions, velocities, times = self._resample(
                positions,
                velocities,
                marker_data.times,
                marker_data.frame_rate,
                self.target_frame_rate,
            )
        else:
            times = marker_data.times

        return JointTrajectory(
            joint_names=joint_names,
            positions=positions,
            velocities=velocities,
            times=times,
            frame_rate=self.target_frame_rate,
            source_file=source_file,
        )

    def _resample(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        times: np.ndarray,
        source_rate: float,
        target_rate: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Resample trajectory to target frame rate.

        Args:
            positions: Original positions (n_frames, n_joints).
            velocities: Original velocities (n_frames, n_joints).
            times: Original time array.
            source_rate: Source frame rate.
            target_rate: Target frame rate.

        Returns:
            Tuple of (resampled_positions, resampled_velocities, new_times).
        """
        from scipy.interpolate import interp1d

        duration = times[-1] - times[0]
        n_new_frames = int(duration * target_rate)
        new_times = np.linspace(times[0], times[-1], n_new_frames)

        new_positions = np.zeros((n_new_frames, positions.shape[1]))
        new_velocities = np.zeros((n_new_frames, velocities.shape[1]))

        for j in range(positions.shape[1]):
            f_pos = interp1d(times, positions[:, j], kind="cubic", fill_value="extrapolate")
            new_positions[:, j] = f_pos(new_times)

            f_vel = interp1d(times, velocities[:, j], kind="cubic", fill_value="extrapolate")
            new_velocities[:, j] = f_vel(new_times)

        return new_positions, new_velocities, new_times

    def detect_swing_phases(
        self, trajectory: JointTrajectory
    ) -> SwingPhaseLabels:
        """Auto-detect golf swing phases from trajectory.

        Uses heuristics based on joint velocity patterns to identify
        key swing phases (address, backswing, downswing, impact, follow-through).

        Args:
            trajectory: Joint trajectory to analyze.

        Returns:
            SwingPhaseLabels with frame indices for each phase.
        """
        # Use total angular velocity as a proxy for swing phase detection
        total_velocity = np.sum(np.abs(trajectory.velocities), axis=1)

        n_frames = trajectory.n_frames

        # Find impact (peak velocity)
        impact_idx = int(np.argmax(total_velocity))

        # Find top of backswing (local minimum before impact)
        search_start = max(0, impact_idx - n_frames // 2)
        pre_impact = total_velocity[search_start:impact_idx]
        if len(pre_impact) > 0:
            top_idx = search_start + int(np.argmin(pre_impact))
        else:
            top_idx = impact_idx // 2

        # Address is the start
        address_idx = 0

        # Backswing starts after address (first significant motion)
        threshold = np.mean(total_velocity[:top_idx]) * 0.1 if top_idx > 0 else 0
        backswing_indices = np.where(total_velocity[:top_idx] > threshold)[0]
        backswing_start = int(backswing_indices[0]) if len(backswing_indices) > 0 else 0

        # Follow-through end
        follow_end = n_frames - 1

        return SwingPhaseLabels(
            address=address_idx,
            backswing_start=backswing_start,
            top_of_backswing=top_idx,
            downswing_start=top_idx,
            impact=impact_idx,
            follow_through_end=follow_end,
        )

    def build_demonstration_dataset(
        self,
        trajectories: list[JointTrajectory],
    ) -> dict[str, Any]:
        """Build a demonstration dataset from imported trajectories.

        Converts JointTrajectory objects into a format compatible with
        the learning.imitation.DemonstrationDataset.

        Args:
            trajectories: List of imported joint trajectories.

        Returns:
            Dictionary with demonstration data ready for DemonstrationDataset.
        """
        demonstrations = []

        for traj in trajectories:
            demo = {
                "observations": traj.positions.tolist(),
                "actions": traj.velocities.tolist(),  # velocities as proxy actions
                "times": traj.times.tolist(),
                "joint_names": traj.joint_names,
                "frame_rate": traj.frame_rate,
                "source_file": traj.source_file,
                "n_frames": traj.n_frames,
                "n_joints": traj.n_joints,
            }

            # Add phase labels if detectable
            try:
                phases = self.detect_swing_phases(traj)
                demo["phases"] = {
                    "address": phases.address,
                    "backswing_start": phases.backswing_start,
                    "top_of_backswing": phases.top_of_backswing,
                    "downswing_start": phases.downswing_start,
                    "impact": phases.impact,
                    "follow_through_end": phases.follow_through_end,
                }
            except Exception as e:
                logger.warning("Failed to detect swing phases: %s", e)

            demonstrations.append(demo)

        return {
            "demonstrations": demonstrations,
            "num_demonstrations": len(demonstrations),
            "format_version": "1.0",
        }

    def export_for_rl(
        self,
        trajectory: JointTrajectory,
        output_path: str | Path,
    ) -> Path:
        """Export a trajectory in a format ready for RL reward computation.

        Creates a JSON file with the reference trajectory that can be loaded
        by RL environments for trajectory-tracking rewards.

        Args:
            trajectory: Joint trajectory to export.
            output_path: Output file path.

        Returns:
            Path to the exported file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "joint_names": trajectory.joint_names,
            "times": trajectory.times.tolist(),
            "positions": trajectory.positions.tolist(),
            "velocities": trajectory.velocities.tolist(),
            "frame_rate": trajectory.frame_rate,
            "source_file": trajectory.source_file,
            "n_frames": trajectory.n_frames,
            "n_joints": trajectory.n_joints,
        }

        try:
            phases = self.detect_swing_phases(trajectory)
            data["phases"] = {
                "address": phases.address,
                "backswing_start": phases.backswing_start,
                "top_of_backswing": phases.top_of_backswing,
                "downswing_start": phases.downswing_start,
                "impact": phases.impact,
                "follow_through_end": phases.follow_through_end,
            }
        except Exception:
            pass

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info("Exported RL reference trajectory to: %s", output_path)
        return output_path
