"""Export trajectory data for use in MuJoCo, Drake, and other engines.

Provides export functionality to various formats:
- MuJoCo: JSON and MJCF keyframe format
- Drake: YAML trajectory format
- OpenSim: STO motion file format
- Generic: CSV and NPZ formats
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

from motion_training.club_trajectory_parser import ClubTrajectory
from motion_training.dual_hand_ik_solver import TrajectoryIKResult


@dataclass
class ExportMetadata:
    """Metadata for exported trajectory."""

    source_file: str = ""
    model_name: str = "golfer"
    num_frames: int = 0
    duration: float = 0.0
    timestep: float = 0.0
    num_dof: int = 0
    convergence_rate: float = 0.0
    export_format: str = ""
    coordinate_system: str = "XYZ"


class TrajectoryExporter:
    """Export trajectories to various formats for physics engines."""

    SUPPORTED_FORMATS = ["mujoco", "drake", "opensim", "csv", "npz", "json"]

    def __init__(
        self,
        ik_result: TrajectoryIKResult,
        trajectory: ClubTrajectory | None = None,
        model_name: str = "golfer",
    ) -> None:
        """Initialize exporter.

        Args:
            ik_result: IK solving results with joint configurations
            trajectory: Optional club trajectory for additional data
            model_name: Name of the model
        """
        self.ik_result = ik_result
        self.trajectory = trajectory
        self.model_name = model_name

        self.q_traj = ik_result.q_trajectory
        self.times = np.array(ik_result.times)
        self.num_frames = len(self.times)
        self.num_dof = self.q_traj.shape[1] if len(self.q_traj) > 0 else 0

        # Compute timestep
        if self.num_frames > 1:
            self.timestep = np.mean(np.diff(self.times))
        else:
            self.timestep = 0.001

    def export(
        self,
        output_path: str | Path,
        format: str = "mujoco",
        **kwargs,
    ) -> Path:
        """Export trajectory to specified format.

        Args:
            output_path: Output file path
            format: Export format (mujoco, drake, opensim, csv, npz, json)
            **kwargs: Format-specific options

        Returns:
            Path to exported file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        format = format.lower()
        if format not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format: {format}. Supported: {self.SUPPORTED_FORMATS}"
            )

        exporters = {
            "mujoco": self._export_mujoco,
            "drake": self._export_drake,
            "opensim": self._export_opensim,
            "csv": self._export_csv,
            "npz": self._export_npz,
            "json": self._export_json,
        }

        return exporters[format](output_path, **kwargs)

    def _create_metadata(self, format: str) -> ExportMetadata:
        """Create export metadata."""
        return ExportMetadata(
            source_file=str(self.trajectory.events if self.trajectory else ""),
            model_name=self.model_name,
            num_frames=self.num_frames,
            duration=(
                float(self.times[-1] - self.times[0]) if len(self.times) > 0 else 0.0
            ),
            timestep=float(self.timestep),
            num_dof=self.num_dof,
            convergence_rate=self.ik_result.convergence_rate,
            export_format=format,
        )

    def _export_mujoco(self, output_path: Path, **kwargs) -> Path:
        """Export for MuJoCo.

        Creates a JSON file with trajectory data that can be loaded
        by MuJoCo simulation scripts.

        Format:
        {
            "metadata": {...},
            "keyframes": [
                {"time": 0.0, "qpos": [...], "qvel": [...]},
                ...
            ]
        }
        """
        # Compute velocities via finite differences
        qvel = np.zeros_like(self.q_traj)
        if self.num_frames > 1:
            for i in range(1, self.num_frames):
                dt = self.times[i] - self.times[i - 1]
                if dt > 0:
                    qvel[i] = (self.q_traj[i] - self.q_traj[i - 1]) / dt

        # Build keyframes
        keyframes = []
        for i in range(self.num_frames):
            keyframes.append(
                {
                    "time": float(self.times[i]),
                    "qpos": self.q_traj[i].tolist(),
                    "qvel": qvel[i].tolist(),
                }
            )

        # Include club trajectory if available
        club_data = None
        if self.trajectory:
            club_data = {
                "grip_positions": self.trajectory.grip_positions.tolist(),
                "club_face_positions": self.trajectory.club_face_positions.tolist(),
                "events": {
                    "address": self.trajectory.events.address,
                    "top": self.trajectory.events.top,
                    "impact": self.trajectory.events.impact,
                    "finish": self.trajectory.events.finish,
                    "club_head_speed_mph": self.trajectory.events.club_head_speed,
                },
            }

        data = {
            "metadata": asdict(self._create_metadata("mujoco")),
            "keyframes": keyframes,
            "club_trajectory": club_data,
        }

        output_path = output_path.with_suffix(".json")
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        return output_path

    def _export_drake(self, output_path: Path, **kwargs) -> Path:
        """Export for Drake.

        Creates a YAML file compatible with Drake's trajectory utilities.
        """
        import yaml

        # Build trajectory data
        trajectory_data = {
            "metadata": asdict(self._create_metadata("drake")),
            "trajectory": {
                "type": "PiecewisePolynomial",
                "order": 1,  # Linear interpolation
                "times": self.times.tolist(),
                "values": self.q_traj.T.tolist(),  # nq x T format
            },
        }

        # Include club path if available
        if self.trajectory:
            trajectory_data["club_path"] = {
                "times": self.times.tolist(),
                "grip_positions": self.trajectory.grip_positions.tolist(),
            }

        output_path = output_path.with_suffix(".yaml")
        with open(output_path, "w") as f:
            yaml.dump(trajectory_data, f, default_flow_style=False)

        return output_path

    def _export_opensim(self, output_path: Path, **kwargs) -> Path:
        """Export for OpenSim.

        Creates an STO (Storage) file format used by OpenSim.
        """
        output_path = output_path.with_suffix(".sto")

        # Generate column names (OpenSim expects specific naming)
        joint_names = kwargs.get("joint_names", [f"q{i}" for i in range(self.num_dof)])

        with open(output_path, "w") as f:
            # Header
            f.write(f"{self.model_name}_motion\n")
            f.write("version=1\n")
            f.write(f"nRows={self.num_frames}\n")
            f.write(f"nColumns={self.num_dof + 1}\n")
            f.write("inDegrees=yes\n")
            f.write("endheader\n")

            # Column labels
            f.write("time\t" + "\t".join(joint_names) + "\n")

            # Data (convert to degrees for OpenSim)
            for i in range(self.num_frames):
                t = self.times[i]
                q_deg = np.rad2deg(self.q_traj[i])
                values = [f"{v:.6f}" for v in q_deg]
                f.write(f"{t:.6f}\t" + "\t".join(values) + "\n")

        return output_path

    def _export_csv(self, output_path: Path, **kwargs) -> Path:
        """Export as CSV."""
        output_path = output_path.with_suffix(".csv")

        import csv

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)

            # Header
            header = ["time"] + [f"q{i}" for i in range(self.num_dof)]
            if kwargs.get("include_errors", True):
                header += ["left_error", "right_error"]
            writer.writerow(header)

            # Data
            for i in range(self.num_frames):
                row = [self.times[i]] + list(self.q_traj[i])
                if kwargs.get("include_errors", True):
                    row += [
                        self.ik_result.left_hand_errors[i],
                        self.ik_result.right_hand_errors[i],
                    ]
                writer.writerow(row)

        return output_path

    def _export_npz(self, output_path: Path, **kwargs) -> Path:
        """Export as NumPy NPZ archive."""
        output_path = output_path.with_suffix(".npz")

        save_dict = {
            "q": self.q_traj,
            "times": self.times,
            "left_errors": np.array(self.ik_result.left_hand_errors),
            "right_errors": np.array(self.ik_result.right_hand_errors),
            "convergence_rate": self.ik_result.convergence_rate,
            "timestep": self.timestep,
        }

        if self.trajectory:
            save_dict["grip_positions"] = self.trajectory.grip_positions
            save_dict["club_face_positions"] = self.trajectory.club_face_positions

        np.savez(output_path, **save_dict)

        return output_path

    def _export_json(self, output_path: Path, **kwargs) -> Path:
        """Export as generic JSON."""
        output_path = output_path.with_suffix(".json")

        data = {
            "metadata": asdict(self._create_metadata("json")),
            "trajectory": {
                "times": self.times.tolist(),
                "joint_positions": self.q_traj.tolist(),
                "left_hand_errors": self.ik_result.left_hand_errors,
                "right_hand_errors": self.ik_result.right_hand_errors,
            },
        }

        if self.trajectory:
            data["club"] = {
                "grip_positions": self.trajectory.grip_positions.tolist(),
                "club_face_positions": self.trajectory.club_face_positions.tolist(),
            }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        return output_path

    def export_all(
        self,
        output_dir: str | Path,
        base_name: str = "trajectory",
    ) -> dict[str, Path]:
        """Export to all supported formats.

        Args:
            output_dir: Output directory
            base_name: Base filename (without extension)

        Returns:
            Dictionary mapping format to output path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {}
        for fmt in self.SUPPORTED_FORMATS:
            try:
                path = self.export(output_dir / base_name, format=fmt)
                results[fmt] = path
            except Exception as e:
                print(f"Warning: Failed to export {fmt}: {e}")

        return results


def export_for_mujoco(
    ik_result: TrajectoryIKResult,
    output_path: str | Path,
    trajectory: ClubTrajectory | None = None,
) -> Path:
    """Convenience function to export for MuJoCo.

    Args:
        ik_result: IK solving results
        output_path: Output file path
        trajectory: Optional club trajectory

    Returns:
        Path to exported file
    """
    exporter = TrajectoryExporter(ik_result, trajectory)
    return exporter.export(output_path, format="mujoco")


def export_for_drake(
    ik_result: TrajectoryIKResult,
    output_path: str | Path,
    trajectory: ClubTrajectory | None = None,
) -> Path:
    """Convenience function to export for Drake.

    Args:
        ik_result: IK solving results
        output_path: Output file path
        trajectory: Optional club trajectory

    Returns:
        Path to exported file
    """
    exporter = TrajectoryExporter(ik_result, trajectory)
    return exporter.export(output_path, format="drake")


def load_trajectory_from_mujoco_json(
    file_path: str | Path,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Load trajectory from MuJoCo JSON export.

    Args:
        file_path: Path to JSON file

    Returns:
        Tuple of (times, q_trajectory)
    """
    with open(file_path) as f:
        data = json.load(f)

    keyframes = data["keyframes"]
    times = np.array([kf["time"] for kf in keyframes])
    q_traj = np.array([kf["qpos"] for kf in keyframes])

    return times, q_traj
