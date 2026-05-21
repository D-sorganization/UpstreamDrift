from __future__ import annotations

import csv

import mujoco
import numpy as np

from ._id_core import InverseDynamicsSolver
from ._id_models import InverseDynamicsResult
from .kinematic_forces import KinematicForceAnalyzer


def _validate_inverse_dynamics_export_inputs(
    times: np.ndarray,
    results: list[InverseDynamicsResult],
) -> int:
    """Validate inputs for inverse dynamics CSV export.

    Args:
        times: Time array [N].
        results: List of InverseDynamicsResult.

    Returns:
        Number of joints (nv) across all results.

    Raises:
        TypeError: If inputs are wrong type.
        ValueError: If inputs are empty or inconsistent.
    """
    if not isinstance(times, np.ndarray):
        raise TypeError(f"times must be numpy array, got {type(times).__name__}")

    if not isinstance(results, list):
        raise TypeError(f"results must be list, got {type(results).__name__}")

    if len(results) == 0:
        raise ValueError("Cannot export empty results list")

    if len(times) != len(results):
        raise ValueError(
            f"Length mismatch: times has {len(times)} elements, "
            f"results has {len(results)} elements"
        )

    for i, result in enumerate(results):
        if not isinstance(result, InverseDynamicsResult):
            raise TypeError(
                f"results[{i}] is {type(result).__name__}, "
                f"expected InverseDynamicsResult"
            )

    nv = len(results[0].joint_torques)
    for i, result in enumerate(results):
        if len(result.joint_torques) != nv:
            raise ValueError(
                f"Inconsistent joint count: results[0] has {nv} joints, "
                f"results[{i}] has {len(result.joint_torques)} joints"
            )
    return nv


def _build_inverse_dynamics_csv_row(
    result: InverseDynamicsResult, time_val: float, nv: int
) -> list[float]:
    """Build a single CSV row from an inverse dynamics result.

    Args:
        result: Single timestep result.
        time_val: Time value for this row.
        nv: Number of joints.

    Returns:
        List of float values for the CSV row.
    """
    if result is None:
        raise ValueError("result must be provided")
    row: list[float] = [time_val]
    for i in range(nv):
        row.append(result.joint_torques[i])
        row.append(
            result.inertial_torques[i] if result.inertial_torques is not None else 0.0
        )
        row.append(
            result.coriolis_torques[i] if result.coriolis_torques is not None else 0.0
        )
        row.append(
            result.gravity_torques[i] if result.gravity_torques is not None else 0.0
        )
    row.append(result.residual_norm)
    return row


def export_inverse_dynamics_to_csv(
    times: np.ndarray,
    results: list[InverseDynamicsResult],
    filepath: str,
) -> None:
    """Export inverse dynamics results to CSV.

    Args:
        times: Time array [N]
        results: List of InverseDynamicsResult
        filepath: Output CSV path

    Raises:
        ValueError: If times and results have mismatched lengths
        TypeError: If results contains non-InverseDynamicsResult items
        ValueError: If results list is empty

    Note:
        FIXED per Assessment A Finding A-007: Added comprehensive input
        validation to prevent malformed CSV output and silent failures.
    """
    if times is None:
        raise ValueError("times must be provided")
    nv = _validate_inverse_dynamics_export_inputs(times, results)

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)

        # Header
        header = ["time"]
        for i in range(nv):
            header.extend(
                [f"torque_{i}", f"inertial_{i}", f"coriolis_{i}", f"gravity_{i}"],
            )
        header.append("residual_norm")
        writer.writerow(header)

        # Data rows
        for t, result in zip(times, results, strict=False):
            writer.writerow(_build_inverse_dynamics_csv_row(result, t, nv))


class InverseDynamicsAnalyzer:
    """High-level analyzer combining inverse dynamics and kinematic forces.

    This class provides the complete analysis pipeline for understanding
    swing dynamics from motion capture data.
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Initialize analyzer.

        Args:
            model: MuJoCo model
            data: MuJoCo data
        """
        if model is None:
            raise ValueError("model must be provided")
        self.id_solver = InverseDynamicsSolver(model, data)
        self.kin_analyzer = KinematicForceAnalyzer(model, data)

    def analyze_captured_motion(
        self,
        times: np.ndarray,
        positions: np.ndarray,
        velocities: np.ndarray,
        accelerations: np.ndarray,
    ) -> dict:
        """Complete analysis of captured motion.

        This is the main method for analyzing motion capture data.
        Computes both kinematic forces (Coriolis, centrifugal) and
        inverse dynamics (required torques).

        Args:
            times: Time array [N]
            positions: Joint positions [N x nv]
            velocities: Joint velocities [N x nv]
            accelerations: Joint accelerations [N x nv]

        Returns:
            Dictionary with comprehensive analysis
        """
        # Kinematic force analysis
        if times is None:
            raise ValueError("times must be provided")
        kinematic_forces = self.kin_analyzer.analyze_trajectory(
            times,
            positions,
            velocities,
            accelerations,
        )

        # Inverse dynamics
        id_results = self.id_solver.solve_inverse_dynamics_trajectory(
            times,
            positions,
            velocities,
            accelerations,
        )

        # Aggregate statistics
        peak_coriolis_power = 0.0
        max_joint_torque = 0.0

        for kf in kinematic_forces:
            peak_coriolis_power = max(peak_coriolis_power, abs(kf.coriolis_power))

        for id_res in id_results:
            max_joint_torque = max(
                max_joint_torque,
                np.max(np.abs(id_res.joint_torques)),
            )

        return {
            "kinematic_forces": kinematic_forces,
            "inverse_dynamics": id_results,
            "statistics": {
                "peak_coriolis_power": peak_coriolis_power,
                "max_joint_torque": max_joint_torque,
                "duration": times[-1] - times[0],
                "num_frames": len(times),
            },
        }

    def compare_swings(self, swing1_data: dict, swing2_data: dict) -> dict:
        """Compare two swing analyses.

        Args:
            swing1_data: First swing analysis
            swing2_data: Second swing analysis

        Returns:
            Comparison metrics
        """
        if swing1_data is None:
            raise ValueError("swing1_data must be provided")
        stats1 = swing1_data["statistics"]
        stats2 = swing2_data["statistics"]

        return {
            "coriolis_power_diff": stats2["peak_coriolis_power"]
            - stats1["peak_coriolis_power"],
            "torque_diff": stats2["max_joint_torque"] - stats1["max_joint_torque"],
            "duration_diff": stats2["duration"] - stats1["duration"],
        }
