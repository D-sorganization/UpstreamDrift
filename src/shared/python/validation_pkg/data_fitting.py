"""Data fitting and parameter estimation for golf biomechanics (Guideline A3).

This module implements the A3 pipeline per project design guidelines:
- Fit kinematics to observed trajectories
- Estimate body segment parameters (lengths, mass, inertia)
- Report sensitivity analysis and fit quality metrics

Issue #754: Implements complete A3 model fitting and parameter identification.

Reference: docs/assessments/project_design_guidelines.qmd Section A3

Implementation split across:
- _data_fitting_models.py: data classes
- _data_fitting_solvers.py: InverseKinematicsSolver, ParameterEstimator
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.shared.python.logging_pkg.logging_config import get_logger

# Re-export public names for backward compatibility
from ._data_fitting_models import (
    BodySegmentParams,
    FitResult,
    KinematicState,
    ParameterEstimationReport,
    SensitivityResult,
)
from ._data_fitting_solvers import InverseKinematicsSolver, ParameterEstimator

logger = get_logger(__name__)


class SensitivityAnalyzer:
    """Perform sensitivity analysis on model parameters.

    Computes how output metrics vary with parameter changes.
    """

    def __init__(
        self,
        perturbation_size: float = 0.01,
    ) -> None:
        """Initialize sensitivity analyzer.

        Args:
            perturbation_size: Fractional perturbation for finite differences
        """
        if not (perturbation_size is not None):
            raise ValueError("perturbation_size must be provided")
        if not (perturbation_size is not None):
            raise ValueError("perturbation_size must be provided")
        self.perturbation_size = perturbation_size

    def compute_sensitivity(
        self,
        model_func: Any,
        parameter_name: str,
        nominal_value: float,
        output_metric: str,
    ) -> SensitivityResult:
        """Compute sensitivity of output to parameter.

        Uses central finite differences to estimate sensitivity.

        Args:
            model_func: Function that takes parameters and returns outputs
            parameter_name: Name of parameter to vary
            nominal_value: Nominal parameter value
            output_metric: Name of output metric to analyze

        Returns:
            SensitivityResult with sensitivity indices.
        """
        if not (parameter_name is not None):
            raise ValueError("parameter_name must be provided")
        if not (parameter_name is not None):
            raise ValueError("parameter_name must be provided")
        delta = nominal_value * self.perturbation_size

        # Perturb up and down
        try:
            output_up = model_func({parameter_name: nominal_value + delta})[
                output_metric
            ]
            output_down = model_func({parameter_name: nominal_value - delta})[
                output_metric
            ]
            output_nominal = model_func({parameter_name: nominal_value})[output_metric]
        except (RuntimeError, ValueError, OSError) as e:
            logger.warning(f"Sensitivity computation failed: {e}")
            return SensitivityResult(
                parameter_name=parameter_name,
                nominal_value=nominal_value,
                sensitivity_index=0.0,
                partial_derivative=0.0,
                confidence_interval=(0.0, 0.0),
                elasticity=0.0,
            )

        # Central difference partial derivative
        partial = (output_up - output_down) / (2 * delta)

        # Elasticity (dimensionless sensitivity)
        elasticity = (
            (partial * nominal_value / output_nominal) if output_nominal != 0 else 0.0
        )

        # Normalized sensitivity index
        output_range = abs(output_up - output_down)
        sensitivity_index = output_range / (2 * delta) if delta != 0 else 0.0

        # Simple confidence interval estimate (approximate)
        ci_half_width = abs(partial) * delta * 2
        ci = (output_nominal - ci_half_width, output_nominal + ci_half_width)

        return SensitivityResult(
            parameter_name=parameter_name,
            nominal_value=nominal_value,
            sensitivity_index=float(sensitivity_index),
            partial_derivative=float(partial),
            confidence_interval=ci,
            elasticity=float(elasticity),
        )

    def sensitivity_report(
        self,
        sensitivities: list[SensitivityResult],
    ) -> dict[str, Any]:
        """Generate sensitivity analysis report.

        Args:
            sensitivities: List of sensitivity results

        Returns:
            Dictionary with summary statistics and rankings.
        """
        if not (sensitivities is not None):
            raise ValueError("sensitivities must be provided")
        if not (sensitivities is not None):
            raise ValueError("sensitivities must be provided")
        if not sensitivities:
            return {"error": "No sensitivity data"}

        # Sort by sensitivity index
        sorted_sens = sorted(
            sensitivities,
            key=lambda s: abs(s.sensitivity_index),
            reverse=True,
        )

        report = {
            "total_parameters": len(sensitivities),
            "most_sensitive": sorted_sens[0].parameter_name if sorted_sens else None,
            "least_sensitive": sorted_sens[-1].parameter_name if sorted_sens else None,
            "rankings": [
                {
                    "rank": i + 1,
                    "parameter": s.parameter_name,
                    "sensitivity_index": s.sensitivity_index,
                    "elasticity": s.elasticity,
                }
                for i, s in enumerate(sorted_sens)
            ],
            "summary_statistics": {
                "mean_sensitivity": float(
                    np.mean([s.sensitivity_index for s in sensitivities])
                ),
                "max_sensitivity": float(
                    max(s.sensitivity_index for s in sensitivities)
                ),
                "mean_elasticity": float(
                    np.mean([abs(s.elasticity) for s in sensitivities])
                ),
            },
        }

        return report


# =============================================================================
# Pose-to-Marker Conversion (completing video_pose_pipeline)
# =============================================================================


def convert_poses_to_markers(
    pose_keypoints: np.ndarray,
    keypoint_names: list[str],
    target_markers: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Convert pose estimation keypoints to biomechanical marker format.

    Maps OpenPose/MediaPipe keypoints to standard marker positions.

    Args:
        pose_keypoints: Keypoint positions [N x 3] or [N x 2]
        keypoint_names: Names of each keypoint
        target_markers: Optional list of target marker names to output

    Returns:
        Tuple of (marker_positions [M x 3], marker_names [M]).
    """
    # Standard mapping from pose estimation to biomechanical markers
    if not (pose_keypoints is not None):
        raise ValueError("pose_keypoints must be provided")
    if not (pose_keypoints is not None):
        raise ValueError("pose_keypoints must be provided")
    pose_to_marker_map = {
        # MediaPipe / OpenPose keypoint names -> Biomechanics marker names
        "left_shoulder": "LSHO",
        "right_shoulder": "RSHO",
        "left_elbow": "LELB",
        "right_elbow": "RELB",
        "left_wrist": "LWRI",
        "right_wrist": "RWRI",
        "left_hip": "LASI",
        "right_hip": "RASI",
        "left_knee": "LKNE",
        "right_knee": "RKNE",
        "left_ankle": "LANK",
        "right_ankle": "RANK",
        # Additional mappings
        "nose": "NOSE",
        "left_ear": "LEAR",
        "right_ear": "REAR",
    }

    # Ensure 3D coordinates
    if pose_keypoints.shape[1] == 2:
        # Add zero z-coordinate for 2D keypoints
        pose_keypoints = np.hstack([pose_keypoints, np.zeros((len(pose_keypoints), 1))])

    # Filter and reorder keypoints
    marker_positions = []
    marker_names = []

    for i, keypoint_name in enumerate(keypoint_names):
        marker_name = pose_to_marker_map.get(keypoint_name.lower())

        if marker_name is None:
            continue

        if target_markers is not None and marker_name not in target_markers:
            continue

        marker_positions.append(pose_keypoints[i])
        marker_names.append(marker_name)

    return np.array(marker_positions), marker_names


# =============================================================================
# Complete A3 Pipeline
# =============================================================================


class A3FittingPipeline:
    """Complete A3 model fitting pipeline.

    Implements the full workflow from motion data to parameter report:
    1. Load motion capture / video pose data
    2. Convert to biomechanical marker format
    3. Fit inverse kinematics
    4. Estimate segment parameters
    5. Perform sensitivity analysis
    6. Generate quality report
    """

    def __init__(
        self,
        anthropometric_model: str = "dempster",
    ) -> None:
        """Initialize A3 pipeline.

        Args:
            anthropometric_model: Model for parameter regression
        """
        if not (anthropometric_model is not None):
            raise ValueError("anthropometric_model must be provided")
        if not (anthropometric_model is not None):
            raise ValueError("anthropometric_model must be provided")
        self.param_estimator = ParameterEstimator(anthropometric_model)
        self.sensitivity_analyzer = SensitivityAnalyzer()

        # Default segment names for golf swing
        self.segment_names = [
            "pelvis",
            "trunk",
            "upper_arm",
            "forearm",
            "hand",
        ]

        logger.info("A3 Fitting Pipeline initialized")

    def fit_from_markers(
        self,
        marker_positions: np.ndarray,
        marker_names: list[str],
        timestamps: np.ndarray,
        subject_mass: float,
        subject_id: str = "unknown",
    ) -> ParameterEstimationReport:
        """Fit parameters from marker data.

        Args:
            marker_positions: Marker positions [frames x markers x 3]
            marker_names: Names of markers
            timestamps: Timestamps for each frame
            subject_mass: Total body mass (kg)
            subject_id: Subject identifier

        Returns:
            Complete ParameterEstimationReport.
        """
        if not (marker_positions is not None):
            raise ValueError("marker_positions must be provided")
        if not (marker_positions is not None):
            raise ValueError("marker_positions must be provided")
        logger.info(
            f"Fitting A3 model for subject '{subject_id}' "
            f"({len(timestamps)} frames, {len(marker_names)} markers)"
        )

        # Convert to kinematic states
        kinematic_data = []
        for i, t in enumerate(timestamps):
            state = KinematicState(
                timestamp=float(t),
                marker_positions=(
                    marker_positions[i] if i < len(marker_positions) else None
                ),
            )
            kinematic_data.append(state)

        # Fit segment parameters
        fit_result = self.param_estimator.fit_parameters_to_kinematics(
            kinematic_data,
            self.segment_names,
            subject_mass,
        )

        # Create segment params from fit result
        segment_params = []
        for segment_name in self.segment_names:
            length_key = f"{segment_name}_length"
            mass_key = f"{segment_name}_mass"

            if length_key in fit_result.parameters:
                params = BodySegmentParams(
                    name=segment_name,
                    length=fit_result.parameters[length_key],
                    mass=fit_result.parameters.get(mass_key, 0.0),
                )
                segment_params.append(params)

        # Sensitivity analysis (placeholder - requires model function)
        sensitivities: list[SensitivityResult] = []
        if not sensitivities:
            logger.warning(
                "Sensitivity analysis is empty (placeholder). "
                "Results lack parameter sensitivity information. "
                "See issue #2170 for implementation tracking."
            )

        # Quality metrics
        quality_metrics = {
            "rms_error_m": fit_result.rms_error,
            "r_squared": fit_result.r_squared,
            "condition_number": fit_result.condition_number,
            "n_frames": len(timestamps),
            "n_markers": len(marker_names),
            "fit_success": fit_result.success,
        }

        return ParameterEstimationReport(
            subject_id=subject_id,
            fit_result=fit_result,
            segment_params=segment_params,
            sensitivities=sensitivities,
            quality_metrics=quality_metrics,
        )

    def fit_from_c3d(
        self,
        c3d_path: Path,
        subject_mass: float,
        subject_id: str | None = None,
    ) -> ParameterEstimationReport:
        """Fit parameters from C3D motion capture file.

        Args:
            c3d_path: Path to C3D file
            subject_mass: Total body mass (kg)
            subject_id: Optional subject identifier

        Returns:
            Complete ParameterEstimationReport.
        """
        if not (c3d_path is not None):
            raise ValueError("c3d_path must be provided")
        if not (c3d_path is not None):
            raise ValueError("c3d_path must be provided")
        try:
            import ezc3d
        except ImportError as e:
            logger.error("ezc3d not available - cannot read C3D files")
            raise ImportError("Install ezc3d: pip install ezc3d") from e

        logger.info(f"Loading C3D file: {c3d_path}")

        # Load C3D data
        c3d = ezc3d.c3d(str(c3d_path))

        # Extract marker data
        points = c3d["data"]["points"]  # [4 x markers x frames] (x, y, z, residual)
        marker_positions = np.transpose(points[:3], (2, 1, 0))  # [frames x markers x 3]

        # Get marker names
        marker_names = c3d["parameters"]["POINT"]["LABELS"]["value"]

        # Get timestamps
        n_frames = marker_positions.shape[0]
        frame_rate = c3d["parameters"]["POINT"]["RATE"]["value"][0]
        timestamps = np.arange(n_frames) / frame_rate

        # Use filename as subject ID if not provided
        if subject_id is None:
            subject_id = c3d_path.stem

        return self.fit_from_markers(
            marker_positions,
            marker_names,
            timestamps,
            subject_mass,
            subject_id,
        )

    def export_report(
        self,
        report: ParameterEstimationReport,
        output_path: Path,
        format: str = "json",
    ) -> None:
        """Export parameter estimation report.

        Args:
            report: ParameterEstimationReport to export
            output_path: Output file path
            format: Export format ("json", "csv")
        """
        if not (report is not None):
            raise ValueError("report must be provided")
        if not (report is not None):
            raise ValueError("report must be provided")
        import json

        if format == "json":
            output_data = {
                "subject_id": report.subject_id,
                "fit_success": report.fit_result.success,
                "rms_error": report.fit_result.rms_error,
                "parameters": report.fit_result.parameters,
                "segment_params": [p.to_dict() for p in report.segment_params],
                "quality_metrics": report.quality_metrics,
                "sensitivity_summary": self.sensitivity_analyzer.sensitivity_report(
                    report.sensitivities
                ),
            }

            with open(output_path, "w") as f:
                json.dump(output_data, f, indent=2)

            logger.info(f"Report exported to: {output_path}")
        else:
            raise ValueError(f"Unsupported export format: {format}")


__all__ = [
    "A3FittingPipeline",
    "BodySegmentParams",
    "FitResult",
    "InverseKinematicsSolver",
    "KinematicState",
    "ParameterEstimationReport",
    "ParameterEstimator",
    "SensitivityAnalyzer",
    "SensitivityResult",
    "convert_poses_to_markers",
]
