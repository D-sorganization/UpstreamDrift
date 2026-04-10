"""Pipeline orchestration helpers for the A3 data-fitting workflow."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.shared.python.logging_pkg.logging_config import get_logger

from .data_fitting_models import (
    BodySegmentParams,
    KinematicState,
    ParameterEstimationReport,
    SensitivityResult,
)
from .data_fitting_parameter_estimator import ParameterEstimator
from .data_fitting_sensitivity import SensitivityAnalyzer

logger = get_logger(__name__)


def convert_poses_to_markers(
    pose_keypoints: np.ndarray,
    keypoint_names: list[str],
    target_markers: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Convert pose estimation keypoints to biomechanical marker format."""
    if not (pose_keypoints is not None):
        raise ValueError("pose_keypoints must be provided")
    pose_to_marker_map = {
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
        "nose": "NOSE",
        "left_ear": "LEAR",
        "right_ear": "REAR",
    }

    if pose_keypoints.shape[1] == 2:
        pose_keypoints = np.hstack([pose_keypoints, np.zeros((len(pose_keypoints), 1))])

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


class A3FittingPipeline:
    """Complete A3 model fitting pipeline."""

    def __init__(
        self,
        anthropometric_model: str = "dempster",
    ) -> None:
        if not (anthropometric_model is not None):
            raise ValueError("anthropometric_model must be provided")
        self.param_estimator = ParameterEstimator(anthropometric_model)
        self.sensitivity_analyzer = SensitivityAnalyzer()
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
        """Fit parameters from marker data."""
        if not (marker_positions is not None):
            raise ValueError("marker_positions must be provided")
        logger.info(
            f"Fitting A3 model for subject '{subject_id}' "
            f"({len(timestamps)} frames, {len(marker_names)} markers)"
        )

        kinematic_data = [
            KinematicState(
                timestamp=float(t),
                marker_positions=(
                    marker_positions[i] if i < len(marker_positions) else None
                ),
            )
            for i, t in enumerate(timestamps)
        ]

        fit_result = self.param_estimator.fit_parameters_to_kinematics(
            kinematic_data,
            self.segment_names,
            subject_mass,
        )

        segment_params = [
            BodySegmentParams(
                name=segment_name,
                length=fit_result.parameters[f"{segment_name}_length"],
                mass=fit_result.parameters.get(f"{segment_name}_mass", 0.0),
            )
            for segment_name in self.segment_names
            if f"{segment_name}_length" in fit_result.parameters
        ]

        sensitivities: list[SensitivityResult] = []
        if not sensitivities:
            logger.warning(
                "Sensitivity analysis is empty (placeholder). "
                "Results lack parameter sensitivity information. "
                "See issue #2170 for implementation tracking."
            )

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
        """Fit parameters from C3D motion capture file."""
        if not (c3d_path is not None):
            raise ValueError("c3d_path must be provided")
        try:
            import ezc3d
        except ImportError as e:
            logger.error("ezc3d not available - cannot read C3D files")
            raise ImportError("Install ezc3d: pip install ezc3d") from e

        logger.info(f"Loading C3D file: {c3d_path}")
        c3d = ezc3d.c3d(str(c3d_path))
        points = c3d["data"]["points"]
        marker_positions = np.transpose(points[:3], (2, 1, 0))
        marker_names = c3d["parameters"]["POINT"]["LABELS"]["value"]

        n_frames = marker_positions.shape[0]
        frame_rate = c3d["parameters"]["POINT"]["RATE"]["value"][0]
        timestamps = np.arange(n_frames) / frame_rate

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
        """Export parameter estimation report."""
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
            return

        raise ValueError(f"Unsupported export format: {format}")
