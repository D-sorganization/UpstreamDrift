"""Parameter-estimation helpers for the A3 data-fitting pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.shared.python.logging_pkg.logging_config import get_logger

from .data_fitting_models import BodySegmentParams, FitResult, KinematicState

logger = get_logger(__name__)


class ParameterEstimator:
    """Estimate body segment parameters from motion data."""

    def __init__(
        self,
        anthropometric_model: str = "dempster",
    ) -> None:
        if not (anthropometric_model is not None):
            raise ValueError("anthropometric_model must be provided")
        self.anthropometric_model = anthropometric_model
        self._load_regression_coefficients()

        logger.info(
            f"Parameter estimator initialized with '{anthropometric_model}' model"
        )

    def _load_regression_coefficients(self) -> None:
        """Load anthropometric regression coefficients."""
        self.coefficients: dict[str, tuple[float, float, float]] = {
            "upper_arm": (0.028, 0.436, 0.322),
            "forearm": (0.016, 0.430, 0.303),
            "hand": (0.006, 0.506, 0.297),
            "thigh": (0.100, 0.433, 0.323),
            "shank": (0.047, 0.433, 0.302),
            "foot": (0.014, 0.500, 0.475),
            "head": (0.081, 0.500, 0.495),
            "trunk": (0.497, 0.500, 0.496),
            "pelvis": (0.142, 0.500, 0.540),
        }

        if self.anthropometric_model == "winter":
            self.coefficients["upper_arm"] = (0.028, 0.436, 0.320)
            self.coefficients["forearm"] = (0.016, 0.430, 0.301)
        elif self.anthropometric_model == "de_leva":
            self.coefficients["upper_arm"] = (0.027, 0.577, 0.285)
            self.coefficients["forearm"] = (0.016, 0.457, 0.276)

    def estimate_segment_length(
        self,
        proximal_markers: np.ndarray,
        distal_markers: np.ndarray,
    ) -> tuple[float, float]:
        """Estimate segment length from marker positions."""
        if not (proximal_markers is not None):
            raise ValueError("proximal_markers must be provided")
        distances = np.sqrt(np.sum((distal_markers - proximal_markers) ** 2, axis=1))

        mean_length = float(np.mean(distances))
        std_length = float(np.std(distances))
        logger.debug(f"Segment length: {mean_length:.4f} +/- {std_length:.4f} m")
        return mean_length, std_length

    def estimate_segment_params(
        self,
        segment_name: str,
        segment_length: float,
        total_body_mass: float,
    ) -> BodySegmentParams:
        """Estimate segment parameters using anthropometric regression."""
        if not (segment_name is not None):
            raise ValueError("segment_name must be provided")
        if segment_name in self.coefficients:
            mass_frac, com_frac, rog_frac = self.coefficients[segment_name]
        else:
            logger.warning(
                f"Unknown segment '{segment_name}', using default coefficients"
            )
            mass_frac, com_frac, rog_frac = 0.02, 0.5, 0.3

        mass = total_body_mass * mass_frac
        radius_gyration = rog_frac * segment_length
        radius = 0.05 * segment_length
        I_xx = mass * (segment_length**2 / 12 + radius_gyration**2)
        I_yy = I_xx
        I_zz = mass * radius**2 / 2

        return BodySegmentParams(
            name=segment_name,
            length=segment_length,
            mass=mass,
            com_position=com_frac,
            inertia=np.array([I_xx, I_yy, I_zz]),
            radius_gyration=rog_frac,
        )

    def _fit_from_anthropometry(
        self,
        segment_names: list[str],
        total_body_mass: float,
        known_lengths: dict[str, float] | None,
    ) -> FitResult:
        """Estimate segment parameters using anthropometric tables only."""
        if not (segment_names is not None):
            raise ValueError("segment_names must be provided")
        logger.warning("No marker data - using anthropometric estimates only")
        params: dict[str, Any] = {}
        for segment_name in segment_names:
            length = known_lengths.get(segment_name, 0.3) if known_lengths else 0.3
            segment_params = self.estimate_segment_params(
                segment_name, length, total_body_mass
            )
            params[f"{segment_name}_length"] = segment_params.length
            params[f"{segment_name}_mass"] = segment_params.mass
        return FitResult(
            success=True,
            parameters=params,
            residuals=np.array([]),
            rms_error=0.0,
            message="Anthropometric estimation (no marker data)",
        )

    def _fit_from_markers(
        self,
        marker_array: np.ndarray,
        segment_names: list[str],
        total_body_mass: float,
        known_lengths: dict[str, float] | None,
    ) -> FitResult:
        """Fit segment parameters from marker position data."""
        if not (marker_array is not None):
            raise ValueError("marker_array must be provided")
        fitted_params: dict[str, Any] = {}
        all_residuals: list[float] = []

        for i, segment_name in enumerate(segment_names):
            if i + 1 >= marker_array.shape[1]:
                break

            proximal = marker_array[:, i, :]
            distal = marker_array[:, i + 1, :]
            mean_length, std_length = self.estimate_segment_length(proximal, distal)

            if known_lengths and segment_name in known_lengths:
                length = known_lengths[segment_name]
                residual = mean_length - length
            else:
                length = mean_length
                residual = std_length

            all_residuals.append(residual)

            segment_params = self.estimate_segment_params(
                segment_name, length, total_body_mass
            )
            fitted_params[f"{segment_name}_length"] = segment_params.length
            fitted_params[f"{segment_name}_mass"] = segment_params.mass
            fitted_params[f"{segment_name}_com"] = segment_params.com_position

        residuals = np.array(all_residuals)
        rms = float(np.sqrt(np.mean(residuals**2))) if len(residuals) > 0 else 0.0

        return FitResult(
            success=True,
            parameters=fitted_params,
            residuals=residuals,
            rms_error=rms,
            message="Segment parameters fitted from marker data",
        )

    def fit_parameters_to_kinematics(
        self,
        kinematic_data: list[KinematicState],
        segment_names: list[str],
        total_body_mass: float,
        known_lengths: dict[str, float] | None = None,
    ) -> FitResult:
        """Fit segment parameters to observed kinematic data."""
        if not (kinematic_data is not None):
            raise ValueError("kinematic_data must be provided")
        if not kinematic_data:
            return FitResult(
                success=False,
                parameters={},
                residuals=np.array([]),
                rms_error=float("inf"),
                message="No kinematic data provided",
            )

        marker_frames = [
            state.marker_positions
            for state in kinematic_data
            if state.marker_positions is not None
        ]

        if not marker_frames:
            return self._fit_from_anthropometry(
                segment_names, total_body_mass, known_lengths
            )

        marker_array = np.array(marker_frames)
        return self._fit_from_markers(
            marker_array, segment_names, total_body_mass, known_lengths
        )
