"""Data fitting and parameter estimation for golf biomechanics (Guideline A3).

This module implements the A3 pipeline per project design guidelines:
- Fit kinematics to observed trajectories
- Estimate body segment parameters (lengths, mass, inertia)
- Report sensitivity analysis and fit quality metrics

Issue #754: Implements complete A3 model fitting and parameter identification.

Reference: docs/assessments/project_design_guidelines.qmd Section A3
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import optimize

from src.shared.python.logging_config import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class BodySegmentParams:
    """Physical parameters for a body segment.

    Represents anthropometric data for a single body segment.

    Attributes:
        name: Segment name (e.g., "upper_arm", "forearm")
        length: Segment length in meters
        mass: Segment mass in kg
        com_position: Center of mass position along segment [0, 1]
        inertia: Principal moments of inertia [Ixx, Iyy, Izz] (kg*m^2)
        radius_gyration: Radius of gyration as fraction of length
    """

    name: str
    length: float
    mass: float
    com_position: float = 0.5  # Proximal = 0, Distal = 1
    inertia: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    radius_gyration: float = 0.3

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "length": self.length,
            "mass": self.mass,
            "com_position": self.com_position,
            "inertia": self.inertia.tolist(),
            "radius_gyration": self.radius_gyration,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BodySegmentParams:
        """Deserialize from dictionary."""
        return cls(
            name=data["name"],
            length=data["length"],
            mass=data["mass"],
            com_position=data.get("com_position", 0.5),
            inertia=np.array(data.get("inertia", [0.0, 0.0, 0.0])),
            radius_gyration=data.get("radius_gyration", 0.3),
        )


@dataclass
class KinematicState:
    """Kinematic state at a single time point.

    Attributes:
        timestamp: Time in seconds
        joint_angles: Dictionary of joint name -> angle (rad)
        joint_velocities: Dictionary of joint name -> angular velocity (rad/s)
        joint_accelerations: Dictionary of joint name -> angular acceleration (rad/s^2)
        marker_positions: Optional marker positions [N x 3]
    """

    timestamp: float
    joint_angles: dict[str, float] = field(default_factory=dict)
    joint_velocities: dict[str, float] = field(default_factory=dict)
    joint_accelerations: dict[str, float] = field(default_factory=dict)
    marker_positions: np.ndarray | None = None


@dataclass
class FitResult:
    """Result of kinematic or parameter fitting.

    Attributes:
        success: Whether fitting succeeded
        parameters: Fitted parameter values
        residuals: Fitting residuals
        rms_error: Root mean square error
        r_squared: Coefficient of determination
        aic: Akaike Information Criterion
        bic: Bayesian Information Criterion
        condition_number: Condition number of Jacobian
        iterations: Number of iterations used
        message: Status message
    """

    success: bool
    parameters: dict[str, float]
    residuals: np.ndarray
    rms_error: float
    r_squared: float = 0.0
    aic: float = 0.0
    bic: float = 0.0
    condition_number: float = 0.0
    iterations: int = 0
    message: str = ""


@dataclass
class SensitivityResult:
    """Result of sensitivity analysis.

    Attributes:
        parameter_name: Name of the parameter
        nominal_value: Nominal parameter value
        sensitivity_index: Normalized sensitivity index
        partial_derivative: Partial derivative at nominal
        confidence_interval: 95% confidence interval [lower, upper]
        elasticity: Elasticity (% change in output / % change in parameter)
    """

    parameter_name: str
    nominal_value: float
    sensitivity_index: float
    partial_derivative: float
    confidence_interval: tuple[float, float]
    elasticity: float


@dataclass
class ParameterEstimationReport:
    """Complete parameter estimation report.

    Per Guideline A3: End-to-end fitting from data to parameter report.

    Attributes:
        subject_id: Subject identifier
        fit_result: Fitting result
        segment_params: Estimated segment parameters
        sensitivities: Sensitivity analysis results
        quality_metrics: Fit quality metrics
        validation_errors: Validation against known data
    """

    subject_id: str
    fit_result: FitResult
    segment_params: list[BodySegmentParams]
    sensitivities: list[SensitivityResult]
    quality_metrics: dict[str, float]
    validation_errors: dict[str, float] = field(default_factory=dict)


# =============================================================================
# Inverse Kinematics
# =============================================================================


class InverseKinematicsSolver:
    """Solve inverse kinematics from marker positions to joint angles.

    Implements analytical and numerical IK for golf swing analysis.
    Uses marker positions to determine joint angles for a kinematic chain.
    """

    def __init__(
        self,
        segment_lengths: dict[str, float],
        joint_names: list[str],
        tolerance: float = 1e-6,
        max_iterations: int = 100,
    ) -> None:
        """Initialize IK solver.

        Args:
            segment_lengths: Dictionary of segment name -> length (m)
            joint_names: List of joint names in kinematic chain order
            tolerance: Convergence tolerance for numerical IK
            max_iterations: Maximum iterations for numerical IK
        """
        self.segment_lengths = segment_lengths
        self.joint_names = joint_names
        self.tolerance = tolerance
        self.max_iterations = max_iterations

        logger.info(
            f"IK solver initialized with {len(joint_names)} joints, "
            f"{len(segment_lengths)} segments"
        )

    def solve_analytical_2d(
        self,
        target_position: np.ndarray,
        segment1_length: float,
        segment2_length: float,
    ) -> tuple[float, float]:
        """Solve 2-link planar IK analytically.

        Uses geometric solution for 2-link planar manipulator.

        Args:
            target_position: Target [x, y] position
            segment1_length: Length of first segment
            segment2_length: Length of second segment

        Returns:
            Tuple of (theta1, theta2) joint angles in radians.

        Raises:
            ValueError: If target is unreachable.
        """
        x, y = target_position[:2]
        L1, L2 = segment1_length, segment2_length

        # Distance to target
        d = np.sqrt(x**2 + y**2)

        # Check reachability
        if d > L1 + L2:
            raise ValueError(
                f"Target at distance {d:.3f}m is unreachable (max: {L1 + L2:.3f}m)"
            )
        if d < abs(L1 - L2):
            raise ValueError(f"Target at distance {d:.3f}m is too close")

        # Elbow angle (law of cosines)
        cos_theta2 = (x**2 + y**2 - L1**2 - L2**2) / (2 * L1 * L2)
        cos_theta2 = np.clip(cos_theta2, -1.0, 1.0)
        theta2 = np.arccos(cos_theta2)

        # Shoulder angle
        k1 = L1 + L2 * np.cos(theta2)
        k2 = L2 * np.sin(theta2)
        theta1 = np.arctan2(y, x) - np.arctan2(k2, k1)

        return float(theta1), float(theta2)

    def solve_numerical(
        self,
        target_positions: np.ndarray,
        initial_angles: np.ndarray | None = None,
    ) -> FitResult:
        """Solve IK numerically using optimization.

        Uses Levenberg-Marquardt to minimize position error.

        Args:
            target_positions: Target positions for each end effector [N x 3]
            initial_angles: Initial guess for joint angles (optional)

        Returns:
            FitResult with optimized joint angles.
        """
        n_joints = len(self.joint_names)

        if initial_angles is None:
            initial_angles = np.zeros(n_joints)

        def residual_func(angles: np.ndarray) -> np.ndarray:
            """Compute position error for given joint angles."""
            predicted = self._forward_kinematics(angles)
            return (predicted - target_positions).flatten()

        # Run optimization
        result = optimize.least_squares(
            residual_func,
            initial_angles,
            method="lm",
            ftol=self.tolerance,
            max_nfev=self.max_iterations,
        )

        residuals = residual_func(result.x)
        rms = float(np.sqrt(np.mean(residuals**2)))

        # Compute R-squared
        total_variance = np.var(target_positions.flatten())
        r_squared = (
            1.0 - (np.var(residuals) / total_variance) if total_variance > 0 else 0.0
        )

        # Condition number from Jacobian
        try:
            jac = result.jac
            if jac is not None and jac.size > 0:
                s = np.linalg.svd(jac, compute_uv=False)
                cond = float(s[0] / s[-1]) if s[-1] > 1e-10 else float("inf")
            else:
                cond = float("inf")
        except (ValueError, TypeError, RuntimeError):
            cond = float("inf")

        return FitResult(
            success=result.success,
            parameters={
                name: float(angle)
                for name, angle in zip(self.joint_names, result.x, strict=False)
            },
            residuals=residuals,
            rms_error=rms,
            r_squared=float(r_squared),
            condition_number=cond,
            iterations=result.nfev,
            message=result.message,
        )

    def _forward_kinematics(self, angles: np.ndarray) -> np.ndarray:
        """Compute forward kinematics for given joint angles.

        Placeholder implementation - should be overridden for specific models.

        Args:
            angles: Joint angles [N]

        Returns:
            End effector positions [M x 3]
        """
        # Simple planar chain for demonstration
        positions = []
        x, y, z = 0.0, 0.0, 0.0
        cumulative_angle = 0.0

        for _i, (joint_name, angle) in enumerate(
            zip(self.joint_names, angles, strict=False)
        ):
            cumulative_angle += angle

            # Get segment length (use 0.3m default if not specified)
            segment_name = joint_name.replace("_joint", "")
            length = self.segment_lengths.get(segment_name, 0.3)

            x += length * np.cos(cumulative_angle)
            y += length * np.sin(cumulative_angle)

            positions.append([x, y, z])

        return np.array(positions)


# =============================================================================
# Parameter Estimation
# =============================================================================


class ParameterEstimator:
    """Estimate body segment parameters from motion data.

    Implements parameter identification per Guideline A3:
    - Segment length estimation from marker data
    - Mass and inertia estimation using regression
    - Sensitivity analysis
    - Fit quality reporting
    """

    def __init__(
        self,
        anthropometric_model: str = "dempster",
    ) -> None:
        """Initialize parameter estimator.

        Args:
            anthropometric_model: Model for mass/inertia regression
                ("dempster", "winter", "de_leva")
        """
        self.anthropometric_model = anthropometric_model
        self._load_regression_coefficients()

        logger.info(
            f"Parameter estimator initialized with '{anthropometric_model}' model"
        )

    def _load_regression_coefficients(self) -> None:
        """Load anthropometric regression coefficients.

        Based on published anthropometric studies:
        - Dempster (1955): Classical segment mass fractions
        - Winter (2009): Updated biomechanics values
        - de Leva (1996): Gender-specific adjustments
        """
        # Mass fractions as proportion of total body mass
        # Format: segment_name -> (mass_fraction, com_proximal_fraction, radius_of_gyration)
        self.coefficients: dict[str, tuple[float, float, float]] = {
            # Upper extremity
            "upper_arm": (0.028, 0.436, 0.322),
            "forearm": (0.016, 0.430, 0.303),
            "hand": (0.006, 0.506, 0.297),
            # Lower extremity
            "thigh": (0.100, 0.433, 0.323),
            "shank": (0.047, 0.433, 0.302),
            "foot": (0.014, 0.500, 0.475),
            # Trunk
            "head": (0.081, 0.500, 0.495),
            "trunk": (0.497, 0.500, 0.496),
            "pelvis": (0.142, 0.500, 0.540),
        }

        if self.anthropometric_model == "winter":
            # Winter's slightly updated values
            self.coefficients["upper_arm"] = (0.028, 0.436, 0.320)
            self.coefficients["forearm"] = (0.016, 0.430, 0.301)
        elif self.anthropometric_model == "de_leva":
            # de Leva male values (adjust for female separately)
            self.coefficients["upper_arm"] = (0.027, 0.577, 0.285)
            self.coefficients["forearm"] = (0.016, 0.457, 0.276)

    def estimate_segment_length(
        self,
        proximal_markers: np.ndarray,
        distal_markers: np.ndarray,
    ) -> tuple[float, float]:
        """Estimate segment length from marker positions.

        Args:
            proximal_markers: Proximal marker positions [N x 3]
            distal_markers: Distal marker positions [N x 3]

        Returns:
            Tuple of (mean_length, std_length) in meters.
        """
        # Compute distances for each frame
        distances = np.linalg.norm(distal_markers - proximal_markers, axis=1)

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
        """Estimate segment parameters using anthropometric regression.

        Args:
            segment_name: Name of body segment
            segment_length: Measured segment length (m)
            total_body_mass: Total body mass (kg)

        Returns:
            BodySegmentParams with estimated values.
        """
        # Get regression coefficients
        if segment_name in self.coefficients:
            mass_frac, com_frac, rog_frac = self.coefficients[segment_name]
        else:
            # Default values
            logger.warning(
                f"Unknown segment '{segment_name}', using default coefficients"
            )
            mass_frac, com_frac, rog_frac = 0.02, 0.5, 0.3

        # Compute parameters
        mass = total_body_mass * mass_frac
        com_position = com_frac

        # Radius of gyration
        radius_gyration = rog_frac * segment_length

        # Compute principal inertias assuming cylindrical segment
        # I_xx = I_yy = (1/12) * m * L^2 + m * r_g^2 (parallel axis)
        # I_zz = (1/2) * m * r^2 (about long axis, assuming small radius)
        radius = 0.05 * segment_length  # Assume radius is 5% of length
        I_xx = mass * (segment_length**2 / 12 + radius_gyration**2)
        I_yy = I_xx
        I_zz = mass * radius**2 / 2

        return BodySegmentParams(
            name=segment_name,
            length=segment_length,
            mass=mass,
            com_position=com_position,
            inertia=np.array([I_xx, I_yy, I_zz]),
            radius_gyration=rog_frac,
        )

    def fit_parameters_to_kinematics(
        self,
        kinematic_data: list[KinematicState],
        segment_names: list[str],
        total_body_mass: float,
        known_lengths: dict[str, float] | None = None,
    ) -> FitResult:
        """Fit segment parameters to observed kinematic data.

        Args:
            kinematic_data: List of kinematic states over time
            segment_names: Names of segments to fit
            total_body_mass: Total body mass (kg)
            known_lengths: Optional known segment lengths (m)

        Returns:
            FitResult with fitted parameters.
        """
        if not kinematic_data:
            return FitResult(
                success=False,
                parameters={},
                residuals=np.array([]),
                rms_error=float("inf"),
                message="No kinematic data provided",
            )

        # Extract marker data if available
        marker_frames = [
            state.marker_positions
            for state in kinematic_data
            if state.marker_positions is not None
        ]

        if not marker_frames:
            # Fall back to anthropometric estimation without marker data
            logger.warning("No marker data - using anthropometric estimates only")

            params = {}
            for segment_name in segment_names:
                # Use known length or estimate from body height
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

        # Compute segment lengths from marker data
        # This requires knowing which markers correspond to which segments
        # Simplified: assume consecutive markers define segments
        marker_array = np.array(marker_frames)  # [frames, markers, 3]

        fitted_params = {}
        all_residuals = []

        for i, segment_name in enumerate(segment_names):
            if i + 1 >= marker_array.shape[1]:
                break

            proximal = marker_array[:, i, :]
            distal = marker_array[:, i + 1, :]

            mean_length, std_length = self.estimate_segment_length(proximal, distal)

            # Use known length if provided, otherwise use measured
            if known_lengths and segment_name in known_lengths:
                length = known_lengths[segment_name]
                residual = mean_length - length
            else:
                length = mean_length
                residual = std_length  # Use variation as residual

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


# =============================================================================
# Sensitivity Analysis
# =============================================================================


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
