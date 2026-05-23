"""Data model classes for A3 biomechanics fitting pipeline.

Extracted from data_fitting.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


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

    @property
    def solver_status(self) -> str:
        """Derive solver status from success."""
        return "success" if self.success else "failure"


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


__all__ = [
    "BodySegmentParams",
    "FitResult",
    "KinematicState",
    "ParameterEstimationReport",
    "SensitivityResult",
]
