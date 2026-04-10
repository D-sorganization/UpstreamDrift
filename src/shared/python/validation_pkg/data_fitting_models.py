"""Shared dataclasses for the A3 data-fitting pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class BodySegmentParams:
    """Physical parameters for a body segment."""

    name: str
    length: float
    mass: float
    com_position: float = 0.5
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
    """Kinematic state at a single time point."""

    timestamp: float
    joint_angles: dict[str, float] = field(default_factory=dict)
    joint_velocities: dict[str, float] = field(default_factory=dict)
    joint_accelerations: dict[str, float] = field(default_factory=dict)
    marker_positions: np.ndarray | None = None


@dataclass
class FitResult:
    """Result of kinematic or parameter fitting."""

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
    """Result of sensitivity analysis."""

    parameter_name: str
    nominal_value: float
    sensitivity_index: float
    partial_derivative: float
    confidence_interval: tuple[float, float]
    elasticity: float


@dataclass
class ParameterEstimationReport:
    """Complete parameter estimation report."""

    subject_id: str
    fit_result: FitResult
    segment_params: list[BodySegmentParams]
    sensitivities: list[SensitivityResult]
    quality_metrics: dict[str, float]
    validation_errors: dict[str, float] = field(default_factory=dict)
