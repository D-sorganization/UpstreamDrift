"""Explicit spatial wrench, contact reaction, and center-of-pressure semantics (MV-06, #10482).

Enforces:
- Declared application frame, point, sign convention, and SI units on every wrench.
- Moment arm calculation on spatial wrench rotation and translation.
- Center of Pressure (CoP) defined only when normal vertical force exceeds threshold.
- Missing telemetry channels remain None (never fabricated zeros).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from src.shared.python.contracts import postcondition, precondition

DEFAULT_FORCE_UNITS: dict[str, str] = {
    "force": "N",
    "torque": "N*m",
    "length": "m",
}


def _validate_vec3(v: Sequence[float], name: str) -> tuple[float, float, float]:
    """Validate 3D vector coordinates for finite real numbers."""
    if len(v) != 3:
        raise ValueError(f"{name} must contain exactly 3 coordinates, got {len(v)}")
    coords = tuple(float(x) for x in v)
    if not all(math.isfinite(x) for x in coords):
        raise ValueError(f"{name} coordinates must be finite numbers, got {coords}")
    return (coords[0], coords[1], coords[2])


@dataclass(frozen=True)
class SpatialWrench:
    """Rigid body wrench with explicit frame, application point, and SI units."""

    application_frame: str
    point_m: tuple[float, float, float]
    force_n: tuple[float, float, float]
    torque_nm: tuple[float, float, float]
    direction_convention: str = "applied_to_body"
    sign_convention: str = "standard_cartesian"
    units: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_FORCE_UNITS))

    def __post_init__(self) -> None:
        if not self.application_frame or not isinstance(self.application_frame, str):
            raise ValueError("application_frame must be a non-empty string")
        object.__setattr__(self, "point_m", _validate_vec3(self.point_m, "point_m"))
        object.__setattr__(self, "force_n", _validate_vec3(self.force_n, "force_n"))
        object.__setattr__(
            self, "torque_nm", _validate_vec3(self.torque_nm, "torque_nm")
        )


@precondition(
    lambda wrench, target_frame, new_point_m, rotation_matrix=None: (
        wrench is not None and bool(target_frame) and len(new_point_m) == 3
    ),
    "Valid source wrench, target frame name, and new 3D point required",
)
def transform_wrench(
    wrench: SpatialWrench,
    target_frame: str,
    new_point_m: Sequence[float],
    rotation_matrix: NDArray[np.float64] | None = None,
) -> SpatialWrench:
    """Transform spatial wrench to a new frame and point of application with moment arm.

    Formulas:
        F_B = R @ F_A
        tau_rot = R @ tau_A
        p_A_in_B = R @ p_A
        r = p_A_in_B - p_B
        tau_B = tau_rot + cross(r, F_B)
    """
    R = (
        np.eye(3, dtype=np.float64)
        if rotation_matrix is None
        else np.asarray(rotation_matrix, dtype=np.float64)
    )
    if R.shape != (3, 3):
        raise ValueError(f"rotation_matrix must have shape (3, 3), got {R.shape}")

    F_A = np.asarray(wrench.force_n, dtype=np.float64)
    tau_A = np.asarray(wrench.torque_nm, dtype=np.float64)
    p_A = np.asarray(wrench.point_m, dtype=np.float64)
    p_B = np.asarray(_validate_vec3(new_point_m, "new_point_m"), dtype=np.float64)

    F_B = R @ F_A
    tau_rot = R @ tau_A
    p_A_in_B = R @ p_A
    r = p_A_in_B - p_B
    moment_arm_torque = np.cross(r, F_B)
    tau_B = tau_rot + moment_arm_torque

    return SpatialWrench(
        application_frame=target_frame,
        point_m=(float(p_B[0]), float(p_B[1]), float(p_B[2])),
        force_n=(float(F_B[0]), float(F_B[1]), float(F_B[2])),
        torque_nm=(float(tau_B[0]), float(tau_B[1]), float(tau_B[2])),
        direction_convention=wrench.direction_convention,
        sign_convention=wrench.sign_convention,
        units=dict(wrench.units),
    )


def compute_center_of_pressure(
    wrench: SpatialWrench,
    f_threshold_n: float = 5.0,
) -> tuple[float, float] | None:
    """Calculate Center of Pressure (CoP) coordinates in the contact plane.

    Returns None when vertical force F_z <= f_threshold_n or non-finite.
    Formulas:
        x_cop = p_x - tau_y / F_z
        y_cop = p_y + tau_x / F_z
    """
    F_z = wrench.force_n[2]
    if not math.isfinite(F_z) or F_z <= f_threshold_n:
        return None

    tau_x = wrench.torque_nm[0]
    tau_y = wrench.torque_nm[1]
    p_x = wrench.point_m[0]
    p_y = wrench.point_m[1]

    x_cop = float(p_x - (tau_y / F_z))
    y_cop = float(p_y + (tau_x / F_z))
    if not (math.isfinite(x_cop) and math.isfinite(y_cop)):
        return None
    return (x_cop, y_cop)


@dataclass(frozen=True)
class ContactReaction:
    """Instantaneous contact and reaction state synchronized with physical time."""

    time_s: float
    net_grf_n: tuple[float, float, float] | None = None
    left_foot_grf_n: tuple[float, float, float] | None = None
    right_foot_grf_n: tuple[float, float, float] | None = None
    net_cop_m: tuple[float, float] | None = None
    left_foot_cop_m: tuple[float, float] | None = None
    right_foot_cop_m: tuple[float, float] | None = None
    contact_status: dict[str, bool] = field(default_factory=dict)
    friction_utilization: dict[str, float | None] = field(default_factory=dict)
    grip_wrench: SpatialWrench | None = None
    closure_residual_m: float | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.time_s):
            raise ValueError("time_s must be a finite number")
