"""Screw Theory Core Kinematics.

This module implements screw theory abstractions across all engines.
It provides the core abstractions for `Twist` and `ScrewAxis` data models,
and provides the math methods necessary to extract instantaneous axes (ISA)
from twists.
This fulfills Guideline C3 ("Instantaneous screw axis (ISA) / twist extraction")
while applying DRY (Don't Repeat Yourself) by abstracting these computations
away from specific engine implementations (MuJoCo, Drake, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import math


@dataclass
class Twist:
    """Spatial twist (6D velocity) in screw theory.

    A twist represents the instantaneous motion of a rigid body as a screw
    motion: rotation about an axis combined with translation along that axis.

    Attributes:
        angular: Angular velocity vector [3] (rad/s)
        linear: Linear velocity vector [3] (m/s)
        body_name: Name of the body this twist describes
        reference_point: Point where linear velocity is measured [3] (m)
    """

    angular: np.ndarray
    linear: np.ndarray
    body_name: str
    reference_point: np.ndarray


@dataclass
class ScrewAxis:
    """Instantaneous Screw Axis (ISA) representation.

    Per Guideline C3, this represents the instantaneous axis of rotation
    and translation (the screw axis) for a rigid body motion.

    Attributes:
        axis_direction: Unit vector along screw axis [3] (dimensionless)
        axis_point: A point on the screw axis [3] (m)
        pitch: Screw pitch (h = v_parallel / ω) [m/rad]
                Special cases:
                - h = 0: Pure rotation about axis
                - h = ∞: Pure translation along axis
                - h finite: Helical motion
        angular_magnitude: Magnitude of angular velocity (rad/s)
        linear_magnitude: Magnitude of linear velocity (m/s)
        is_singular: True if motion is pure translation (ω ≈ 0)
    """

    axis_direction: np.ndarray
    axis_point: np.ndarray
    pitch: float
    angular_magnitude: float
    linear_magnitude: float
    is_singular: bool


def _finite_vector3(value: object, name: str) -> np.ndarray:
    """Return ``value`` as a float ``(3,)`` array, refusing NaN/inf or wrong shape."""
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {arr.shape}")
    if not bool(np.all(np.isfinite(arr))):
        raise ValueError(f"{name} must be finite, got {arr}")
    return arr


def compute_screw_axis(
    twist: Twist,
    singularity_threshold: float = 1e-6,
) -> ScrewAxis:
    """Compute Instantaneous Screw Axis from twist.

    Extracts the screw axis representation logically independent of physics engines.
    - Axis direction (unit vector)
    - Axis location (point on axis)
    - Pitch (ratio of translation to rotation)

    Args:
        twist: Spatial twist to analyze (angular, linear velocities and ref point)
        singularity_threshold: Threshold for detecting pure translation

    Returns:
        ScrewAxis with complete representation

    Raises:
        ValueError: ``twist`` is missing, its angular, linear or reference
            point is not a finite 3-vector, or ``singularity_threshold`` is not
            a finite positive number. A non-finite twist would otherwise yield
            a NaN screw axis reported as a regular (non-singular) motion.
    """
    if not (twist is not None):
        raise ValueError("twist must be provided")
    if not (math.isfinite(singularity_threshold) and singularity_threshold > 0.0):
        raise ValueError(
            f"singularity_threshold must be finite and > 0, got {singularity_threshold}"
        )
    ω = _finite_vector3(twist.angular, "twist.angular")
    v = _finite_vector3(twist.linear, "twist.linear")
    r = _finite_vector3(twist.reference_point, "twist.reference_point")

    # Check for singular case (pure translation, ω ≈ 0)
    ω_mag = float(
        math.hypot(*ω)
    )  # ⚡ Bolt: math.hypot is ~5x faster than np.linalg.norm for small arrays
    v_mag = float(
        math.hypot(*v)
    )  # ⚡ Bolt: math.hypot is ~5x faster than np.linalg.norm for small arrays

    if ω_mag < singularity_threshold:
        # Pure translation: screw axis is at infinity
        if v_mag > singularity_threshold:
            axis_dir = v / v_mag
        else:
            # No motion at all
            axis_dir = np.array([0.0, 0.0, 1.0])

        axis_point = r.copy()
        pitch = float("inf")
        is_singular = True

    else:
        # General case: screw motion
        is_singular = False

        # 1. Axis direction: ŝ = ω / |ω|
        axis_dir = ω / ω_mag

        # 2. Pitch: h = (ω · v) / |ω|²
        pitch = float(np.dot(ω, v) / (ω_mag**2))

        # 3. Axis location: Find point q on axis closest to reference point
        axis_point = r + np.cross(ω, v) / (ω_mag**2)

    return ScrewAxis(
        axis_direction=axis_dir,
        axis_point=axis_point,
        pitch=pitch,
        angular_magnitude=ω_mag,
        linear_magnitude=v_mag,
        is_singular=is_singular,
    )


def compute_screw_endpoints(
    screw: ScrewAxis,
    length: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate line segment for screw axis visualization.

    Args:
        screw: Screw axis to visualize
        length: Length of axis segment to draw [m] (finite, >= 0)

    Returns:
        Tuple of (start_point, end_point) for line segment [3], [3]
    """
    if not (screw is not None):
        raise ValueError("screw must be provided")
    if not (math.isfinite(length) and length >= 0.0):
        raise ValueError(f"length must be finite and >= 0, got {length}")
    if screw.is_singular:
        # Pure translation: draw along velocity direction
        start = screw.axis_point
        end = screw.axis_point + screw.axis_direction * length
    else:
        # Draw axis segment centered at axis_point
        start = screw.axis_point - screw.axis_direction * (length / 2)
        end = screw.axis_point + screw.axis_direction * (length / 2)

    return start, end
