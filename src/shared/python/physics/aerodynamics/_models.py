"""Aerodynamic force models: DragModel, LiftModel, MagnusModel.

Each model calculates one orthogonal force type from ball velocity and spin.

The :class:`DragModel` Reynolds-number correction now delegates to
:func:`src.shared.python.physics.atmosphere.cd_dimpled_sphere` which models
the drag-crisis behaviour of dimpled spheres (Bearman & Harvey 1976,
Mehta 1985); see issue #3504.
"""

from __future__ import annotations

import math

import numpy as np

from src.shared.python.core.physics_constants import (
    AIR_DENSITY_SEA_LEVEL_KG_M3,
    AIR_VISCOSITY_KG_M_S,
    GOLF_BALL_CROSS_SECTIONAL_AREA_M2,
    GOLF_BALL_DRAG_COEFFICIENT,
    GOLF_BALL_LIFT_COEFFICIENT,
    GOLF_BALL_RADIUS_M,
)
from src.shared.python.physics.atmosphere import cd_dimpled_sphere
from src.shared.python.physics.ball_properties import calculate_spin_lift_coefficient

_DEFAULT_DRAG_COEFFICIENT = float(GOLF_BALL_DRAG_COEFFICIENT)


def _as_vector3(name: str, value: np.ndarray) -> np.ndarray:
    """Return a flattened 3-vector for force calculations."""
    if value is None:
        raise ValueError(f"{name} must be provided")
    vector = np.asarray(value, dtype=float).reshape(-1)
    if vector.shape != (3,):
        raise ValueError(f"{name} must contain exactly 3 elements")
    return vector


def _active_motion_vectors(
    velocity: np.ndarray,
    spin: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float] | None:
    """Normalize velocity/spin vectors for spin-induced forces."""
    velocity_vector = _as_vector3("velocity", velocity)
    spin_vector = _as_vector3("spin", spin)
    speed = float(
        math.sqrt(np.dot(velocity_vector, velocity_vector))
    )  # ⚡ Bolt: math.sqrt(np.dot) is ~3x faster than np.linalg.norm
    spin_magnitude = float(
        math.sqrt(np.dot(spin_vector, spin_vector))
    )  # ⚡ Bolt: math.sqrt(np.dot) is ~3x faster than np.linalg.norm
    if speed < 1e-10 or spin_magnitude < 1e-10:
        return None
    return velocity_vector, spin_vector, speed, spin_magnitude


class DragModel:
    """Model for aerodynamic drag force.

    Drag opposes motion and scales with v^2:
        F_drag = -0.5 * rho * Cd * A * |v| * v

    The drag coefficient can optionally be corrected for Reynolds number,
    which accounts for the transition from laminar to turbulent flow
    around the golf ball's dimpled surface.
    """

    def __init__(
        self,
        base_coefficient: float = float(GOLF_BALL_DRAG_COEFFICIENT),
        ball_area: float = float(GOLF_BALL_CROSS_SECTIONAL_AREA_M2),
        ball_radius: float = float(GOLF_BALL_RADIUS_M),
        reynolds_correction: bool = True,
    ) -> None:
        if base_coefficient is None:
            raise ValueError("base_coefficient must be provided")
        if not math.isfinite(base_coefficient) or base_coefficient < 0.0:
            raise ValueError("base_coefficient must be finite and non-negative")
        self.base_coefficient = float(base_coefficient)
        self.ball_area = ball_area
        self.ball_radius = ball_radius
        self.reynolds_correction = reynolds_correction

    def calculate(
        self,
        velocity: np.ndarray,
        air_density: float = float(AIR_DENSITY_SEA_LEVEL_KG_M3),
    ) -> np.ndarray:
        """Calculate drag force.

        Args:
            velocity: Ball velocity relative to air [m/s]
            air_density: Air density [kg/m^3]

        Returns:
            Drag force vector [N]
        """
        velocity_vector = _as_vector3("velocity", velocity)
        speed = float(
            math.sqrt(np.dot(velocity_vector, velocity_vector))
        )  # ⚡ Bolt: math.sqrt(np.dot) is ~3x faster than np.linalg.norm
        if speed < 1e-10:
            return np.zeros_like(velocity_vector)

        cd = self._effective_coefficient_from_speed(speed, air_density)
        force_magnitude = 0.5 * air_density * cd * self.ball_area * speed * speed
        return -force_magnitude * velocity_vector / speed

    def get_effective_coefficient(
        self,
        velocity: np.ndarray,
        air_density: float = float(AIR_DENSITY_SEA_LEVEL_KG_M3),
    ) -> float:
        """Get drag coefficient, optionally corrected for Reynolds number.

        Uses the smoothed drag-crisis model from
        :func:`src.shared.python.physics.atmosphere.cd_dimpled_sphere`
        (Bearman & Harvey 1976, Mehta 1985). Reynolds numbers outside the
        supported [1e3, 1e7] range fall back to the nearest endpoint so the
        integrator never sees a discontinuity.
        """
        velocity_vector = _as_vector3("velocity", velocity)
        if not self.reynolds_correction:
            return self.base_coefficient

        speed = float(
            math.sqrt(np.dot(velocity_vector, velocity_vector))
        )  # ⚡ Bolt: math.sqrt(np.dot) is ~3x faster than np.linalg.norm
        if speed < 1e-10:
            return self.base_coefficient

        return self._effective_coefficient_from_speed(speed, air_density)

    def _effective_coefficient_from_speed(
        self,
        speed: float,
        air_density: float,
    ) -> float:
        """Return the Reynolds-corrected drag coefficient for a known speed."""
        if not self.reynolds_correction:
            return self.base_coefficient

        viscosity = float(AIR_VISCOSITY_KG_M_S)
        diameter = 2 * self.ball_radius
        re = air_density * speed * diameter / viscosity
        # Clamp to the model's supported range; outside it the underlying
        # correlation extrapolation is meaningless for golf balls.
        re_clamped = max(1.0e3, min(1.0e7, re))
        cd = cd_dimpled_sphere(
            re_clamped,
            base_cd=_DEFAULT_DRAG_COEFFICIENT,
        )
        return cd * self.base_coefficient / _DEFAULT_DRAG_COEFFICIENT


class LiftModel:
    """Model for spin-induced lift force.

    Lift from backspin acts perpendicular to velocity in the
    plane defined by the spin axis and velocity vector.
    """

    def __init__(
        self,
        base_coefficient: float = float(GOLF_BALL_LIFT_COEFFICIENT),
        ball_area: float = float(GOLF_BALL_CROSS_SECTIONAL_AREA_M2),
        ball_radius: float = float(GOLF_BALL_RADIUS_M),
        max_coefficient: float = float("inf"),
    ) -> None:
        if base_coefficient is None:
            raise ValueError("base_coefficient must be provided")
        self.base_coefficient = base_coefficient
        self.ball_area = ball_area
        self.ball_radius = ball_radius
        self.max_coefficient = max_coefficient

    def calculate(
        self,
        velocity: np.ndarray,
        spin: np.ndarray,
        air_density: float = float(AIR_DENSITY_SEA_LEVEL_KG_M3),
    ) -> np.ndarray:
        """Calculate lift force from spin."""
        motion = _active_motion_vectors(velocity, spin)
        if motion is None:
            return np.zeros(3)
        velocity_vector, spin_vector, speed, spin_magnitude = motion

        spin_axis = spin_vector / spin_magnitude
        lift_dir = np.cross(spin_axis, velocity_vector)
        lift_norm = float(
            math.hypot(lift_dir[0], lift_dir[1], lift_dir[2])
        )  # ⚡ Bolt: Explicit component unpacking is faster than *args expansion

        if lift_norm < 1e-10:
            return np.zeros_like(velocity_vector)

        lift_dir = lift_dir / lift_norm
        spin_ratio = self.ball_radius * spin_magnitude / speed
        cl = self._compute_lift_coefficient(spin_ratio)
        force_magnitude = 0.5 * air_density * cl * self.ball_area * speed**2
        return force_magnitude * lift_dir

    def _compute_lift_coefficient(self, spin_ratio: float) -> float:
        """Compute lift coefficient based on spin ratio."""
        if spin_ratio is None:
            raise ValueError("spin_ratio must be provided")
        return min(
            self.max_coefficient,
            calculate_spin_lift_coefficient(float(spin_ratio)),
        )


class MagnusModel:
    """Model for the single spin-induced Magnus/lift force.

    The spin force is perpendicular to both velocity and spin axis, causing
    lift for backspin and hook/slice curvature for sidespin.
    """

    def __init__(
        self,
        coefficient: float = float("inf"),
        ball_area: float = float(GOLF_BALL_CROSS_SECTIONAL_AREA_M2),
        ball_radius: float = float(GOLF_BALL_RADIUS_M),
    ) -> None:
        if coefficient is None:
            raise ValueError("coefficient must be provided")
        self.coefficient = coefficient
        self.ball_area = ball_area
        self.ball_radius = ball_radius

    def calculate(
        self,
        velocity: np.ndarray,
        spin: np.ndarray,
        air_density: float = float(AIR_DENSITY_SEA_LEVEL_KG_M3),
    ) -> np.ndarray:
        """Calculate Magnus force."""
        motion = _active_motion_vectors(velocity, spin)
        if motion is None:
            return np.zeros(3)
        velocity_vector, spin_vector, speed, spin_magnitude = motion

        magnus_dir = np.cross(spin_vector, velocity_vector)
        magnus_norm = float(
            math.hypot(magnus_dir[0], magnus_dir[1], magnus_dir[2])
        )  # ⚡ Bolt: Explicit component unpacking is faster than *args expansion

        if magnus_norm < 1e-10:
            return np.zeros_like(velocity_vector)

        magnus_dir = magnus_dir / magnus_norm
        spin_param = self.ball_radius * spin_magnitude / speed
        cm = self._compute_magnus_coefficient(spin_param)
        force_magnitude = 0.5 * air_density * cm * self.ball_area * speed**2
        return force_magnitude * magnus_dir

    def _compute_magnus_coefficient(self, spin_param: float) -> float:
        """Compute Magnus/lift coefficient based on spin parameter."""
        return min(
            self.coefficient,
            calculate_spin_lift_coefficient(float(spin_param)),
        )


__all__ = ["DragModel", "LiftModel", "MagnusModel"]
