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
    MAGNUS_COEFFICIENT,
)
from src.shared.python.physics.atmosphere import cd_dimpled_sphere


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
        self.base_coefficient = base_coefficient
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
        if velocity is None:
            raise ValueError("velocity must be provided")
        speed = float(
            math.sqrt(np.dot(velocity, velocity))
        )  # ⚡ Bolt: math.sqrt(np.dot) is ~3x faster than np.linalg.norm
        if speed < 1e-10:
            return np.zeros(3)

        cd = self.get_effective_coefficient(velocity, air_density)
        force_magnitude = 0.5 * air_density * cd * self.ball_area * speed**2
        return -force_magnitude * velocity / speed

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
        if velocity is None:
            raise ValueError("velocity must be provided")
        if not self.reynolds_correction:
            return self.base_coefficient

        speed = float(
            math.sqrt(np.dot(velocity, velocity))
        )  # ⚡ Bolt: math.sqrt(np.dot) is ~3x faster than np.linalg.norm
        if speed < 1e-10:
            return self.base_coefficient

        viscosity = float(AIR_VISCOSITY_KG_M_S)
        diameter = 2 * self.ball_radius
        re = air_density * speed * diameter / viscosity
        # Clamp to the model's supported range; outside it the underlying
        # correlation extrapolation is meaningless for golf balls.
        re_clamped = max(1.0e3, min(1.0e7, re))
        return cd_dimpled_sphere(re_clamped, base_cd=self.base_coefficient)


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
        max_coefficient: float = 0.4,
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
        if velocity is None:
            raise ValueError("velocity must be provided")
        speed = float(
            math.sqrt(np.dot(velocity, velocity))
        )  # ⚡ Bolt: math.sqrt(np.dot) is ~3x faster than np.linalg.norm
        spin_magnitude = float(
            math.sqrt(np.dot(spin, spin))
        )  # ⚡ Bolt: math.sqrt(np.dot) is ~3x faster than np.linalg.norm

        if speed < 1e-10 or spin_magnitude < 1e-10:
            return np.zeros(3)

        spin_axis = spin / spin_magnitude
        lift_dir = np.cross(spin_axis, velocity)
        lift_norm = float(math.hypot(*lift_dir))

        if lift_norm < 1e-10:
            return np.zeros(3)

        lift_dir = lift_dir / lift_norm
        spin_ratio = self.ball_radius * spin_magnitude / speed
        cl = self._compute_lift_coefficient(spin_ratio)
        force_magnitude = 0.5 * air_density * cl * self.ball_area * speed**2
        return force_magnitude * lift_dir

    def _compute_lift_coefficient(self, spin_ratio: float) -> float:
        """Compute lift coefficient based on spin ratio."""
        if spin_ratio is None:
            raise ValueError("spin_ratio must be provided")
        cl = self.max_coefficient * (1 - math.exp(-spin_ratio / 0.1))
        return min(cl, self.max_coefficient)


class MagnusModel:
    """Model for Magnus force from spin.

    The Magnus effect creates a force perpendicular to both velocity
    and spin axis, causing hook/slice for sidespin.
    """

    def __init__(
        self,
        coefficient: float = float(MAGNUS_COEFFICIENT),
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
        if velocity is None:
            raise ValueError("velocity must be provided")
        speed = float(
            math.sqrt(np.dot(velocity, velocity))
        )  # ⚡ Bolt: math.sqrt(np.dot) is ~3x faster than np.linalg.norm
        spin_magnitude = float(
            math.sqrt(np.dot(spin, spin))
        )  # ⚡ Bolt: math.sqrt(np.dot) is ~3x faster than np.linalg.norm

        if speed < 1e-10 or spin_magnitude < 1e-10:
            return np.zeros(3)

        magnus_dir = np.cross(spin, velocity)
        magnus_norm = float(math.hypot(*magnus_dir))

        if magnus_norm < 1e-10:
            return np.zeros(3)

        magnus_dir = magnus_dir / magnus_norm
        spin_param = self.ball_radius * spin_magnitude / speed
        cm = self._compute_magnus_coefficient(spin_param)
        force_magnitude = 0.5 * air_density * cm * self.ball_area * speed**2
        return force_magnitude * magnus_dir

    def _compute_magnus_coefficient(self, spin_param: float) -> float:
        """Compute Magnus coefficient based on spin parameter."""
        return self.coefficient * min(spin_param / 0.2, 1.0)


__all__ = ["DragModel", "LiftModel", "MagnusModel"]
