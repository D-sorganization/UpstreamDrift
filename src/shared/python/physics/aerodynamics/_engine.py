"""AerodynamicsEngine: unified force calculation orchestrator.

Combines DragModel and one spin-induced Magnus/lift model with optional wind and
environment randomization. The legacy lift output slot stays zero in combined
kernels so spin lift is not double-counted.
"""

from __future__ import annotations

import math

import numpy as np

from src.shared.python.core.contracts import precondition
from src.shared.python.core.physics_constants import AIR_DENSITY_SEA_LEVEL_KG_M3

from ._config import AerodynamicsConfig
from ._environment import EnvironmentRandomizer
from ._models import DragModel, LiftModel, MagnusModel
from ._wind import WindModel


class AerodynamicsEngine:
    """Unified aerodynamics calculation engine.

    Combines aerodynamic force models with optional wind and
    environment randomization. Drag and the single spin force can be toggled
    on/off; the legacy lift slot remains present for compatibility.

    Example:
        >>> config = AerodynamicsConfig(drag_enabled=True, lift_enabled=True)
        >>> engine = AerodynamicsEngine(config)
        >>> forces = engine.compute_forces(velocity, spin)
        >>> print(forces['total'])
    """

    def __init__(
        self,
        config: AerodynamicsConfig | None = None,
        wind_model: WindModel | None = None,
        randomization: EnvironmentRandomizer | None = None,
        air_density: float = float(AIR_DENSITY_SEA_LEVEL_KG_M3),
    ) -> None:
        if air_density is None:
            raise ValueError("air_density must be provided")
        self.config = config or AerodynamicsConfig()
        self.wind_model = wind_model
        self.randomization = randomization
        self._base_air_density = air_density
        self._current_air_density = air_density

        self._drag = DragModel(
            base_coefficient=self.config.drag_coefficient,
            ball_area=self.config.ball_area,
            ball_radius=self.config.ball_radius,
            reynolds_correction=self.config.reynolds_correction_enabled,
        )
        self._lift = LiftModel(
            base_coefficient=self.config.lift_coefficient,
            ball_area=self.config.ball_area,
            ball_radius=self.config.ball_radius,
        )
        self._magnus = MagnusModel(
            coefficient=self.config.magnus_coefficient,
            ball_area=self.config.ball_area,
            ball_radius=self.config.ball_radius,
        )

    @precondition(
        lambda self, velocity, spin, t=0.0, position=None, resample=False: (
            np.ndim(velocity) == 1 and len(velocity) == 3
        ),
        "velocity must be a 1-D array of length 3",
    )
    @precondition(
        lambda self, velocity, spin, t=0.0, position=None, resample=False: (
            np.ndim(spin) == 1 and len(spin) == 3
        ),
        "spin must be a 1-D array of length 3",
    )
    def compute_forces(
        self,
        velocity: np.ndarray,
        spin: np.ndarray,
        t: float = 0.0,
        position: np.ndarray | None = None,
        resample: bool = False,
    ) -> dict[str, np.ndarray]:
        """Compute all aerodynamic forces.

        Args:
            velocity: Ball velocity [m/s]
            spin: Angular velocity [rad/s]
            t: Current time [s] (for wind variation)
            position: Current position [m] (for wind gradient)
            resample: Resample random environment

        Returns:
            Dictionary with 'drag', 'lift', 'magnus', and 'total' forces [N]
        """
        if velocity is None:
            raise ValueError("velocity must be provided")
        if position is None:
            position = np.zeros(3)

        if resample and self.randomization:
            self._current_air_density = self.randomization.randomize_air_density(
                self._base_air_density
            )

        wind = np.zeros(3)
        if self.wind_model:
            wind = self.wind_model.get_wind_at(t, position)

        rel_velocity = velocity - wind

        drag = np.zeros(3)
        lift = np.zeros(3)
        magnus = np.zeros(3)

        if self.config.is_drag_active():
            drag = self._drag.calculate(rel_velocity, self._current_air_density)

        if self.config.is_magnus_active():
            magnus = self._lift.calculate(rel_velocity, spin, self._current_air_density)

        total = drag + magnus

        return {
            "drag": drag,
            "lift": lift,
            "magnus": magnus,
            "total": total,
        }

    @precondition(
        lambda self, velocity, spin, mass, t=0.0, position=None, resample=False: (
            mass > 0
        ),
        "mass must be positive (non-zero, non-negative) to avoid ZeroDivisionError",
    )
    def compute_acceleration(
        self,
        velocity: np.ndarray,
        spin: np.ndarray,
        mass: float,
        t: float = 0.0,
        position: np.ndarray | None = None,
        resample: bool = False,
    ) -> np.ndarray:
        """Compute acceleration from aerodynamic forces.

        Args:
            velocity: Ball velocity [m/s]
            spin: Angular velocity [rad/s]
            mass: Ball mass [kg] — must be positive
            t: Current time [s]
            position: Current position [m]
            resample: Resample random environment

        Returns:
            Acceleration vector [m/s^2]

        Raises:
            PreconditionError: If mass <= 0
        """
        if velocity is None:
            raise ValueError("velocity must be provided")
        forces = self.compute_forces(velocity, spin, t, position, resample)
        return forces["total"] / mass

    def compute_spin_decay(self, spin: np.ndarray, dt: float) -> np.ndarray:
        """Compute spin decay over time step (exponential decay).

        Args:
            spin: Current angular velocity [rad/s]
            dt: Time step [s]

        Returns:
            Updated spin after decay [rad/s]
        """
        if spin is None:
            raise ValueError("spin must be provided")
        decay_factor = math.exp(-self.config.spin_decay_rate * dt)
        return spin * decay_factor


__all__ = ["AerodynamicsEngine"]
