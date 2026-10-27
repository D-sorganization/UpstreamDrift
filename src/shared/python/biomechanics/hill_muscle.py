"""Hill-type muscle model implementation.

This module implements a standard Hill-type muscle model commonly used in biomechanics
and robotics (e.g., OpenSim, MuJoCo).

Components:
1. Contractile Element (CE): Generates active force (f_l * f_v * a)
2. Parallel Elastic Element (PEE): Passive resistance to stretch
3. Series Elastic Element (SEE): Tendon elasticity

The total muscle-tendon force is:
F_mt = F_tendon = (F_CE + F_PEE) * cos(alpha)

Reference:
- Hill (1938), "The Heat of Shortening and the Dynamic Constants of Muscle"
- Zajac (1989), "Muscle and Tendon: Properties, Models, Scaling..."
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from src.shared.python.core.contracts import ensure, require
from src.shared.python.logging_pkg.logging_config import get_logger

logger = logging.getLogger(__name__)
logger = get_logger(__name__)


@dataclass
class MuscleParameters:
    """Parameters defining a specific muscle."""

    F_max: float  # Maximum isometric force [N]
    l_opt: float  # Optimal fiber length [m]
    l_slack: float  # Tendon slack length [m]
    v_max: float = 10.0  # Max contraction velocity [l_opt/s] (default ~10)
    pennation_angle: float = 0.0  # Pennation angle at optimal length [rad]
    damping: float = 0.05  # Passive damping [N*s/m] (stabilization)

    def __post_init__(self) -> None:
        """Validate parameters."""
        if self.F_max <= 0:
            msg = "F_max must be positive"
            raise ValueError(msg)
        if self.l_opt <= 0:
            msg = "l_opt must be positive"
            raise ValueError(msg)
        if self.l_slack <= 0:
            msg = "l_slack must be positive"
            raise ValueError(msg)


@dataclass
class MuscleState:
    """Current state of the muscle."""

    activation: float = 0.0  # Current activation [0, 1]
    l_CE: float = 0.0  # Current fiber length [m]
    v_CE: float = 0.0  # Current fiber velocity [m/s]
    l_MT: float = 0.0  # Current muscle-tendon length [m]


class HillMuscleModel:
    """Standard Hill-type muscle model.

    Computes forces based on length, velocity, and activation.

    Force generation:
    F_CE = F_max * a * f_l(l_CE) * f_v(v_CE)
    F_PEE = F_max * f_p(l_CE)
    F_total = (F_CE + F_PEE) * cos(alpha)
    """

    #: Default width of the active force-length Gaussian curve.
    #: From Thelen (2003), J. Biomech. Eng., 125(1), pp. 70-77.
    DEFAULT_FORCE_LENGTH_GAMMA: float = 0.45
    DEFAULT_FORCE_LENGTH_WIDTH: float = 0.45  # Backward compatible alias for gamma
    DEFAULT_TENDON_STRAIN_AT_FMAX: float = 0.04  # Thelen (2003) reference strain

    def __init__(
        self,
        params: MuscleParameters,
        force_length_width: float | None = None,
        force_length_gamma: float | None = None,
    ) -> None:
        """Initialize muscle model.

        Args:
            params: MuscleParameters dataclass
            force_length_width: Alias for Gaussian shape parameter gamma (dimensionless).
            force_length_gamma: Gaussian shape parameter gamma (dimensionless).
                Default 0.45 from Thelen (2003).
        """
        self.params = params
        gamma = (
            force_length_gamma if force_length_gamma is not None else force_length_width
        )
        self._force_length_gamma = (
            gamma if gamma is not None else self.DEFAULT_FORCE_LENGTH_GAMMA
        )
        self._force_length_width = self._force_length_gamma

    def force_length_active(self, l_norm: float) -> float:
        """Active force-length relationship (Gaussian curve).

        From Thelen (2003), J. Biomech. Eng., 125(1), pp. 70-77, Eq. 2:
            f_L = exp(-(l_norm - 1)^2 / gamma)
        with shape parameter gamma = 0.45.

        Args:
            l_norm: Normalized fiber length (l_CE / l_opt)

        Returns:
            Force multiplier [0, 1]
        """
        if l_norm is None:
            raise ValueError("l_norm must be provided")
        if l_norm <= 0:
            return 0.0
        gamma = self._force_length_gamma
        return float(np.exp(-((l_norm - 1.0) ** 2) / gamma))

    def force_length_passive(self, l_norm: float) -> float:
        """Passive force-length relationship (Exponential spring).

        Args:
            l_norm: Normalized fiber length (l_CE / l_opt)

        Returns:
            Force multiplier [0, inf)
        """
        if l_norm is None:
            raise ValueError("l_norm must be provided")
        if l_norm <= 1.0:
            return 0.0
        # Typical exponential passive curve
        k_passive = 4.0
        return float(
            (np.exp(k_passive * (l_norm - 1.0)) - 1.0) / (np.exp(k_passive) - 1.0)
        )

    def force_velocity(self, v_norm: float) -> float:
        """Force-velocity relationship (Hill's Hyperbola).

        Args:
            v_norm: Normalized velocity (v_CE / v_max_m_s)
                   Positive = lengthening (eccentric)
                   Negative = shortening (concentric)

        Returns:
            Force multiplier [0, 1.4]
        """
        # Concentric (shortening)
        if v_norm is None:
            raise ValueError("v_norm must be provided")
        if v_norm < 0:
            # Hill's hyperbola: clamp v_norm to prevent division by zero
            # The denominator (1 - v_norm / 0.25) = 0 when v_norm = 0.25
            # Since v_norm < 0 for concentric, we clamp at -0.99 * v_max
            v_norm_clamped = max(v_norm, -0.99)
            return float((1 + v_norm_clamped) / (1 - v_norm_clamped / 0.25))

        # Eccentric (lengthening)
        # Force increases up to ~1.4x F_max
        return float((1 + v_norm * 1.4 / 0.10) / (1 + v_norm / 0.10))

    def tendon_force(
        self, l_tendon_norm: float, epsilon_0: float | None = None
    ) -> float:
        """Tendon force-length relationship (Non-linear toe + linear spring).

        From Thelen (2003), J. Biomech. Eng., 125(1), pp. 70-77, Eq. 3,
        and Zajac (1989):
        Normalized tendon force reaches F_max (1.0) at the reference strain
        epsilon_0 (typically 0.033 - 0.04).

        Args:
            l_tendon_norm: Normalized tendon length (l_tendon / l_slack)
            epsilon_0: Tendon strain at F_max (default 0.04 per Thelen 2003).

        Returns:
            Force multiplier [0, inf)
        """
        if l_tendon_norm is None:
            raise ValueError("l_tendon_norm must be provided")
        if l_tendon_norm <= 1.0:
            return 0.0
        strain = l_tendon_norm - 1.0
        eps_0 = (
            epsilon_0 if epsilon_0 is not None else self.DEFAULT_TENDON_STRAIN_AT_FMAX
        )
        f_toe = 0.33
        k_toe = 3.0
        eps_toe = 0.609 * eps_0
        k_lin = 1.75 / eps_0
        if strain <= eps_toe:
            return float(
                f_toe * (np.exp(k_toe * strain / eps_toe) - 1.0) / (np.exp(k_toe) - 1.0)
            )
        return float(f_toe + k_lin * (strain - eps_toe))

    def compute_force(self, state: MuscleState) -> float:
        """Compute total muscle force generated at the tendon.

        Design by Contract:
            Preconditions:
                - activation in [0, 1]
            Postconditions:
                - returned force >= 0

        Args:
            state: Current MuscleState

        Returns:
            Force at the tendon [N]
        """
        if state is None:
            raise ValueError("state must be provided")
        require(
            0.0 <= state.activation <= 1.0,
            "activation must be in [0, 1]",
            state.activation,
        )

        # Normalize inputs
        l_norm = state.l_CE / self.params.l_opt
        v_norm = state.v_CE / (self.params.v_max * self.params.l_opt)

        # Force components
        f_l = self.force_length_active(l_norm)
        f_v = self.force_velocity(v_norm)
        f_p = self.force_length_passive(l_norm)

        # Active and passive muscle forces
        F_active = self.params.F_max * state.activation * f_l * f_v
        F_passive = self.params.F_max * f_p

        # Damping (parallel)
        F_damping = self.params.damping * state.v_CE

        # Total fiber force projected to tendon line of action
        cos_alpha = np.cos(self.params.pennation_angle)  # Simplified (constant angle)

        F_total = (F_active + F_passive + F_damping) * cos_alpha

        result = float(max(0.0, F_total))
        ensure(result >= 0, "muscle force must be non-negative", result)
        return result


# Example usage
if __name__ == "__main__":
    # Define a generic muscle (e.g., Biceps)
    biceps_params = MuscleParameters(
        F_max=1000.0,
        l_opt=0.15,
        l_slack=0.20,
        v_max=10.0,
    )

    muscle = HillMuscleModel(biceps_params)

    # Test state
    state = MuscleState(
        activation=0.8,
        l_CE=0.15,
        v_CE=0.0,
        l_MT=0.35,  # At optimal length  # Isometric
    )

    force = muscle.compute_force(state)
    logger.info(f"Muscle force: {force:.1f} N")

    # Verify scaling
    F_muscle = muscle.compute_force(state)

    logger.info("Biceps muscle force test:")
    logger.info(f"  Activation: {state.activation * 100:.0f}%")
    logger.info(
        f"  Fiber length: {state.l_CE:.3f} m (opt: {biceps_params.l_opt:.3f} m)"
    )
    logger.info(f"  Fiber velocity: {state.v_CE:.3f} m/s")
    logger.info(f"  Force: {F_muscle:.1f} N (max: {biceps_params.F_max:.0f} N)")
    logger.info(f"  Force/F_max: {F_muscle / biceps_params.F_max * 100:.1f}%")
