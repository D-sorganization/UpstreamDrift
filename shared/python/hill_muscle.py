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

logger = logging.getLogger(__name__)


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

    def __init__(self, params: MuscleParameters):
        """Initialize muscle model.

        Args:
            params: MuscleParameters dataclass
        """
        self.params = params

    def force_length_active(self, l_norm: float) -> float:
        """Active force-length relationship (Gaussian-like curve).

        Args:
            l_norm: Normalized fiber length (l_CE / l_opt)

        Returns:
            Force multiplier [0, 1]
        """
        # Width of the force-length curve
        width = 0.56
        return float(np.exp(-((l_norm - 1.0) ** 2) / width**2))

    def force_length_passive(self, l_norm: float) -> float:
        """Passive force-length relationship (Exponential spring).

        Args:
            l_norm: Normalized fiber length (l_CE / l_opt)

        Returns:
            Force multiplier [0, inf)
        """
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
            Force multiplier [0, 1.8]
        """
        # Concentric (shortening)
        if v_norm < 0:
            # Hill's hyperbola: clamp v_norm to prevent division by zero
            # The denominator (1 - v_norm / 0.25) = 0 when v_norm = 0.25
            # Since v_norm < 0 for concentric, we clamp at -0.99 * v_max
            v_norm_clamped = max(v_norm, -0.99)
            return float((1 + v_norm_clamped) / (1 - v_norm_clamped / 0.25))

        # Eccentric (lengthening)
        # Force increases up to ~1.4x F_max
        return float((1 + v_norm * 1.4 / 0.10) / (1 + v_norm / 0.10))

    def tendon_force(self, l_tendon_norm: float) -> float:
        """Tendon force-length relationship (Non-linear spring).

        Args:
            l_tendon_norm: Normalized tendon length (l_tendon / l_slack)

        Returns:
            Force multiplier [0, inf)
        """
        if l_tendon_norm <= 1.0:
            return 0.0
        # Non-linear stiffening region (toe region)
        # Often modeled as exponential or quadratic.
        # Here using a simple quadratic for stability.
        strain = l_tendon_norm - 1.0
        return float(10.0 * strain**2)  # Stiffness coefficient

    def compute_force(self, state: MuscleState) -> float:
        """Compute total muscle force generated at the tendon.

        Args:
            state: Current MuscleState

        Returns:
            Force at the tendon [N]
        """
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

        return float(max(0.0, F_total))


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
        activation=0.8, l_CE=0.15, v_CE=0.0, l_MT=0.35  # At optimal length  # Isometric
    )

    force = muscle.compute_force(state)
    print(f"Muscle force: {force:.1f} N")  # noqa: T201

    # Verify scaling
    F_muscle = muscle.compute_force(state)

    print("Biceps muscle force test:")  # noqa: T201
    print(f"  Activation: {state.activation*100:.0f}%")  # noqa: T201
    print(  # noqa: T201
        f"  Fiber length: {state.l_CE:.3f} m (opt: {biceps_params.l_opt:.3f} m)"
    )
    print(f"  Fiber velocity: {state.v_CE:.3f} m/s")  # noqa: T201
    print(f"  Force: {F_muscle:.1f} N (max: {biceps_params.F_max:.0f} N)")  # noqa: T201
    print(f"  Force/F_max: {F_muscle/biceps_params.F_max*100:.1f}%")  # noqa: T201
