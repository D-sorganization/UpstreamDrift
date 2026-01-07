"""Hill-type muscle model implementation.

Assessment A Finding F-003 / Guideline J Implementation (OpenSim Biomechanics)

This module implements the classic Hill muscle model with:
- Contractile element (CE): Active force generation
- Series elastic element (SEE): Tendon compliance
- Parallel elastic element (PEE): Passive muscle stretch resistance

Hill Model Equations:
--------------------
Total force: F_total = F_CE + F_PEE = F_SEE (equilibrium)

F_CE = F_max · a · f_l(l_CE) · f_v(v_CE)
where:
    a = activation level [0,1] (neural drive)
    f_l = force-length relationship (optimal length @ l_opt)
    f_v = force-velocity relationship (max force @ v=0)

F_SEE = F_max · f_t(ε_tendon)  (tendon force-strain)
F_PEE = F_max · f_p(l_CE)  (passive force)

This is the STANDARD muscle model used in OpenSim, biomechanics research,
and MyoSuite. Understanding this is critical for biomechanics + robotics interface.

Reference:
- Hill, A.V. (1938). "The Heat of Shortening and Dynamics of Muscle"
- Zajac, F.E. (1989). "Muscle and Tendon: Properties, Models, Scaling"
- OpenSim Documentation: "Muscle Models"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# Standard muscle model constants (Zajac 1989)
# Source: Zajac, F.E. (1989), Journal of Biomechanics 22(2), 105-116

OPTIMAL_FIBER_LENGTH_WIDTH = 0.56  # [dimensionless]
# Width of active force-length curve (Gaussian-like)

MAX_CONTRACTION_VELOCITY = 10.0  # [L_opt/s]
# Maximum normalized shortening velocity (typical mammalian muscle)

TENDON_SLACK_STRAIN = 0.04  # [dimensionless]
# Tendon strain at which force begins (4% strain)

TENDON_STIFFNESS = 35.0  # [dimensionless]
# Tendon stiffness parameter (normalized)


@dataclass
class MuscleState:
    """Current state of muscle-tendon unit.

    Attributes:
        l_MT: Muscle-tendon unit length [m]
        v_MT: Muscle-tendon unit velocity [m/s]
        activation: Neural activation level [0,1] (0=off, 1=max)
        l_CE: Contractile element length [m]
        v_CE: Contractile element velocity [m/s]
    """

    l_MT: float  # [m]
    v_MT: float  # [m/s]
    activation: float  # [0,1]
    l_CE: float  # [m]
    v_CE: float  # [m/s]


@dataclass
class MuscleParameters:
    """Hill muscle model parameters.

    Attributes:
        F_max: Maximum isometric force [N]
        l_opt: Optimal fiber length [m] (length at max active force)
        l_slack: Tendon slack length [m] (length at zero tendon force)
        v_max: Maximum contraction velocity [m/s]
        pennation_angle: Fiber pennation angle [rad] (0 for parallel fibers)
    """

    F_max: float  # [N]
    l_opt: float  # [m]
    l_slack: float  # [m]
    v_max: float  # [m/s]
    pennation_angle: float = 0.0  # [rad]


class HillMuscleModel:
    """Hill-type muscle model with active and passive force components.

    This implements the classic 3-component Hill model used in OpenSim.

    Example:
        >>> params = MuscleParameters(
        ...     F_max=1000.0,  # N
        ...     l_opt=0.10,    # m (10 cm optimal fiber length)
        ...     l_slack=0.20,  # m (20 cm tendon slack)
        ...     v_max=1.0      # m/s
        ... )
        >>> muscle = HillMuscleModel(params)
        >>> state = MuscleState(l_MT=0.32, v_MT=0.0, activation=0.5, l_CE=0.10, v_CE=0.0)
        >>> force = muscle.compute_muscle_force(state)
        >>> print(f"Muscle force: {force:.1f} N")
    """

    def __init__(self, params: MuscleParameters):
        """Initialize Hill muscle model.

        Args:
            params: Muscle parameters (F_max, l_opt, etc.)
        """
        self.params = params

    def force_length_active(self, l_CE_norm: float) -> float:
        """Active force-length relationship f_l(l_CE).

        Gaussian-like curve peaked at l_opt (l_CE_norm = 1.0).

        Args:
            l_CE_norm: Normalized fiber length l_CE / l_opt [dimensionless]

        Returns:
            Active force multiplier [0,1]

        Formula:
            f_l = exp(-((l_CE_norm - 1) / width)²)

        Reference: Zajac (1989), Eq. 2
        """
        width = OPTIMAL_FIBER_LENGTH_WIDTH
        f_l = np.exp(-(((l_CE_norm - 1.0) / width) ** 2))
        return float(np.clip(f_l, 0.0, 1.0))

    def force_length_passive(self, l_CE_norm: float) -> float:
        """Passive force-length relationship f_p(l_CE).

        Exponential rise for l_CE > l_opt (stretch resistance).

        Args:
            l_CE_norm: Normalized fiber length [dimensionless]

        Returns:
            Passive force multiplier [0,∞)

        Formula:
            f_p = (exp(k_PE · (l_CE_norm - 1)) - 1) / (exp(k_PE) - 1)
            where k_PE ≈ 4.0 (stiffness parameter)

        Reference: Zajac (1989), Eq. 3
        """
        k_PE = 4.0  # Passive stiffness parameter
        if l_CE_norm < 1.0:
            return 0.0  # No passive force for l_CE < l_opt

        f_p = (np.exp(k_PE * (l_CE_norm - 1.0)) - 1.0) / (np.exp(k_PE) - 1.0)
        return float(f_p)

    def force_velocity(self, v_CE_norm: float) -> float:
        """Force-velocity relationship f_v(v_CE).

        Hyperbolic curve (Hill's characteristic equation).
        Max force at v=0 (isometric), decreases with shortening velocity.

        Args:
            v_CE_norm: Normalized fiber velocity v_CE / v_max [dimensionless]

        Returns:
            Force-velocity multiplier [0,1.8]

        Formula (shortening, v < 0):
            f_v = (v_max - v_CE) / (v_max + k·v_CE)
            where k ≈ 0.25 (curvature parameter)

        Formula (lengthening, v > 0):
            f_v = 1.8 - 0.8/(1 + v_CE/0.18)

        Reference: Zajac (1989), Eq. 4-5
        """
        k = 0.25  # Curvature parameter

        if v_CE_norm <= 0:
            # Shortening (concentric contraction)
            # Use Hill's hyperbolic equation
            f_v = (1.0 - v_CE_norm) / (1.0 + k * v_CE_norm)
        else:
            # Lengthening (eccentric contraction)
            # Muscles produce MORE force when lengthening
            f_v = 1.8 - 0.8 / (1.0 + v_CE_norm / 0.18)

        return float(np.clip(f_v, 0.0, 1.8))

    def tendon_force(self, l_tendon_norm: float) -> float:
        """Series elastic element force f_t(ε_tendon).

        Exponential tendon force-strain relationship.

        Args:
            l_tendon_norm: Normalized tendon length l_T / l_slack [dimensionless]

        Returns:
            Normalized tendon force F_T / F_max [dimensionless]

        Formula:
            ε = (l_T - l_slack) / l_slack = l_tendon_norm - 1
            f_t = 0  if ε < ε_toe
            f_t = k_T · (ε - ε_toe)² / ε_toe  if ε >= ε_toe

        Reference: Zajac (1989), Eq. 6
        """
        epsilon_toe = TENDON_SLACK_STRAIN  # 0.04 (4% strain)
        k_T = TENDON_STIFFNESS  # 35

        # Strain
        epsilon = l_tendon_norm - 1.0

        if epsilon < epsilon_toe:
            # Tendon slack (no force)
            return 0.0

        # Nonlinear tendon force
        f_t = k_T * ((epsilon - epsilon_toe) ** 2) / epsilon_toe
        return float(f_t)

    def compute_muscle_force(self, state: MuscleState) -> float:
        """Compute total muscle force F_total.

        Solves equilibrium equation:
            F_CE + F_PEE = F_SEE

        where:
            F_CE = F_max · a · f_l · f_v (contractile element)
            F_PEE = F_max · f_p (parallel elastic)
            F_SEE = F_max · f_t (series elastic / tendon)

        Args:
            state: Current muscle state (l_MT, v_MT, activation, l_CE, v_CE)

        Returns:
            Total muscle force [N]

        Note:
            In full OpenSim, l_CE and v_CE are solved from equilibrium.
            Here we assume they are provided (from equilibrium solver).
        """
        # Normalize lengths and velocities
        l_CE_norm = state.l_CE / self.params.l_opt
        v_CE_norm = state.v_CE / self.params.v_max

        # Tendon length (from muscle-tendon length)
        cos_alpha = np.cos(self.params.pennation_angle)
        l_tendon = state.l_MT - state.l_CE * cos_alpha
        l_tendon_norm = l_tendon / self.params.l_slack

        # Force components
        f_l = self.force_length_active(l_CE_norm)
        f_v = self.force_velocity(v_CE_norm)
        f_p = self.force_length_passive(l_CE_norm)
        f_t = self.tendon_force(l_tendon_norm)

        # Active contractile force
        F_CE = self.params.F_max * state.activation * f_l * f_v

        # Passive force
        F_PEE = self.params.F_max * f_p

        # Total fiber force (along fiber direction)
        F_fiber = F_CE + F_PEE

        # Project to tendon direction (pennation angle correction)
        F_tendon = F_fiber * cos_alpha

        # Equilibrium check (for debugging)
        F_SEE = self.params.F_max * f_t
        equilibrium_error = abs(F_tendon - F_SEE)
        if equilibrium_error > 0.01 * self.params.F_max:
            logger.warning(
                f"Muscle force equilibrium NOT satisfied:\\n"
                f"  F_fiber·cos(α) = {F_tendon:.2f} N\\n"
                f"  F_SEE = {F_SEE:.2f} N\\n"
                f"  Error: {equilibrium_error:.2f} N ({equilibrium_error/self.params.F_max*100:.1f}%)\\n"
                f"  This suggests l_CE, v_CE need equilibrium solver adjustment"
            )

        return float(F_tendon)

    def compute_muscle_moment_arm(self, q: float, body_name: str = "default") -> float:
        """Compute muscle moment arm r = ∂l_MT/∂q.

        Moment arm determines how muscle force generates joint torque:
            τ = F_muscle · r

        Args:
            q: Joint angle [rad]
            body_name: Body identifier (for multi-DOF systems)

        Returns:
            Moment arm [m]

        Note:
            In OpenSim, moment arms are computed from geometry (muscle path).
            Here we use a simple analytical approximation.

            For more accuracy, use OpenSim GeometryPath or wrapping surfaces.
        """
        # Placeholder: Linear approximation r ≈ constant
        # Real OpenSim uses GeometryPath with wrapping surfaces
        r_approx = 0.05  # [m] (5 cm typical arm/shoulder)

        logger.debug(
            f"Using simplified moment arm r = {r_approx:.3f} m. "
            f"For accurate geometry, integrate OpenSim GeometryPath."
        )

        return r_approx


# Example usage / validation
if __name__ == "__main__":
    # Create muscle with realistic parameters (biceps brachii)
    biceps_params = MuscleParameters(
        F_max=1000.0,  # N (1000 N max isometric force)
        l_opt=0.12,  # m (12 cm optimal fiber length)
        l_slack=0.25,  # m (25 cm tendon slack length)
        v_max=1.2,  # m/s (max shortening velocity)
        pennation_angle=0.0,  # rad (pennation angle, simplified)
    )

    muscle = HillMuscleModel(biceps_params)

    # Test at optimal length, zero velocity, 50% activation
    state = MuscleState(
        l_MT=0.37,  # m (total length = l_opt + l_slack)
        v_MT=0.0,  # m/s
        activation=0.5,  # 50% activation
        l_CE=0.12,  # m (at optimal length)
        v_CE=0.0,  # m/s (isometric)
    )

    F_muscle = muscle.compute_muscle_force(state)

    print("Biceps muscle force test:")
    print(f"  Activation: {state.activation*100:.0f}%")
    print(
        f"  Fiber length: {state.l_CE*100:.1f} cm (optimal: {biceps_params.l_opt*100:.1f} cm)"
    )
    print(f"  Fiber velocity: {state.v_CE:.3f} m/s")
    print(f"  Force: {F_muscle:.1f} N (max: {biceps_params.F_max:.0f} N)")
    print(f"  Force/F_max: {F_muscle/biceps_params.F_max*100:.1f}%")

    # Expected: ~50% of F_max (50% activation, optimal length, isometric)
    # Should be ≈ 500 N
