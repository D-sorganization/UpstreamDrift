"""Muscle equilibrium solver for Hill muscle models.

This module solves the muscle-tendon equilibrium equation to find fiber
length (l_CE) and velocity (v_CE) given muscle-tendon unit length (l_MT).

Equilibrium Equation:
--------------------
F_fiber(l_CE, v_CE, a) = F_tendon(l_MT - l_CE)

where:
    F_fiber = (F_max · a · f_l(l_CE) · f_v(v_CE) + F_max · f_p(l_CE)) · cos(α)
    F_tendon = F_max · f_t((l_MT - l_CE·cos(α) - l_slack) / l_slack)

This is a nonlinear root-finding problem solved via Newton-Raphson iteration.

In OpenSim, this is called the "muscle equilibrium computation" and is critical
for muscle-driven forward dynamics.

Reference:
- Millard et al. (2013), "Flexing Computational Muscle", J. Biomech. Eng.
- OpenSim Documentation: "Muscle Equilibrium"
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import newton

from src.shared.python.logging_config import get_logger

if TYPE_CHECKING:
    from shared.python.hill_muscle import HillMuscleModel

logger = get_logger(__name__)

# Solver parameters
MAX_ITERATIONS = 100  # Maximum Newton-Raphson iterations
TOLERANCE = 1e-6  # Convergence tolerance [m]
INITIAL_GUESS_RATIO = 0.9  # l_CE / l_opt initial guess


class EquilibriumSolver:
    """Solves muscle-tendon equilibrium for fiber length.

    Given l_MT (muscle-tendon unit length) and activation, this solves for
    l_CE (contractile element length) such that fiber and tendon forces balance.

    Example:
        >>> from shared.python.hill_muscle import HillMuscleModel, MuscleParameters
        >>> params = MuscleParameters(F_max=1000.0, l_opt=0.12, l_slack=0.25, v_max=1.2)
        >>> muscle = HillMuscleModel(params)
        >>> solver = EquilibriumSolver(muscle)
        >>> l_CE = solver.solve_fiber_length(l_MT=0.37, activation=0.5)
        >>> print(f"Fiber length: {l_CE:.4f} m")
    """

    def __init__(self, muscle: HillMuscleModel):
        """Initialize equilibrium solver.

        Args:
            muscle: HillMuscleModel instance
        """
        self.muscle = muscle

    def _equilibrium_residual(
        self, l_CE: float, l_MT: float, activation: float, v_CE: float = 0.0
    ) -> float:
        """Compute equilibrium residual F_fiber - F_tendon.

        At equilibrium, this should be zero.

        Args:
            l_CE: Contractile element length [m]
            activation: Activation level [0,1]
            l_MT: Muscle-tendon unit length [m]
            v_CE: Fiber velocity [m/s] (default: 0 for static equilibrium)

        Returns:
            Residual force [N] (zero at equilibrium)
        """
        # Normalize lengths
        l_CE_norm = l_CE / self.muscle.params.l_opt
        v_CE_norm = v_CE / self.muscle.params.v_max

        # Tendon length (accounting for pennation)
        cos_alpha = np.cos(self.muscle.params.pennation_angle)
        l_tendon = l_MT - l_CE * cos_alpha
        l_tendon_norm = l_tendon / self.muscle.params.l_slack

        # Force components
        f_l = self.muscle.force_length_active(l_CE_norm)
        f_v = self.muscle.force_velocity(v_CE_norm)
        f_p = self.muscle.force_length_passive(l_CE_norm)
        f_t = self.muscle.tendon_force(l_tendon_norm)

        # Fiber force (along fiber direction)
        F_CE = self.muscle.params.F_max * activation * f_l * f_v
        F_PEE = self.muscle.params.F_max * f_p
        F_fiber = (F_CE + F_PEE) * cos_alpha

        # Tendon force
        F_tendon = self.muscle.params.F_max * f_t

        # Residual (should be zero at equilibrium)
        residual = F_fiber - F_tendon

        return float(residual)

    def solve_fiber_length(
        self,
        l_MT: float,
        activation: float,
        v_CE: float = 0.0,
        initial_guess: float | None = None,
    ) -> float:
        """Solve for fiber length at equilibrium.

        Uses Newton-Raphson to find l_CE such that F_fiber = F_tendon.

        Args:
            l_MT: Muscle-tendon unit length [m]
            activation: Activation level [0,1]
            v_CE: Fiber velocity [m/s] (default: 0 for static)
            initial_guess: Initial guess for l_CE [m] (default: 0.9 · l_opt)

        Returns:
            Equilibrium fiber length l_CE [m]

        Raises:
            RuntimeError: If solver fails to converge

        Example:
            >>> l_CE = solver.solve_fiber_length(l_MT=0.37, activation=0.5)
        """
        if initial_guess is None:
            initial_guess = INITIAL_GUESS_RATIO * self.muscle.params.l_opt

        # Define residual function for newton()
        def residual_func(l_CE: float) -> float:
            return self._equilibrium_residual(l_CE, l_MT, activation, v_CE)

        try:
            # Newton-Raphson solver from scipy
            l_CE_solution = newton(
                residual_func,
                x0=initial_guess,
                maxiter=MAX_ITERATIONS,
                tol=TOLERANCE,
            )

            # Validate solution
            residual = residual_func(l_CE_solution)
            if abs(residual) > TOLERANCE * self.muscle.params.F_max:
                logger.warning(
                    f"Equilibrium solver converged but residual is large:\\n"
                    f"  l_CE = {l_CE_solution:.6f} m\\n"
                    f"  Residual = {residual:.2f} N ({abs(residual) / self.muscle.params.F_max * 100:.2f}% of F_max)\\n"
                    f"  This may indicate numerical issues or invalid configuration"
                )

            return float(l_CE_solution)

        except RuntimeError as e:
            logger.error(
                f"Muscle equilibrium solver FAILED to converge:\\n"
                f"  l_MT = {l_MT:.6f} m\\n"
                f"  activation = {activation:.3f}\\n"
                f"  v_CE = {v_CE:.6f} m/s\\n"
                f"  Initial guess: {initial_guess:.6f} m\\n"
                f"  Max iterations: {MAX_ITERATIONS}\\n"
                f"  Error: {e}"
            )
            raise RuntimeError(
                f"Muscle equilibrium solver failed for l_MT={l_MT:.6f}m, a={activation:.3f}"
            ) from e

    def solve_fiber_velocity(
        self,
        l_MT: float,
        v_MT: float,
        activation: float,
        l_CE: float,
        dt: float = 0.001,
    ) -> float:
        """Solve for fiber velocity from muscle-tendon velocity.

        Uses finite difference:
            v_CE ≈ (l_CE(t+dt) - l_CE(t)) / dt

        where l_CE(t+dt) is solved from l_MT(t+dt) = l_MT(t) + v_MT·dt.

        Args:
            l_MT: Current muscle-tendon length [m]
            v_MT: Muscle-tendon velocity [m/s]
            activation: Activation level [0,1]
            l_CE: Current fiber length [m]
            dt: Time step [s] (default: 0.001s = 1ms)

        Returns:
            Fiber velocity v_CE [m/s]

        Note:
            This is an approximation. For exact v_CE, solve the implicit
            differentiated equilibrium equation (more complex).
        """
        # Future muscle-tendon length
        l_MT_next = l_MT + v_MT * dt

        # Solve for future fiber length
        try:
            l_CE_next = self.solve_fiber_length(
                l_MT_next, activation, v_CE=0.0, initial_guess=l_CE
            )

            # Finite difference velocity
            v_CE = (l_CE_next - l_CE) / dt

            return float(v_CE)

        except RuntimeError:
            logger.warning(
                f"Failed to solve fiber velocity via finite difference.\\n"
                f"  Returning zero velocity as fallback.\\n"
                f"  l_MT = {l_MT:.6f} m, v_MT = {v_MT:.6f} m/s"
            )
            return 0.0


def compute_equilibrium_state(
    muscle: HillMuscleModel,
    l_MT: float,
    v_MT: float,
    activation: float,
    initial_l_CE: float | None = None,
) -> tuple[float, float]:
    """Convenience function to compute both l_CE and v_CE at equilibrium.

    Args:
        muscle: HillMuscleModel instance
        l_MT: Muscle-tendon unit length [m]
        v_MT: Muscle-tendon unit velocity [m/s]
        activation: Activation level [0,1]
        initial_l_CE: Initial guess for fiber length [m] (optional)

    Returns:
        Tuple of (l_CE, v_CE) at equilibrium [m, m/s]

    Example:
        >>> from shared.python.hill_muscle import HillMuscleModel, MuscleParameters
        >>> from shared.python.muscle_equilibrium import compute_equilibrium_state
        >>> params = MuscleParameters(F_max=1000.0, l_opt=0.12, l_slack=0.25, v_max=1.2)
        >>> muscle = HillMuscleModel(params)
        >>> l_CE, v_CE = compute_equilibrium_state(muscle, l_MT=0.37, v_MT=0.0, activation=0.5)
        >>> print(f"Fiber length: {l_CE:.4f} m, velocity: {v_CE:.4f} m/s")
    """
    solver = EquilibriumSolver(muscle)

    # Solve for fiber length
    l_CE = solver.solve_fiber_length(
        l_MT, activation, v_CE=0.0, initial_guess=initial_l_CE
    )

    # Solve for fiber velocity
    if abs(v_MT) > 1e-10:
        v_CE = solver.solve_fiber_velocity(l_MT, v_MT, activation, l_CE)
    else:
        v_CE = 0.0  # Static case

    return l_CE, v_CE


# Example usage / validation
if __name__ == "__main__":
    from shared.python.hill_muscle import HillMuscleModel, MuscleParameters

    # Create muscle
    params = MuscleParameters(
        F_max=1000.0,  # N
        l_opt=0.12,  # m
        l_slack=0.25,  # m
        v_max=1.2,  # m/s
        pennation_angle=0.0,  # rad
    )

    muscle = HillMuscleModel(params)
    solver = EquilibriumSolver(muscle)

    # Test 1: Solve at optimal total length
    l_MT_test = params.l_opt + params.l_slack  # 0.37 m
    activation_test = 0.5

    print("=" * 60)  # noqa: T201
    print("Muscle Equilibrium Solver Test")  # noqa: T201
    print("=" * 60)  # noqa: T201
    print("Muscle parameters:")  # noqa: T201
    print(f"  F_max = {params.F_max:.0f} N")  # noqa: T201
    print(f"  l_opt = {params.l_opt * 100:.1f} cm")  # noqa: T201
    print(f"  l_slack = {params.l_slack * 100:.1f} cm")  # noqa: T201
    print("\\nTest case:")  # noqa: T201
    print(f"  l_MT = {l_MT_test * 100:.2f} cm")  # noqa: T201
    print(f"  activation = {activation_test * 100:.0f}%")  # noqa: T201

    l_CE_solution = solver.solve_fiber_length(l_MT_test, activation_test)

    print("\\nSolution:")  # noqa: T201
    print(f"  l_CE = {l_CE_solution * 100:.2f} cm")  # noqa: T201
    print(f"  l_CE / l_opt = {l_CE_solution / params.l_opt:.3f}")  # noqa: T201

    # Verify equilibrium
    residual = solver._equilibrium_residual(l_CE_solution, l_MT_test, activation_test)
    print(f"  Residual = {residual:.2e} N (should be ~0)")  # noqa: T201

    print("\\n" + "=" * 60)  # noqa: T201
    print("✓ Equilibrium solver test complete")  # noqa: T201
    print("=" * 60)  # noqa: T201
