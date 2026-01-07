"""Activation dynamics for muscle models.

This module implements the first-order filter that converts neural excitation
to muscle activation, accounting for activation/deactivation time constants.

Activation Dynamics:
-------------------
da/dt = (u - a) / τ(u, a)

where:
    u = neural excitation [0,1] (input from nervous system)
    a = muscle activation [0,1] (controls force generation)
    τ = time constant [s] (faster activation than deactivation)

Time constants (Winters et al. 1988):
    τ_act = 0.01 s (10 ms activation)
    τ_deact = 0.04 s (40 ms deactivation)

Why This Matters:
----------------
Muscles cannot instantly activate/deactivate. Neural signals take time to:
1. Propagate down motor neurons
2. Release Ca²⁺ from sarcoplasmic reticulum
3. Bind to troponin (cross-bridge formation)
4. Pump Ca²⁺ back (relaxation)

This delay is CRITICAL for:
- Realistic muscle control (no instantaneous force changes)
- Neural control optimization (must plan ahead)
- MyoSuite integration (RL policies learn activation timing)

Reference:
- Winters et al. (1988), "Whole muscle mechanics in physiological movements"
- Zajac (1989), "Muscle coordination of movement"
- OpenSim Documentation: "ActivationFiberLengthMuscle"
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Standard activation time constants (Winters et al. 1988)
# Source: Winters, J.M. (1988), Acta Anatomica 132(4), 219-233

ACTIVATION_TIME_CONSTANT = 0.01  # [s] (10 ms)
# Time to reach 63% of target activation during excitation

DEACTIVATION_TIME_CONSTANT = 0.04  # [s] (40 ms)
# Time to decay to 37% of initial activation during relaxation

MIN_ACTIVATION = 0.01  # [dimensionless]
# Minimum activation (prevents numerical issues at a=0)


class ActivationDynamics:
    """First-order activation dynamics model.

    Converts neural excitation u(t) to muscle activation a(t) with
    physiological time delays.

    Example:
        >>> dynamics = ActivationDynamics()
        >>> u = 1.0  # Full excitation
        >>> a = 0.0  # Initially inactive
        >>> dt = 0.001  # 1 ms timestep
        >>>
        >>> for _ in range(100):  # Simulate 100 ms
        ...     a = dynamics.update(u, a, dt)
        >>> print(f"Activation after 100ms: {a:.2f}")  # Should be ~0.98 (nearly full)
    """

    def __init__(
        self,
        tau_act: float = ACTIVATION_TIME_CONSTANT,
        tau_deact: float = DEACTIVATION_TIME_CONSTANT,
    ):
        """Initialize activation dynamics.

        Args:
            tau_act: Activation time constant [s] (default: 0.01s)
            tau_deact: Deactivation time constant [s] (default: 0.04s)
        """
        self.tau_act = tau_act
        self.tau_deact = tau_deact

    def time_constant(self, u: float, a: float) -> float:
        """Compute time constant τ(u, a).

        Uses faster time constant during activation, slower during deactivation.

        Args:
            u: Neural excitation [0,1]
            a: Current activation [0,1]

        Returns:
            Time constant [s]

        Formula:
            τ = τ_act  if u > a (activation)
            τ = τ_deact  if u <= a (deactivation)
        """
        if u > a:
            # Activation (excitation increasing)
            return self.tau_act
        else:
            # Deactivation (excitation decreasing or constant)
            return self.tau_deact

    def activation_derivative(self, u: float, a: float) -> float:
        """Compute da/dt.

        Args:
            u: Neural excitation [0,1]
            a: Current activation [0,1]

        Returns:
            Activation derivative [1/s]

        Formula:
            da/dt = (u - a) / τ(u, a)
        """
        tau = self.time_constant(u, a)
        da_dt = (u - a) / tau
        return float(da_dt)

    def update(self, u: float, a: float, dt: float) -> float:
        """Update activation using forward Euler integration.

        Args:
            u: Neural excitation [0,1]
            a: Current activation [0,1]
            dt: Time step [s]

        Returns:
            Updated activation [0,1]

        Formula:
            a_new = a + (u - a)/τ(u,a) · dt
        """
        da_dt = self.activation_derivative(u, a)
        a_new = a + da_dt * dt

        # Clamp to [MIN_ACTIVATION, 1.0]
        a_new = float(np.clip(a_new, MIN_ACTIVATION, 1.0))

        return a_new

    def update_rk4(self, u: float, a: float, dt: float) -> float:
        """Update activation using 4th-order Runge-Kutta (more accurate).

        RK4 is more stable and accurate than forward Euler, especially for
        larger timesteps.

        Args:
            u: Neural excitation [0,1]
            a: Current activation [0,1]
            dt: Time step [s]

        Returns:
            Updated activation [0,1]

        Note:
            Use this for dt > 1ms or when accuracy is critical.
        """
        # RK4 stages
        k1 = self.activation_derivative(u, a)
        k2 = self.activation_derivative(u, a + 0.5 * k1 * dt)
        k3 = self.activation_derivative(u, a + 0.5 * k2 * dt)
        k4 = self.activation_derivative(u, a + k3 * dt)

        # Weighted average
        a_new = a + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6.0

        # Clamp
        a_new = float(np.clip(a_new, MIN_ACTIVATION, 1.0))

        return a_new


def simulate_activation_response(
    u_target: float,
    duration: float,
    dt: float = 0.001,
    method: str = "euler",
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate activation response to step excitation.

    Useful for visualizing activation dynamics behavior.

    Args:
        u_target: Target excitation level [0,1]
        duration: Simulation duration [s]
        dt: Time step [s] (default: 0.001s = 1ms)
        method: Integration method ("euler" or "rk4")

    Returns:
        Tuple of (time, activation) arrays

    Example:
        >>> t, a = simulate_activation_response(u_target=1.0, duration=0.1)
        >>> import matplotlib.pyplot as plt
        >>> plt.plot(t, a)
        >>> plt.xlabel("Time [s]")
        >>> plt.ylabel("Activation")
        >>> plt.title("Step Response (0 → 1)")
        >>> plt.show()
    """
    dynamics = ActivationDynamics()

    n_steps = int(duration / dt)
    time = np.linspace(0, duration, n_steps)
    activation = np.zeros(n_steps)

    # Initial state
    a = MIN_ACTIVATION

    for i in range(n_steps):
        activation[i] = a

        # Update
        if method == "rk4":
            a = dynamics.update_rk4(u_target, a, dt)
        else:
            a = dynamics.update(u_target, a, dt)

    return time, activation


# Example usage / validation
if __name__ == "__main__":
    print("=" * 60)
    print("Activation Dynamics Test")
    print("=" * 60)

    dynamics = ActivationDynamics()

    # Test 1: Step response (0 → 1)
    print("\\nTest 1: Step response (0 → 1)")
    print(f"  τ_act = {dynamics.tau_act*1000:.1f} ms")

    u = 1.0  # Full excitation
    a = MIN_ACTIVATION  # Start at rest
    dt = 0.001  # 1 ms

    # Simulate 50 ms
    t_total = 0.050  # 50 ms
    n_steps = int(t_total / dt)

    print("\\n  Time [ms]  Activation [%]")
    print("  ---------  --------------")

    for i in range(n_steps + 1):
        if i % 10 == 0:  # Print every 10 ms
            print(f"  {i*dt*1000:5.0f}      {a*100:5.1f}")

        if i < n_steps:
            a = dynamics.update(u, a, dt)

    # Test 2: Deactivation (1 → 0)
    print("\\nTest 2: Deactivation (1 → 0)")
    print(f"  τ_deact = {dynamics.tau_deact*1000:.1f} ms")

    u = 0.0  # Zero excitation
    a = 1.0  # Start fully activated
    dt = 0.001

    t_total = 0.100  # 100 ms
    n_steps = int(t_total / dt)

    print("\\n  Time [ms]  Activation [%]")
    print("  ---------  --------------")

    for i in range(n_steps + 1):
        if i % 20 == 0:  # Print every 20 ms
            print(f"  {i*dt*1000:5.0f}      {a*100:5.1f}")

        if i < n_steps:
            a = dynamics.update(u, a, dt)

    print("\\n" + "=" * 60)
    print("✓ Activation dynamics test complete")
    print("=" * 60)
    print("\\nKey observation:")
    print(
        f"  Activation ({dynamics.tau_act*1000:.0f}ms) is ~{dynamics.tau_deact/dynamics.tau_act:.0f}× faster than deactivation ({dynamics.tau_deact*1000:.0f}ms)"
    )
    print("  This asymmetry is physiologically realistic (Ca²⁺ release vs. pump)")
