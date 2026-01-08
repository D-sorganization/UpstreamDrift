"""Activation dynamics for Hill muscle models.

This module models the time delay between neural excitation (u) and
muscle activation (a). This is a first-order differential equation
that represents calcium dynamics in the muscle fibers.

Dynamics Equation:
-----------------
da/dt = (u - a) / τ(u, a)

where:
    τ = τ_act * (0.5 + 1.5*a)   if u > a (Activation)
    τ = τ_deact / (0.5 + 1.5*a) if u <= a (Deactivation)

Typical values:
    τ_act ≈ 10 ms (fast)
    τ_deact ≈ 40 ms (slow)

Reference:
- Thelen (2003), "Adjustment of Muscle Mechanics Model Parameters...", J. Biomech. Eng.
- Zajac (1989), "Muscle and Tendon: Properties, Models, Scaling...", CRC Crit. Rev. Biomed. Eng.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class ActivationDynamics:
    """Models neural excitation to muscle activation dynamics.

    Implements a first-order differential equation with different time constants
    for activation (rise) and deactivation (fall).

    Example:
        >>> dynamics = ActivationDynamics(tau_act=0.010, tau_deact=0.040)
        >>> a = 0.0  # Initial activation
        >>> u = 1.0  # Step input excitation
        >>> for _ in range(100):
        ...     a = dynamics.update(u, a, dt=0.001)
        >>> print(f"Activation after 100ms: {a:.2f}")  # Should be ~0.98 (nearly full)
    """

    def __init__(
        self,
        tau_act: float = 0.010,
        tau_deact: float = 0.040,
        min_activation: float = 0.001,
    ):
        """Initialize activation dynamics.

        Args:
            tau_act: Time constant for activation (rise) [s]
            tau_deact: Time constant for deactivation (fall) [s]
            min_activation: Minimum activation floor (to prevent division by zero)
        """
        if tau_act <= 0 or tau_deact <= 0:
            msg = "Time constants must be positive"
            raise ValueError(msg)

        self.tau_act = tau_act
        self.tau_deact = tau_deact
        self.min_activation = min_activation

    def compute_derivative(self, u: float, a: float) -> float:
        """Compute activation derivative da/dt.

        Args:
            u: Neural excitation [0, 1]
            a: Current activation [0, 1]

        Returns:
            Time derivative da/dt [1/s]
        """
        # Clamp inputs
        u = np.clip(u, self.min_activation, 1.0)
        a = np.clip(a, self.min_activation, 1.0)

        if u > a:
            # Activation (rise)
            tau = self.tau_act * (0.5 + 1.5 * a)
        else:
            # Deactivation (fall)
            tau = self.tau_deact / (0.5 + 1.5 * a)

        dadt = (u - a) / tau
        return float(dadt)

    def update(self, u: float, a: float, dt: float) -> float:
        """Update activation state by one time step.

        Uses Euler integration: a(t+dt) = a(t) + da/dt * dt

        Args:
            u: Neural excitation [0, 1]
            a: Current activation [0, 1]
            dt: Time step [s]

        Returns:
            New activation level a(t+dt) [0, 1]
        """
        dadt = self.compute_derivative(u, a)
        a_new = a + dadt * dt

        # Clamp result
        return float(np.clip(a_new, self.min_activation, 1.0))


# Example usage / validation
if __name__ == "__main__":

    dynamics = ActivationDynamics(tau_act=0.010, tau_deact=0.040)
    dt = 0.001  # 1 ms
    duration = 0.200  # 200 ms

    # Simulation arrays
    times = np.arange(0, duration, dt)
    u_signal = np.zeros_like(times)
    a_response = np.zeros_like(times)

    # Step input: 0 -> 1 at t=0, 1 -> 0 at t=100ms
    u_signal[0:100] = 1.0
    u_signal[100:] = 0.0

    # Simulate
    a = 0.0
    print("=" * 60)  # noqa: T201
    print("Activation Dynamics Test")  # noqa: T201
    print("=" * 60)  # noqa: T201
    print("\\nTest 1: Step response (0 → 1)")  # noqa: T201
    print(f"  τ_act = {dynamics.tau_act*1000:.1f} ms")  # noqa: T201

    print("\\n  Time [ms]  Activation [%]")  # noqa: T201
    print("  ---------  --------------")  # noqa: T201

    for i, _t in enumerate(times):
        u = u_signal[i]
        a = dynamics.update(u, a, dt)
        a_response[i] = a

        # Print check points
        if i % 20 == 0 and i < 120:
            print(f"  {i*dt*1000:5.0f}      {a*100:5.1f}")  # noqa: T201

    print("\\nTest 2: Deactivation (1 → 0)")  # noqa: T201
    print(f"  τ_deact = {dynamics.tau_deact*1000:.1f} ms")  # noqa: T201
    print("\\n  Time [ms]  Activation [%]")  # noqa: T201
    print("  ---------  --------------")  # noqa: T201

    # Print check points for deactivation
    for i in range(100, 160, 10):
        print(f"  {i*dt*1000:5.0f}      {a_response[i]*100:5.1f}")  # noqa: T201

    print("\\n" + "=" * 60)  # noqa: T201
    print("✓ Activation dynamics test complete")  # noqa: T201
    print("=" * 60)  # noqa: T201
    print("\\nKey observation:")  # noqa: T201
    print(  # noqa: T201
        "  Activation is faster (10ms) than deactivation (40ms)"
    )  # noqa: T201
    print(  # noqa: T201
        "  This asymmetry is physiologically realistic (Ca²⁺ release vs. pump)"
    )
