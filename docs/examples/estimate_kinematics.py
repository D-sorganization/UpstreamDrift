#!/usr/bin/env python3
"""End-to-end example: estimate velocity and acceleration from positions.

A common calc/estimation task is recovering joint velocities and
accelerations from a sampled position signal (e.g. mocap or a simulation
recording). This example uses central finite differences — the same approach
used by the kinematics utilities in ``src/shared/python/pose_interchange/`` —
and validates the estimate against a known analytic signal.

Run it directly from the repository root::

    python docs/examples/estimate_kinematics.py
"""

from __future__ import annotations

import numpy as np


def estimate_derivatives(
    positions: np.ndarray, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate velocity and acceleration via central finite differences.

    Args:
        positions: Sampled positions, shape ``(frames,)`` or ``(frames, dof)``.
        dt: Constant sample spacing in seconds.

    Returns:
        Tuple ``(velocity, acceleration)`` with the same shape as ``positions``.
    """
    if dt <= 0:
        raise ValueError("dt must be positive")
    if positions.shape[0] < 3:
        raise ValueError("need at least 3 samples for a central difference")

    velocity = np.gradient(positions, dt, axis=0)
    acceleration = np.gradient(velocity, dt, axis=0)
    return velocity, acceleration


def main() -> int:
    """Estimate kinematics for a sinusoidal signal and report the error."""
    dt = 1.0 / 240.0
    t = np.arange(0.0, 1.0, dt)
    omega = 2.0 * np.pi  # 1 Hz

    # Analytic signal: x = sin(wt) -> v = w cos(wt), a = -w^2 sin(wt).
    position = np.sin(omega * t)
    true_velocity = omega * np.cos(omega * t)
    true_acceleration = -(omega**2) * np.sin(omega * t)

    velocity, acceleration = estimate_derivatives(position, dt)

    # Compare on the interior, where central differences are most accurate.
    interior = slice(1, -1)
    vel_err = float(np.max(np.abs(velocity[interior] - true_velocity[interior])))
    acc_err = float(
        np.max(np.abs(acceleration[interior] - true_acceleration[interior]))
    )

    print(f"Max velocity error:     {vel_err:.4e}")
    print(f"Max acceleration error: {acc_err:.4e}")
    assert vel_err < 1e-2, "velocity estimate should track the analytic signal"
    assert acc_err < 5e-1, "acceleration estimate should track the analytic signal"
    print("Kinematics estimation example completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
