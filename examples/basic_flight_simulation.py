"""Demonstrate BallFlightSimulator: simulate a golf shot trajectory.

Usage::

    python3 examples/basic_flight_simulation.py

Prints a summary table of trajectory points (time, x, z, speed) and
the estimated carry distance to landing.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.shared.python.physics.ball_flight_physics import (
    BallFlightSimulator,
    BallProperties,
    EnvironmentalConditions,
    LaunchConditions,
)


def main() -> None:
    """Run a basic driver-shot simulation and print results."""
    # --- Ball and environment setup ---
    ball = BallProperties(
        mass=0.0459,  # kg (regulation golf ball)
        diameter=0.04267,  # m (regulation diameter)
        cd0=0.25,
        cd1=0.0,
        cd2=0.0,
        cl0=0.15,
        cl1=0.0,
        cl2=0.0,
    )
    env = EnvironmentalConditions(
        gravity=9.81,
        air_density=1.225,  # kg/m³ at sea level
    )

    # --- Launch conditions: driver shot ---
    launch = LaunchConditions(
        velocity=70.0,  # m/s (~157 mph ball speed)
        launch_angle=np.radians(12.0),
        azimuth_angle=0.0,
        spin_rate=2700.0,  # rpm (back-spin)
    )

    simulator = BallFlightSimulator(ball=ball, env=env)
    trajectory = simulator.simulate_trajectory(launch, max_time=8.0, dt=0.05)

    # --- Print summary every 0.5 s ---
    print(f"{'Time(s)':>8}  {'X(m)':>8}  {'Z(m)':>8}  {'Speed(m/s)':>10}")
    print("-" * 42)
    for pt in trajectory[::10]:
        print(
            f"{pt.time:8.2f}  {pt.position[0]:8.1f}  {pt.position[2]:8.1f}"
            f"  {pt.speed:10.2f}"
        )

    # --- Carry distance: last point before height < 0 ---
    landing_pts = [p for p in trajectory if p.height <= 0.0]
    if len(landing_pts) >= 2:
        carry_m = float(landing_pts[-1].position[0])
        carry_yd = carry_m * 1.0936
        print(f"\nEstimated carry: {carry_m:.1f} m  ({carry_yd:.0f} yd)")
    else:
        print("\nBall did not land within simulated time window.")


if __name__ == "__main__":
    main()
