"""Demonstrate BallFlightSimulator: simulate a golf shot trajectory.

Usage::

    python3 examples/basic_flight_simulation.py

Prints a summary table of trajectory points (time, x, z, speed) and
the estimated carry distance to landing.
"""

from __future__ import annotations

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

    # --- Print summary every 0.5 s (every 10 points at dt=0.05) ---
    print("\n=== Golf Shot Trajectory ===\n")
    print(f"{'Time (s)':<10} {'X (m)':<12} {'Z (m)':<12} {'Speed (m/s)':<15}")
    print("-" * 50)
    for pt in trajectory[::10]:
        print(
            f"{pt.time:<10.2f} {pt.position[0]:<12.2f} {pt.position[2]:<12.2f} "
            f"{pt.speed:<15.2f}"
        )

    # --- Carry distance: last point before height < 0 ---
    landing_pts = [p for p in trajectory if p.height <= 0.0]
    if len(landing_pts) >= 2:
        carry_m = float(landing_pts[-1].position[0])
        carry_yards = carry_m * 1.0936
        print("\n=== Results ===\n")
        print(f"Carry Distance: {carry_m:.2f} m ({carry_yards:.2f} yards)")
        print(f"Flight Time: {landing_pts[-1].time:.2f} s")
        print(f"Max Height: {max(p.height for p in trajectory):.2f} m")
    else:
        print("\n[Warning] Landing point not found or insufficient data")
        print(f"Final height: {trajectory[-1].height:.2f} m")


if __name__ == "__main__":
    main()
