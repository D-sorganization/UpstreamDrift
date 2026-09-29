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
_this_file = Path(__file__).resolve()
_project_root = _this_file.parents[1]
sys.path.insert(0, str(_project_root))

import numpy as np  # noqa: E402
from src.shared.python.physics.ball_flight_physics import (  # noqa: E402
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
        gravity=9.80665,
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

    # --- Print summary every 0.5 s (index step of 10) ---
    print("Trajectory Summary (every 0.5 seconds):")
    print(f"{'t (s)':<12} {'x (m)':<12} {'z (m)':<15} {'|v| (m/s)':<12}")
    print("-" * 51)
    for pt in trajectory[::10]:
        print(
            f"{pt.time:<12.3f} {pt.position[0]:<12.2f} "
            f"{pt.height:<15.2f} {pt.speed:<12.2f}"
        )

    print("\nPhysics: trajectory simulates lift and drag forces on the ball.")

    # --- Carry distance: last point before height < 0 ---
    landing_pts = [p for p in trajectory if p.height <= 0.0]
    if len(landing_pts) >= 2:
        carry_m = float(landing_pts[-1].position[0])
        carry_yards = carry_m * 1.0936
        print(f"\nCarry distance: {carry_m:.2f} m ({carry_yards:.2f} yd)")
    else:
        print("\nBall did not land (insufficient trajectory data)")
    print("Physics: Simulation incorporates aerodynamic drag and Magnus lift.")

    print(
        "\nPhysics: Drag slows the ball down, while lift (from backspin) keeps it in the air longer."
    )


if __name__ == "__main__":
    main()
    print("\nPhysics: Includes aerodynamic effects like drag and lift.")
