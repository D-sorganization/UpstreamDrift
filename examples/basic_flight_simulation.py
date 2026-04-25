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
    header = f"{'t (s)':>8} {'x (m)':>10} {'z (m)':>8} {'|v| (m/s)':>12}"
    print(header)
    print("-" * len(header))
    for pt in trajectory[::10]:
        speed = float(np.linalg.norm(pt.velocity))
        print(f"{pt.time:8.2f} {pt.position[0]:10.2f} {pt.height:8.2f} {speed:12.2f}")

    # --- Carry distance: last point before height < 0 ---
    landing_pts = [p for p in trajectory if p.height <= 0.0]
    if len(landing_pts) >= 2:
        carry_m = float(landing_pts[-1].position[0])
        carry_yd = carry_m * 1.0936
        print(f"\nCarry distance: {carry_m:.1f} m ({carry_yd:.1f} yd)")
    else:
        print(
            "\nBall did not return to ground within the simulated window "
            "(increase max_time or check launch conditions)."
        )

    print(
        "\nPhysics: drag Cd=0.25 and lift Cl=0.15 were applied via the "
        "BallFlightSimulator; gravity is 9.81 m/s² and air density 1.225 kg/m³ "
        "(sea level). Higher spin → more lift and longer carry."
    )


if __name__ == "__main__":
    main()
