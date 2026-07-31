#!/usr/bin/env python3
"""
Generate the portfolio demo output CSV.
Run via: python scripts/ci/generate_portfolio_demo_output.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

# Allow running from repo root
_this_file = Path(__file__).resolve()
_project_root = _this_file.parents[2]
sys.path.insert(0, str(_project_root))

import numpy as np  # noqa: E402
from src.shared.python.physics.ball_flight_physics import (  # noqa: E402
    BallFlightSimulator,
    BallProperties,
    EnvironmentalConditions,
    LaunchConditions,
)


def generate_portfolio_demo_output(output_path: Path | None = None) -> None:
    if output_path is None:
        output_path = (
            _project_root / "docs" / "portfolio" / "golf_modeling_demo_output.csv"
        )

    # Match the values in docs/portfolio/golf_modeling_demo.md

    ball = BallProperties(
        mass=0.0459,
        diameter=0.04267,
        cd0=0.25,
        cd1=0.0,
        cd2=0.0,
        cl0=0.15,
        cl1=0.0,
        cl2=0.0,
    )
    env = EnvironmentalConditions(
        gravity=9.80665,
        air_density=1.225,
    )

    launch = LaunchConditions(
        velocity=70.0,
        launch_angle=np.radians(12.0),
        azimuth_angle=0.0,
        spin_rate=2700.0,
    )

    simulator = BallFlightSimulator(ball=ball, env=env)
    trajectory = simulator.simulate_trajectory(launch, max_time=10.0, dt=0.01)

    # Calculate outputs
    landing_pts = [p for p in trajectory if p.height <= 0.0]
    landing_pt = landing_pts[-1] if len(landing_pts) >= 2 else trajectory[-1]

    carry_m = float(landing_pt.position[0])
    carry_yd = carry_m * 1.0936
    peak_height = float(max(p.height for p in trajectory))
    flight_time = float(landing_pt.time)

    rows = [
        {
            "quantity": "ball_speed",
            "category": "measured_input",
            "value": "70.0",
            "unit": "m/s",
            "source": "driver launch-condition fixture",
        },
        {
            "quantity": "launch_angle",
            "category": "measured_input",
            "value": "12.0",
            "unit": "deg",
            "source": "driver launch-condition fixture",
        },
        {
            "quantity": "backspin",
            "category": "measured_input",
            "value": "2700",
            "unit": "rpm",
            "source": "driver launch-condition fixture",
        },
        {
            "quantity": "air_density",
            "category": "assumption",
            "value": "1.225",
            "unit": "kg/m^3",
            "source": "sea-level standard atmosphere",
        },
        {
            "quantity": "gravity",
            "category": "assumption",
            "value": "9.80665",
            "unit": "m/s^2",
            "source": "default simulation environment",
        },
        {
            "quantity": "mass",
            "category": "assumption",
            "value": "0.0459",
            "unit": "kg",
            "source": "custom ball properties",
        },
        {
            "quantity": "diameter",
            "category": "assumption",
            "value": "0.04267",
            "unit": "m",
            "source": "custom ball properties",
        },
        {
            "quantity": "cd0",
            "category": "assumption",
            "value": "0.25",
            "unit": "dimensionless",
            "source": "custom ball properties",
        },
        {
            "quantity": "cd1",
            "category": "assumption",
            "value": "0.0",
            "unit": "s/rad",
            "source": "custom ball properties",
        },
        {
            "quantity": "cd2",
            "category": "assumption",
            "value": "0.0",
            "unit": "(s/rad)^2",
            "source": "custom ball properties",
        },
        {
            "quantity": "cl0",
            "category": "assumption",
            "value": "0.15",
            "unit": "dimensionless",
            "source": "custom ball properties",
        },
        {
            "quantity": "cl1",
            "category": "assumption",
            "value": "0.0",
            "unit": "s/rad",
            "source": "custom ball properties",
        },
        {
            "quantity": "cl2",
            "category": "assumption",
            "value": "0.0",
            "unit": "(s/rad)^2",
            "source": "custom ball properties",
        },
        {
            "quantity": "carry_distance",
            "category": "simulated_output",
            "value": f"{carry_m:.1f}",
            "unit": "m",
            "source": "portfolio reference fixture",
        },
        {
            "quantity": "carry_distance",
            "category": "simulated_output",
            "value": f"{carry_yd:.1f}",
            "unit": "yd",
            "source": "portfolio reference fixture",
        },
        {
            "quantity": "peak_height",
            "category": "simulated_output",
            "value": f"{peak_height:.1f}",
            "unit": "m",
            "source": "portfolio reference fixture",
        },
        {
            "quantity": "flight_time",
            "category": "simulated_output",
            "value": f"{flight_time:.1f}",
            "unit": "s",
            "source": "portfolio reference fixture",
        },
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["quantity", "category", "value", "unit", "source"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {output_path}")


if __name__ == "__main__":
    generate_portfolio_demo_output()
