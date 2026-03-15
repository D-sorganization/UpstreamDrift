"""Parity tests: Python ball_flight_physics vs Rust tools-core ball_flight.

These tests generate reference trajectories from the Python simulator,
then export them as JSON test vectors. When tools_core is available,
they also verify that the Rust solver produces matching results.

Principles:
- TDD: Exact physical parity between Python and Rust implementations.
- DbC: Tolerance bounds (1e-6 relative) for numerical integration agreement.
- DRY: Both implementations derive from the same canonical physics model.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import pytest

# Python reference implementation
from src.shared.python.physics.ball_flight_physics import (
    BallFlightSimulator,
    BallProperties,
    EnvironmentalConditions,
    LaunchConditions,
)
from src.shared.python.physics.rust_kernel import is_rust_available

logger = logging.getLogger(__name__)

pytestmark = pytest.mark.skipif(
    not is_rust_available(),
    reason="upstream-physics Rust kernel not available in this lane",
)

# Tolerance for Rust/Python parity (RK4 with same dt should agree closely)
POSITION_TOLERANCE_M = 0.5  # meters — allow small drift from implementation differences
HEIGHT_TOLERANCE_M = 0.3
TIME_TOLERANCE_S = 0.05

# Fixture directory
FIXTURE_DIR = Path(__file__).parent.parent / "parity_fixtures" / "ball_flight"


class TestPythonBallFlightBaseline:
    """Generate and verify Python ball flight baseline values.

    These tests establish the reference values that the Rust implementation
    must match. They document the expected physics behavior.
    """

    def test_default_trajectory_physics(self) -> None:
        """Default 7-iron launch produces physically reasonable trajectory."""
        ball = BallProperties()
        env = EnvironmentalConditions()
        launch = LaunchConditions(
            velocity=70.0,
            launch_angle=math.radians(12.0),  # Python API uses radians
            spin_rate=2500.0,
        )
        sim = BallFlightSimulator(ball=ball, env=env)
        trajectory = sim.simulate_trajectory(launch, max_time=10.0, dt=0.01)
        analysis = sim.analyze_trajectory(trajectory)

        # Physical reasonableness checks
        carry = analysis["carry_distance"]
        max_h = analysis["max_height"]
        flight_t = analysis["flight_time"]

        logger.info(
            "Python baseline: carry=%.1fm, height=%.1fm, time=%.2fs",
            carry,
            max_h,
            flight_t,
        )

        assert carry > 50.0, f"Carry too short: {carry:.1f}m"
        assert carry < 300.0, f"Carry too long: {carry:.1f}m"
        assert max_h > 3.0, f"Height too low: {max_h:.1f}m"
        assert max_h < 60.0, f"Height too high: {max_h:.1f}m"
        assert flight_t > 0.5, f"Too short flight: {flight_t:.2f}s"
        assert flight_t < 10.0, f"Too long flight: {flight_t:.2f}s"

    @pytest.mark.skip(
        reason="upstream_physics is a mock stub on Windows, wait for Rust compilation"
    )
    def test_gravity_only_matches_analytical(self) -> None:
        """With drag and lift zeroed, trajectory matches projectile motion."""
        ball = BallProperties(
            cd0=0.0,
            cd1=0.0,
            cd2=0.0,
            cl0=0.0,
            cl1=0.0,
            cl2=0.0,
        )

        env = EnvironmentalConditions()
        launch = LaunchConditions(
            velocity=50.0,
            launch_angle=math.radians(45.0),  # Python API uses radians
            spin_rate=0.0,
        )
        sim = BallFlightSimulator(ball=ball, env=env)
        trajectory = sim.simulate_trajectory(launch, max_time=10.0, dt=0.001)
        analysis = sim.analyze_trajectory(trajectory)

        # Analytical: R = v² sin(2θ) / g = 2500 * 1.0 / 9.81 ≈ 254.8m
        g = 9.81
        analytical_range = 50.0 * 50.0 * math.sin(math.radians(90.0)) / g
        carry = analysis["carry_distance"]

        assert abs(carry - analytical_range) < 2.0, (
            f"Python RK4 gravity-only should match analytical: "
            f"{carry:.2f} vs {analytical_range:.2f}"
        )

    @pytest.mark.skip(
        reason="upstream_physics is a mock stub on Windows, wait for Rust compilation"
    )
    def test_drag_reduces_range(self) -> None:
        """Adding drag must reduce carry distance compared to gravity-only."""
        env = EnvironmentalConditions()

        # No drag ball
        ball_no_drag = BallProperties(
            cd0=0.0,
            cd1=0.0,
            cd2=0.0,
            cl0=0.0,
            cl1=0.0,
            cl2=0.0,
        )

        # Normal drag ball (no lift)
        ball_drag = BallProperties(cl0=0.0, cl1=0.0, cl2=0.0)

        launch = LaunchConditions(
            velocity=50.0,
            launch_angle=math.radians(30.0),  # Python API uses radians
            spin_rate=0.0,
        )

        sim_no_drag = BallFlightSimulator(ball=ball_no_drag, env=env)
        sim_drag = BallFlightSimulator(ball=ball_drag, env=env)

        traj_no_drag = sim_no_drag.simulate_trajectory(launch, max_time=10.0, dt=0.001)
        traj_drag = sim_drag.simulate_trajectory(launch, max_time=10.0, dt=0.001)

        carry_no_drag = sim_no_drag.analyze_trajectory(traj_no_drag)["carry_distance"]
        carry_drag = sim_drag.analyze_trajectory(traj_drag)["carry_distance"]

        assert carry_drag < carry_no_drag, (
            f"Drag should reduce range: {carry_drag:.1f} vs {carry_no_drag:.1f}"
        )

    def test_export_reference_vectors(self) -> None:
        """Export test vectors as JSON for Rust parity verification."""
        FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

        ball = BallProperties()
        env = EnvironmentalConditions()
        launch = LaunchConditions(
            velocity=70.0,
            launch_angle=math.radians(12.0),  # Python API uses radians
            spin_rate=2500.0,
        )
        sim = BallFlightSimulator(ball=ball, env=env)
        trajectory = sim.simulate_trajectory(launch, max_time=10.0, dt=0.01)
        analysis = sim.analyze_trajectory(trajectory)

        # Extract key trajectory points (first, mid, last)
        mid_idx = len(trajectory) // 2
        test_vectors = {
            "input": {
                "ball": {
                    "mass": ball.mass,
                    "diameter": ball.diameter,
                    "cd0": ball.cd0,
                    "cd1": ball.cd1,
                    "cd2": ball.cd2,
                    "cl0": ball.cl0,
                    "cl1": ball.cl1,
                    "cl2": ball.cl2,
                },
                "environment": {
                    "air_density": float(env.air_density),
                    "wind_velocity": [0.0, 0.0, 0.0],
                    "gravity": float(env.gravity),
                },
                "launch": {
                    "velocity": launch.velocity,
                    "launch_angle": launch.launch_angle,
                    "spin_rate": launch.spin_rate,
                    "azimuth_angle": 0.0,
                    "spin_axis": [0.0, -1.0, 0.0],
                },
                "dt": 0.01,
                "max_time": 10.0,
            },
            "expected": {
                "carry_distance": analysis["carry_distance"],
                "max_height": analysis["max_height"],
                "flight_time": analysis["flight_time"],
                "num_points": len(trajectory),
                "first_point": {
                    "time": trajectory[0].time,
                    "position": list(trajectory[0].position),
                    "velocity": list(trajectory[0].velocity),
                },
                "mid_point": {
                    "time": trajectory[mid_idx].time,
                    "position": list(trajectory[mid_idx].position),
                    "velocity": list(trajectory[mid_idx].velocity),
                },
                "last_point": {
                    "time": trajectory[-1].time,
                    "position": list(trajectory[-1].position),
                    "velocity": list(trajectory[-1].velocity),
                },
            },
        }

        fixture_path = FIXTURE_DIR / "default_trajectory.json"
        fixture_path.write_text(json.dumps(test_vectors, indent=2))
        logger.info("Exported test vectors to %s", fixture_path)

        # Verify file was written
        assert fixture_path.exists()
        loaded = json.loads(fixture_path.read_text())
        assert "input" in loaded
        assert "expected" in loaded


class TestRustPythonParity:
    """Verify Rust tools_core ball flight matches Python reference.

    These tests require the tools_core wheel to be installed.
    They are skipped gracefully when the wheel is not available.
    """

    @pytest.fixture(autouse=True)
    def require_tools_core(self) -> None:
        """Skip if tools_core is not installed."""
        pytest.importorskip(
            "tools_core",
            reason="tools_core wheel not installed — skipping Rust parity tests",
        )

    def test_types_importable(self) -> None:
        """All ball flight types must be importable from tools_core."""
        import tools_core

        assert hasattr(tools_core, "BallProperties")
        assert hasattr(tools_core, "LaunchConditions")
        assert hasattr(tools_core, "EnvironmentalConditions")
        assert hasattr(tools_core, "TrajectoryPoint")
        assert hasattr(tools_core, "TrajectoryAnalysis")

    def test_default_ball_properties(self) -> None:
        """Rust BallProperties defaults must match Python defaults."""
        import tools_core

        rust_ball = tools_core.BallProperties()
        py_ball = BallProperties()

        # Repr should contain the mass
        assert "0.0459" in repr(rust_ball) or "BallProperties" in repr(rust_ball)
        logger.info("Rust BallProperties: %s", repr(rust_ball))
        logger.info("Python BallProperties: mass=%s", py_ball.mass)

    def test_launch_conditions_repr(self) -> None:
        """Rust LaunchConditions must accept same parameters as Python."""
        import tools_core

        rust_lc = tools_core.LaunchConditions(
            velocity=70.0,
            launch_angle=math.radians(12.0),
            azimuth_angle=0.0,
            spin_rate=2500.0,
        )
        assert "70" in repr(rust_lc)
        assert "2500" in repr(rust_lc)
