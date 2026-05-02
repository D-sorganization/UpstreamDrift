"""CI-runnable regression benchmarks for critical hot paths (issue #3510).

These benchmarks intentionally use ``time.perf_counter`` rather than
pytest-benchmark so they stay dependency-light and runnable in the default
CI lane behind the ``benchmark`` marker.

Each test:
1. Runs a focused hot path N times.
2. Computes the median per-call wall-clock time.
3. Compares it against the checked-in baseline at
   ``tests/benchmarks/baseline.json`` with a generous regression
   multiplier (5x). New metrics auto-pass on the first run so a baseline
   can be recorded without breaking CI.

Run locally:

    python3 -m pytest tests/benchmarks/test_regression_benchmarks.py \
        -m benchmark --timeout=120 -v
"""

from __future__ import annotations

import numpy as np
import pytest
from src.api.models.requests import SimulationRequest
from src.shared.python.physics.aerodynamics import (
    AerodynamicsEngine,
    DragModel,
)
from src.shared.python.physics.ball_launch_conditions import LaunchConditions
from src.shared.python.physics.ball_simulator import BallFlightSimulator

from tests.benchmarks.regression_helpers import (
    assert_within_regression_threshold,
    measure_median_seconds,
)

pytestmark = pytest.mark.benchmark


# ---------------------------------------------------------------------------
# Aerodynamic force calculations
# ---------------------------------------------------------------------------


def test_drag_force_calculation_regression() -> None:
    """Median drag force calculation should stay below 5x baseline."""
    model = DragModel()
    velocity = np.array([60.0, 5.0, 10.0])

    median = measure_median_seconds(
        model.calculate, velocity, iterations=500, warmup=20
    )

    assert_within_regression_threshold("drag_force_calculation", median)


def test_aerodynamics_engine_compute_forces_regression() -> None:
    """Median compute_forces (drag+lift+magnus) should stay below 5x baseline."""
    engine = AerodynamicsEngine()
    velocity = np.array([60.0, 5.0, 10.0])
    spin = np.array([10.0, -250.0, 30.0])

    median = measure_median_seconds(
        engine.compute_forces, velocity, spin, iterations=500, warmup=20
    )

    assert_within_regression_threshold("aerodynamics_engine_compute_forces", median)


# ---------------------------------------------------------------------------
# Ball flight integrator inner loop
# ---------------------------------------------------------------------------


def test_ball_flight_force_step_regression() -> None:
    """Median pure-Python ball flight force step should stay below 5x baseline.

    Targets ``BallFlightSimulator._calculate_forces_single`` which is the
    inner per-step force computation invoked by both the legacy Python RK4
    integrator and the post-processing of Rust kernel outputs.
    """
    simulator = BallFlightSimulator()
    launch = LaunchConditions(
        velocity=70.0,
        launch_angle=float(np.radians(12.0)),
        azimuth_angle=0.0,
        spin_rate=3000.0,
        spin_axis=np.array([0.0, 1.0, 0.0]),
    )
    velocity = np.array([60.0, 0.0, 20.0])
    omega = launch.spin_rate * 2.0 * np.pi / 60.0

    median = measure_median_seconds(
        simulator._calculate_forces_single,
        velocity,
        omega,
        launch,
        iterations=500,
        warmup=20,
    )

    assert_within_regression_threshold("ball_flight_force_step", median)


# ---------------------------------------------------------------------------
# Pydantic request deserialization
# ---------------------------------------------------------------------------


def test_simulation_request_deserialization_regression() -> None:
    """Median SimulationRequest.model_validate should stay below 5x baseline."""
    payload = {
        "engine_type": "mujoco",
        "model_path": "models/sample.urdf",
        "duration": 1.0,
        "timestep": 0.001,
        "initial_state": {"q": [0.0, 0.0, 0.0]},
    }

    median = measure_median_seconds(
        SimulationRequest.model_validate,
        payload,
        iterations=2000,
        warmup=50,
    )

    assert_within_regression_threshold("simulation_request_model_validate", median)
