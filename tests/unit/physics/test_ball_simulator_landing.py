"""Landing-only conversion must preserve the full native flight result."""

import math

import numpy as np
import pytest

from src.shared.python.physics.ball_launch_conditions import LaunchConditions
from src.shared.python.physics.ball_simulator import BallFlightSimulator

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "azimuth,angle,spin", [(0, 12, 2500), (3, 23, 6500), (-4, 35, 9000)]
)
def test_landing_matches_full_native_trajectory(azimuth, angle, spin):
    pytest.importorskip("upstream_physics")
    launch = LaunchConditions(
        velocity=50,
        launch_angle=math.radians(angle),
        azimuth_angle=math.radians(azimuth),
        spin_rate=spin,
        spin_axis=np.array([0.0, -1.0, 0.0]),
    )
    simulator = BallFlightSimulator()
    trajectory = simulator.simulate_trajectory(launch, max_time=20, dt=0.02)
    prev, end = trajectory[-2].position, trajectory[-1].position
    expected = prev + prev[2] / (prev[2] - end[2]) * (end - prev)
    np.testing.assert_allclose(
        simulator.simulate_landing(launch, max_time=20, dt=0.02),
        expected,
        rtol=0,
        atol=1e-12,
    )


def test_landing_rejects_incomplete_flight():
    pytest.importorskip("upstream_physics")
    launch = LaunchConditions(velocity=50, launch_angle=0.3, spin_rate=2500)
    with pytest.raises(RuntimeError, match="before landing"):
        BallFlightSimulator().simulate_landing(launch, max_time=0.1, dt=0.02)


@pytest.mark.parametrize(
    "max_time,dt", [(float("nan"), 0.02), (20, float("inf")), (0, 0.02), (20, -1)]
)
def test_landing_rejects_invalid_time_settings(max_time, dt):
    launch = LaunchConditions(velocity=50, launch_angle=0.3, spin_rate=2500)
    with pytest.raises(ValueError, match="finite and positive"):
        BallFlightSimulator().simulate_landing(launch, max_time=max_time, dt=dt)


def test_landing_skips_full_postprocessing(monkeypatch):
    pytest.importorskip("upstream_physics")
    simulator = BallFlightSimulator()

    def forbidden(*args):
        raise AssertionError("landing-only path reconstructed full trajectory")

    monkeypatch.setattr(simulator, "_post_process_rust", forbidden)
    landing = simulator.simulate_landing(
        LaunchConditions(velocity=50, launch_angle=0.3, spin_rate=2500),
        max_time=20,
        dt=0.02,
    )
    assert landing[0] > 0
    assert abs(landing[2]) < 1e-12
