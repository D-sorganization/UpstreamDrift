"""Issue #14937: reported forces must match the integrated aerodynamic model."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.shared.python.physics.ball_launch_conditions import LaunchConditions
from src.shared.python.physics.ball_simulator import BallFlightSimulator
from src.shared.python.physics.ball_enhanced_simulator import (
    EnhancedBallFlightSimulator,
)
from src.shared.python.physics.rust_kernel import is_rust_available

pytestmark = [pytest.mark.unit, pytest.mark.scientific]

requires_rust = pytest.mark.skipif(
    not is_rust_available(),
    reason="upstream-physics Rust kernel not available",
)


def _driver_launch() -> LaunchConditions:
    return LaunchConditions(
        velocity=70.0,
        launch_angle=math.radians(12.0),
        spin_rate=3000.0,
        azimuth_angle=0.0,
        spin_axis=np.array([0.0, -1.0, 0.0]),
    )


def _newton_residual(point: object, mass: float) -> float:
    forces = point.forces
    if "total" in forces:
        aero = np.asarray(forces["total"], dtype=float)
    else:
        aero = (
            forces["drag"]
            + forces.get("magnus", np.zeros(3))
            + forces.get("lift", np.zeros(3))
        )
    total = forces["gravity"] + aero
    return float(np.linalg.norm(mass * point.acceleration - total))


@requires_rust
def test_ball_flight_simulator_newton_consistency_per_step() -> None:
    sim = BallFlightSimulator()
    traj = sim.simulate_trajectory(_driver_launch(), max_time=0.35, dt=0.01)
    assert traj and traj[0].forces_model == "rust_kernel_integrated"
    for point in traj:
        assert _newton_residual(point, sim.ball.mass) < 1e-9


@requires_rust
def test_ball_flight_velocity_finite_difference_matches_reported_accel() -> None:
    sim = BallFlightSimulator()
    dt = 0.01
    traj = sim.simulate_trajectory(_driver_launch(), max_time=0.35, dt=dt)
    assert len(traj) > 3
    for prev, point, nxt in zip(traj[:-2], traj[1:-1], traj[2:], strict=True):
        step = nxt.time - prev.time
        assert step > 0
        fd_accel = (nxt.velocity - prev.velocity) / step
        rtol = (dt**4) + 1e-6
        np.testing.assert_allclose(fd_accel, point.acceleration, rtol=rtol, atol=1e-3)


@requires_rust
def test_legacy_forces_flag_tags_polynomial_posthoc() -> None:
    sim = BallFlightSimulator()
    traj = sim.simulate_trajectory(
        _driver_launch(), max_time=0.1, dt=0.01, legacy_forces=True
    )
    assert traj[0].forces_model == "python_polynomial_posthoc"


def test_enhanced_custom_force_provider_is_newton_consistent() -> None:
    def constant_lift(_v, _s, _t, _p):
        return np.array([0.0, 0.0, 2.5])

    sim = EnhancedBallFlightSimulator(force_provider=constant_lift)
    launch = _driver_launch()
    traj = sim.simulate_trajectory(launch, max_time=0.05, dt=0.01)
    assert traj[0].forces_model == "custom_force_provider"
    for point in traj:
        assert _newton_residual(point, sim.ball.mass) < 1e-9
        assert np.allclose(point.forces["drag"], 0.0)
        np.testing.assert_allclose(point.forces["total"], np.array([0.0, 0.0, 2.5]))
