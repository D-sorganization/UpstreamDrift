"""Regression test for issue #8930 (defect A).

``BallFlightSimulator._post_process_rust`` looped over every trajectory point
and called the scalar ``_calculate_forces_single`` path once per point, even
though a fully vectorized ``_calculate_forces_batch`` path already existed.
This test locks in the fix: force calculation must run as a single batched
call (dispatching through ``vel.ndim > 1``), and the per-point results must
be numerically identical to what the old scalar path produced.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.physics.ball_launch_conditions import LaunchConditions
from src.shared.python.physics.ball_simulator import BallFlightSimulator

pytestmark = pytest.mark.unit


class _FakePoint:
    """Minimal stand-in for the Rust ``TrajectoryPoint`` PyO3 wrapper."""

    def __init__(
        self, t: float, x: float, y: float, z: float, vx: float, vy: float, vz: float
    ) -> None:
        self.t, self.x, self.y, self.z = t, x, y, z
        self.vx, self.vy, self.vz = vx, vy, vz


class _FakeRustResult:
    def __init__(self, points: list[_FakePoint]) -> None:
        self._points = points

    def get_points(self) -> list[_FakePoint]:
        return self._points


def _make_points(n: int) -> list[_FakePoint]:
    return [
        _FakePoint(
            t=i * 0.01,
            x=i * 0.5,
            y=0.0,
            z=max(0.0, i * 0.2 - 0.01 * i * i),
            vx=40.0 - 0.1 * i,
            vy=1.0,
            vz=10.0 - 0.2 * i,
        )
        for i in range(n)
    ]


@pytest.fixture
def launch() -> LaunchConditions:
    return LaunchConditions(
        velocity=50.0,
        launch_angle=0.2,
        azimuth_angle=0.0,
        spin_rate=3000.0,
        spin_axis=np.array([0.0, 1.0, 0.0]),
    )


def test_post_process_rust_batches_force_calculation(
    monkeypatch: pytest.MonkeyPatch, launch: LaunchConditions
) -> None:
    """The Rust post-processing path must call force calc once, not per point."""
    sim = BallFlightSimulator()
    points = _make_points(25)
    rust_result = _FakeRustResult(points)

    calls: list[np.ndarray] = []
    original = sim._calculate_forces

    def spy(vel: np.ndarray, launch_arg: LaunchConditions) -> dict[str, np.ndarray]:
        calls.append(vel)
        return original(vel, launch_arg)

    monkeypatch.setattr(sim, "_calculate_forces", spy)

    result = sim._post_process_rust(rust_result, launch)

    assert len(calls) == 1, (
        "expected exactly one batched _calculate_forces call, "
        f"got {len(calls)} (per-point scalar dispatch was not eliminated)"
    )
    assert calls[0].ndim > 1, "batched call must dispatch through the (3, N) path"
    assert len(result) == len(points)


def test_post_process_rust_matches_scalar_reference(launch: LaunchConditions) -> None:
    """Vectorized output must equal the original per-point scalar computation."""
    sim = BallFlightSimulator()
    points = _make_points(17)
    rust_result = _FakeRustResult(points)

    result = sim._post_process_rust(rust_result, launch)

    omega = launch.spin_rate * 2 * np.pi / 60
    for point, tp in zip(points, result, strict=True):
        vel = np.array([point.vx, point.vy, point.vz])
        pos = np.array([point.x, point.y, point.z])
        drag_ref, magnus_ref = sim._calculate_forces_single(vel, omega, launch)
        gravity_ref = np.array([0.0, 0.0, -sim.ball.mass * sim.environment.gravity])
        acc_ref = (gravity_ref + drag_ref + magnus_ref) / sim.ball.mass

        assert tp.time == pytest.approx(point.t)
        np.testing.assert_allclose(tp.position, pos)
        np.testing.assert_allclose(tp.velocity, vel)
        np.testing.assert_allclose(tp.acceleration, acc_ref, atol=1e-10)
        np.testing.assert_allclose(tp.forces["drag"], drag_ref, atol=1e-10)
        np.testing.assert_allclose(tp.forces["magnus"], magnus_ref, atol=1e-10)


def test_post_process_rust_handles_empty_points(launch: LaunchConditions) -> None:
    """No trajectory points must not crash the batched path."""
    sim = BallFlightSimulator()
    result = sim._post_process_rust(_FakeRustResult([]), launch)
    assert result == []
