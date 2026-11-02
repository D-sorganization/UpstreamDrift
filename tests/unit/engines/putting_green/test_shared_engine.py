"""Unit tests for SharedPuttingEngine (#14141 step 10a).

Covers PhysicsEngine protocol conformance and the precompute-then-sample
invariants (non-increasing speed, eventual rest, time bookkeeping);
expected values are invariants or facts derived outside the physics
(e.g. the ball mass constant), never copied from a model run.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.engines.physics_engines.putting_green.python.shared_engine import (
    SharedPuttingEngine,
)
from src.engines.loaders import load_putting_green_engine
from src.shared.python.core.physics_constants import GRAVITY_M_S2
from src.shared.python.engine_core.interfaces import PhysicsEngine

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

#: Cap on the number of 2 ms steps a flat-green putt may take to settle.
_MAX_STEPS = 5000


def _run_to_rest(engine: SharedPuttingEngine) -> list[float]:
    """Step until the ball is at rest, recording speed at each step."""
    speeds: list[float] = []
    for _ in range(_MAX_STEPS):
        engine.step()
        _, v = engine.get_state()
        speed = float(np.hypot(v[0], v[1]))
        speeds.append(speed)
        if speed == 0.0:
            break
    return speeds


class TestProtocolConformance:
    def test_isinstance_physics_engine(self) -> None:
        engine = SharedPuttingEngine()
        assert isinstance(engine, PhysicsEngine)

    def test_loader_returns_shared_engine(self, tmp_path) -> None:
        engine = load_putting_green_engine(tmp_path)
        assert isinstance(engine, SharedPuttingEngine)
        assert isinstance(engine, PhysicsEngine)


class TestStateAndStepping:
    def test_default_state_is_at_rest_at_origin(self) -> None:
        engine = SharedPuttingEngine()
        q, v = engine.get_state()
        assert np.allclose(q, 0.0)
        assert np.allclose(v, 0.0)
        assert engine.get_time() == 0.0

    def test_set_state_round_trip(self) -> None:
        engine = SharedPuttingEngine()
        q0 = np.array([1.0, -0.5])
        v0 = np.array([1.5, 0.2])
        engine.set_state(q0, v0)
        q, v = engine.get_state()
        assert np.allclose(q, q0)
        assert np.allclose(v, v0)

    def test_step_advances_time_by_dt(self) -> None:
        engine = SharedPuttingEngine()
        engine.set_state(np.array([0.0, 0.0]), np.array([1.0, 0.0]))
        dt = 0.01
        total = 0.0
        for _ in range(10):
            engine.step(dt)
            total += dt
        assert engine.get_time() == pytest.approx(total)

    def test_speed_is_non_increasing_on_flat_green(self) -> None:
        engine = SharedPuttingEngine()
        engine.set_state(np.array([0.0, 0.0]), np.array([2.0, 0.0]))
        speeds = _run_to_rest(engine)
        assert all(a >= b - 1e-9 for a, b in zip(speeds, speeds[1:], strict=False))

    def test_ball_eventually_reaches_rest(self) -> None:
        engine = SharedPuttingEngine()
        engine.set_state(np.array([0.0, 0.0]), np.array([1.5, 0.3]))
        speeds = _run_to_rest(engine)
        assert speeds[-1] == 0.0

    def test_step_after_rest_holds_position(self) -> None:
        engine = SharedPuttingEngine()
        engine.set_state(np.array([0.0, 0.0]), np.array([0.8, 0.0]))
        _run_to_rest(engine)
        q_rest, v_rest = engine.get_state()
        for _ in range(50):
            engine.step()
        q_after, v_after = engine.get_state()
        assert np.allclose(q_after, q_rest)
        assert np.allclose(v_after, 0.0)
        assert np.allclose(v_rest, 0.0)

    def test_reset_returns_to_origin_at_rest(self) -> None:
        engine = SharedPuttingEngine()
        engine.set_state(np.array([2.0, 1.0]), np.array([1.0, 0.0]))
        engine.step()
        engine.reset()
        q, v = engine.get_state()
        assert np.allclose(q, 0.0)
        assert np.allclose(v, 0.0)
        assert engine.get_time() == 0.0


class TestInvalidInputsRejected:
    @pytest.mark.parametrize("bad_dt", [0.0, -0.01])
    def test_non_positive_dt_rejected(self, bad_dt: float) -> None:
        engine = SharedPuttingEngine()
        with pytest.raises(ValueError):
            engine.step(dt=bad_dt)

    def test_non_finite_state_rejected(self) -> None:
        engine = SharedPuttingEngine()
        with pytest.raises(ValueError):
            engine.set_state(np.array([float("nan"), 0.0]), np.array([0.0, 0.0]))

    def test_wrong_shape_state_rejected(self) -> None:
        engine = SharedPuttingEngine()
        with pytest.raises(ValueError):
            engine.set_state(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0]))


class TestDynamicsQueries:
    def test_mass_matrix_is_ball_mass_identity(self) -> None:
        from src.shared.python.core.physics_constants import GOLF_BALL_MASS_KG

        engine = SharedPuttingEngine()
        mass_matrix = engine.compute_mass_matrix()
        assert np.allclose(mass_matrix, np.eye(2) * float(GOLF_BALL_MASS_KG))

    def test_zvcf_matches_gravity_forces_over_mass(self) -> None:
        engine = SharedPuttingEngine()
        q = np.array([0.0, 0.0])
        zvcf = engine.compute_zvcf(q)
        engine.set_state(q, np.zeros(2))
        gravity = engine.compute_gravity_forces()
        assert np.allclose(zvcf, gravity / engine.ball_mass)

    def test_control_acceleration_is_force_over_mass(self) -> None:
        engine = SharedPuttingEngine()
        tau = np.array([1.0, -2.0])
        accel = engine.compute_control_acceleration(tau)
        assert np.allclose(accel, tau / engine.ball_mass)

    def test_jacobian_identity_for_ball_none_otherwise(self) -> None:
        engine = SharedPuttingEngine()
        jac = engine.compute_jacobian("ball")
        assert jac is not None
        assert np.allclose(jac["linear"], np.eye(2))
        assert engine.compute_jacobian("not-a-body") is None

    def test_contact_force_at_rest_is_the_weight_on_a_flat_green(self) -> None:
        engine = SharedPuttingEngine()
        force = engine.compute_contact_forces()
        assert force.shape == (3,)
        assert np.allclose(force, [0.0, 0.0, engine.ball_mass * GRAVITY_M_S2])

    def test_rolling_contact_friction_opposes_the_velocity(self) -> None:
        engine = SharedPuttingEngine()
        engine.set_state(np.zeros(2), np.array([1.5, 0.5]))
        force = engine.compute_contact_forces()
        _q, v = engine.get_state()
        assert float(np.dot(force[:2], v)) < 0.0
        assert force[2] == pytest.approx(engine.ball_mass * GRAVITY_M_S2)
