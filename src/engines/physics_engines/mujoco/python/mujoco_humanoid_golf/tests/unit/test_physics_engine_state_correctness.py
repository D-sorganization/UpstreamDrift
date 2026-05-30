"""Tests for MuJoCo physics engine correctness fixes (issue #6638 F2/F3/F6).

Tests use ``pytest.importorskip`` so CI jobs without MuJoCo simply skip.

Coverage:
    F2 — compute_inverse_dynamics preserves sim state (qacc unchanged after call)
    F3 — compute_gravity_forces is velocity-independent (pure g(q), not C+g)
    F6 — model_name returns a real name, not the constant "MuJoCo Model"
"""

from __future__ import annotations

import pytest

# Skip the whole module early if MuJoCo is absent
mujoco = pytest.importorskip("mujoco")

import numpy as np  # noqa: E402 (after importorskip guard)

# ── Minimal MuJoCo test XML ─────────────────────────────────────────────────

_MINIMAL_XML = """
<mujoco model="test_pendulum">
  <option timestep="0.01" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="link1" pos="0 0 0.5">
      <joint name="hinge" type="hinge" axis="0 1 0"/>
      <geom type="capsule" size="0.05 0.25" mass="1.0"/>
    </body>
  </worldbody>
  <actuator>
    <motor joint="hinge" gear="1"/>
  </actuator>
</mujoco>
"""


@pytest.fixture()
def engine():
    """Return a loaded MuJoCoPhysicsEngine fixture."""
    from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine import (
        MuJoCoPhysicsEngine,
    )

    eng = MuJoCoPhysicsEngine()
    eng.load_from_string(_MINIMAL_XML)
    return eng


@pytest.fixture()
def unloaded_engine():
    """Return an unloaded MuJoCoPhysicsEngine fixture."""
    from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine import (
        MuJoCoPhysicsEngine,
    )

    return MuJoCoPhysicsEngine()


# ── F2: compute_inverse_dynamics state restoration ──────────────────────────


class TestInverseDynamicsStateRestoration:
    """F2 — compute_inverse_dynamics must not mutate persistent qacc."""

    def test_qacc_unchanged_after_call(self, engine) -> None:
        """data.qacc must be restored to its pre-call value (F2)."""
        assert engine.data is not None
        nv: int = engine.model.nv

        # Set a distinctive initial qacc
        initial_qacc = np.full(nv, 3.14)
        engine.data.qacc[:] = initial_qacc

        # Call inverse dynamics with a *different* qacc
        test_qacc = np.zeros(nv)
        _tau = engine.compute_inverse_dynamics(test_qacc)

        # qacc must be restored
        np.testing.assert_array_almost_equal(
            engine.data.qacc,
            initial_qacc,
            decimal=12,
            err_msg="compute_inverse_dynamics mutated data.qacc (F2 regression)",
        )

    def test_inverse_dynamics_returns_finite_torques(self, engine) -> None:
        """Torques from inverse dynamics must be finite arrays."""
        nv: int = engine.model.nv
        tau = engine.compute_inverse_dynamics(np.zeros(nv))
        assert len(tau) == nv
        assert np.all(np.isfinite(tau)), "Inverse dynamics returned non-finite torques"

    def test_qacc_restored_even_after_nv_mismatch_error(self, engine) -> None:
        """qacc must not be left corrupted if a dimension error is raised (F2)."""
        assert engine.data is not None
        nv: int = engine.model.nv
        initial_qacc = np.full(nv, 9.9)
        engine.data.qacc[:] = initial_qacc

        wrong_qacc = np.zeros(nv + 5)  # wrong size — should raise
        with pytest.raises(Exception):  # noqa: B017 (intentional broad catch)
            engine.compute_inverse_dynamics(wrong_qacc)

        # qacc must be unchanged (call should have raised before writing)
        np.testing.assert_array_almost_equal(engine.data.qacc, initial_qacc, decimal=12)


# ── F3: compute_gravity_forces velocity independence ────────────────────────


class TestGravityForcesVelocityIndependence:
    """F3 — compute_gravity_forces must return pure g(q), not C(q,v)v + g(q)."""

    def test_gravity_forces_velocity_independent(self, engine) -> None:
        """g(q) at v=0 and v≠0 must be identical (F3).

        At v=0, qfrc_bias == g(q).  At v≠0, qfrc_bias == C(q,v)v + g(q).
        The fix temporarily zeroes qvel so both evaluations yield g(q).
        """
        assert engine.data is not None
        nv: int = engine.model.nv

        engine.data.qpos[:] = 0.5  # fixed, nonzero configuration

        # g(q) at zero velocity
        engine.data.qvel[:] = 0.0
        engine.forward()
        g_at_zero_vel = engine.compute_gravity_forces()

        # g(q) at large velocity (Coriolis would be significant if not zeroed)
        engine.data.qvel[:] = 10.0
        engine.forward()
        g_at_nonzero_vel = engine.compute_gravity_forces()

        np.testing.assert_array_almost_equal(
            g_at_zero_vel,
            g_at_nonzero_vel,
            decimal=8,
            err_msg=(
                "compute_gravity_forces changed with velocity — "
                "Coriolis term is leaking into the output (F3 regression)"
            ),
        )
        assert len(g_at_zero_vel) == nv

    def test_gravity_forces_finite(self, engine) -> None:
        """Gravity forces must always be finite."""
        g = engine.compute_gravity_forces()
        assert np.all(np.isfinite(g)), "Gravity forces are non-finite"

    def test_gravity_does_not_mutate_qvel(self, engine) -> None:
        """compute_gravity_forces must restore qvel after computation (F3)."""
        assert engine.data is not None
        nv: int = engine.model.nv
        initial_qvel = np.full(nv, 5.0)
        engine.data.qvel[:] = initial_qvel
        engine.forward()

        _g = engine.compute_gravity_forces()

        np.testing.assert_array_almost_equal(
            engine.data.qvel,
            initial_qvel,
            decimal=12,
            err_msg="compute_gravity_forces mutated data.qvel (F3 side-effect)",
        )

    def test_gravity_does_not_mutate_qpos(self, engine) -> None:
        """compute_gravity_forces must not alter qpos."""
        assert engine.data is not None
        initial_qpos = engine.data.qpos.copy()

        _g = engine.compute_gravity_forces()

        np.testing.assert_array_almost_equal(
            engine.data.qpos,
            initial_qpos,
            decimal=12,
            err_msg="compute_gravity_forces mutated data.qpos",
        )


# ── F6: model_name reflects real name ───────────────────────────────────────


class TestModelName:
    """F6 — model_name must return the real model name, not a constant."""

    def test_model_name_before_load_is_none(self, unloaded_engine) -> None:
        """model_name returns 'None' when no model is loaded."""
        assert unloaded_engine.model_name == "None"

    def test_model_name_not_constant_mujoco_model(self, engine) -> None:
        """model_name must not be 'MuJoCo Model' after loading a named model (F6).

        The XML has ``<mujoco model="test_pendulum">``.
        """
        name = engine.model_name
        assert name != "MuJoCo Model", (
            f"model_name returned the constant 'MuJoCo Model' instead "
            f"of the real XML model name (F6 regression); got: {name!r}"
        )

    def test_model_name_is_non_empty_string(self, engine) -> None:
        """model_name must be a non-empty string after loading."""
        name = engine.model_name
        assert isinstance(name, str)
        assert len(name) > 0
