"""Regression tests for MuJoCo engine state/query bugs (issue #6638).

Covers:
- F2: ``compute_inverse_dynamics`` must not mutate persistent ``qacc``.
- F3: ``compute_gravity_forces`` must return velocity-independent gravity g(q),
      not the velocity-dependent bias forces C(q,v)v + g(q).
- F6: ``model_name`` must reflect the loaded model, not a constant string.

These use a real MuJoCo model so they anchor to actual physics rather than
shape assertions.
"""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine import (  # noqa: E501
    MuJoCoPhysicsEngine,
)

# A named 2-DOF planar double pendulum under gravity.
_MJCF = """
<mujoco model="double_pendulum_test">
  <option gravity="0 0 -9.81"/>
  <worldbody>
    <body name="upper" pos="0 0 0">
      <joint name="j1" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.5" size="0.02" mass="1"/>
      <body name="lower" pos="0 0 -0.5">
        <joint name="j2" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.5" size="0.02" mass="1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="j1"/>
    <motor joint="j2"/>
  </actuator>
</mujoco>
"""


@pytest.fixture
def engine() -> MuJoCoPhysicsEngine:
    eng = MuJoCoPhysicsEngine()
    eng.load_from_string(_MJCF)
    eng.set_state(np.array([0.3, -0.2]), np.zeros(2))
    return eng


def test_compute_inverse_dynamics_restores_qacc(engine: MuJoCoPhysicsEngine) -> None:
    """F2: a 'pure query' must leave persistent qacc unchanged."""
    qacc_before = engine.data.qacc.copy()
    tau = engine.compute_inverse_dynamics(np.array([1.5, -0.7]))
    assert tau.shape == (2,)
    np.testing.assert_array_equal(engine.data.qacc, qacc_before)


def test_compute_gravity_forces_is_velocity_independent(
    engine: MuJoCoPhysicsEngine,
) -> None:
    """F3: gravity g(q) must not change with velocity."""
    # At rest.
    engine.set_state(np.array([0.3, -0.2]), np.zeros(2))
    grav_v0 = engine.compute_gravity_forces()

    # Same configuration, nonzero velocity (introduces Coriolis terms in bias).
    engine.set_state(np.array([0.3, -0.2]), np.array([2.5, -1.8]))
    grav_vnonzero = engine.compute_gravity_forces()

    np.testing.assert_allclose(grav_v0, grav_vnonzero, atol=1e-9)

    # Bias forces SHOULD differ at nonzero velocity (sanity: they include
    # Coriolis/centrifugal), confirming gravity is genuinely isolated.
    bias_nonzero = engine.compute_bias_forces()
    assert not np.allclose(grav_vnonzero, bias_nonzero)


def test_gravity_query_restores_velocity(engine: MuJoCoPhysicsEngine) -> None:
    """F3: computing gravity must not clobber the real qvel."""
    engine.set_state(np.array([0.3, -0.2]), np.array([2.5, -1.8]))
    _ = engine.compute_gravity_forces()
    np.testing.assert_allclose(engine.data.qvel, np.array([2.5, -1.8]))


def test_model_name_reflects_loaded_model(engine: MuJoCoPhysicsEngine) -> None:
    """F6: model_name must decode the real model name, not a constant."""
    assert engine.model_name == "double_pendulum_test"
