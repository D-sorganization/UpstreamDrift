"""Regression: MuJoCo control-at-state must respect ``q`` when nv != nu (#14943)."""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytestmark = pytest.mark.unit

_UNDERACTUATED_MJCF = """
<mujoco model="underactuated_pendulum">
  <worldbody>
    <body name="link1" pos="0 0 1">
      <joint name="j1" type="hinge" axis="0 1 0"/>
      <geom type="capsule" size="0.04" fromto="0 0 0 0.4 0 0" mass="1"/>
      <body name="link2" pos="0.4 0 0">
        <joint name="j2" type="hinge" axis="0 1 0"/>
        <geom type="capsule" size="0.04" fromto="0 0 0 0.4 0 0" mass="1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="m2" joint="j2" gear="1"/>
  </actuator>
</mujoco>
"""


def test_control_acceleration_at_state_uses_supplied_q_when_nv_ne_nu() -> None:
    import mujoco
    from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine import (
        MuJoCoPhysicsEngine,
    )

    engine = MuJoCoPhysicsEngine()
    engine.load_from_string(_UNDERACTUATED_MJCF)
    assert engine.model is not None and engine.data is not None
    assert engine.model.nv == 2
    assert engine.model.nu == 1
    assert engine.model.nv != engine.model.nu

    q_a = np.array([0.2, -0.3])
    q_b = np.array([0.9, 0.4])
    tau = np.array([0.0, 1.5])

    engine.data.qpos[: engine.model.nq] = q_a
    engine.data.qvel[:] = 0.0
    mujoco.mj_forward(engine.model, engine.data)

    at_b = engine.compute_control_acceleration_at_state(q_b, tau)

    engine.data.qpos[: engine.model.nq] = q_b
    engine.data.qvel[:] = 0.0
    mujoco.mj_forward(engine.model, engine.data)
    reference = engine.compute_control_acceleration(tau)

    np.testing.assert_allclose(at_b, reference, rtol=1e-9, atol=1e-9)
    assert not np.allclose(at_b, np.zeros(engine.model.nv))
    assert np.linalg.norm(at_b - reference) == pytest.approx(0.0, abs=1e-9)

    engine.data.qpos[: engine.model.nq] = q_a
    mujoco.mj_forward(engine.model, engine.data)
    at_b_while_at_a = engine.compute_control_acceleration_at_state(q_b, tau)
    np.testing.assert_allclose(at_b_while_at_a, at_b, rtol=1e-9, atol=1e-9)
