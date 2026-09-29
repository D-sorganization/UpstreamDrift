"""Canonical ZVCF and OpenSim analytic-Jacobian tests (#7051, #8586).

Two acceptance criteria from issue #7051:

1. **Drake ZVCF zeros nonzero actuation.** Under the ratified terminology
   contract, ``compute_zvcf`` fixes configuration and zeros both velocity and
   declared applied control. Nonzero control therefore cannot change ZVCF.

2. **OpenSim Jacobian is analytic.** ``compute_jacobian`` now uses Simbody's
   analytic station/system Jacobian; this is regression-checked against the
   retained finite-difference path.

Both are gated by the matching ``requires_*`` marker plus an availability
``skipif`` so missing wheels skip cleanly.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.engine_core.engine_availability import (
    is_engine_available,
)

pytestmark = [pytest.mark.unit]

# A 1-DOF actuated pendulum so Drake exposes one actuator on the joint.
_ACTUATED_PENDULUM_URDF = """<?xml version="1.0"?>
<robot name="actuated_pendulum">
  <link name="base_link">
    <inertial>
      <mass value="0.001"/>
      <inertia ixx="1e-6" ixy="0" ixz="0" iyy="1e-6" iyz="0" izz="1e-6"/>
    </inertial>
  </link>
  <link name="rod">
    <inertial>
      <origin xyz="0 0 -1.0"/>
      <mass value="1.0"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>
  <joint name="pivot" type="revolute">
    <parent link="base_link"/>
    <child link="rod"/>
    <origin xyz="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-3.14159" upper="3.14159" effort="100" velocity="10"/>
  </joint>
  <transmission name="pivot_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="pivot"><hardwareInterface>EffortJointInterface</hardwareInterface></joint>
    <actuator name="pivot_motor"><mechanicalReduction>1</mechanicalReduction></actuator>
  </transmission>
</robot>
"""


@pytest.mark.requires_drake
@pytest.mark.skipif(not is_engine_available("drake"), reason="pydrake not installed")
def test_drake_zvcf_is_independent_of_applied_actuation() -> None:
    """Canonical Drake ZVCF zeros the declared applied-control channel."""
    from src.engines.physics_engines.drake.python.drake_physics_engine import (
        DrakePhysicsEngine,
    )

    engine = DrakePhysicsEngine()
    engine.load_from_string(_ACTUATED_PENDULUM_URDF, extension="urdf")

    q = np.array([0.3], dtype=np.float64)
    engine.set_state(q, np.array([0.0]))
    engine.forward()

    if engine.plant.num_actuators() == 0:
        pytest.skip("Loaded plant exposes no actuator; cannot drive ZVCF.")

    # Baseline: zero control.
    engine.set_control(np.zeros(engine.plant.num_actuators()))
    a_zero = np.atleast_1d(engine.compute_zvcf(q))

    # Nonzero control.
    u = np.full(engine.plant.num_actuators(), 2.5, dtype=np.float64)
    engine.set_control(u)
    a_ctrl = np.atleast_1d(engine.compute_zvcf(q))

    np.testing.assert_allclose(a_ctrl, a_zero, atol=1e-12)


@pytest.mark.requires_opensim
@pytest.mark.skipif(not is_engine_available("opensim"), reason="opensim not installed")
def test_opensim_jacobian_analytic_matches_finite_difference() -> None:
    """OpenSim analytic Simbody Jacobian matches the FD baseline (#7051)."""
    from pathlib import Path

    from src.engines.physics_engines.opensim.python.opensim_physics_engine import (
        OpenSimPhysicsEngine,
    )

    repo_root = Path(__file__).resolve().parents[2]
    osim = (
        repo_root
        / "src"
        / "shared"
        / "models"
        / "opensim"
        / "examples"
        / "pendulum_1dof.osim"
    )

    engine = OpenSimPhysicsEngine()
    engine.load_from_path(str(osim))
    engine.set_state(np.array([0.4]), np.array([0.0]))
    engine.forward()

    body_name = "pendulum_link"
    body = engine._model.getBodySet().get(body_name)

    analytic = engine._compute_jacobian_analytic(body)
    if analytic is None:
        pytest.skip("Installed OpenSim lacks the analytic Simbody Jacobian API.")

    fd = engine._compute_jacobian_finite_difference(body)
    assert fd is not None

    # Analytic vs FD regression baseline (FD is 2nd-order; loosen tolerance).
    np.testing.assert_allclose(analytic["linear"], fd["linear"], atol=1e-3)
    assert analytic["spatial"].shape == (6, engine._state.getNU())
    assert np.all(np.isfinite(analytic["spatial"]))
