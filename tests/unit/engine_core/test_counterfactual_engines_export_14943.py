"""MuJoCo, Pinocchio, and export stamping for #14943 follow-on."""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pytest

from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.counterfactuals import (
    CounterfactualResult,
)
from src.shared.python.engine_core.counterfactual_contract import (
    COUNTERFACTUAL_DEFINITION_VERSION,
)
from src.shared.python.engine_core.engine_availability import require_engine
from src.shared.python.motion_matching.counterfactual import AccelerationDecomposition

pytestmark = pytest.mark.unit

DOUBLE_PENDULUM_URDF = (
    Path(__file__).resolve().parents[2] / "fixtures" / "models" / "double_pendulum.urdf"
)


def test_mujoco_counterfactual_result_stamps_definition_version() -> None:
    z = np.zeros(3)
    result = CounterfactualResult(
        type="zvcf",
        observed_acceleration=z,
        counterfactual_acceleration=z,
        delta_acceleration=z,
        observed_position=None,
        counterfactual_position=None,
        delta_position=None,
        torque_attributed_effect=z,
        velocity_attributed_effect=None,
    )
    assert result.definition_version == COUNTERFACTUAL_DEFINITION_VERSION


def test_acceleration_decomposition_export_stamp() -> None:
    a = np.array([0.0, 1.0])
    decomp = AccelerationDecomposition(a_grav=a, a_drift=a, a_ctrl=a)
    assert decomp.definition_version == COUNTERFACTUAL_DEFINITION_VERSION
    assert decomp.export_stamp() == {
        "counterfactual_definition_version": COUNTERFACTUAL_DEFINITION_VERSION,
    }


@require_engine("pinocchio")
def test_pinocchio_compute_zvcf_emits_deprecation() -> None:
    import pinocchio as pin

    from src.engines.physics_engines.pinocchio.python.pinocchio_physics_engine import (
        PinocchioPhysicsEngine,
    )

    if not DOUBLE_PENDULUM_URDF.is_file():
        pytest.skip("double_pendulum.urdf fixture missing")
    engine = PinocchioPhysicsEngine()
    try:
        engine.load_from_path(str(DOUBLE_PENDULUM_URDF))
    except (ImportError, OSError, ValueError, RuntimeError):
        pytest.skip("Pinocchio model not loadable in this environment")
    q = np.zeros(engine.model.nq)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        legacy = engine.compute_zvcf(q)
    assert any(isinstance(w.message, DeprecationWarning) for w in caught)
    gravity = engine.compute_gravity_only_acceleration(q)
    np.testing.assert_allclose(legacy, gravity)


@require_engine("mujoco")
def test_mujoco_compute_control_acceleration_at_state_evaluates_at_q_and_restores() -> (
    None
):
    import mujoco

    from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine import (
        MuJoCoPhysicsEngine,
    )
    from src.shared.python.engine_core.mujoco_compat import full_mass_matrix

    if not DOUBLE_PENDULUM_URDF.is_file():
        pytest.skip("double_pendulum.urdf fixture missing")
    engine = MuJoCoPhysicsEngine()
    try:
        engine.load_from_path(str(DOUBLE_PENDULUM_URDF))
    except (ImportError, OSError, ValueError, RuntimeError):
        pytest.skip("MuJoCo model not loadable in this environment")

    assert engine.data is not None
    assert engine.model is not None

    q_current = np.array([0.5, -0.4])
    v_current = np.array([0.2, 0.1])
    engine.data.qpos[:] = q_current
    engine.data.qvel[:] = v_current
    mujoco.mj_forward(engine.model, engine.data)

    q_target = np.array([1.2, 0.7])
    tau_nv = np.array([1.5, -2.5])

    acc = engine.compute_control_acceleration_at_state(q_target, tau_nv)

    np.testing.assert_allclose(engine.data.qpos, q_current)
    np.testing.assert_allclose(engine.data.qvel, v_current)

    engine.data.qpos[:] = q_target
    engine.data.qvel[:] = 0.0
    engine.data.ctrl[:] = 0.0
    mujoco.mj_forward(engine.model, engine.data)
    mass_target = full_mass_matrix(mujoco, engine.model, engine.data)
    expected_acc = np.linalg.solve(mass_target, tau_nv)
    np.testing.assert_allclose(acc, expected_acc)


@require_engine("pinocchio")
def test_pinocchio_restores_kinematics_after_counterfactual_queries() -> None:
    import pinocchio as pin

    from src.engines.physics_engines.pinocchio.python.pinocchio_physics_engine import (
        PinocchioPhysicsEngine,
    )

    if not DOUBLE_PENDULUM_URDF.is_file():
        pytest.skip("double_pendulum.urdf fixture missing")
    engine = PinocchioPhysicsEngine()
    try:
        engine.load_from_path(str(DOUBLE_PENDULUM_URDF))
    except (ImportError, OSError, ValueError, RuntimeError):
        pytest.skip("Pinocchio model not loadable in this environment")

    assert engine.data is not None
    assert engine.model is not None

    q_current = np.array([0.3, -0.5])
    v_current = np.array([0.4, 0.2])
    engine.q = q_current.copy()
    engine.v = v_current.copy()
    pin.forwardKinematics(engine.model, engine.data, engine.q, engine.v)
    saved_placement = engine.data.oMi[engine.model.njoints - 1].translation.copy()

    q_target = np.array([1.4, 0.9])
    tau = np.array([0.8, -0.5])

    _ = engine.compute_gravity_only_acceleration(q_target)
    np.testing.assert_allclose(engine.q, q_current)
    np.testing.assert_allclose(engine.v, v_current)
    np.testing.assert_allclose(
        engine.data.oMi[engine.model.njoints - 1].translation, saved_placement
    )

    _ = engine.compute_control_acceleration_at_state(q_target, tau)
    np.testing.assert_allclose(engine.q, q_current)
    np.testing.assert_allclose(engine.v, v_current)
    np.testing.assert_allclose(
        engine.data.oMi[engine.model.njoints - 1].translation, saved_placement
    )
