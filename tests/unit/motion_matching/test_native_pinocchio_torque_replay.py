"""Real native Pinocchio frozen-input replay and admission regressions."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.unit

URDF = """<robot name="native_replay_probe">
<link name="world"/>
<link name="base"><inertial><mass value="1"/><inertia ixx="0.1"
iyy="0.1" izz="0.1" ixy="0" ixz="0" iyz="0"/></inertial></link>
<joint name="root" type="floating"><parent link="world"/>
<child link="base"/></joint>
<link name="arm"><inertial><origin xyz="0 0 -0.2"/><mass value="0.5"/>
<inertia ixx="0.02" iyy="0.02" izz="0.02" ixy="0" ixz="0" iyz="0"/>
</inertial></link>
<joint name="hinge" type="revolute"><parent link="base"/>
<child link="arm"/><axis xyz="0 1 0"/>
<limit lower="-3" upper="3" effort="2" velocity="100"/></joint>
<transmission name="hinge_motor"><type>transmission_interface/SimpleTransmission</type>
<joint name="hinge"><hardwareInterface>EffortJointInterface</hardwareInterface></joint>
<actuator name="motor"><hardwareInterface>EffortJointInterface</hardwareInterface>
<mechanicalReduction>1</mechanicalReduction></actuator></transmission></robot>"""


@pytest.fixture
def native_fixture(tmp_path: Path) -> tuple[Path, Any, Any]:
    pin = pytest.importorskip("pinocchio")
    if not isinstance(getattr(pin, "__file__", None), str):
        pytest.skip("real native binding required; mock is not evidence")
    from src.engines.physics_engines.pinocchio.python import native_torque_replay
    from src.engines.physics_engines.pinocchio.python.pinocchio_physics_engine import (
        PinocchioPhysicsEngine,
    )

    path = tmp_path / "native.urdf"
    path.write_text(URDF, encoding="utf-8")
    engine = PinocchioPhysicsEngine()
    engine.load_from_path(str(path))
    q = pin.neutral(engine.model)
    q[2] = 1
    v = np.zeros(engine.model.nv)
    v[-1] = 0.1
    engine.set_state(q, v)
    return path, engine, native_torque_replay


def test_native_pinocchio_reproduces_frozen_saturated_feedback(
    native_fixture: tuple[Path, Any, Any],
) -> None:
    path, engine, adapter = native_fixture
    initial_q, initial_v = engine.q.copy(), engine.v.copy()
    assert engine.model.nq == 8 and engine.model.nv == 7
    times = np.arange(21) * 0.001
    positions, velocities, inputs = [initial_q], [initial_v], []
    for _ in times[1:]:
        torque = float(np.clip(50 * (0.8 - engine.q[-1]) - engine.v[-1], -2, 2))
        inputs.append([torque])
        generalized = np.zeros(engine.model.nv)
        generalized[-1] = torque
        engine.set_control(generalized)
        engine.step(0.001, integrator="rk4")
        positions.append(engine.q.copy())
        velocities.append(engine.v.copy())
    values = np.array(inputs + [inputs[-1]])
    bundle = adapter.build_native_pinocchio_torque_bundle(
        path, initial_q, initial_v, times, values
    )
    replay = adapter.replay_native_pinocchio_torque_bundle(bundle, path)
    np.testing.assert_array_equal(replay.qpos[0], initial_q)
    np.testing.assert_allclose(replay.qpos, positions, atol=1e-12, rtol=0)
    np.testing.assert_allclose(replay.qvel, velocities, atol=1e-12, rtol=0)
    np.testing.assert_array_equal(replay.applied_actuator_torques, values[:-1])
    expected = np.zeros((len(times) - 1, engine.model.nv))
    expected[:, -1] = values[:-1, 0]
    np.testing.assert_array_equal(replay.generalized_actuator_torques, expected)
    assert np.any(np.abs(values) == 2)
    assert not replay.qpos.flags.writeable


@pytest.mark.parametrize(
    "defect",
    ["missing_state", "quaternion", "nonfinite_state", "limit", "grid", "sentinel"],
)
def test_native_pinocchio_rejects_unqualified_state_and_inputs(
    native_fixture: tuple[Path, Any, Any], defect: str
) -> None:
    path, engine, adapter = native_fixture
    q, v = engine.q.copy(), engine.v.copy()
    times, values = np.arange(3) * 0.001, np.zeros((3, 1))
    if defect == "missing_state":
        v = v[:-1]
    elif defect == "quaternion":
        q[3:7] = 0
    elif defect == "nonfinite_state":
        v[-1] = np.nan
    elif defect == "limit":
        values[:] = 3
    elif defect == "grid":
        times[1] = 0.0005
    else:
        values[-1] = 0.5
    with pytest.raises(ValueError):
        adapter.build_native_pinocchio_torque_bundle(path, q, v, times, values)


@pytest.mark.parametrize("defect", ["collision", "mimic", "reduction"])
def test_native_pinocchio_rejects_ignored_source_semantics(
    native_fixture: tuple[Path, Any, Any], defect: str
) -> None:
    path, engine, adapter = native_fixture
    raw = URDF
    if defect == "reduction":
        raw = raw.replace("<mechanicalReduction>1", "<mechanicalReduction>2")
    else:
        raw = raw.replace("</robot>", f"<{defect}/></robot>")
    path.write_text(raw, encoding="utf-8")
    with pytest.raises(ValueError):
        adapter.build_native_pinocchio_torque_bundle(
            path, engine.q, engine.v, np.arange(3) * 0.001, np.zeros((3, 1))
        )


def test_native_pinocchio_rejects_model_replacement(
    native_fixture: tuple[Path, Any, Any],
) -> None:
    path, engine, adapter = native_fixture
    bundle = adapter.build_native_pinocchio_torque_bundle(
        path, engine.q, engine.v, np.arange(3) * 0.001, np.zeros((3, 1))
    )
    path.write_text(URDF.replace('mass value="0.5"', 'mass value="0.7"'))
    with pytest.raises(ValueError, match="identity|model|policy"):
        adapter.replay_native_pinocchio_torque_bundle(bundle, path)


def test_native_pinocchio_refinement_reduces_independent_terminal_error(
    native_fixture: tuple[Path, Any, Any],
) -> None:
    import pinocchio as pin

    path, engine, adapter = native_fixture
    endpoints = []
    for dt in (0.016, 0.008, 0.004, 0.001):
        times = np.arange(round(0.256 / dt) + 1) * dt
        values = np.full((len(times), 1), 1.5)
        bundle = adapter.build_native_pinocchio_torque_bundle(
            path, engine.q, engine.v, times, values, time_step=dt
        )
        replay = adapter.replay_native_pinocchio_torque_bundle(bundle, path)
        endpoints.append((replay.qpos[-1], replay.qvel[-1]))
    reference_q, reference_v = endpoints[-1]
    errors = [
        np.linalg.norm(pin.difference(engine.model, q, reference_q))
        + np.linalg.norm(v - reference_v)
        for q, v in endpoints[:-1]
    ]
    assert errors[0] > errors[1] > errors[2]
    assert errors[2] < errors[0] / 4
