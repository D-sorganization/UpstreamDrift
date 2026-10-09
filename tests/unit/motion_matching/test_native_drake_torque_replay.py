"""Provider-enabled native Drake frozen torque replay regressions."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.engines.physics_engines.drake.python.native_torque_replay import (
    build_native_drake_torque_bundle,
    replay_native_drake_torque_bundle,
)

pytestmark = pytest.mark.unit
URDF = '<robot name="replay_probe"><link name="base"><inertial><mass value="1"/><inertia ixx="0.1" iyy="0.1" izz="0.1" ixy="0" ixz="0" iyz="0"/></inertial></link><link name="arm"><inertial><origin xyz="0 0 -0.2"/><mass value="0.5"/><inertia ixx="0.02" iyy="0.02" izz="0.02" ixy="0" ixz="0" iyz="0"/></inertial></link><joint name="hinge" type="revolute"><parent link="base"/><child link="arm"/><axis xyz="0 1 0"/><limit lower="-3" upper="3" effort="2" velocity="100"/></joint><transmission name="hinge_motor"><type>transmission_interface/SimpleTransmission</type><joint name="hinge"><hardwareInterface>EffortJointInterface</hardwareInterface></joint><actuator name="motor"><hardwareInterface>EffortJointInterface</hardwareInterface><mechanicalReduction>1</mechanicalReduction></actuator></transmission></robot>'


@pytest.fixture
def native_fixture(tmp_path: Path) -> tuple[Path, Any, Any]:
    pytest.importorskip("pydrake.multibody.plant")
    from pydrake.multibody.parsing import Parser
    from pydrake.multibody.plant import MultibodyPlant

    path = tmp_path / "native_replay.urdf"
    path.write_text(URDF, encoding="utf-8")
    plant = MultibodyPlant(0.001)
    plant.SetUseSampledOutputPorts(False)
    Parser(plant).AddModels(str(path))
    plant.Finalize()
    context = plant.CreateDefaultContext()
    state = plant.GetPositionsAndVelocities(context)
    state[4:7] = (0, 0, 1)
    state[-1] = 0.1
    plant.SetPositionsAndVelocities(context, state)
    return path, plant, context


def test_native_drake_reproduces_recorded_held_feedback_torque(
    native_fixture: tuple[Path, Any, Any],
) -> None:
    from pydrake.systems.analysis import Simulator

    path, plant, context = native_fixture
    initial = context.get_discrete_state_vector().CopyToVector()
    assert plant.num_positions() != plant.num_velocities()
    assert context.num_abstract_states() == 0
    simulator = Simulator(plant, context)
    simulator.Initialize()
    times = np.arange(21) * 0.001
    states = [plant.GetPositionsAndVelocities(context).copy()]
    inputs = []
    for target_time in times[1:]:
        state = plant.GetPositionsAndVelocities(context)
        torque = float(np.clip(50 * (0.8 - state[7]) - state[-1], -2, 2))
        inputs.append([torque])
        plant.get_actuation_input_port().FixValue(context, np.array([torque]))
        simulator.AdvanceTo(float(target_time))
        states.append(plant.GetPositionsAndVelocities(context).copy())
    values = np.array(inputs + [inputs[-1]])
    assert np.any(np.abs(values) == 2)
    bundle = build_native_drake_torque_bundle(path, initial, times, values)
    replay = replay_native_drake_torque_bundle(bundle, path)
    np.testing.assert_array_equal(replay.discrete_states[0], initial)
    np.testing.assert_allclose(replay.qpos, np.array(states)[:, :8], atol=1e-12, rtol=0)
    np.testing.assert_allclose(replay.qvel, np.array(states)[:, 8:], atol=1e-12, rtol=0)
    assert not replay.qpos.flags.writeable


@pytest.mark.parametrize("defect", ["missing_state", "quaternion", "limit", "grid"])
def test_native_drake_rejects_incomplete_or_silently_repaired_inputs(
    native_fixture: tuple[Path, Any, Any], defect: str
) -> None:
    path, plant, context = native_fixture
    initial = context.get_discrete_state_vector().CopyToVector()
    times, values = np.arange(3) * 0.001, np.zeros((3, 1))
    if defect == "missing_state":
        initial = initial[:-1]
    elif defect == "quaternion":
        initial[:4] = 0
    elif defect == "limit":
        values[:] = 3
    else:
        times[1] = 0.0005
    with pytest.raises(ValueError):
        build_native_drake_torque_bundle(path, initial, times, values)


def test_native_drake_rejects_ignored_nonunit_mechanical_reduction(
    native_fixture: tuple[Path, Any, Any],
) -> None:
    path, plant, context = native_fixture
    path.write_text(
        URDF.replace("<mechanicalReduction>1", "<mechanicalReduction>2"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="reduction|transmission"):
        build_native_drake_torque_bundle(
            path,
            context.get_discrete_state_vector().CopyToVector(),
            np.arange(3) * 0.001,
            np.zeros((3, 1)),
        )


def test_native_drake_rejects_changed_model_after_bundle_export(
    native_fixture: tuple[Path, Any, Any],
) -> None:
    path, plant, context = native_fixture
    bundle = build_native_drake_torque_bundle(
        path,
        context.get_discrete_state_vector().CopyToVector(),
        np.arange(3) * 0.001,
        np.zeros((3, 1)),
    )
    path.write_text(
        URDF.replace('mass value="0.5"', 'mass value="0.7"'), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="identity|model|policy"):
        replay_native_drake_torque_bundle(bundle, path)
