"""Issue #13421: golfer URDF must receive torque when theta is nonzero."""

from __future__ import annotations

import sys
from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.engines.physics_engines.drake.python.motion_matching.humanoid_urdf import (
    EXPECTED_NUM_REVOLUTE_DOF,
    CANONICAL_URDF,
    ensure_revolute_joint_actuators,
)
from src.engines.physics_engines.drake.python.motion_matching.simulate import (
    COEFFS_PER_JOINT,
    DEFAULT_GOLFER_URDF,
    SimOptions,
    is_drake_available,
    simulate_with_coefficients,
)

pytestmark = [pytest.mark.unit]


def test_ensure_revolute_joint_actuators_raises_when_joint_index_missing() -> None:
    """DbC: plants without ``get_joint`` cannot be retrofitted."""
    plant = MagicMock()
    plant.num_actuators.return_value = 0
    del plant.get_joint
    with pytest.raises(TypeError, match="get_joint"):
        ensure_revolute_joint_actuators(plant)


def test_ensure_revolute_joint_actuators_noop_when_actuators_present() -> None:
    plant = MagicMock()
    plant.num_actuators.return_value = EXPECTED_NUM_REVOLUTE_DOF
    n = ensure_revolute_joint_actuators(plant)
    assert n == EXPECTED_NUM_REVOLUTE_DOF
    plant.AddJointActuator.assert_not_called()


class _PreFinalizePlant:
    """Unfinalized Drake plant: plant-level ``num_velocities()`` raises.

    Real pydrake raises ``RuntimeError`` for plant-wide state-size queries
    before ``Finalize()``; joint-level queries are allowed (#13017).
    """

    def __init__(self, joints: list[MagicMock]) -> None:
        self._joints = joints
        self.actuators: list[str] = []

    def num_velocities(self) -> int:
        msg = "Pre-finalize calls to 'num_velocities()' are not allowed"
        raise RuntimeError(msg)

    def num_joints(self) -> int:
        return len(self._joints)

    def get_joint(self, index: int) -> MagicMock:
        return self._joints[int(index)]

    def num_actuators(self) -> int:
        return len(self.actuators)

    def AddJointActuator(self, name: str, joint: MagicMock) -> None:  # noqa: N802
        self.actuators.append(name)


def _joint(name: str, nv: int, type_name: str) -> MagicMock:
    joint = MagicMock(name=name)
    joint.name.return_value = name
    joint.num_velocities.return_value = nv
    joint.type_name.return_value = type_name
    return joint


@pytest.fixture
def _fake_joint_index() -> Generator[None, None, None]:
    tree = MagicMock()
    tree.JointIndex = int
    with patch.dict(sys.modules, {"pydrake.multibody.tree": tree}):
        yield


@pytest.mark.usefixtures("_fake_joint_index")
def test_retrofit_runs_on_an_unfinalized_plant() -> None:
    """The retrofit runs pre-Finalize, so it must not query plant nv."""
    joints = [_joint(f"j{i}", 1, "revolute") for i in range(3)]
    joints.append(_joint("club_weld", 0, "weld"))
    plant = _PreFinalizePlant(joints)

    assert ensure_revolute_joint_actuators(plant) == 3
    assert plant.actuators == ["j0_motor", "j1_motor", "j2_motor"]


@pytest.mark.usefixtures("_fake_joint_index")
def test_retrofit_still_rejects_unactuated_multi_dof_joints() -> None:
    """A ball joint the retrofit skips leaves DOFs unactuated: fail closed."""
    joints = [
        _joint("root", 6, "quaternion_floating"),
        _joint("hip", 1, "revolute"),
        _joint("shoulder", 3, "ball_rpy"),
    ]
    with pytest.raises(ValueError, match="expected 4 actuated DOFs"):
        ensure_revolute_joint_actuators(_PreFinalizePlant(joints))


_PYDRAKE_KEYS = [
    "pydrake",
    "pydrake.all",
    "pydrake.multibody",
    "pydrake.multibody.parsing",
    "pydrake.multibody.plant",
    "pydrake.multibody.tree",
    "pydrake.systems",
    "pydrake.systems.analysis",
    "pydrake.systems.framework",
    "pydrake.systems.primitives",
    "pydrake.math",
]


def _make_plant_mock(n_actuators: int = 0) -> MagicMock:
    plant = MagicMock(name="MultibodyPlant")
    plant.num_positions.return_value = 25
    plant.num_velocities.return_value = 25
    plant.num_actuators.return_value = n_actuators
    plant.num_multibody_states.return_value = 50
    plant.GetPositions.return_value = np.zeros(25)
    plant.GetVelocities.return_value = np.zeros(25)
    plant.HasBodyNamed.return_value = False
    plant.get_actuation_input_port.return_value = MagicMock(name="ActuationPort")
    return plant


@pytest.fixture
def _mocked_pydrake_zero_actuators() -> Generator[dict[str, MagicMock], None, None]:
    """Plant reports zero actuators (URDF without transmissions)."""
    mocks: dict[str, MagicMock] = {key: MagicMock() for key in _PYDRAKE_KEYS}
    plant = _make_plant_mock(n_actuators=0)
    scene_graph = MagicMock(name="SceneGraph")
    mocks["pydrake.multibody.plant"].AddMultibodyPlantSceneGraph = MagicMock(
        return_value=(plant, scene_graph)
    )

    class _FakeLeafSystem:
        def __init__(self) -> None:
            self._ports: list[Any] = []

        def DeclareVectorOutputPort(
            self, name: str, model_vector: Any, calc_callback: Any
        ) -> Any:
            port = MagicMock(name=f"OutputPort[{name}]")
            self._ports.append(port)
            return port

        def get_output_port(self, idx: int) -> Any:
            return self._ports[idx] if self._ports else MagicMock()

    framework_mod = mocks["pydrake.systems.framework"]
    framework_mod.LeafSystem = _FakeLeafSystem
    framework_mod.BasicVector = MagicMock(side_effect=lambda n: MagicMock())
    framework_mod.DiagramBuilder = MagicMock(return_value=MagicMock())

    sim_instance = MagicMock(name="Simulator")
    sim_context = MagicMock(name="DiagramContext")
    sim_instance.get_context.return_value = sim_context
    mocks["pydrake.systems.analysis"].Simulator = MagicMock(return_value=sim_instance)
    mocks["pydrake.systems.primitives"].VectorLogSink = MagicMock(
        return_value=MagicMock()
    )

    with patch.dict(sys.modules, mocks):
        yield {"plant": plant, "mocks": mocks}


def test_simulate_raises_when_actuation_cannot_be_wired(
    _mocked_pydrake_zero_actuators: dict[str, MagicMock],
) -> None:
    """Never return success with phantom torques when the actuation port is open."""
    theta = np.zeros(EXPECTED_NUM_REVOLUTE_DOF * COEFFS_PER_JOINT)
    theta[0] = 50.0
    with (
        patch(
            "src.engines.physics_engines.drake.python.motion_matching.simulate."
            "load_humanoid_into_plant",
            return_value=MagicMock(),
        ),
        pytest.raises(ValueError, match="cannot wire polynomial torque"),
    ):
        simulate_with_coefficients(
            theta,
            options=SimOptions(simulation_time_s=0.01, sample_rate_hz=1000.0),
        )


@pytest.mark.requires_drake
def test_live_nonzero_theta_changes_motion_vs_zero_theta() -> None:
    """200 N·m constant joint torques must move the plant relative to zero torque."""
    if not is_drake_available():
        pytest.skip("pydrake runtime is not installed on host")
    if not DEFAULT_GOLFER_URDF.exists():
        pytest.skip(f"default golfer URDF missing: {DEFAULT_GOLFER_URDF}")

    opts = SimOptions(
        urdf_path=DEFAULT_GOLFER_URDF,
        simulation_time_s=0.05,
        sample_rate_hz=1000.0,
    )
    n_act = EXPECTED_NUM_REVOLUTE_DOF
    zero = simulate_with_coefficients(np.zeros(n_act * COEFFS_PER_JOINT), opts)
    torque = np.zeros(n_act * COEFFS_PER_JOINT)
    torque[::COEFFS_PER_JOINT] = 200.0
    driven = simulate_with_coefficients(torque, opts)

    assert zero.solver_status == "success"
    assert driven.solver_status == "success"
    assert np.max(driven.tau) > 0.0
    max_q_delta = float(np.max(np.abs(driven.q - zero.q)))
    assert max_q_delta > 1.0e-4, (
        "nonzero theta produced identical q to zero theta (torque not applied)"
    )


@pytest.mark.requires_drake
def test_load_humanoid_adds_actuators_for_transmissionless_urdf() -> None:
    """``load_humanoid_into_plant`` retrofits actuators before finalize."""
    if not is_drake_available():
        pytest.skip("pydrake runtime is not installed on host")
    from pydrake.multibody.plant import MultibodyPlant

    from src.engines.physics_engines.drake.python.motion_matching.humanoid_urdf import (
        load_humanoid_into_plant,
    )

    plant = MultibodyPlant(1e-3)
    load_humanoid_into_plant(plant, CANONICAL_URDF)
    assert plant.num_actuators() == EXPECTED_NUM_REVOLUTE_DOF
