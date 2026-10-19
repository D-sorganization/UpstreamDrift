"""Contract tests for get_joint_names() across physics engines (#12990).

Validates the invariant:
- len(get_joint_names()) == nv
- Names are unique and non-empty
- On double pendulum, engines return ['upper_joint', 'lower_joint']
- Multi-DOF joints expand cleanly per DOF
- Uninitialized engines return []
- API routes receive non-empty joint names from engines
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.shared.python.engine_core._simulation_interface import SimulationInterface

pytestmark = [pytest.mark.unit]

REPO_ROOT = Path(__file__).resolve().parents[2]
DOUBLE_PENDULUM_URDF = REPO_ROOT / "tests/fixtures/models/double_pendulum.urdf"
MUJOCO_GOLFER_XML = (
    REPO_ROOT / "src/engines/physics_engines/mujoco/models/generated/golfer.xml"
)


class DummySimulationEngine(SimulationInterface):
    """Minimal implementation of SimulationInterface for contract testing."""

    def __init__(self, nv: int = 0) -> None:
        self.nv = nv

    @property
    def model_name(self) -> str:
        return "dummy"

    def load_from_path(self, path: str) -> None:
        pass

    def load_from_string(self, content: str, extension: str | None = None) -> None:
        pass

    def reset(self) -> None:
        pass

    def step(self, dt: float | None = None) -> None:
        pass

    def forward(self) -> None:
        pass

    def get_state(self) -> tuple[np.ndarray, np.ndarray]:
        if self.nv == 0:
            return np.array([]), np.array([])
        return np.zeros(self.nv), np.zeros(self.nv)

    def set_state(self, q: np.ndarray, v: np.ndarray) -> None:
        pass

    def set_control(self, u: np.ndarray) -> None:
        pass

    def get_time(self) -> float:
        return 0.0

    def save_checkpoint(self) -> Any:
        return {}

    def restore_checkpoint(self, checkpoint: Any) -> None:
        pass

    @property
    def engine_type(self) -> str:
        return "dummy"


class TestSimulationInterfaceBaseContract:
    """Test the default get_joint_names() in SimulationInterface."""

    def test_uninitialized_returns_empty(self) -> None:
        engine = DummySimulationEngine(nv=0)
        assert engine.get_joint_names() == []

    def test_default_returns_generic_names_matching_nv(self) -> None:
        engine = DummySimulationEngine(nv=3)
        names = engine.get_joint_names()
        assert names == ["dof_0", "dof_1", "dof_2"]
        assert len(names) == 3
        assert len(set(names)) == len(names)


class TestMuJoCoJointNamesContract:
    """Test MuJoCo get_joint_names() contract."""

    def test_mujoco_uninitialized_returns_empty(self) -> None:
        from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine import (
            MuJoCoPhysicsEngine,
        )

        engine = MuJoCoPhysicsEngine()
        assert engine.get_joint_names() == []

    def test_mujoco_double_pendulum_returns_exact_names(self) -> None:
        from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine import (
            MuJoCoPhysicsEngine,
        )

        engine = MuJoCoPhysicsEngine()
        engine.load_from_path(str(DOUBLE_PENDULUM_URDF))
        names = engine.get_joint_names()
        assert len(names) == engine.model.nv == 2
        assert names == ["upper_joint", "lower_joint"]
        assert len(set(names)) == len(names)

    def test_mujoco_golfer_matches_nv_with_multi_dof_expansion(self) -> None:
        from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine import (
            MuJoCoPhysicsEngine,
        )

        if not MUJOCO_GOLFER_XML.exists():
            pytest.skip(f"Golfer XML not found at {MUJOCO_GOLFER_XML}")

        engine = MuJoCoPhysicsEngine()
        engine.load_from_path(str(MUJOCO_GOLFER_XML))
        names = engine.get_joint_names()

        # Golfer has 44 joints (njnt=44) including a 6-DOF free joint, totaling nv=49
        assert len(names) == engine.model.nv == 49
        assert len(names) > engine.model.njnt
        assert len(set(names)) == len(names)
        assert all(isinstance(n, str) and len(n) > 0 for n in names)
        # Verify 6-DOF pelvis free joint is expanded with indices
        assert names[:6] == [f"pelvis_free_{i}" for i in range(6)]


class TestPinocchioJointNamesContract:
    """Test Pinocchio get_joint_names() contract."""

    def test_pinocchio_uninitialized_returns_empty(self) -> None:
        from src.engines.physics_engines.pinocchio.python.pinocchio_physics_engine import (
            PinocchioPhysicsEngine,
        )

        engine = PinocchioPhysicsEngine()
        assert engine.get_joint_names() == []

    def test_pinocchio_mock_double_pendulum(self) -> None:
        from src.engines.physics_engines.pinocchio.python.pinocchio_physics_engine import (
            PinocchioPhysicsEngine,
        )

        engine = PinocchioPhysicsEngine()
        model = MagicMock()
        model.nv = 2
        model.njoints = 3
        model.names = ["universe", "upper_joint", "lower_joint"]

        j0 = MagicMock()
        j0.nv = 0
        j0.idx_v = 0
        j1 = MagicMock()
        j1.nv = 1
        j1.idx_v = 0
        j2 = MagicMock()
        j2.nv = 1
        j2.idx_v = 1
        model.joints = [j0, j1, j2]
        engine.model = model

        names = engine.get_joint_names()
        assert names == ["upper_joint", "lower_joint"]
        assert len(names) == 2
        assert len(set(names)) == len(names)

    def test_pinocchio_mock_multi_dof_joint(self) -> None:
        from src.engines.physics_engines.pinocchio.python.pinocchio_physics_engine import (
            PinocchioPhysicsEngine,
        )

        engine = PinocchioPhysicsEngine()
        model = MagicMock()
        model.nv = 4
        model.njoints = 3
        model.names = ["universe", "root_joint", "elbow_joint"]

        j0 = MagicMock()
        j0.nv = 0
        j0.idx_v = 0
        j1 = MagicMock()
        j1.nv = 3
        j1.idx_v = 0
        j2 = MagicMock()
        j2.nv = 1
        j2.idx_v = 3
        model.joints = [j0, j1, j2]
        engine.model = model

        names = engine.get_joint_names()
        assert len(names) == 4
        assert names == ["root_joint_0", "root_joint_1", "root_joint_2", "elbow_joint"]
        assert len(set(names)) == len(names)


@pytest.fixture
def drake_builder_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``DrakePhysicsEngine()`` constructible when pydrake is absent.

    Without pydrake the engine module binds ``DiagramBuilder`` and
    ``AddMultibodyPlantSceneGraph`` to ``None``, so the constructor cannot run.
    The stub models what real Drake returns for a freshly constructed engine:
    an empty ``MultibodyPlant`` with zero generalized velocities. When pydrake
    is installed the real builder is used unchanged.
    """
    from src.engines.physics_engines.drake.python import drake_physics_engine as dpe

    if dpe.DRAKE_AVAILABLE:
        return

    empty_plant = MagicMock(name="EmptyMultibodyPlant")
    empty_plant.num_velocities.return_value = 0
    empty_plant.num_joints.return_value = 0
    monkeypatch.setattr(dpe, "DiagramBuilder", MagicMock(name="DiagramBuilder"))
    monkeypatch.setattr(
        dpe,
        "AddMultibodyPlantSceneGraph",
        MagicMock(return_value=(empty_plant, MagicMock(name="SceneGraph"))),
    )


@pytest.mark.usefixtures("drake_builder_stub")
class TestDrakeJointNamesContract:
    """Test Drake get_joint_names() contract."""

    def test_drake_uninitialized_returns_empty(self) -> None:
        from src.engines.physics_engines.drake.python.drake_physics_engine import (
            DrakePhysicsEngine,
        )

        engine = DrakePhysicsEngine()
        assert engine.get_joint_names() == []

    def test_drake_mock_double_pendulum(self) -> None:
        from src.engines.physics_engines.drake.python.drake_physics_engine import (
            DrakePhysicsEngine,
        )

        engine = DrakePhysicsEngine()
        plant = MagicMock()
        plant.num_velocities.return_value = 2
        plant.num_joints.return_value = 2

        j0 = MagicMock()
        j0.name.return_value = "upper_joint"
        j0.num_velocities.return_value = 1
        j0.velocity_start.return_value = 0

        j1 = MagicMock()
        j1.name.return_value = "lower_joint"
        j1.num_velocities.return_value = 1
        j1.velocity_start.return_value = 1

        plant.get_joint.side_effect = lambda idx: [j0, j1][int(idx)]
        engine.plant = plant

        names = engine.get_joint_names()
        assert names == ["upper_joint", "lower_joint"]
        assert len(names) == 2
        assert len(set(names)) == len(names)

    def test_drake_mock_multi_dof_joint(self) -> None:
        from src.engines.physics_engines.drake.python.drake_physics_engine import (
            DrakePhysicsEngine,
        )

        engine = DrakePhysicsEngine()
        plant = MagicMock()
        plant.num_velocities.return_value = 3
        plant.num_joints.return_value = 2

        j0 = MagicMock()
        j0.name.return_value = "ball_joint"
        j0.num_velocities.return_value = 2
        j0.velocity_start.return_value = 0

        j1 = MagicMock()
        j1.name.return_value = "wrist_joint"
        j1.num_velocities.return_value = 1
        j1.velocity_start.return_value = 2

        plant.get_joint.side_effect = lambda idx: [j0, j1][int(idx)]
        engine.plant = plant

        names = engine.get_joint_names()
        assert len(names) == 3
        assert names == ["ball_joint_0", "ball_joint_1", "wrist_joint"]
        assert len(set(names)) == len(names)

    def test_drake_mock_fallback_when_no_joints(self) -> None:
        from src.engines.physics_engines.drake.python.drake_physics_engine import (
            DrakePhysicsEngine,
        )

        engine = DrakePhysicsEngine()
        plant = MagicMock()
        plant.num_velocities.return_value = 2
        plant.num_joints.return_value = 0
        plant.num_actuators.return_value = 0
        engine.plant = plant

        names = engine.get_joint_names()
        assert names == ["dof_0", "dof_1"]
        assert len(names) == 2


class TestOpenSimJointNamesContract:
    """Test OpenSim get_joint_names() contract."""

    def test_opensim_uninitialized_returns_empty(self) -> None:
        from src.engines.physics_engines.opensim.python.opensim_physics_engine import (
            OpenSimPhysicsEngine,
        )

        engine = OpenSimPhysicsEngine()
        assert engine.get_joint_names() == []

    def test_opensim_mock_coordinate_names(self) -> None:
        from src.engines.physics_engines.opensim.python.opensim_physics_engine import (
            OpenSimPhysicsEngine,
        )

        engine = OpenSimPhysicsEngine()
        engine._model = MagicMock()
        engine._state = MagicMock()
        engine._joint_names = ["upper_joint", "lower_joint"]

        names = engine.get_joint_names()
        assert names == ["upper_joint", "lower_joint"]
        assert len(names) == 2
        assert len(set(names)) == len(names)


class TestApiRoutesJointNamesIntegration:
    """Test the 5 API routes depend on non-empty engine get_joint_names()."""

    def test_actuator_controls_route(self) -> None:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from src.api.dependencies import get_engine_manager
        from src.api.routes.actuator_controls import router

        mock_engine = MagicMock()
        mock_engine.engine_type = "test_engine"
        mock_engine.get_joint_names.return_value = ["upper_joint", "lower_joint"]
        mock_engine.get_state.return_value = {"torques": [0.0, 0.0]}
        mock_engine.get_joint_limits.return_value = [(-1.0, 1.0), (-2.0, 2.0)]
        del mock_engine.joint_names

        mgr = MagicMock()
        mgr.get_active_engine.return_value = mock_engine

        test_app = FastAPI()
        test_app.include_router(router)
        test_app.dependency_overrides[get_engine_manager] = lambda: mgr

        client = TestClient(test_app)
        resp = client.get("/simulation/actuators")
        assert resp.status_code == 200
        data = resp.json()
        assert data["n_actuators"] == 2
        assert [a["name"] for a in data["actuators"]] == ["upper_joint", "lower_joint"]

    def test_analysis_tools_route(self) -> None:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from src.api.dependencies import get_engine_manager
        from src.api.routes.analysis_tools import router

        mock_engine = MagicMock()
        mock_engine.get_state.return_value = (
            np.array([0.1, 0.2]),
            np.array([0.3, 0.4]),
        )
        mock_engine.get_joint_names.return_value = ["upper_joint", "lower_joint"]
        mock_engine.get_applied_torques.return_value = np.array([1.0, 2.0])

        mgr = MagicMock()
        mgr.get_active_physics_engine.return_value = mock_engine

        test_app = FastAPI()
        test_app.include_router(router)
        test_app.dependency_overrides[get_engine_manager] = lambda: mgr

        client = TestClient(test_app)
        resp = client.get("/simulation/measurements")
        assert resp.status_code == 200
        data = resp.json()
        assert "joint_angles" in data
        assert len(data["joint_angles"]) == 2
        assert data["joint_angles"][0]["joint_name"] == "upper_joint"
        assert data["joint_angles"][1]["joint_name"] == "lower_joint"

    def test_simulation_runs_service(self) -> None:
        from src.api.services.simulation_runs import SimulationRunStore

        mock_engine = MagicMock()
        mock_engine.get_joint_names.return_value = ["upper_joint", "lower_joint"]
        del mock_engine.joint_names

        store = SimulationRunStore()
        run = store.register_completed_run(
            run_id="test_run_12990",
            engine=mock_engine,
            recorder=MagicMock(),
        )
        assert run.joint_names == ["upper_joint", "lower_joint"]

    def test_aip_methods_get_property(self) -> None:
        from src.api.aip.methods import _model_query

        mock_engine = MagicMock()
        mock_engine.get_joint_names.return_value = ["upper_joint", "lower_joint"]
        del mock_engine.joint_names

        mgr = MagicMock()
        mgr.get_active_engine.return_value = mock_engine

        result = _model_query(
            property_name="joints",
            _context={"engine_manager": mgr},
        )
        assert result == {"joints": ["upper_joint", "lower_joint"]}

    def test_control_interface(self) -> None:
        from src.shared.python.control_interface import ControlInterface

        mock_engine = MagicMock()
        mock_engine.get_joint_names.return_value = ["upper_joint", "lower_joint"]
        del mock_engine.joint_names
        mock_engine.get_state.return_value = (
            np.array([0.0, 0.0]),
            np.array([0.0, 0.0]),
        )

        ctrl = ControlInterface(engine=mock_engine)
        assert ctrl.joint_names == ["upper_joint", "lower_joint"]
        state = ctrl.get_state()
        assert state["joint_names"] == ["upper_joint", "lower_joint"]
        assert len(state["joints"]) == 2
        assert [j["name"] for j in state["joints"]] == ["upper_joint", "lower_joint"]
