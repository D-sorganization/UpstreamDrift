"""Tests for DrakePhysicsEngine wrapper.

Drake's ``__init__`` calls into ``pydrake`` at construction, so we install
fake submodules into ``sys.modules`` and reload the wrapper module so the
``DRAKE_AVAILABLE`` import branch resolves to our fakes. This exercises
the wrapper-level logic (state translation, argument validation, factory
routing) without requiring a real Drake install.
"""

from __future__ import annotations

import importlib
import sys
import types
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest


def _build_pydrake_stubs() -> dict[str, Any]:
    """Construct a stub pydrake module tree and return submodules."""
    pydrake = types.ModuleType("pydrake")
    pydrake.math = types.ModuleType("pydrake.math")
    pydrake.multibody = types.ModuleType("pydrake.multibody")
    pydrake.multibody.parsing = types.ModuleType("pydrake.multibody.parsing")
    pydrake.multibody.plant = types.ModuleType("pydrake.multibody.plant")
    pydrake.multibody.tree = types.ModuleType("pydrake.multibody.tree")
    pydrake.systems = types.ModuleType("pydrake.systems")
    pydrake.systems.analysis = types.ModuleType("pydrake.systems.analysis")
    pydrake.systems.framework = types.ModuleType("pydrake.systems.framework")
    pydrake.all = types.ModuleType("pydrake.all")

    # Builders / objects used at runtime
    def fake_add_msg(builder, dt):
        plant = MagicMock(name="plant")
        plant.time_step.return_value = dt
        plant.num_velocities.return_value = 2
        plant.num_actuators.return_value = 0
        scene = MagicMock(name="scene_graph")
        return (plant, scene)

    pydrake.all.AddMultibodyPlantSceneGraph = fake_add_msg
    pydrake.all.DiagramBuilder = MagicMock(name="DiagramBuilder")
    pydrake.all.JacobianWrtVariable = MagicMock()
    pydrake.all.LoadModelDirectives = MagicMock()
    pydrake.all.MultibodyPlant = MagicMock()
    pydrake.all.Parser = MagicMock(name="Parser")
    pydrake.all.ProcessModelDirectives = MagicMock()
    pydrake.all.RigidTransform = MagicMock()
    pydrake.all.RotationMatrix = MagicMock()
    pydrake.multibody.tree.JointActuatorIndex = MagicMock()
    pydrake.systems.analysis.Simulator = MagicMock(name="Simulator")

    return {
        "pydrake": pydrake,
        "pydrake.math": pydrake.math,
        "pydrake.multibody": pydrake.multibody,
        "pydrake.multibody.parsing": pydrake.multibody.parsing,
        "pydrake.multibody.plant": pydrake.multibody.plant,
        "pydrake.multibody.tree": pydrake.multibody.tree,
        "pydrake.systems": pydrake.systems,
        "pydrake.systems.analysis": pydrake.systems.analysis,
        "pydrake.systems.framework": pydrake.systems.framework,
        "pydrake.all": pydrake.all,
    }


@pytest.fixture
def drake_module(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Inject fake pydrake into sys.modules and reload the wrapper module."""
    stubs = _build_pydrake_stubs()
    for name, mod in stubs.items():
        monkeypatch.setitem(sys.modules, name, mod)

    # Patch availability flag before reload.
    import src.shared.python.engine_core.engine_availability as avail

    monkeypatch.setattr(avail, "DRAKE_AVAILABLE", True)

    from src.engines.physics_engines.drake.python import drake_physics_engine as mod

    mod = importlib.reload(mod)
    yield mod
    # Leave stubs in place; pytest's monkeypatch will revert sys.modules and the
    # availability flag, but we deliberately do not reload again — reloading
    # without pydrake would re-trigger NameError errors during teardown.


class TestConstructionAndState:
    def test_construct_default_timestep(self, drake_module: Any) -> None:
        eng = drake_module.DrakePhysicsEngine()
        assert eng._is_finalized is False
        assert eng.model_name_str == ""

    def test_engine_not_initialized_until_finalised(self, drake_module: Any) -> None:
        eng = drake_module.DrakePhysicsEngine()
        assert eng.is_initialized is False

    def test_model_name_property(self, drake_module: Any) -> None:
        eng = drake_module.DrakePhysicsEngine()
        eng.model_name_str = "TestModel"
        assert eng.model_name == "TestModel"

    def test_get_time_default_zero(self, drake_module: Any) -> None:
        eng = drake_module.DrakePhysicsEngine()
        assert eng.get_time() == 0.0

    def test_get_state_empty_when_no_context(self, drake_module: Any) -> None:
        eng = drake_module.DrakePhysicsEngine()
        q, v = eng.get_state()
        assert q.size == 0 and v.size == 0

    def test_get_joint_names_no_actuators_uses_dofs(self, drake_module: Any) -> None:
        eng = drake_module.DrakePhysicsEngine()
        names = eng.get_joint_names()
        # plant.num_velocities returns 2, num_actuators returns 0
        assert names == ["dof_0", "dof_1"]

    def test_get_full_state_uninit(self, drake_module: Any) -> None:
        eng = drake_module.DrakePhysicsEngine()
        st = eng.get_full_state()
        assert st["q"].size == 0
        assert st["M"] is None


class TestArgumentValidation:
    def test_set_state_uninit_warns_no_raise(self, drake_module: Any) -> None:
        eng = drake_module.DrakePhysicsEngine()
        # plant_context is None, should warn and return.
        eng.set_state(np.zeros(2), np.zeros(2))

    def test_set_control_uninit_warns_no_raise(self, drake_module: Any) -> None:
        eng = drake_module.DrakePhysicsEngine()
        eng.set_control(np.zeros(2))


class TestLoadFromString:
    def test_load_from_string_success(self, drake_module: Any) -> None:
        eng = drake_module.DrakePhysicsEngine()
        parser_mock = MagicMock()
        drake_module.Parser.return_value = parser_mock
        # Avoid finalization complexity by stubbing _ensure_finalized.
        eng._ensure_finalized = MagicMock()
        eng.load_from_string("<urdf/>", extension="urdf")
        assert eng.model_name_str == "StringLoadedModel"
        parser_mock.AddModelsFromString.assert_called_once()

    def test_load_from_string_default_extension(self, drake_module: Any) -> None:
        eng = drake_module.DrakePhysicsEngine()
        parser_mock = MagicMock()
        drake_module.Parser.return_value = parser_mock
        eng._ensure_finalized = MagicMock()
        eng.load_from_string("<urdf/>")
        ext_used = parser_mock.AddModelsFromString.call_args.args[1]
        assert ext_used == "urdf"

    def test_load_from_string_propagates_error(self, drake_module: Any) -> None:
        eng = drake_module.DrakePhysicsEngine()
        parser_mock = MagicMock()
        parser_mock.AddModelsFromString.side_effect = RuntimeError("bad xml")
        drake_module.Parser.return_value = parser_mock
        with pytest.raises(RuntimeError):
            eng.load_from_string("garbage", extension="urdf")


class TestLoadFromPath:
    def test_load_from_path_extracts_model_name(self, drake_module: Any) -> None:
        eng = drake_module.DrakePhysicsEngine()
        parser_mock = MagicMock()
        drake_module.Parser.return_value = parser_mock
        eng._ensure_finalized = MagicMock()
        eng.load_from_path("/tmp/foo/MyModel.urdf")
        assert eng.model_name_str == "MyModel"

    def test_load_from_path_failure_propagates(self, drake_module: Any) -> None:
        eng = drake_module.DrakePhysicsEngine()
        parser_mock = MagicMock()
        parser_mock.AddModels.side_effect = ValueError("nope")
        drake_module.Parser.return_value = parser_mock
        eng._ensure_finalized = MagicMock()
        with pytest.raises(ValueError):
            eng.load_from_path("/tmp/x.urdf")
