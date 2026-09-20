"""Unit contract tests for the stateful Pink backend wrapper."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

pytestmark = pytest.mark.unit

SOLVER_MODULE = "src.engines.physics_engines.pinocchio.python.dtack.ik.pink_solver"
BACKEND_MODULE = (
    "src.engines.physics_engines.pinocchio.python.dtack.backends.pink_backend"
)


def _restore_module(module_name: str, previous: ModuleType | None) -> None:
    current = sys.modules.pop(module_name, None)
    parent = sys.modules.get(module_name.rpartition(".")[0])
    attribute = module_name.rpartition(".")[2]
    if previous is not None:
        sys.modules[module_name] = previous
        if parent is not None:
            setattr(parent, attribute, previous)
    elif parent is not None and getattr(parent, attribute, None) is current:
        delattr(parent, attribute)


@pytest.fixture
def backend_module(monkeypatch: pytest.MonkeyPatch):
    previous_solver = sys.modules.get(SOLVER_MODULE)
    previous_backend = sys.modules.get(BACKEND_MODULE)

    class Configuration:
        def __init__(self, model, data, q, **kwargs):
            self.model = model
            self.data = data
            self.q = np.asarray(q).copy()
            self.kwargs = kwargs

    model = SimpleNamespace(nq=3, nv=2)
    robot = SimpleNamespace(
        model=model,
        data=object(),
        q0=np.zeros(3),
        visual_model=object(),
        collision_model=object(),
    )
    build = MagicMock(return_value=robot)
    fake_pin = ModuleType("pinocchio")
    fake_pin.RobotWrapper = SimpleNamespace(BuildFromURDF=build)
    fake_pin.integrate = MagicMock()
    fake_pin.Model = object
    fake_pin.Data = object
    fake_pin.GeometryModel = object

    fake_pink = ModuleType("pink")
    fake_pink.Configuration = Configuration
    fake_pink.Task = object
    fake_pink.solve_ik = MagicMock()
    fake_pink.tasks = SimpleNamespace(Task=object)

    monkeypatch.setitem(sys.modules, "pinocchio", fake_pin)
    monkeypatch.setitem(sys.modules, "pink", fake_pink)
    sys.modules.pop(SOLVER_MODULE, None)
    sys.modules.pop(BACKEND_MODULE, None)
    importlib.import_module(SOLVER_MODULE)
    module = importlib.import_module(BACKEND_MODULE)
    yield SimpleNamespace(module=module, pin=fake_pin, pink=fake_pink, robot=robot)
    for module_name, previous in (
        (SOLVER_MODULE, previous_solver),
        (BACKEND_MODULE, previous_backend),
    ):
        _restore_module(module_name, previous)


def test_backend_delegates_to_shared_validated_step(
    backend_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = backend_module.module.PINKBackend(Path("robot.urdf"))
    q_init = np.array([1.0, -2.0, 0.5])
    q_next = np.array([1.1, -1.9, 0.5])
    updated_configuration = SimpleNamespace(q=q_next)
    shared_step = MagicMock(
        return_value=SimpleNamespace(q_next=q_next, configuration=updated_configuration)
    )
    monkeypatch.setattr(backend_module.module, "_solve_pink_step", shared_step)
    tasks = {"posture": object(), "frame": object()}
    constraints = [object()]
    limits = [object()]

    result = backend.solve_ik(
        tasks,
        q_init,
        0.02,
        "quadprog",
        damping=4e-6,
        constraints=constraints,
        limits=limits,
    )

    np.testing.assert_array_equal(result, q_next)
    assert backend.configuration is updated_configuration
    call = shared_step.call_args
    assert call.args[:4] == (
        backend.robot.model,
        backend.robot.data,
        q_init,
        list(tasks.values()),
    )
    options = call.args[4]
    assert options.dt == 0.02
    assert options.settings.solver == "quadprog"
    assert options.settings.damping == 4e-6
    assert options.constraints is constraints
    assert options.limits is limits


def test_backend_configuration_receives_collision_geometry(backend_module) -> None:
    backend = backend_module.module.PINKBackend(Path("robot.urdf"))

    assert backend.robot.collision_model is not None
    assert (
        backend.configuration.kwargs["collision_model"] is backend.robot.collision_model
    )


def test_backend_does_not_convert_solver_failure_to_unchanged_pose(
    backend_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = backend_module.module.PINKBackend("robot.urdf")
    failure = RuntimeError("hard constraint infeasible")
    monkeypatch.setattr(
        backend_module.module, "_solve_pink_step", MagicMock(side_effect=failure)
    )

    with pytest.raises(RuntimeError, match="hard constraint infeasible") as caught:
        backend.solve_ik({}, np.zeros(3))

    assert caught.value is failure


def test_missing_dependency_hint_points_to_runtime_guide(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous_solver = sys.modules.get(SOLVER_MODULE)
    previous_backend = sys.modules.get(BACKEND_MODULE)
    try:
        monkeypatch.setitem(sys.modules, "pink", None)
        monkeypatch.setitem(sys.modules, "pinocchio", None)
        sys.modules.pop(SOLVER_MODULE, None)
        sys.modules.pop(BACKEND_MODULE, None)
        module = importlib.import_module(BACKEND_MODULE)

        with pytest.raises(
            ImportError, match="Optional Pink dependencies.*pinocchio.md"
        ):
            module.PINKBackend("robot.urdf")
    finally:
        _restore_module(SOLVER_MODULE, previous_solver)
        _restore_module(BACKEND_MODULE, previous_backend)
