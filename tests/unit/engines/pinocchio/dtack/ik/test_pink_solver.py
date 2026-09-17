"""Unit contract tests for the Pink differential-IK adapter."""

from __future__ import annotations

import importlib
import inspect
import builtins
import sys
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

pytestmark = pytest.mark.unit

MODULE_NAME = "src.engines.physics_engines.pinocchio.python.dtack.ik.pink_solver"


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
def pink_module(monkeypatch: pytest.MonkeyPatch):
    previous_module = sys.modules.get(MODULE_NAME)
    state: dict[str, Any] = {
        "velocity": np.array([0.4, -0.2]),
        "integrated": np.array([0.14, 0.18, 0.3]),
    }
    configurations: list[Any] = []

    class Configuration:
        def __init__(self, model, data, q, **kwargs):
            self.model = model
            self.data = data
            self.q = np.asarray(q).copy()
            self.kwargs = kwargs
            self.updates: list[np.ndarray] = []
            configurations.append(self)

        def update(self, q=None):
            if q is not None:
                self.q = np.asarray(q).copy()
                self.updates.append(self.q.copy())

    def solve_ik(
        configuration,
        tasks,
        dt,
        solver,
        damping=1e-12,
        limits=None,
        constraints=None,
    ):
        if "error" in state:
            raise state["error"]
        return state["velocity"]

    solve_mock = MagicMock(side_effect=solve_ik)
    solve_mock.__signature__ = inspect.signature(solve_ik)
    fake_pink = ModuleType("pink")
    fake_pink.Configuration = Configuration
    fake_pink.Task = object
    fake_pink.solve_ik = solve_mock

    integrate = MagicMock(side_effect=lambda model, q, delta: state["integrated"])
    fake_pin = ModuleType("pinocchio")
    fake_pin.integrate = integrate
    fake_pin.Model = object
    fake_pin.Data = object
    fake_pin.GeometryModel = object

    monkeypatch.setitem(sys.modules, "pink", fake_pink)
    monkeypatch.setitem(sys.modules, "pinocchio", fake_pin)
    sys.modules.pop(MODULE_NAME, None)
    module = importlib.import_module(MODULE_NAME)
    yield SimpleNamespace(
        module=module,
        pink=fake_pink,
        pin=fake_pin,
        state=state,
        configurations=configurations,
    )
    _restore_module(MODULE_NAME, previous_module)


def _solver(harness):
    model = SimpleNamespace(nq=3, nv=2)
    solver = harness.module.PinkSolver(model, object(), object(), object())
    return solver, model


def test_shared_solve_path_stays_within_parameter_budget(pink_module) -> None:
    parameters = inspect.signature(pink_module.module._solve_pink_step).parameters

    assert len(parameters) <= 8


def test_solve_validates_and_integrates_one_velocity_step(pink_module) -> None:
    solver, model = _solver(pink_module)
    q = np.array([0.1, 0.2, 0.3])
    constraints = [object()]
    limits = [object()]

    result = solver.solve(
        q,
        [object()],
        0.1,
        pink_module.module.SolverSettings("quadprog", 2e-6),
        constraints=constraints,
        limits=limits,
    )

    np.testing.assert_array_equal(result, pink_module.state["integrated"])
    pink_module.pin.integrate.assert_called_once()
    integrate_args = pink_module.pin.integrate.call_args.args
    assert integrate_args[0] is model
    np.testing.assert_array_equal(integrate_args[1], q)
    np.testing.assert_array_equal(
        integrate_args[2], pink_module.state["velocity"] * 0.1
    )
    solve_kwargs = pink_module.pink.solve_ik.call_args.kwargs
    assert solve_kwargs == {
        "solver": "quadprog",
        "damping": 2e-6,
        "constraints": constraints,
        "limits": limits,
    }
    assert solver.configuration is pink_module.configurations[0]
    assert solver.configuration.kwargs["collision_model"] is solver.collision_model
    assert solver.visual_model is not solver.collision_model
    np.testing.assert_array_equal(solver.configuration.q, result)
    np.testing.assert_array_equal(solver.configuration.updates[-1], result)


def test_none_constraints_and_limits_retain_pink_defaults(pink_module) -> None:
    solver, _ = _solver(pink_module)
    solver.solve(np.array([0.1, 0.2, 0.3]), [], 0.1)

    kwargs = pink_module.pink.solve_ik.call_args.kwargs
    assert "constraints" not in kwargs
    assert "limits" not in kwargs


def test_consecutive_solves_refresh_cached_configuration(pink_module) -> None:
    solver, _ = _solver(pink_module)
    first_q = np.array([0.1, 0.2, 0.3])
    second_q = np.array([-2.0, 1.5, 0.7])
    solver.solve(first_q, [], 0.1)
    pink_module.state["integrated"] = np.array([-1.96, 1.48, 0.7])

    result = solver.solve(second_q, [], 0.1)

    assert len(pink_module.configurations) == 1
    configuration = pink_module.configurations[0]
    np.testing.assert_array_equal(configuration.updates[-2], second_q)
    np.testing.assert_array_equal(configuration.updates[-1], result)


def test_failed_solve_cache_tracks_attempt_before_distant_retry(pink_module) -> None:
    solver, _ = _solver(pink_module)
    failed_q = np.array([2.0, -1.5, 0.7])
    failure = RuntimeError("infeasible QP")
    pink_module.state["error"] = failure

    with pytest.raises(RuntimeError, match="infeasible QP"):
        solver.solve(failed_q, [], 0.1)

    np.testing.assert_array_equal(solver.configuration.q, failed_q)
    del pink_module.state["error"]
    retry_q = np.array([-4.0, 3.0, -0.5])
    pink_module.state["integrated"] = np.array([-3.96, 2.98, -0.5])

    result = solver.solve(retry_q, [], 0.1)

    configuration = pink_module.configurations[0]
    np.testing.assert_array_equal(configuration.updates[-2], retry_q)
    np.testing.assert_array_equal(configuration.updates[-1], result)


@pytest.mark.parametrize(
    ("q", "dt", "settings", "match"),
    [
        (np.zeros(2), 0.1, None, "q_init must have shape \\(3,\\)"),
        (np.array([0.0, np.nan, 0.0]), 0.1, None, "q_init must be finite"),
        (np.zeros(3), 0.0, None, "dt must be positive and finite"),
        (np.zeros(3), np.nan, None, "dt must be positive and finite"),
        (np.zeros(3), 0.1, ("", 1e-6), "solver must be a nonempty string"),
        (np.zeros(3), 0.1, ("quadprog", -1.0), "damping must be nonnegative"),
        (np.zeros(3), 0.1, ("quadprog", np.inf), "damping must be nonnegative"),
    ],
)
def test_invalid_inputs_are_rejected(pink_module, q, dt, settings, match) -> None:
    solver, _ = _solver(pink_module)
    solver_settings = (
        None
        if settings is None
        else pink_module.module.SolverSettings(settings[0], settings[1])
    )

    with pytest.raises(ValueError, match=match):
        solver.solve(q, [], dt, solver_settings)

    pink_module.pink.solve_ik.assert_not_called()
    pink_module.pin.integrate.assert_not_called()


@pytest.mark.parametrize("dimension", [True, 3.0, np.nan, np.inf])
@pytest.mark.parametrize("attribute", ["nq", "nv"])
def test_model_dimensions_must_be_exact_finite_integers(
    pink_module, attribute, dimension
) -> None:
    model = SimpleNamespace(nq=3, nv=2)
    setattr(model, attribute, dimension)
    solver = pink_module.module.PinkSolver(model, object(), object(), object())

    with pytest.raises(
        ValueError, match=f"model.{attribute} must be a nonnegative integer"
    ):
        solver.solve(np.zeros(3), [], 0.1)

    pink_module.pink.solve_ik.assert_not_called()
    pink_module.pin.integrate.assert_not_called()


@pytest.mark.parametrize(
    ("velocity", "match"),
    [
        (np.zeros(3), "velocity must have shape \\(2,\\)"),
        (np.array([0.0, np.nan]), "velocity must be finite"),
    ],
)
def test_invalid_solver_velocity_is_rejected(pink_module, velocity, match) -> None:
    solver, _ = _solver(pink_module)
    pink_module.state["velocity"] = velocity

    with pytest.raises(ValueError, match=match):
        solver.solve(np.zeros(3), [], 0.1)

    pink_module.pin.integrate.assert_not_called()


@pytest.mark.parametrize(
    ("integrated", "match"),
    [
        (np.zeros(2), "q_next must have shape \\(3,\\)"),
        (np.array([0.0, np.inf, 0.0]), "q_next must be finite"),
    ],
)
def test_invalid_integrated_result_is_rejected(pink_module, integrated, match) -> None:
    solver, _ = _solver(pink_module)
    pink_module.state["integrated"] = integrated

    with pytest.raises(ValueError, match=match):
        solver.solve(np.zeros(3), [], 0.1)


def test_solver_failure_is_propagated_with_context(pink_module) -> None:
    solver, _ = _solver(pink_module)
    failure = RuntimeError("infeasible QP")
    pink_module.state["error"] = failure

    with pytest.raises(RuntimeError, match="infeasible QP") as caught:
        solver.solve(np.zeros(3), [], 0.1)

    assert caught.value is failure
    assert any("Pink IK solve failed" in note for note in caught.value.__notes__)


def test_constraints_fail_explicitly_when_pink_api_cannot_accept_them(
    pink_module,
) -> None:
    solver, _ = _solver(pink_module)

    def old_solve_ik(configuration, tasks, dt, solver, damping=1e-12):
        return np.zeros(2)

    pink_module.pink.solve_ik = old_solve_ik

    with pytest.raises(RuntimeError, match="does not support constraints"):
        solver.solve(np.zeros(3), [], 0.1, constraints=[object()])


def test_missing_dependency_hint_points_to_runtime_guide(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous_module = sys.modules.get(MODULE_NAME)
    try:
        monkeypatch.setitem(sys.modules, "pink", None)
        monkeypatch.setitem(sys.modules, "pinocchio", None)
        sys.modules.pop(MODULE_NAME, None)
        module = importlib.import_module(MODULE_NAME)

        with pytest.raises(
            ImportError, match="Optional Pink dependencies.*pinocchio.md"
        ):
            module.PinkSolver(object(), object(), object(), object())
    finally:
        _restore_module(MODULE_NAME, previous_module)


def test_native_loader_oserror_is_treated_as_optional_dependency_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous_module = sys.modules.get(MODULE_NAME)
    original_import = builtins.__import__

    def fail_pink_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "pink":
            raise OSError("native Pink library failed to load")
        return original_import(name, globals, locals, fromlist, level)

    try:
        monkeypatch.setattr(builtins, "__import__", fail_pink_import)
        sys.modules.pop(MODULE_NAME, None)
        module = importlib.import_module(MODULE_NAME)

        with pytest.raises(ImportError, match="Optional Pink dependencies"):
            module.PinkSolver(object(), object(), object(), object())
    finally:
        _restore_module(MODULE_NAME, previous_module)
