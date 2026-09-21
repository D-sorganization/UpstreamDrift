"""Contract tests for global-coefficient Crocoddyl action assembly."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.shared.python.motion_matching import crocoddyl_polynomial as subject
from src.shared.python.motion_matching.crocoddyl_polynomial import (
    CrocoddylSmoothnessError,
    PolynomialCostQuadratic,
    build_polynomial_shooting_problem,
)

pytestmark = pytest.mark.unit


class _Data:
    def __init__(self, model: Any) -> None:
        nx, nu = model.state.nx, model.nu
        self.xnext = np.zeros(nx)
        self.cost = 0.0
        self.Fx = np.zeros((nx, nx))
        self.Fu = np.zeros((nx, nu))
        self.Lx = np.zeros(nx)
        self.Lu = np.zeros(nu)
        self.Lxx = np.zeros((nx, nx))
        self.Lxu = np.zeros((nx, nu))
        self.Luu = np.zeros((nu, nu))


class _Action:
    def __init__(self, state: Any, nu: int, nr: int = 0) -> None:
        del nr
        self.state = state
        self.nu = nu
        self.u_lb = np.full(nu, -np.inf)
        self.u_ub = np.full(nu, np.inf)

    def createData(self) -> _Data:
        return _Data(self)


class _State:
    def __init__(self, nx: int) -> None:
        self.nx = nx
        self.ndx = nx


class _Problem:
    def __init__(self, x0: np.ndarray, running: list[Any], terminal: Any) -> None:
        self.x0 = x0
        self.runningModels = running
        self.terminalModel = terminal
        self.runningDatas = [model.createData() for model in running]
        self.terminalData = terminal.createData()
        self.T = len(running)

    def calc(self, xs: list[np.ndarray], us: list[np.ndarray]) -> float:
        for model, data, x, u in zip(
            self.runningModels, self.runningDatas, xs[:-1], us, strict=True
        ):
            model.calc(data, x, u)
        self.terminalModel.calc(self.terminalData, xs[-1])
        return sum(data.cost for data in self.runningDatas) + self.terminalData.cost

    def calcDiff(self, xs: list[np.ndarray], us: list[np.ndarray]) -> float:
        for model, data, x, u in zip(
            self.runningModels, self.runningDatas, xs[:-1], us, strict=True
        ):
            model.calcDiff(data, x, u)
        self.terminalModel.calcDiff(self.terminalData, xs[-1])
        return self.calc(xs, us)


_FAKE_CROCODDYL = SimpleNamespace(
    StateVector=_State,
    ActionModelAbstract=_Action,
    ShootingProblem=_Problem,
)


class _Stepper:
    def __init__(self, *, differentiable: bool = True) -> None:
        self.control = SimpleNamespace(
            n_coordinates=1,
            n_parameters=2,
            duration_s=1.0,
        )
        self.differentiable = differentiable
        self.linearize_calls = 0

    @staticmethod
    def _blocks(dt_s: float) -> tuple[np.ndarray, np.ndarray]:
        state = np.array([[1.0, dt_s], [0.0, 1.0]])
        coefficients = np.array([[0.5 * dt_s**2, 0.25 * dt_s**2], [dt_s, 0.5 * dt_s]])
        return state, coefficients

    def step(
        self,
        state: np.ndarray,
        parameters: np.ndarray,
        *,
        time_s: float,
        dt_s: float,
    ) -> np.ndarray:
        del time_s
        state_block, coefficient_block = self._blocks(dt_s)
        return state_block @ state + coefficient_block @ parameters

    def linearize(
        self,
        state: np.ndarray,
        parameters: np.ndarray,
        *,
        time_s: float,
        dt_s: float,
    ) -> Any:
        self.linearize_calls += 1
        state_block, coefficient_block = self._blocks(dt_s)
        return SimpleNamespace(
            next_state=self.step(state, parameters, time_s=time_s, dt_s=dt_s),
            dnext_dstate=state_block,
            dnext_dcoefficients=coefficient_block,
            differentiable=self.differentiable,
        )


class _QuadraticCost:
    def __init__(self, hessian: np.ndarray, gradient: np.ndarray) -> None:
        self.hessian = np.asarray(hessian, dtype=float)
        self.linear = np.asarray(gradient, dtype=float)

    def value(self, state: np.ndarray, parameters: np.ndarray, time_s: float) -> float:
        vector = np.r_[state, parameters]
        return float(
            0.5 * vector @ self.hessian @ vector + self.linear @ vector + time_s
        )

    def quadraticize(
        self, state: np.ndarray, parameters: np.ndarray, time_s: float
    ) -> PolynomialCostQuadratic:
        vector = np.r_[state, parameters]
        return PolynomialCostQuadratic(
            self.value(state, parameters, time_s),
            self.hessian @ vector + self.linear,
            self.hessian.copy(),
        )


@pytest.fixture(autouse=True)
def fake_crocoddyl(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subject, "_qualified_crocoddyl", lambda: _FAKE_CROCODDYL)


def _bundle(*, differentiable: bool = True, bounds=None):
    hessian = np.array(
        [
            [2.0, 0.2, 0.4, 0.0],
            [0.2, 3.0, 0.0, -0.3],
            [0.4, 0.0, 4.0, 0.1],
            [0.0, -0.3, 0.1, 5.0],
        ]
    )
    running = _QuadraticCost(hessian, np.array([0.1, -0.2, 0.3, -0.4]))
    terminal = _QuadraticCost(2.0 * hessian, np.array([-0.2, 0.1, 0.0, 0.2]))
    stepper = _Stepper(differentiable=differentiable)
    bundle = build_polynomial_shooting_problem(
        stepper,
        np.array([0.0, 0.2, 0.5]),
        np.array([0.3, -0.1]),
        running,
        terminal,
        coefficient_bounds=bounds,
    )
    return bundle, stepper, running, terminal


def test_assembly_has_one_lift_then_autonomous_flows() -> None:
    bundle, _, _, _ = _bundle(bounds=(np.full(2, -1.0), np.full(2, 2.0)))

    assert bundle.problem.x0.tolist() == [0.3, -0.1, 0.0, 0.0]
    assert [model.nu for model in bundle.problem.runningModels] == [2, 0, 0]
    assert bundle.problem.terminalModel.nu == 0
    np.testing.assert_array_equal(bundle.problem.runningModels[0].u_lb, [-1.0, -1.0])
    np.testing.assert_array_equal(bundle.problem.runningModels[0].u_ub, [2.0, 2.0])


def test_lift_exact_blocks_and_reused_data_are_cleared() -> None:
    bundle, _, _, _ = _bundle()
    model = bundle.problem.runningModels[0]
    data = model.createData()
    for name in ("Fx", "Fu", "Lx", "Lu", "Lxx", "Lxu", "Luu"):
        getattr(data, name).fill(7.0)
    state = np.array([0.4, -0.2, 8.0, 9.0])
    coefficients = np.array([0.6, -0.7])

    model.calc(data, state, coefficients)
    model.calcDiff(data, state, coefficients)

    np.testing.assert_array_equal(data.xnext, [0.4, -0.2, 0.6, -0.7])
    np.testing.assert_array_equal(data.Fx, np.diag([1.0, 1.0, 0.0, 0.0]))
    np.testing.assert_array_equal(data.Fu, np.vstack((np.zeros((2, 2)), np.eye(2))))
    assert data.cost == 0.0
    for name in ("Lx", "Lu", "Lxx", "Lxu", "Luu"):
        assert not np.any(getattr(data, name))


def test_flow_augmented_jacobian_and_running_rate_scaling() -> None:
    bundle, stepper, running, _ = _bundle()
    model = bundle.problem.runningModels[1]
    data = model.createData()
    state = np.array([0.2, -0.4, 0.6, -0.8])

    model.calc(data, state, np.zeros(0))
    model.calcDiff(data, state, np.zeros(0))

    direct = stepper.linearize(state[:2], state[2:], time_s=0.0, dt_s=0.2)
    expected_fx = np.block(
        [
            [direct.dnext_dstate, direct.dnext_dcoefficients],
            [np.zeros((2, 2)), np.eye(2)],
        ]
    )
    np.testing.assert_allclose(data.Fx, expected_fx, rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(data.xnext, np.r_[direct.next_state, state[2:]])
    quadratic = running.quadraticize(state[:2], state[2:], 0.0)
    assert data.cost == pytest.approx(0.2 * quadratic.value)
    np.testing.assert_allclose(data.Lx, 0.2 * quadratic.gradient)
    np.testing.assert_allclose(data.Lxx, 0.2 * quadratic.hessian)

    direction = np.array([0.2, -0.4, 0.1, 0.7])
    epsilon = 1e-7
    plus = model.createData()
    minus = model.createData()
    model.calc(plus, state + epsilon * direction, np.zeros(0))
    model.calc(minus, state - epsilon * direction, np.zeros(0))
    measured = (plus.xnext - minus.xnext) / (2.0 * epsilon)
    np.testing.assert_allclose(data.Fx @ direction, measured, rtol=2e-9, atol=2e-9)


def test_terminal_x_only_dispatch_is_unscaled_and_calls_no_dynamics() -> None:
    bundle, stepper, _, terminal = _bundle()
    warm = bundle.warm_start(np.array([0.2, -0.3]))

    bundle.problem.calc(warm.xs, warm.us)
    bundle.problem.calcDiff(warm.xs, warm.us)

    terminal_state = warm.xs[-1]
    quadratic = terminal.quadraticize(terminal_state[:2], terminal_state[2:], 0.5)
    data = bundle.problem.terminalData
    assert data.cost == pytest.approx(quadratic.value)
    np.testing.assert_allclose(data.Lx, quadratic.gradient)
    np.testing.assert_allclose(data.Lxx, quadratic.hessian)
    assert stepper.linearize_calls == 2


def test_warm_start_and_diagnostics_use_independent_replay() -> None:
    bundle, _, running, terminal = _bundle(bounds=(np.full(2, -0.5), np.full(2, 0.5)))
    coefficients = np.array([0.2, -0.3])
    warm = bundle.warm_start(coefficients)
    diagnostic = bundle.diagnose(warm.xs, warm.us)

    assert [control.shape for control in warm.us] == [(2,), (0,), (0,)]
    assert len(warm.xs) == 4
    assert diagnostic.max_physical_defect == 0.0
    assert diagnostic.max_coefficient_defect == 0.0
    assert diagnostic.max_coefficient_node_mismatch == 0.0
    assert diagnostic.max_bound_violation == 0.0
    assert diagnostic.max_replay_gap == 0.0
    expected_cost = sum(
        (bundle.time_grid[index + 1] - time) * running.value(state, coefficients, time)
        for index, (time, state) in enumerate(
            zip(bundle.time_grid[:-1], diagnostic.replay_states[:-1], strict=True)
        )
    ) + terminal.value(diagnostic.replay_states[-1], coefficients, 0.5)
    assert diagnostic.replay_cost == pytest.approx(expected_cost)
    assert diagnostic.node_cost == pytest.approx(expected_cost)
    assert diagnostic.cost_gap == pytest.approx(0.0, abs=1e-15)
    np.testing.assert_array_equal(diagnostic.initial_physical_defect, np.zeros(2))

    corrupted = [state.copy() for state in warm.xs]
    corrupted[0][0] -= 0.1
    corrupted[1][0] += 0.25
    corrupted[2][2] += 0.4
    changed = bundle.diagnose(corrupted, warm.us)
    assert changed.max_physical_defect >= 0.25
    assert changed.max_coefficient_defect >= 0.4
    assert changed.max_coefficient_node_mismatch >= 0.4
    assert changed.max_replay_gap >= 0.25
    assert changed.replay_cost == pytest.approx(diagnostic.replay_cost)
    np.testing.assert_array_equal(changed.initial_physical_defect, [-0.1, 0.0])
    assert changed.node_cost != pytest.approx(changed.replay_cost)
    assert changed.cost_gap == pytest.approx(changed.node_cost - changed.replay_cost)


def test_false_contact_branch_rejects_smooth_linearization() -> None:
    bundle, _, _, _ = _bundle(differentiable=False)
    model = bundle.problem.runningModels[1]
    with pytest.raises(CrocoddylSmoothnessError, match="differentiable"):
        model.calcDiff(model.createData(), np.zeros(4), np.zeros(0))


@pytest.mark.parametrize(
    ("time_grid", "initial_state", "bounds", "match"),
    [
        ([0.0], [0.0, 0.0], None, "time_grid"),
        ([0.0, 0.0], [0.0, 0.0], None, "strictly"),
        ([-0.1, 0.1], [0.0, 0.0], None, "nonnegative"),
        ([0.0, 1.1], [0.0, 0.0], None, "horizon"),
        ([0.0, 0.1], [0.0], None, "initial_state"),
        ([0.0, 0.1], [0.0, np.nan], None, "finite"),
        ([0.0, 0.1], [0.0, 0.0], ([-1.0], [1.0]), "shape"),
        ([0.0, 0.1], [0.0, 0.0], ([1.0, -1.0], [0.0, 1.0]), "ordered"),
    ],
)
def test_build_contract_validation(time_grid, initial_state, bounds, match) -> None:
    stepper = _Stepper()
    cost = _QuadraticCost(np.eye(4), np.zeros(4))
    with pytest.raises(ValueError, match=match):
        build_polynomial_shooting_problem(
            stepper,
            time_grid,
            initial_state,
            cost,
            cost,
            coefficient_bounds=bounds,
        )


@pytest.mark.parametrize("kind", ["shape", "finite", "symmetric"])
def test_quadratic_contract_is_checked_at_use(kind: str) -> None:
    bundle, _, running, _ = _bundle()
    model = bundle.problem.runningModels[1]
    original = running.quadraticize

    def invalid(*args):
        result = original(*args)
        if kind == "shape":
            return PolynomialCostQuadratic(
                result.value, result.gradient[:-1], result.hessian
            )
        if kind == "finite":
            gradient = result.gradient.copy()
            gradient[0] = np.nan
            return PolynomialCostQuadratic(result.value, gradient, result.hessian)
        hessian = result.hessian.copy()
        hessian[0, 1] += 1e-5
        return PolynomialCostQuadratic(result.value, result.gradient, hessian)

    running.quadraticize = invalid
    with pytest.raises(ValueError, match=kind):
        model.calcDiff(model.createData(), np.zeros(4), np.zeros(0))


def test_warm_start_and_diagnostic_shapes_are_exact() -> None:
    bundle, _, _, _ = _bundle()
    with pytest.raises(ValueError, match="parameters"):
        bundle.warm_start(np.zeros(3))
    warm = bundle.warm_start(np.zeros(2))
    with pytest.raises(ValueError, match="state trajectory"):
        bundle.diagnose(warm.xs[:-1], warm.us)
    with pytest.raises(ValueError, match="control trajectory"):
        bundle.diagnose(warm.xs, warm.us[:-1])


def test_diagnostic_rejects_nonfinite_accumulated_node_cost() -> None:
    class CoefficientCost:
        @staticmethod
        def value(state, parameters, time_s):
            del state, time_s
            return float(abs(parameters[0]))

        @staticmethod
        def quadraticize(state, parameters, time_s):
            del state, parameters, time_s
            return PolynomialCostQuadratic(0.0, np.zeros(4), np.zeros((4, 4)))

    stepper = _Stepper()
    cost = CoefficientCost()
    bundle = build_polynomial_shooting_problem(
        stepper, [0.0, 0.2, 0.5], [0.0, 0.0], cost, cost
    )
    warm = bundle.warm_start(np.zeros(2))
    defectful = [state.copy() for state in warm.xs]
    for state in defectful[1:]:
        state[2] = 1.3e308

    with pytest.raises(ValueError, match="shooting node cost"):
        bundle.diagnose(defectful, warm.us)
