"""Unit tests for native Crocoddyl full-body optimal control assembly and solve (#10338).

Tests:
1. Pure problem assembly contracts (dimension matching, grid monotonicity, input validation).
2. Fail-closed error reporting when native Crocoddyl stack is absent/unhealthy.
3. Toy-plant assembly and mock FDDP solve with convergence diagnostics and receipt extraction.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.engines.physics_engines.pinocchio.python import (
    crocoddyl_problem as prob_subject,
)
from src.engines.physics_engines.pinocchio.python.crocoddyl_problem import (
    CrocoddylProblemBundle,
    CrocoddylProblemConfig,
    build_native_crocoddyl_problem,
)
from src.engines.physics_engines.pinocchio.python.full_body_fit import (
    FullBodyFitOptions,
    FullBodyFitReceipt,
    solve_full_body_fddp,
)
from src.shared.python.optimization.crocoddyl_backend import CrocoddylNotAvailableError

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Mock Crocoddyl Architecture for Unit Testing
# ---------------------------------------------------------------------------


class _MockData:
    def __init__(self, model: Any) -> None:
        self.cost = 0.0


class _MockActionModel:
    def __init__(self, state: Any, nu: int, nr: int = 0) -> None:
        self.state = state
        self.nu = nu
        self.u_lb = np.full(nu, -np.inf)
        self.u_ub = np.full(nu, np.inf)

    def createData(self) -> _MockData:
        return _MockData(self)


class _MockState:
    def __init__(self, nx: int) -> None:
        self.nx = nx
        self.ndx = nx


class _MockProblem:
    def __init__(self, x0: np.ndarray, running: list[Any], terminal: Any) -> None:
        self.x0 = x0
        self.runningModels = running
        self.terminalModel = terminal
        self.runningDatas = [m.createData() for m in running]
        self.terminalData = terminal.createData()
        self.T = len(running)


class _MockSolverFDDP:
    def __init__(self, problem: Any) -> None:
        self.problem = problem
        self.cost = 12.34
        self.iter = 5
        self.stoppingCriteria = 1e-7
        self.stepLength = 1.0
        self.th_stop = 1e-6
        self.th_gap_tol = 1e-6
        nx = problem.x0.size
        nu = problem.runningModels[0].nu if problem.runningModels else 0
        self.xs = [np.zeros(nx) for _ in range(problem.T + 1)]
        self.us = [np.zeros(nu) for _ in range(problem.T)]

    def solve(
        self, xs: list, us: list, max_iter: int, is_feasible: bool, init_reg: float
    ) -> bool:
        return True


_FAKE_CROCODDYL = SimpleNamespace(
    StateVector=_MockState,
    ActionModelAbstract=_MockActionModel,
    ShootingProblem=_MockProblem,
    SolverFDDP=_MockSolverFDDP,
)


def _make_dummy_spec(nq: int = 41) -> dict[str, Any]:
    return {
        "coordinate_order": [f"coord_{i}" for i in range(nq)],
        "schema_version": "full-body-v1",
    }


# ---------------------------------------------------------------------------
# Unit Test Cases
# ---------------------------------------------------------------------------


def test_problem_assembly_contract_validation() -> None:
    """Invalid coordinate order, monotonic grid violations, and shape mismatches must fail closed."""
    spec = _make_dummy_spec(4)
    time_grid = np.array([0.0, 0.05, 0.1])
    targets = np.zeros((3, 2, 3))
    valid = np.ones((3, 2), dtype=bool)
    labels = ("MarkerA", "MarkerB")

    # Empty coordinate order
    with pytest.raises(ValueError, match="coordinate_order"):
        build_native_crocoddyl_problem({}, time_grid, targets, valid, labels)

    # Decreasing time grid
    with pytest.raises(ValueError, match="strictly increasing"):
        build_native_crocoddyl_problem(
            spec, [0.1, 0.05], targets[:2], valid[:2], labels
        )

    # Mismatched target shape
    with pytest.raises(ValueError, match="marker_targets"):
        build_native_crocoddyl_problem(
            spec, time_grid, np.zeros((2, 2, 3)), valid, labels
        )


def test_fail_closed_when_crocoddyl_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CrocoddylNotAvailableError must be raised when backend is not importable."""
    monkeypatch.setattr(
        prob_subject,
        "require_crocoddyl",
        lambda: (_ for _ in ()).throw(CrocoddylNotAvailableError("crocoddyl missing")),
    )
    prob_subject.qualified_crocoddyl.cache_clear()

    spec = _make_dummy_spec(4)
    time_grid = np.array([0.0, 0.1])
    targets = np.zeros((2, 1, 3))
    valid = np.ones((2, 1), dtype=bool)

    with pytest.raises(CrocoddylNotAvailableError, match="crocoddyl missing"):
        build_native_crocoddyl_problem(spec, time_grid, targets, valid, ("M1",))


def test_toy_assembly_and_mock_fddp_solve(monkeypatch: pytest.MonkeyPatch) -> None:
    """Assembles shooting problem with mock crocoddyl and runs solve_full_body_fddp."""
    monkeypatch.setattr(prob_subject, "qualified_crocoddyl", lambda: _FAKE_CROCODDYL)
    from src.engines.physics_engines.pinocchio.python import (
        full_body_fit as fit_subject,
    )

    monkeypatch.setattr(fit_subject, "qualified_crocoddyl", lambda: _FAKE_CROCODDYL)

    spec = _make_dummy_spec(4)
    time_grid = np.array([0.0, 0.05, 0.1])
    targets = np.zeros((3, 2, 3))
    valid = np.ones((3, 2), dtype=bool)
    labels = ("MarkerA", "MarkerB")

    bundle = build_native_crocoddyl_problem(
        spec,
        time_grid,
        targets,
        valid,
        labels,
        config=CrocoddylProblemConfig(dt_s=0.05),
    )

    assert isinstance(bundle, CrocoddylProblemBundle)
    assert bundle.nq == 4
    assert bundle.nv == 4
    assert bundle.nu == 4
    assert len(bundle.problem.runningModels) == 2

    # Execute FDDP solver
    opts = FullBodyFitOptions(max_iterations=10)
    receipt = solve_full_body_fddp(
        bundle,
        options=opts,
        target_markers=targets,
        valid_mask=valid,
    )

    assert isinstance(receipt, FullBodyFitReceipt)
    assert receipt.converged is True
    assert receipt.status == "terminal"
    assert receipt.iterations == 5
    assert receipt.final_cost == pytest.approx(12.34)
    assert receipt.xs.shape == (3, 8)
    assert receipt.us.shape == (2, 4)
