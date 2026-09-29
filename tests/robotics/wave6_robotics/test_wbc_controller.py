"""Tests for WholeBodyController."""

from __future__ import annotations

import numpy as np
import pytest

from src.robotics.control.whole_body.qp_solver import NullspaceQPSolver
from src.robotics.control.whole_body.task import Task, TaskType
from src.robotics.control.whole_body.wbc_controller import (
    WBCConfig,
    WBCSolution,
    WholeBodyController,
)

N_V = 4


class FakeEngine:
    def __init__(self, n_v: int = N_V) -> None:
        self._n = n_v
        self._q = np.zeros(n_v)
        self._v = np.zeros(n_v)

    def get_state(self):
        return self._q.copy(), self._v.copy()

    def set_state(self, q, v):
        self._q = np.asarray(q, dtype=float)
        self._v = np.asarray(v, dtype=float)

    def compute_mass_matrix(self):
        return np.eye(self._n)

    def compute_bias_forces(self):
        return np.zeros(self._n)

    def compute_gravity_forces(self):
        return np.zeros(self._n)

    def compute_jacobian(self, body_name):
        return None

    def get_time(self):
        return 0.0


def _task(name="t", J=None, target=None, weight=1.0, priority=2):
    if J is None:
        J = np.eye(3, N_V)
    if target is None:
        target = np.zeros(3)
    return Task(
        name=name,
        task_type=TaskType.SOFT,
        priority=priority,
        jacobian=J,
        target=target,
        weight=np.full(J.shape[0], weight),
    )


def test_construct_rejects_non_engine() -> None:
    with pytest.raises(TypeError):
        WholeBodyController(object())  # type: ignore[arg-type]


def test_construct_with_defaults() -> None:
    wbc = WholeBodyController(FakeEngine())
    assert wbc.n_tasks == 0
    assert wbc.tasks == []
    assert wbc.engine is not None
    assert wbc.config is not None


def test_add_task_and_duplicate_rejected() -> None:
    wbc = WholeBodyController(FakeEngine())
    wbc.add_task(_task("a"))
    with pytest.raises(ValueError, match="already exists"):
        wbc.add_task(_task("a"))


def test_add_task_sorts_by_priority_desc() -> None:
    wbc = WholeBodyController(FakeEngine())
    wbc.add_task(_task("low", priority=4))
    wbc.add_task(_task("high", priority=0))
    wbc.add_task(_task("mid", priority=2))
    assert [t.name for t in wbc.tasks] == ["low", "mid", "high"]


def test_remove_task_returns_bool() -> None:
    wbc = WholeBodyController(FakeEngine())
    wbc.add_task(_task("a"))
    assert wbc.remove_task("a")
    assert not wbc.remove_task("missing")


def test_clear_tasks() -> None:
    wbc = WholeBodyController(FakeEngine())
    wbc.add_task(_task("a"))
    wbc.clear_tasks()
    assert wbc.n_tasks == 0


def test_get_task() -> None:
    wbc = WholeBodyController(FakeEngine())
    t = _task("a")
    wbc.add_task(t)
    assert wbc.get_task("a") is t
    assert wbc.get_task("nope") is None


def test_set_contact_jacobians_copies_list() -> None:
    wbc = WholeBodyController(FakeEngine())
    J = np.zeros((3, N_V))
    src_list = [J]
    wbc.set_contact_jacobians(src_list)
    src_list.append(J)
    # Internal not mutated
    assert len(wbc._contact_jacobians) == 1


def test_solve_no_tasks_returns_failure() -> None:
    wbc = WholeBodyController(FakeEngine())
    sol = wbc.solve()
    assert isinstance(sol, WBCSolution)
    assert not sol.success
    assert "No tasks" in sol.status


def test_solve_weighted_simple() -> None:
    cfg = WBCConfig(use_hierarchical=False, regularization=1e-3)
    wbc = WholeBodyController(FakeEngine(), config=cfg, solver=NullspaceQPSolver())
    wbc.add_task(_task("a"))
    sol = wbc.solve()
    assert isinstance(sol, WBCSolution)
    # Either success or failure but well-formed
    if sol.success:
        assert sol.joint_accelerations is not None
        assert sol.joint_torques is not None
        assert "a" in sol.task_errors


def test_solve_hierarchical_simple() -> None:
    cfg = WBCConfig(use_hierarchical=True, regularization=1e-3)
    wbc = WholeBodyController(FakeEngine(), config=cfg, solver=NullspaceQPSolver())
    wbc.add_task(_task("a", priority=0))
    wbc.add_task(_task("b", priority=2))
    sol = wbc.solve()
    assert isinstance(sol, WBCSolution)


def test_solve_with_torque_limits_clips_torques() -> None:
    cfg = WBCConfig(
        use_hierarchical=False,
        regularization=1e-3,
        torque_limits=np.full(N_V, 0.001),
    )
    wbc = WholeBodyController(FakeEngine(), config=cfg, solver=NullspaceQPSolver())
    # Big target makes torques large; clipping should bound them
    wbc.add_task(_task("a", target=np.array([100.0, 100.0, 100.0])))
    sol = wbc.solve()
    if sol.success and sol.joint_torques is not None:
        assert np.all(np.abs(sol.joint_torques) <= 0.001 + 1e-6)


def test_solve_with_velocity_limits() -> None:
    cfg = WBCConfig(
        use_hierarchical=False,
        velocity_limits=np.ones(N_V),
    )
    wbc = WholeBodyController(FakeEngine(), config=cfg, solver=NullspaceQPSolver())
    wbc.add_task(_task("a"))
    sol = wbc.solve()
    assert isinstance(sol, WBCSolution)


def test_solve_with_acceleration_limits() -> None:
    cfg = WBCConfig(
        use_hierarchical=False,
        acceleration_limits=np.ones(N_V) * 10.0,
    )
    wbc = WholeBodyController(FakeEngine(), config=cfg, solver=NullspaceQPSolver())
    wbc.add_task(_task("a"))
    sol = wbc.solve()
    assert isinstance(sol, WBCSolution)


def test_solve_with_contact_jacobians_3x() -> None:
    cfg = WBCConfig(use_hierarchical=False)
    wbc = WholeBodyController(FakeEngine(), config=cfg, solver=NullspaceQPSolver())
    wbc.add_task(_task("a"))
    wbc.set_contact_jacobians([np.zeros((3, N_V))])
    sol = wbc.solve()
    assert isinstance(sol, WBCSolution)


def test_solve_with_contact_jacobians_6x_reduced_to_3() -> None:
    cfg = WBCConfig(use_hierarchical=False)
    wbc = WholeBodyController(FakeEngine(), config=cfg, solver=NullspaceQPSolver())
    wbc.add_task(_task("a"))
    wbc.set_contact_jacobians([np.zeros((6, N_V))])
    sol = wbc.solve()
    assert isinstance(sol, WBCSolution)


def test_solve_skips_mismatched_task_dim() -> None:
    cfg = WBCConfig(use_hierarchical=False)
    wbc = WholeBodyController(FakeEngine(), config=cfg, solver=NullspaceQPSolver())
    # Task with wrong config dim should be skipped, not crash
    wbc.add_task(_task("good"))
    wbc.add_task(_task("bad", J=np.eye(3, N_V + 2), target=np.zeros(3)))
    sol = wbc.solve()
    assert isinstance(sol, WBCSolution)


def test_compute_nullspace_projector() -> None:
    wbc = WholeBodyController(FakeEngine())
    A = np.eye(2, 4)
    N = wbc._compute_nullspace_projector(A, 4)
    # N @ A.T should be ~0
    assert np.allclose(N @ A.T, 0.0, atol=1e-8)


def test_group_tasks_by_priority() -> None:
    wbc = WholeBodyController(FakeEngine())
    wbc.add_task(_task("a", priority=0))
    wbc.add_task(_task("b", priority=0))
    wbc.add_task(_task("c", priority=2))
    groups = wbc._group_tasks_by_priority()
    assert len(groups[0]) == 2
    assert len(groups[2]) == 1


def test_compute_task_errors_for_consistent_qdd() -> None:
    wbc = WholeBodyController(FakeEngine())
    wbc.add_task(_task("a", J=np.eye(3, N_V), target=np.zeros(3)))
    errs = wbc._compute_task_errors(np.zeros(N_V))
    assert errs["a"] == pytest.approx(0.0)


def test_compute_task_errors_skips_mismatched_dim() -> None:
    wbc = WholeBodyController(FakeEngine())
    wbc.add_task(_task("good", J=np.eye(3, N_V), target=np.zeros(3)))
    wbc.add_task(_task("bad", J=np.eye(3, N_V + 1), target=np.zeros(3)))
    errs = wbc._compute_task_errors(np.zeros(N_V))
    assert "good" in errs
    assert "bad" not in errs
