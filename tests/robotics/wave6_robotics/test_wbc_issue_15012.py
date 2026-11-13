"""Regression tests for issue #15012 — WBC contact torques and hierarchy."""

from __future__ import annotations

import numpy as np
import pytest

from src.robotics.control.whole_body.qp_solver import (
    NullspaceQPSolver,
    create_default_solver,
)
from src.robotics.control.whole_body.task import Task, TaskType
from src.robotics.control.whole_body.wbc_controller import (
    WBCConfig,
    WholeBodyController,
)
from tests.robotics.wave6_robotics.test_wbc_controller import FakeEngine, N_V


class BiasedFakeEngine(FakeEngine):
    """FakeEngine with configurable bias forces."""

    def __init__(self, n_v: int = N_V, nle: np.ndarray | None = None) -> None:
        super().__init__(n_v=n_v)
        self._nle = np.zeros(n_v) if nle is None else np.asarray(nle, dtype=float)

    def compute_bias_forces(self):
        return self._nle.copy()


def _one_dof_task(name: str, target_qdd: float, priority: int) -> Task:
    J = np.zeros((1, N_V))
    J[0, 0] = 1.0
    return Task(
        name=name,
        task_type=TaskType.SOFT,
        priority=priority,
        jacobian=J,
        target=np.array([target_qdd]),
        weight=np.array([1.0]),
    )


@pytest.mark.unit
def test_fixed_base_default_config_nonzero_contact_torques() -> None:
    """Default WBCConfig must not zero torques on fixed-base models with contact."""
    nle = np.array([0.0, 0.0, 2.0, 0.0])
    cfg = WBCConfig(use_hierarchical=False, regularization=1e-4)
    wbc = WholeBodyController(
        BiasedFakeEngine(nle=nle),
        config=cfg,
        solver=create_default_solver(),
    )
    wbc.add_task(
        Task(
            name="posture",
            task_type=TaskType.SOFT,
            priority=2,
            jacobian=np.eye(3, N_V),
            target=np.zeros(3),
            weight=np.ones(3),
        )
    )
    J_c = np.zeros((3, N_V))
    J_c[2, 2] = 1.0
    wbc.set_contact_jacobians([J_c])

    sol = wbc.solve()
    assert sol.success
    assert sol.joint_torques is not None
    assert np.linalg.norm(sol.joint_torques) > 1e-6


@pytest.mark.unit
def test_contact_support_yields_nonzero_actuated_torques() -> None:
    """Actuated DOFs must not receive zero torque when contacts are active."""
    nle = np.array([0.0, 0.0, 2.0, 0.0])
    cfg = WBCConfig(
        use_hierarchical=False,
        floating_base_dofs=0,
        regularization=1e-4,
    )
    wbc = WholeBodyController(
        BiasedFakeEngine(nle=nle),
        config=cfg,
        solver=create_default_solver(),
    )
    wbc.add_task(
        Task(
            name="posture",
            task_type=TaskType.SOFT,
            priority=2,
            jacobian=np.eye(3, N_V),
            target=np.zeros(3),
            weight=np.ones(3),
        )
    )
    J_c = np.zeros((3, N_V))
    J_c[2, 2] = 1.0
    wbc.set_contact_jacobians([J_c])

    sol = wbc.solve()
    assert sol.success
    assert sol.joint_torques is not None
    assert np.linalg.norm(sol.joint_torques) > 1e-6, (
        f"Expected non-zero torques with contact; got {sol.joint_torques}"
    )


@pytest.mark.unit
def test_floating_base_rows_only_constrained_with_contact() -> None:
    """Unactuated rows satisfy dynamics; actuated torques stay non-zero."""
    n_v = 10
    n_fb = 6
    nle = np.linspace(0.5, 5.0, n_v)

    class Engine10(BiasedFakeEngine):
        def __init__(self) -> None:
            super().__init__(n_v=n_v, nle=nle)

    cfg = WBCConfig(use_hierarchical=False, floating_base_dofs=n_fb)
    wbc = WholeBodyController(Engine10(), config=cfg, solver=create_default_solver())
    wbc.add_task(
        Task(
            name="soft",
            task_type=TaskType.SOFT,
            priority=1,
            jacobian=np.eye(3, n_v),
            target=np.zeros(3),
            weight=np.ones(3),
        )
    )
    J_c = np.zeros((3, n_v))
    J_c[:, 6] = 1.0
    wbc.set_contact_jacobians([J_c])

    sol = wbc.solve()
    assert sol.success
    assert sol.joint_torques is not None
    actuated = sol.joint_torques[n_fb:]
    assert np.max(np.abs(actuated)) > 1e-6


@pytest.mark.unit
def test_hierarchy_respects_priority_zero_highest() -> None:
    """Higher-priority (numerically lower) task wins when targets conflict."""
    cfg = WBCConfig(use_hierarchical=True, regularization=1e-4, floating_base_dofs=0)
    wbc = WholeBodyController(FakeEngine(), config=cfg, solver=NullspaceQPSolver())
    wbc.add_task(_one_dof_task("high", target_qdd=1.0, priority=0))
    wbc.add_task(_one_dof_task("low", target_qdd=-1.0, priority=5))
    sol = wbc.solve()

    assert sol.success
    assert sol.joint_accelerations is not None
    assert sol.joint_accelerations[0] > 0.5

    wbc.clear_tasks()
    wbc.add_task(_one_dof_task("low", target_qdd=-1.0, priority=5))
    wbc.add_task(_one_dof_task("high", target_qdd=1.0, priority=0))
    sol2 = wbc.solve()
    assert sol2.success
    assert sol2.joint_accelerations is not None
    assert sol2.joint_accelerations[0] > 0.5


@pytest.mark.unit
def test_tasks_sorted_lowest_priority_number_first() -> None:
    wbc = WholeBodyController(FakeEngine())
    wbc.add_task(_one_dof_task("p4", 0.0, priority=4))
    wbc.add_task(_one_dof_task("p0", 0.0, priority=0))
    wbc.add_task(_one_dof_task("p2", 0.0, priority=2))
    assert [t.name for t in wbc.tasks] == ["p0", "p2", "p4"]
