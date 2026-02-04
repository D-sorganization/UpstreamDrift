"""Control module for robotics applications.

This module provides control algorithms for robot actuation:
- Whole-body control with hierarchical task prioritization
- Operational space control for task-space tracking
- QP-based optimization for constraint handling

Example:
    >>> from src.robotics.control import WholeBodyController, Task
    >>>
    >>> wbc = WholeBodyController(engine)
    >>> wbc.add_task(Task.com_tracking(target_com, weight=1.0))
    >>> wbc.add_task(Task.posture(target_q, weight=0.1))
    >>> torques = wbc.solve()
"""

from __future__ import annotations

from src.robotics.control.whole_body.qp_solver import (
    QPProblem,
    QPSolution,
    QPSolver,
)
from src.robotics.control.whole_body.task import (
    Task,
    TaskType,
    create_com_task,
    create_ee_task,
    create_posture_task,
)
from src.robotics.control.whole_body.wbc_controller import (
    WBCConfig,
    WBCSolution,
    WholeBodyController,
)

__all__ = [
    # Tasks
    "Task",
    "TaskType",
    "create_com_task",
    "create_posture_task",
    "create_ee_task",
    # Controller
    "WholeBodyController",
    "WBCConfig",
    "WBCSolution",
    # Solver
    "QPSolver",
    "QPProblem",
    "QPSolution",
]
