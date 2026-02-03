"""Whole-body control module.

Provides hierarchical task-space control for humanoid robots
using quadratic programming for constraint optimization.
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
    "Task",
    "TaskType",
    "create_com_task",
    "create_posture_task",
    "create_ee_task",
    "WholeBodyController",
    "WBCConfig",
    "WBCSolution",
    "QPSolver",
    "QPProblem",
    "QPSolution",
]
