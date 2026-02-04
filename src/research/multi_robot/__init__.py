"""Multi-Robot Coordination.

This module provides multi-robot simulation and coordination:
- Multi-robot system management
- Formation control
- Cooperative manipulation
- Task allocation
"""

from __future__ import annotations

from src.research.multi_robot.coordination import (
    CooperativeManipulation,
    FormationConfig,
    FormationController,
)
from src.research.multi_robot.system import (
    MultiRobotSystem,
    Task,
    TaskCoordinator,
)

__all__ = [
    "MultiRobotSystem",
    "Task",
    "TaskCoordinator",
    "FormationController",
    "FormationConfig",
    "CooperativeManipulation",
]
