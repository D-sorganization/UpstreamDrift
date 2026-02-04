"""Motion planning module.

This module provides sampling-based motion planners:
- RRT (Rapidly-exploring Random Trees)
- RRT* (Optimal RRT with rewiring)
- PRM (Probabilistic Roadmap) - future
- Bi-directional variants - future

Example:
    >>> from src.robotics.planning.motion import RRTPlanner, RRTStarPlanner
    >>> from src.robotics.planning.collision import CollisionChecker
    >>>
    >>> collision_checker = CollisionChecker(engine)
    >>> planner = RRTPlanner(collision_checker)
    >>> path = planner.plan(q_start, q_goal)
"""

from __future__ import annotations

from src.robotics.planning.motion.planner_base import (
    MotionPlanner,
    PlannerConfig,
    PlannerResult,
    PlannerStatus,
)
from src.robotics.planning.motion.rrt import RRTConfig, RRTPlanner
from src.robotics.planning.motion.rrt_star import RRTStarConfig, RRTStarPlanner

__all__ = [
    # Base
    "MotionPlanner",
    "PlannerConfig",
    "PlannerResult",
    "PlannerStatus",
    # RRT
    "RRTPlanner",
    "RRTConfig",
    # RRT*
    "RRTStarPlanner",
    "RRTStarConfig",
]
