"""Inverse Kinematics solver using Pink."""

from __future__ import annotations

import logging
import typing
from dataclasses import dataclass

import pink
import pinocchio as pin
from pink import Task

if typing.TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SolverSettings:
    """Settings for the IK solver."""

    solver: str = "quadprog"
    damping: float = 1e-6


class PinkSolver:
    """Inverse Kinematics solver wrapper for Pink."""

    def __init__(
        self,
        robot_model: pin.Model,
        robot_data: pin.Data,
        robot_visual: pin.GeometryModel,
        robot_collision: pin.GeometryModel,
    ) -> None:
        """Initialize Pink solver.

        Args:
            robot_model: Pinocchio setup model
            robot_data: Pinocchio setup data
            robot_visual: Pinocchio visual model
            robot_collision: Pinocchio collision model
        """
        # Pink expects a 'Configuration' object usually, but can work with models.
        # We'll maintain the pinocchio model references.
        self.model = robot_model
        self.data = robot_data
        self.visual_model = robot_visual
        self.collision_model = robot_collision

        # Pink configuration is created during solve or cached if appropriate
        # but for simple usage we might just recreate it or update it.
        # A Pink 'Configuration' binds a model to a specific joint configuration `q`.

    def solve(
        self,
        q_init: np.ndarray,
        tasks: list[Task],
        dt: float,
        settings: SolverSettings | None = None,
    ) -> np.ndarray:
        """Solve differential IK for one step.

        Args:
            q_init: Current joint configuration
            tasks: List of Pink tasks to satisfy (e.g. FrameTask, PostureTask)
            dt: Time step for velocity integration
            settings: Solver settings (algorithm, damping)

        Returns:
            New joint configuration q_next
        """
        if settings is None:
            settings = SolverSettings()

        # Create a Pink configuration at the current state
        # Note: Depending on pink version, signature might vary.
        # Assuming standard pink.Configuration usage.

        configuration = pink.Configuration(self.model, self.data, q_init)

        # Solve delta_q or velocity
        # pink.solve_ik returns the velocity (v) usually to achieve tasks
        velocity = pink.solve_ik(
            configuration,
            tasks,
            dt,
            solver=settings.solver,
            damping=settings.damping,
        )

        # Integrate velocity to update configuration: q_next = q + v * dt
        result = pin.integrate(self.model, q_init, velocity * dt)
        return np.array(result, dtype=np.float64)
