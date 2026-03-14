"""Inverse Kinematics solver using Pink."""

from __future__ import annotations

import typing
from dataclasses import dataclass

from src.shared.python.logging_pkg.logging_config import get_logger

# Guard pink/pinocchio imports — these are optional heavy dependencies
try:
    import pink
    import pinocchio as pin
    from pink import Task

    PINK_SOLVER_AVAILABLE = True
except ImportError:
    PINK_SOLVER_AVAILABLE = False
    pink = None  # type: ignore[assignment]
    pin = None  # type: ignore[assignment]
    Task = None  # type: ignore[assignment,misc]

if typing.TYPE_CHECKING:
    import numpy as np

logger = get_logger(__name__)


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

        Raises:
            ImportError: If pink or pinocchio are not installed.
        """
        if not PINK_SOLVER_AVAILABLE:
            raise ImportError(
                "Pink and pinocchio are required for PinkSolver. "
                "Install with: pip install pink pinocchio"
            )

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
        assert q_init is not None, "q_init must be provided"
        assert q_init is not None, "q_init must be provided"
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
