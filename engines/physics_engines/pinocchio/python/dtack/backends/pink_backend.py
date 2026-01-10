"""PINK backend wrapper for inverse kinematics."""

from __future__ import annotations

import logging
import typing
from pathlib import Path

if typing.TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    import pinocchio as pin

logger = logging.getLogger(__name__)

# PINK availability check
try:
    import pink  # noqa: F401
    import pinocchio as pin

    PINK_AVAILABLE = True
except ImportError:
    PINK_AVAILABLE = False
    logger.warning("PINK not available. Install with: pip install pink")


class PINKBackend:
    """PINK backend for inverse kinematics.

    This backend provides:
    - IK task definition
    - Closed-loop IK solving
    - Task-space control
    """

    def __init__(self, model_path: Path | str) -> None:
        """Initialize PINK backend.

        Args:
            model_path: Path to model file or canonical YAML specification

        Raises:
            ImportError: If PINK is not installed
        """
        if not PINK_AVAILABLE:
            msg = "PINK is required but not installed. Install with: pip install pink"
            raise ImportError(msg)

        self.model_path = Path(model_path)

        # Load robot model
        # Using pinocchio to load the model (URDF usually)
        try:
            self.robot = pin.RobotWrapper.BuildFromURDF(str(self.model_path))
            self.configuration = pink.Configuration(
                self.robot.model, self.robot.data, self.robot.q0
            )
            logger.info(f"PINK backend initialized with model: {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model for PINK: {e}")
            raise

    def solve_ik(
        self,
        tasks: dict[str, pink.tasks.Task],
        q_init: npt.NDArray[np.float64],
        dt: float = 1e-3,
        solver: str = "quadprog",
    ) -> npt.NDArray[np.float64]:
        """Solve inverse kinematics for given tasks.

        Args:
            tasks: Dictionary of task objects (Pink tasks)
            q_init: Initial joint configuration
            dt: Time step for integration
            solver: QP solver to use

        Returns:
            Joint configuration satisfying tasks
        """
        if not PINK_AVAILABLE:
            raise ImportError("Pink not installed")

        try:
            # Update configuration to q_init
            self.configuration.q = q_init

            # Solve
            # pink.solve_ik takes (configuration, tasks, dt, solver, ...)
            # It returns the velocity dq. We need to integrate or pink might return new q?
            # Looking at standard pink usage:
            # velocity = pink.solve_ik(configuration, tasks, dt, solver)
            # configuration.integrate_inplace(velocity, dt)

            task_list = list(tasks.values())
            velocity = pink.solve_ik(self.configuration, task_list, dt, solver=solver)

            # Integrate to get new q
            q_next = pin.integrate(self.robot.model, q_init, velocity * dt)

            # Update internal configuration for next step consistency if this object persists state?
            # method signature implies stateless solve based on q_init,
            # but we can update self.configuration
            self.configuration.q = q_next

            return q_next

        except Exception as e:
            logger.error(f"PINK IK solve failed: {e}")
            return q_init
