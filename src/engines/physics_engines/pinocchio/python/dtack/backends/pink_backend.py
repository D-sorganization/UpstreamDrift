"""Stateful backend wrapper around the shared validated Pink solve step."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from src.engines.physics_engines.pinocchio.python.dtack.ik.pink_solver import (
    _INSTALL_HINT,
    SolverSettings,
    _PinkConfigurationCache,
    _PinkStepOptions,
    _solve_pink_step,
)
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

try:
    import pink
    import pinocchio as pin

    PINK_AVAILABLE = True
except (ImportError, OSError):
    PINK_AVAILABLE = False
    pink = None  # type: ignore[assignment]
    pin = None  # type: ignore[assignment]
    logger.warning("PINK is not available. %s", _INSTALL_HINT)


class PINKBackend:
    """PINK backend for stateful differential inverse kinematics."""

    def __init__(self, model_path: Path | str) -> None:
        """Load a Pinocchio robot and initialize Pink's kinematic cache."""
        if not PINK_AVAILABLE:
            raise ImportError(f"PINK is required but not installed. {_INSTALL_HINT}")
        self.model_path = Path(model_path)
        try:
            self.robot: Any = pin.RobotWrapper.BuildFromURDF(str(self.model_path))
            self.configuration = pink.Configuration(
                self.robot.model,
                self.robot.data,
                self.robot.q0,
                collision_model=self.robot.collision_model,
            )
            self._cache = _PinkConfigurationCache(
                configuration=self.configuration,
                collision_model=self.robot.collision_model,
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            exc.add_note(f"Failed to initialize PINK backend from {self.model_path}")
            raise
        logger.info("PINK backend initialized with model: %s", self.model_path)

    def solve_ik(
        self,
        tasks: Mapping[str, Any],
        q_init: npt.NDArray[np.float64],
        dt: float = 1e-3,
        solver: str = "quadprog",
        *,
        damping: float = 1e-12,
        constraints: Iterable[Any] | None = None,
        limits: Iterable[Any] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Solve one step, preserving hard constraints and Pink limit defaults.

        After input validation, the configuration cache is refreshed to ``q_init``
        before the QP solve and to the integrated result after success.
        """
        if not PINK_AVAILABLE:
            raise ImportError(f"PINK is required but not installed. {_INSTALL_HINT}")
        options = _PinkStepOptions(
            dt=dt,
            settings=SolverSettings(solver=solver, damping=damping),
            constraints=constraints,
            limits=limits,
        )
        result = _solve_pink_step(
            self.robot.model,
            self.robot.data,
            q_init,
            list(tasks.values()),
            options,
            self._cache,
        )
        self._cache.configuration = result.configuration
        self.configuration = result.configuration
        return result.q_next
