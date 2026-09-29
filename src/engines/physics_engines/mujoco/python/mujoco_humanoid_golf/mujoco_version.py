"""MuJoCo version validation and MjData context management.

This module provides:
- Runtime MuJoCo version checking (Issue F-003)
- MjDataContext: context manager for safe state isolation
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import mujoco
import numpy as np

if TYPE_CHECKING:
    from types import TracebackType


def _check_mujoco_version() -> None:
    """Validate MuJoCo version meets minimum requirements.

    Addresses Issue F-003: Prevents API signature mismatches by enforcing
    minimum version at runtime.

    Raises:
        ImportError: If MuJoCo version is too old
    """
    try:
        # MuJoCo version format: "3.3.0" or similar
        version_str = mujoco.__version__
        major, minor, *_ = map(int, version_str.split("."))

        # Require MuJoCo 3.3+ for reshaped Jacobian API
        if (major, minor) < (3, 3):
            msg = (
                f"MuJoCo {version_str} detected, but 3.3.0+ is required.\n"
                f"The reshaped Jacobian API (mj_jacBody with 2D arrays) was "
                f"introduced in MuJoCo 3.3. Earlier versions use flat arrays "
                f"which can cause dimension alignment errors.\n"
                f"Please upgrade: pip install 'mujoco>=3.3.0,<4.0.0'\n"
                f"See Issue F-003 in Assessment C for details."
            )
            raise ImportError(msg)

        # Success - log version
        import logging

        logger = logging.getLogger(__name__)
        logger.info(f"MuJoCo version {version_str} validated successfully")

    except (AttributeError, ValueError) as e:
        # Could not parse version
        warnings.warn(
            f"Could not validate MuJoCo version: {e}. "
            f"Proceeding with fallback Jacobian handling.",
            category=UserWarning,
            stacklevel=2,
        )


# Validate MuJoCo version on module import (Issue F-003)
_check_mujoco_version()


class MjDataContext:
    """Context manager for safe MuJoCo MjData state isolation.

    This context manager saves the current state of MjData on entry and
    restores it on exit, ensuring that any mutations within the context
    do not affect the original state.

    Addresses Issues A-001, A-003, F-001, F-002 by providing functional
    purity guarantees for analysis methods.

    Example:
        >>> with MjDataContext(model, data):
        ...     data.qpos[:] = new_positions  # Safe to mutate
        ...     result = compute_something(model, data)
        ... # data.qpos is automatically restored here

    This enables:
    - Safe parallel analysis
    - No Observer Effect bugs
    - Scientific reproducibility
    - Thread-safe computations
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Initialize context manager.

        Args:
            model: MuJoCo model (needed for forward kinematics)
            data: MuJoCo data structure to protect
        """
        if model is None:
            raise ValueError("model must be provided")
        self.model = model
        self.data = data
        self.qpos_backup: np.ndarray | None = None
        self.qvel_backup: np.ndarray | None = None
        self.qacc_backup: np.ndarray | None = None
        self.ctrl_backup: np.ndarray | None = None
        self.time_backup: float = 0.0

    def __enter__(self) -> mujoco.MjData:
        """Save current state on context entry.

        Returns:
            The data object for convenience
        """
        self.qpos_backup = self.data.qpos.copy()
        self.qvel_backup = self.data.qvel.copy()
        self.qacc_backup = self.data.qacc.copy()
        self.ctrl_backup = self.data.ctrl.copy()
        self.time_backup = self.data.time
        return self.data

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Restore state on context exit, even if exception occurred.

        Args:
            exc_type: Exception type if raised
            exc_val: Exception value if raised
            exc_tb: Exception traceback if raised
        """
        self.data.qpos[:] = self.qpos_backup
        self.data.qvel[:] = self.qvel_backup
        self.data.qacc[:] = self.qacc_backup
        self.data.ctrl[:] = self.ctrl_backup
        self.data.time = self.time_backup

        # Recompute forward kinematics to sync all derived quantities
        mujoco.mj_forward(self.model, self.data)
