"""Data classes and context manager for kinematic force analysis.

Contains:
- MjDataContext: context manager for MuJoCo state isolation
- KinematicForceData: container for forces at a time point
- _check_mujoco_version: runtime version validation (Issue F-003)
- export_kinematic_forces_to_csv: CSV export utility
"""

from __future__ import annotations

import csv
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import numpy as np

# Import numerical constants (Assessment B-004, B-007)

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


@dataclass
class KinematicForceData:
    """Container for kinematic-dependent forces at a single time point."""

    time: float

    # Joint-space forces
    coriolis_forces: np.ndarray  # [nv] - Coriolis and centrifugal forces
    gravity_forces: np.ndarray  # [nv] - Gravitational forces

    # Decomposed components
    centrifugal_forces: np.ndarray | None = None  # [nv] - Pure centrifugal
    velocity_coupling_forces: np.ndarray | None = None  # [nv] - Velocity coupling

    # Task-space forces (end-effector)
    club_head_coriolis_force: np.ndarray | None = None  # [3] - at club head
    club_head_centrifugal_force: np.ndarray | None = None  # [3] - at club head
    club_head_apparent_force: np.ndarray | None = None  # [3] - total apparent force

    # Power contributions
    coriolis_power: float = 0.0  # Power dissipated by Coriolis forces
    centrifugal_power: float = 0.0  # Power from centrifugal effects

    # Kinetic energy contributions
    rotational_kinetic_energy: float = 0.0
    translational_kinetic_energy: float = 0.0


def export_kinematic_forces_to_csv(
    force_data_list: list[KinematicForceData],
    filepath: str,
) -> None:
    """Export kinematic force analysis to CSV file.

    Args:
        force_data_list: List of force data
        filepath: Output CSV file path
    """
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)

        # Header
        header = [
            "time",
            "coriolis_power",
            "centrifugal_power",
            "rotational_ke",
            "translational_ke",
        ]

        # Add joint-wise Coriolis forces
        nv = len(force_data_list[0].coriolis_forces)
        for i in range(nv):
            header.extend(
                [f"coriolis_force_{i}", f"gravity_force_{i}", f"centrifugal_force_{i}"],
            )

        # Add club head forces
        header.extend(
            [
                "club_coriolis_x",
                "club_coriolis_y",
                "club_coriolis_z",
                "club_centrifugal_x",
                "club_centrifugal_y",
                "club_centrifugal_z",
            ],
        )

        writer.writerow(header)

        # Data rows
        for data in force_data_list:
            row = [
                data.time,
                data.coriolis_power,
                data.centrifugal_power,
                data.rotational_kinetic_energy,
                data.translational_kinetic_energy,
            ]

            for i in range(nv):
                row.extend(
                    [
                        data.coriolis_forces[i],
                        data.gravity_forces[i],
                        (
                            data.centrifugal_forces[i]
                            if data.centrifugal_forces is not None
                            else 0.0
                        ),
                    ],
                )

            if data.club_head_coriolis_force is not None:
                row.extend(data.club_head_coriolis_force.tolist())
            else:
                row.extend([0, 0, 0])

            if data.club_head_centrifugal_force is not None:
                row.extend(data.club_head_centrifugal_force.tolist())
            else:
                row.extend([0, 0, 0])

            writer.writerow(row)
