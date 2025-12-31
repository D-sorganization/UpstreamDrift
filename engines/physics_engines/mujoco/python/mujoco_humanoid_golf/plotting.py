"""Advanced plotting and visualization for golf swing analysis.

This module re-exports the shared plotting functionality with a compatibility layer
for the existing MuJoCo implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
from shared.python.plotting import GolfSwingPlotter as SharedGolfSwingPlotter
from shared.python.plotting import MplCanvas

if TYPE_CHECKING:
    from engines.physics_engines.mujoco.python.mujoco_humanoid_golf.biomechanics import (  # noqa: E501
        SwingRecorder,
    )

__all__ = ["GolfSwingPlotter", "MplCanvas"]


class GolfSwingPlotter(SharedGolfSwingPlotter):
    """Compatibility wrapper for GolfSwingPlotter."""

    def __init__(
        self,
        recorder: SwingRecorder,
        model: mujoco.MjModel | None = None,
    ) -> None:
        """Initialize plotter with recorded data.

        Args:
            recorder: SwingRecorder with recorded swing data
            model: Optional MuJoCo model for joint names
        """
        # Create joint names list if model is provided
        joint_names = None
        if model is not None:
            joint_names = []
            for i in range(model.nq):
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
                joint_names.append(name if name else f"Joint {i}")

        super().__init__(recorder, joint_names)
        self.model = model
