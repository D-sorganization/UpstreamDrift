"""MyoSuite adapter implementation."""

from __future__ import annotations

import numpy as np

from src.shared.python.pose_interchange.protocol import PoseConventionAdapter


class MyosuiteAdapter(PoseConventionAdapter):
    """Adapter for MyoSuite pose conventions."""

    engine_name = "myosuite"

    def apply_convention(self, pose: np.ndarray) -> np.ndarray:
        """Apply MyoSuite convention."""
        return pose.copy()

    def remove_convention(self, pose: np.ndarray) -> np.ndarray:
        """Remove MyoSuite convention."""
        return pose.copy()
