"""MyoSuite :class:`PoseConventionAdapter` implementation."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from src.shared.python.pose_interchange.canonical import CanonicalPose


class MyoSuiteAdapter:
    """Adapter translating CanonicalPose to/from MyoSuite joint space."""

    engine_name: str = "myosuite"

    def canonical_to_engine(
        self,
        pose: CanonicalPose,
    ) -> dict[str, float | npt.NDArray[np.float64]]:
        """Map canonical pose to MyoSuite qpos."""
        return {}

    def engine_to_canonical(
        self,
        engine_pose: dict[str, float | npt.NDArray[np.float64]],
    ) -> CanonicalPose:
        """Map MyoSuite qpos to canonical pose."""
        return CanonicalPose(
            address_pose={},
            root_transform=np.eye(4),
            contact_wrenches={},
        )
