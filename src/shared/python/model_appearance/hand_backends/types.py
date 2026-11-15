"""Shared types for grip hand visual backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

from src.shared.python.model_appearance.geometry import Mesh

if TYPE_CHECKING:
    from src.shared.python.model_appearance.hand_backends.joint_angles import (
        HandJointAngles,
    )
    from src.shared.python.model_appearance.hand_backends.skinning import (
        SkinnedHandAsset,
    )

HandSide = Literal["lead", "trail"]


@dataclass(frozen=True)
class HandVisualResult:
    """One hand mesh in the hand-grip frame.

    ``mesh`` and ``skin_rgb`` are what a renderer needs. The rest lets a
    caller audit the pose: skinned backends report the joint angles, the gap
    between the grip and each phalanx, and the bone transforms and rigid frame
    that produced every vertex. ``frame_focus`` marks the vertices a renderer
    frames its camera on (the hand, not the forearm stub); ``None`` means all.
    """

    backend: str
    side: HandSide
    mesh: Mesh
    skin_rgb: tuple[float, float, float]
    vertex_normals: NDArray[np.float64] | None = None
    joint_angles: HandJointAngles | None = None
    contact_gaps_m: dict[str, float] = field(default_factory=dict)
    grip_radius_m: float | None = None
    bone_transforms: NDArray[np.float64] | None = None
    frame_from_rest: NDArray[np.float64] | None = None
    rest_asset: SkinnedHandAsset | None = None
    frame_focus: NDArray[np.bool_] | None = None
