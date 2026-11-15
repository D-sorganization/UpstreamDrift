"""Shared types for hand visual backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

from src.shared.python.model_appearance.geometry import Mesh

HandVisualSide = Literal["lead", "trail"]
HandVisualMeshes = dict[str, Mesh]

BACKEND_CAPSULE: Final[str] = "capsule"
BACKEND_MYOHAND: Final[str] = "myohand"
DEFAULT_HAND_VISUAL_BACKEND: Final[str] = BACKEND_CAPSULE


@dataclass(frozen=True)
class HandBackendInfo:
    """Registry metadata for one visual backend."""

    name: str
    research_only: bool = False
    description: str = ""


MYOHAND_BACKEND_INFO = HandBackendInfo(
    name=BACKEND_MYOHAND,
    research_only=True,
    description=(
        "MyoHub myo_sim anatomical bone STL meshes only; skin is separate capsule "
        "geoms in MJCF, not a renderable surface mesh. MoBL upstream lineage — "
        "research-only until #13763 owner sign-off."
    ),
)

__all__ = [
    "BACKEND_CAPSULE",
    "BACKEND_MYOHAND",
    "DEFAULT_HAND_VISUAL_BACKEND",
    "HandBackendInfo",
    "HandVisualMeshes",
    "HandVisualSide",
    "MYOHAND_BACKEND_INFO",
]
