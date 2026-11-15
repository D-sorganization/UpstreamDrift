"""Shared types for hand visual backends."""

from __future__ import annotations

from typing import Final, Literal

from src.shared.python.model_appearance.geometry import Mesh

HandVisualSide = Literal["lead", "trail"]
HandVisualMeshes = dict[str, Mesh]

BACKEND_MYOHAND: Final[str] = "myohand"
BACKEND_CAPSULE: Final[str] = "capsule"

__all__ = [
    "BACKEND_CAPSULE",
    "BACKEND_MYOHAND",
    "HandVisualMeshes",
    "HandVisualSide",
]
