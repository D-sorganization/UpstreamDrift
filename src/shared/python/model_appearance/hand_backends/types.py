"""Shared types for grip hand visual backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.shared.python.model_appearance.geometry import Mesh

HandSide = Literal["lead", "trail"]


@dataclass(frozen=True)
class HandVisualResult:
    """One hand mesh in the hand-grip frame."""

    backend: str
    side: HandSide
    mesh: Mesh
    skin_rgb: tuple[float, float, float]
