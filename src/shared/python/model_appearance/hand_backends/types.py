"""Shared types for parallel grip hand visual backends (#15122)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from src.shared.python.model_appearance.geometry import Mesh

HandSide = Literal["lead", "trail"]
JointAngles = Mapping[str, float] | NDArray[np.float64] | Sequence[float] | None

Array = NDArray[np.float64]


@dataclass(frozen=True)
class HandVisualResult:
    """Skin mesh and material in the hand-grip frame."""

    mesh: Mesh
    material_rgba: tuple[float, float, float, float]


__all__ = ["Array", "HandSide", "HandVisualResult", "JointAngles"]
