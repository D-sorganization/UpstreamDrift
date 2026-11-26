"""Bounded visual-only deformations in the canonical unit-head frame.

The same smooth mapping is applied to skin and accessory vertices. Parameters
are artistic controls, not biometric measurements; x is forward, y left, z up.
"""

from dataclasses import dataclass, fields
import math

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class HeadMorph:
    """Signed visual adjustments, each in [-0.2, 0.2]; zero is identity."""

    width: float = 0.0
    jaw: float = 0.0
    nose: float = 0.0
    mouth: float = 0.0
    eyes: float = 0.0

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not -0.2 <= value <= 0.2
            ):
                raise ValueError(
                    f"head morph {item.name} must be finite and within [-0.2, 0.2]"
                )


def warp_vertices(
    vertices: NDArray[np.float64], morph: HeadMorph
) -> NDArray[np.float64]:
    """Warp head-length coordinates, keeping neck plane and heights fixed."""
    points = np.asarray(vertices, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("head vertices must be a finite N by 3 array")
    if morph == HeadMorph():
        return points.copy()
    x, y, z = points.T
    neck = np.clip(z / 0.2, 0.0, 1.0)
    neck = neck * neck * (3.0 - 2.0 * neck)

    def window(center: float, radius: float) -> NDArray[np.float64]:
        return np.exp(-(((z - center) / radius) ** 2))

    front = np.clip((x + 0.05) / 0.35, 0.0, 1.0)
    out = points.copy()
    out[:, 1] = y * (
        1
        + neck
        * (
            morph.width
            + morph.jaw * window(0.28, 0.16)
            + front
            * (morph.mouth * window(0.276, 0.08) + morph.eyes * window(0.532, 0.08))
        )
    )
    out[:, 0] = x + neck * morph.nose * 0.12 * front * window(0.43, 0.12) * np.exp(
        -((y / 0.10) ** 2)
    )
    return out
