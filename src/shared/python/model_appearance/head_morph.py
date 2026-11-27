"""Bounded visual-only deformations in the canonical unit-head frame.

The same smooth mapping is applied to skin and accessory vertices. Parameters
are artistic controls, not biometric measurements; x is forward, y left, z up.
"""

from dataclasses import dataclass, fields
import math

import numpy as np
from numpy.typing import NDArray

LEGACY_MORPH_FIELDS = ("width", "jaw", "nose", "mouth", "eyes")


@dataclass(frozen=True)
class HeadMorph:
    """Signed visual adjustments, each in [-0.2, 0.2]; zero is identity."""

    width: float = 0.0
    jaw: float = 0.0
    nose: float = 0.0
    mouth: float = 0.0
    eyes: float = 0.0
    cheeks: float = 0.0
    chin: float = 0.0
    brow: float = 0.0
    nose_width: float = 0.0
    nose_bridge: float = 0.0
    nose_length: float = 0.0
    eye_opening: float = 0.0
    face_length: float = 0.0
    ears: float = 0.0

    def __post_init__(self) -> None:
        """Require every artistic adjustment to be finite and bounded."""
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
    """Warp head-length coordinates, keeping the neck and skull height fixed."""
    points = np.asarray(vertices, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("head vertices must be a finite N by 3 array")
    if morph == HeadMorph():
        return points.copy()
    x, y, z = points.T
    neck = np.clip(z / 0.2, 0.0, 1.0)
    neck = neck * neck * (3.0 - 2.0 * neck)

    def window(center: float, radius: float) -> NDArray[np.float64]:
        """Return a smooth vertical influence centered on a facial feature."""
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
    if any(
        getattr(morph, f.name)
        for f in fields(morph)
        if f.name not in LEGACY_MORPH_FIELDS
    ):
        out = _portrait_warp(points, out, morph, neck * front)
    return out


def _portrait_warp(
    points: NDArray[np.float64],
    out: NDArray[np.float64],
    morph: HeadMorph,
    face: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compact facial influence fields leave neck, vertex and hats anchored.

    All parts use the same original coordinates, including eye layers, hair
    and facial hair. Controls describe an artistic shape, not measured age.
    """
    x, y, z = points.T

    def band(
        value: NDArray[np.float64], centre: float, radius: float
    ) -> NDArray[np.float64]:
        return np.maximum(1 - ((value - centre) / radius) ** 2, 0) ** 2

    bridge = band(y, 0, 0.10) * band(z, 0.51, 0.15) * face
    nose = band(y, 0, 0.12) * band(z, 0.42, 0.13) * face
    cheek = band(np.abs(y), 0.19, 0.16) * band(z, 0.43, 0.17) * face
    chin = band(z, 0.21, 0.15) * face
    brow = band(z, 0.61, 0.095) * face
    eye = band(np.abs(y), 0.124, 0.10) * band(z, 0.538, 0.075) * face
    ear = band(np.abs(y), 0.34, 0.15) * band(z, 0.55, 0.22)
    ear *= np.clip((0.35 - x) / 0.25, 0, 1)
    out[:, 0] += (
        0.14 * morph.nose_bridge * bridge
        + 0.12 * morph.cheeks * cheek
        + 0.16 * morph.chin * chin
        + 0.15 * morph.brow * brow
    )
    out[:, 1] += y * (
        2 * morph.nose_width * nose
        + 0.5 * morph.cheeks * cheek
        + 2 * morph.chin * chin * band(y, 0, 0.17)
        + 0.5 * morph.ears * ear
    )
    out[:, 2] += (
        -0.12 * morph.nose_length * nose
        + 2 * morph.eye_opening * (z - 0.538) * eye
        + morph.face_length * (z - 0.57) * band(z, 0.39, 0.25) * face
    )
    return out
