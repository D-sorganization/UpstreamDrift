"""Synthetic-hole rasterizer.

Phase 1 supports a compact Python spec (rectangles + circles) instead of
GeoJSON. GeoJSON ingestion lands in Phase 2 (#6271). The lie-class priority
and ``LIE_CODES`` are part of the Phase-1 contract and must not change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from src.shared.python.contracts import require

# Lie codes — fixed integer values to allow integer indexing in the solver.
LIE_CODES: dict[str, int] = {
    "tee": 0,
    "fairway": 1,
    "rough": 2,
    "trees": 3,
    "sand": 4,
    "water": 5,
    "ob": 6,
    "green": 7,
    "holed": 8,
}
LIE_NAMES: dict[int, str] = {v: k for k, v in LIE_CODES.items()}

# Priority order — higher index wins when polygons overlap. Green wins over
# everything except holed. (Spec §3.6.)
LIE_PRIORITY: tuple[str, ...] = (
    "tee",
    "fairway",
    "rough",
    "trees",
    "sand",
    "water",
    "ob",
    "green",
    "holed",
)


LieClass = Literal["tee", "fairway", "rough", "trees", "sand", "water", "ob", "green"]


@dataclass(frozen=True)
class RectFeature:
    """Axis-aligned rectangle in hole-frame yards. xmin/xmax along-target,
    ymin/ymax lateral (positive = left)."""

    lie: LieClass
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    def __post_init__(self) -> None:
        require(self.xmax > self.xmin and self.ymax > self.ymin, "degenerate rect")


@dataclass(frozen=True)
class CircleFeature:
    """Circle centered at (cx, cy) with radius (yards)."""

    lie: LieClass
    cx: float
    cy: float
    radius: float

    def __post_init__(self) -> None:
        require(self.radius > 0, "radius must be > 0")


Feature = RectFeature | CircleFeature


@dataclass(frozen=True)
class SyntheticHole:
    """A hand-coded hole specification — sufficient for Phase 1 tests."""

    name: str
    par: int
    tee: tuple[float, float]
    pin: tuple[float, float]
    bbox: tuple[float, float, float, float]  # xmin, xmax, ymin, ymax
    features: tuple[Feature, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        require(self.par >= 3, "par must be ≥ 3", self.par)
        xmin, xmax, ymin, ymax = self.bbox
        require(xmax > xmin and ymax > ymin, "degenerate bbox")
        require(xmin <= self.tee[0] <= xmax, "tee outside bbox")
        require(xmin <= self.pin[0] <= xmax, "pin outside bbox")


@dataclass(frozen=True)
class LieRaster:
    """Rasterized hole: integer lie codes on a uniform yard grid."""

    codes: NDArray[np.int8]  # shape (nx, ny)
    resolution_yd: float
    origin: tuple[float, float]  # (xmin, ymin) of grid cell (0, 0)
    pin: tuple[float, float]
    tee: tuple[float, float]

    def world_to_index(self, x: float, y: float) -> tuple[int, int]:
        ix = int((x - self.origin[0]) / self.resolution_yd)
        iy = int((y - self.origin[1]) / self.resolution_yd)
        return ix, iy

    def lie_at(self, x: float, y: float) -> int:
        nx, ny = self.codes.shape
        ix, iy = self.world_to_index(x, y)
        if 0 <= ix < nx and 0 <= iy < ny:
            return int(self.codes[ix, iy])
        return LIE_CODES["ob"]  # off-grid → OB

    @property
    def shape(self) -> tuple[int, int]:
        return self.codes.shape  # type: ignore[return-value]


def rasterize_synthetic(hole: SyntheticHole, resolution_yd: float = 1.0) -> LieRaster:
    """Rasterize a synthetic hole using point-in-polygon (no anti-aliasing).

    Anti-aliasing across hazard boundaries is *wrong* — see spec §3.6 note.
    """
    require(resolution_yd > 0, "resolution_yd must be > 0", resolution_yd)
    xmin, xmax, ymin, ymax = hole.bbox
    nx = int(np.ceil((xmax - xmin) / resolution_yd))
    ny = int(np.ceil((ymax - ymin) / resolution_yd))

    # Default everything to rough (the universal "off-fairway but in play" state).
    codes = np.full((nx, ny), LIE_CODES["rough"], dtype=np.int8)

    # Cell centres in world frame.
    xs = xmin + (np.arange(nx) + 0.5) * resolution_yd
    ys = ymin + (np.arange(ny) + 0.5) * resolution_yd
    xx, yy = np.meshgrid(xs, ys, indexing="ij")

    # Apply features in priority order so higher-priority lies overwrite lower.
    for lie in LIE_PRIORITY:
        if lie == "holed":
            continue
        lie_code = LIE_CODES[lie]
        for feat in hole.features:
            if feat.lie != lie:
                continue
            mask = _feature_mask(feat, xx, yy)
            codes[mask] = lie_code

    # Pin location → 'holed' cell.
    px, py = hole.pin
    pix = int((px - xmin) / resolution_yd)
    piy = int((py - ymin) / resolution_yd)
    if 0 <= pix < nx and 0 <= piy < ny:
        codes[pix, piy] = LIE_CODES["holed"]

    return LieRaster(
        codes=codes,
        resolution_yd=resolution_yd,
        origin=(xmin, ymin),
        pin=hole.pin,
        tee=hole.tee,
    )


def _feature_mask(
    feat: Feature, xx: NDArray[np.float64], yy: NDArray[np.float64]
) -> NDArray[np.bool_]:
    if isinstance(feat, RectFeature):
        return (
            (xx >= feat.xmin)
            & (xx <= feat.xmax)
            & (yy >= feat.ymin)
            & (yy <= feat.ymax)
        )
    return ((xx - feat.cx) ** 2 + (yy - feat.cy) ** 2) <= feat.radius**2
