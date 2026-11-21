"""MuJoCo scene overlays for segment kinetic-energy globes (#15389)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.engines.physics_engines.mujoco.python.visual_layer import add_scene_marker
from src.shared.python.segment_kinetic_energy.contracts import (
    SegmentKineticEnergyFrame,
    SegmentKineticEnergyPeak,
)
from src.shared.python.segment_kinetic_energy.render_globes import (
    globe_rgba_for_ke,
    radius_for_ke,
)
from src.shared.python.segment_kinetic_energy.shading import (
    REFERENCE_SWING_PEAK_SEGMENT_KE_J,
)

__all__ = [
    "GrfPlateOutline",
    "GlobeDrawReceipt",
    "add_com_rgb_axes_to_scene",
    "add_grf_plate_outlines_to_scene",
    "add_ke_globes_to_scene",
]


@dataclass(frozen=True)
class GrfPlateOutline:
    corners_m: tuple[tuple[float, float, float], ...]
    rgba: tuple[float, float, float, float] = (0.45, 0.45, 0.5, 0.35)


@dataclass(frozen=True)
class GlobeDrawReceipt:
    globes_drawn: int
    peak_highlights: int
    axes_drawn: int
    plates_drawn: int


def _peak_lookup(
    peaks: Sequence[SegmentKineticEnergyPeak],
) -> dict[str, float]:
    return {p.body: p.peak_total_j for p in peaks}


def add_ke_globes_to_scene(
    scene: Any,
    frame: SegmentKineticEnergyFrame,
    peaks: Sequence[SegmentKineticEnergyPeak],
    *,
    max_radius_m: float = 0.22,
) -> GlobeDrawReceipt:
    _peak_lookup(peaks)
    ref_peak = REFERENCE_SWING_PEAK_SEGMENT_KE_J
    globes = 0
    for seg in frame.segments:
        if seg.total_j <= 0.0:
            continue
        radius = radius_for_ke(
            seg.total_j, reference_peak_j=ref_peak, max_radius_m=max_radius_m
        )
        rgba = globe_rgba_for_ke(seg.total_j, reference_peak_j=ref_peak)
        add_scene_marker(scene, np.asarray(seg.com_m, dtype=float), radius, rgba)
        globes += 1
    return GlobeDrawReceipt(
        globes_drawn=globes, peak_highlights=0, axes_drawn=0, plates_drawn=0
    )


def add_com_rgb_axes_to_scene(
    scene: Any,
    com_m: Sequence[float],
    rot3x3: np.ndarray,
    *,
    length_m: float = 0.18,
) -> int:
    import mujoco

    origin = np.asarray(com_m, dtype=float)
    rot = np.asarray(rot3x3, dtype=float).reshape(3, 3)
    colors = (
        (1.0, 0.2, 0.2, 0.95),
        (0.2, 0.9, 0.2, 0.95),
        (0.2, 0.4, 1.0, 0.95),
    )
    drawn = 0
    for axis_idx, rgba in enumerate(colors):
        direction = rot[:, axis_idx]
        end = origin + direction * length_m
        if scene.ngeom >= scene.maxgeom:
            break
        geom = scene.geoms[scene.ngeom]
        width = 0.004
        if hasattr(mujoco, "mjv_connector"):
            mujoco.mjv_connector(
                geom,
                mujoco.mjtGeom.mjGEOM_ARROW,
                width,
                origin,
                end,
            )
        else:
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_ARROW,
                np.array([width, width, length_m]),
                origin,
                rot,
                np.asarray(rgba, dtype=np.float32),
            )
        geom.rgba[:] = rgba
        scene.ngeom += 1
        drawn += 1
    return drawn


def add_grf_plate_outlines_to_scene(
    scene: Any,
    plates: Sequence[GrfPlateOutline],
) -> int:
    import mujoco

    drawn = 0
    for plate in plates:
        corners = plate.corners_m
        if len(corners) < 4:
            continue
        for i in range(4):
            start = np.asarray(corners[i], dtype=float)
            end = np.asarray(corners[(i + 1) % 4], dtype=float)
            if scene.ngeom >= scene.maxgeom:
                return drawn
            geom = scene.geoms[scene.ngeom]
            if hasattr(mujoco, "mjv_connector"):
                mujoco.mjv_connector(
                    geom,
                    mujoco.mjtGeom.mjGEOM_ARROW,
                    0.003,
                    start,
                    end,
                )
            else:
                mujoco.mjv_initGeom(
                    geom,
                    mujoco.mjtGeom.mjGEOM_ARROW,
                    np.array([0.003, 0.003, float(np.linalg.norm(end - start))]),
                    start,
                    np.eye(3).reshape(-1),
                    np.asarray(plate.rgba, dtype=np.float32),
                )
            geom.rgba[:] = plate.rgba
            scene.ngeom += 1
            drawn += 1
    return drawn
