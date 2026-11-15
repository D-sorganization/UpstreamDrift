"""Procedural anatomical skin lofted over the capsule skeleton."""

from __future__ import annotations

import math

import numpy as np

from src.shared.python.grip_spec.types import GripSpec
from src.shared.python.hand_size import hand_mesh_scale
from src.shared.python.model_appearance.geometry import (
    LoftOptions,
    Mesh,
    closed_grid_mesh,
    concat_meshes,
    ellipsoid_mesh,
    lofted_segment,
)
from src.shared.python.model_appearance.grip_hand_mesh import (
    FINGER_NAMES,
    HandMeshParams,
    build_grip_hand_mesh,
)
from src.shared.python.model_appearance.grip_pose import Hand
from src.shared.python.model_appearance.hand_backends.contract import (
    HandJointAngles,
    HandRole,
    HandVisualResult,
)
from src.shared.python.model_appearance.hand_backends.metrics import palm_arch_angle_deg
from src.shared.python.model_appearance.hand_backends.settings import (
    DEFAULT_HAND_VISUAL_SETTINGS,
    HandVisualSettings,
)

BACKEND_ID = "procedural_skin"


def _enforce_minimum_radial_distance(
    mesh: Mesh,
    grip_radius_m: float,
    *,
    inner_floor_m: float,
) -> Mesh:
    """Push any vertex strictly inside ``inner_floor_m`` radially outward."""
    verts = np.asarray(mesh.vertices, dtype=float).copy()
    rho = np.hypot(verts[:, 0], verts[:, 1])
    too_close = rho < inner_floor_m
    if not too_close.any():
        return mesh
    scale = inner_floor_m / np.maximum(rho[too_close], 1e-12)
    verts[too_close, 0] *= scale
    verts[too_close, 1] *= scale
    return Mesh(verts, mesh.faces.copy())


def _component_axis_radius(
    vertices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    centroid = vertices.mean(axis=0)
    centered = vertices - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    axis = axis / np.linalg.norm(axis)
    projections = centered @ axis
    start = centroid + axis * float(projections.min())
    end = centroid + axis * float(projections.max())
    radial = np.linalg.norm(centered - np.outer(projections, axis), axis=1)
    radius = float(np.percentile(radial, 92))
    return start, end, max(radius, 1e-4)


def _smooth_segment(
    start: np.ndarray,
    end: np.ndarray,
    radius: float,
    *,
    rings: int,
    sides: int,
    knuckle_scale: float = 1.12,
    pad_scale: float = 0.92,
) -> Mesh:
    mid = 0.5 * (start + end)
    knuckle = lofted_segment(
        start,
        mid,
        radius * knuckle_scale,
        LoftOptions(rings=rings, sides=sides, end_scale=pad_scale),
    )
    pad = lofted_segment(
        mid,
        end,
        radius * pad_scale,
        LoftOptions(rings=max(4, rings - 2), sides=sides, end_scale=0.85),
    )
    return concat_meshes(knuckle, pad)


def _arched_palm_mesh(
    grip_radius_m: float,
    baked_mesh: Mesh,
    joint_angles: HandJointAngles,
    params: HandMeshParams,
    settings: HandVisualSettings,
) -> Mesh:
    """Deformable palm grid with transverse and longitudinal arches."""
    verts = baked_mesh.vertices
    z_lo, z_hi = float(verts[:, 2].min()), float(verts[:, 2].max())
    rho_lo = float(np.percentile(np.hypot(verts[:, 0], verts[:, 1]), 15))
    rho_hi = float(np.percentile(np.hypot(verts[:, 0], verts[:, 1]), 85))
    curl = joint_angles.palm_finger_depth
    transverse = 0.004 + 0.010 * curl
    longitudinal = 0.003 + 0.008 * (1.0 - curl)
    rings = settings.palm_grid_rings
    sides = settings.palm_grid_sides
    theta = np.linspace(-0.35, 1.45, sides)
    stations = np.linspace(z_lo, z_hi, rings)
    inner = grip_radius_m + params.palm_clearance_m
    thickness = params.scale * params.palm_thickness_m
    grid = []
    for frac, z in enumerate(stations):
        t = frac / max(1, rings - 1)
        for th in theta:
            arch_t = transverse * math.sin(math.pi * t)
            arch_l = longitudinal * math.sin(math.pi * (th / math.pi))
            r = max(
                inner + 0.0005,
                inner + thickness * (0.55 + 0.45 * math.sin(th)) + arch_t + arch_l,
            )
            grid.append(
                np.array(
                    [r * math.cos(th), r * math.sin(th), z],
                    dtype=float,
                )
            )
    arr = np.asarray(grid, dtype=float).reshape(rings, sides, 3)
    outer = arr[:, arr.shape[1] // 4, :]
    top = outer.mean(axis=0) + np.array([0.0, 0.0, -0.008], dtype=float)
    bottom = arr[-1].mean(axis=0) + np.array([0.006, 0.0, 0.006], dtype=float)
    palm = closed_grid_mesh(arr, top, bottom)
    dorsal_centre = np.array(
        [
            0.85 * (rho_lo + rho_hi) * 0.5,
            0.0,
            0.5 * (z_lo + z_hi) + 0.012,
        ],
        dtype=float,
    )
    dorsum = ellipsoid_mesh(
        dorsal_centre,
        np.array([0.022, 0.028, 0.040], dtype=float) * params.scale,
        np.array([0.0, 0.0, 1.0], dtype=float),
        rings=settings.procedural_skin_rings,
        sides=settings.procedural_skin_sides,
    )
    return concat_meshes(palm, dorsum)


def _thenar_eminences(
    grip_radius_m: float,
    params: HandMeshParams,
    hand: Hand,
    settings: HandVisualSettings,
) -> list[Mesh]:
    k = params.scale
    thumb_side = -1.0 if hand is Hand.LEAD else 1.0
    thenar = ellipsoid_mesh(
        np.array(
            [grip_radius_m + 0.020, thumb_side * 0.014, -0.008],
            dtype=float,
        ),
        np.array([0.016, 0.014, 0.022], dtype=float) * k,
        np.array([1.0, 0.0, 0.0], dtype=float),
        rings=settings.procedural_skin_rings,
        sides=settings.procedural_skin_sides,
    )
    hypo = ellipsoid_mesh(
        np.array(
            [grip_radius_m + 0.014, -thumb_side * 0.026, 0.018],
            dtype=float,
        ),
        np.array([0.012, 0.011, 0.018], dtype=float) * k,
        np.array([0.0, 1.0, 0.0], dtype=float),
        rings=settings.procedural_skin_rings,
        sides=settings.procedural_skin_sides,
    )
    return [thenar, hypo]


def build_hand_visual(
    grip_spec: GripSpec,
    side: HandRole,
    joint_angles: HandJointAngles,
    *,
    grip_radius_m: float,
    settings: HandVisualSettings = DEFAULT_HAND_VISUAL_SETTINGS,
) -> HandVisualResult:
    if not isinstance(joint_angles, HandJointAngles):
        raise TypeError("joint_angles must be HandJointAngles")
    hand = Hand.LEAD if side == "lead" else Hand.TRAIL
    pose = grip_spec.to_grip_pose()
    scale = hand_mesh_scale(grip_spec.hand_size.length_m)
    params = HandMeshParams(scale=scale, sides=settings.procedural_skin_sides)
    reference = build_grip_hand_mesh(
        hand,
        grip_radius_m=float(grip_radius_m),
        pose=pose,
        params=params,
    )
    shells: list[Mesh] = []
    for comp in reference.components:
        if comp.finger in {*FINGER_NAMES, "thumb"} and "_" in comp.name:
            start, end, radius = _component_axis_radius(comp.mesh.vertices)
            shells.append(
                _smooth_segment(
                    start,
                    end,
                    radius,
                    rings=settings.procedural_skin_rings,
                    sides=settings.procedural_skin_sides,
                )
            )
        elif comp.finger == "palm":
            shells.append(
                _arched_palm_mesh(
                    float(grip_radius_m),
                    comp.mesh,
                    joint_angles,
                    params,
                    settings,
                )
            )
    shells.extend(_thenar_eminences(float(grip_radius_m), params, hand, settings))
    mesh = concat_meshes(*shells)
    inner_floor = float(grip_radius_m) - settings.shaft_penetration_tolerance_m + 1e-9
    mesh = _enforce_minimum_radial_distance(
        mesh, float(grip_radius_m), inner_floor_m=inner_floor
    )
    arch = palm_arch_angle_deg(mesh)
    return HandVisualResult(
        backend_id=BACKEND_ID,
        role=side,
        mesh=mesh,
        grip_radius_m=float(grip_radius_m),
        material=settings.skin_material,
        palm_arch_angle_deg=arch,
    )


__all__ = ["BACKEND_ID", "build_hand_visual"]
