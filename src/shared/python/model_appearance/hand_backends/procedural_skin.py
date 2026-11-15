"""Lofted procedural skin (epic #15150 track — distinct from implicit SDF lane).

One Catmull–Rom tapered tube per digit, cupped palm shell, thenar pad, thumb sweep.
Does not use marching-cubes whole-hand SDF (see ``sdf_meshing`` for the SDF lane).
"""

from __future__ import annotations

import math

import numpy as np

from src.shared.python.grip_spec.types import GripSpec
from src.shared.python.hand_size import hand_mesh_scale
from src.shared.python.model_appearance.geometry import (
    Mesh,
    closed_grid_mesh,
    concat_meshes,
    ellipsoid_mesh,
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
from src.shared.python.model_appearance.hand_backends.sdf_meshing import (
    catmull_rom_spline,
    tube_mesh_from_polyline,
)
from src.shared.python.model_appearance.hand_backends.skeleton_extract import (
    finger_polyline,
    palm_z_extent,
    thumb_polyline,
    uniform_scale_skeleton,
)


BACKEND_ID = "procedural_skin"


def _enforce_minimum_radial_distance(
    mesh: Mesh,
    grip_radius_m: float,
    *,
    inner_floor_m: float,
) -> Mesh:
    verts = np.asarray(mesh.vertices, dtype=float).copy()
    rho = np.hypot(verts[:, 0], verts[:, 1])
    too_close = rho < inner_floor_m
    if not too_close.any():
        return mesh
    scale = inner_floor_m / np.maximum(rho[too_close], 1e-12)
    verts[too_close, 0] *= scale
    verts[too_close, 1] *= scale
    return Mesh(verts, mesh.faces.copy())


def _continuous_finger_tube(
    points: np.ndarray,
    *,
    base_radius_m: float,
    tip_radius_m: float,
    sides: int,
) -> Mesh:
    """One tapered tube along a Catmull–Rom spine (no segment bulbs)."""
    return tube_mesh_from_polyline(
        points,
        base_radius_m,
        tip_radius_m,
        sides=sides,
        spline_samples=max(56, 12 * len(points)),
    )


def _palm_cup_mesh(
    knuckles: np.ndarray,
    *,
    grip_radius_m: float,
    z_lo: float,
    joint_angles: HandJointAngles,
    anthro: object,
    settings: HandVisualSettings,
) -> Mesh:
    """Cupped metacarpal shell with visible knuckle ridge."""
    from src.shared.python.model_appearance.hand_backends.settings import (
        HandAnthropometryReference,
    )

    if not isinstance(anthro, HandAnthropometryReference):
        raise TypeError("anthro must be HandAnthropometryReference")
    curl = joint_angles.palm_finger_depth
    transverse = 0.003 + 0.005 * curl
    clearance = 0.0015
    inner = grip_radius_m + clearance
    width = anthro.reference_palm_width_m
    rings = 10
    sides = settings.procedural_skin_sides
    z_knuckle = float(np.mean(knuckles[:, 2]))
    zs = np.linspace(z_lo - 0.006, z_knuckle + 0.004, rings)
    theta = np.linspace(-0.5 * math.pi, 0.85 * math.pi, sides)
    grid: list[list[np.ndarray]] = []
    for frac, z in enumerate(zs):
        t = frac / max(1, rings - 1)
        arch = transverse * math.sin(math.pi * t)
        for th in theta:
            wave = 0.0018 * math.cos(4.0 * th)
            palmar = inner + 0.0035 + arch + wave
            dorsal_bump = 0.011 * (1.0 - t)
            rho = palmar + dorsal_bump
            taper_y = 0.5 + 0.5 * math.cos(th)
            rho += width * 0.04 * taper_y * (1.0 - t)
            grid.append(
                np.array([rho * math.cos(th), rho * math.sin(th), z], dtype=float)
            )
    arr = np.asarray(grid, dtype=float).reshape(rings, sides, 3)
    dorsal_pole = np.array(
        [
            inner + 0.024,
            0.0,
            0.5 * (z_lo + z_knuckle) + 0.006,
        ],
        dtype=float,
    )
    palmar_pole = np.mean(knuckles, axis=0) - np.array([0.006, 0.0, 0.0])
    return closed_grid_mesh(arr, dorsal_pole, palmar_pole)


def _thenar_pad(
    thumb_root: np.ndarray,
    hand: Hand,
    *,
    settings: HandVisualSettings,
    scale: float,
) -> Mesh:
    side = -1.0 if hand is Hand.LEAD else 1.0
    centre = thumb_root + np.array([0.003, side * 0.011, -0.004])
    return ellipsoid_mesh(
        centre,
        np.array([0.014, 0.012, 0.018], dtype=float) * scale,
        np.array([1.0, 0.0, 0.0], dtype=float),
        rings=settings.procedural_skin_sides,
        sides=settings.procedural_skin_sides,
    )


def _thumb_tube(
    thumb_line: np.ndarray,
    *,
    base_r: float,
    tip_r: float,
    sides: int,
) -> Mesh:
    dense = catmull_rom_spline(thumb_line, 40)
    return tube_mesh_from_polyline(dense, base_r, tip_r, sides=sides, spline_samples=40)


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
    finger_lines = [finger_polyline(reference, f) for f in FINGER_NAMES]
    thumb_line = thumb_polyline(reference)
    z_lo, z_hi = palm_z_extent(reference)
    measured_len = float(reference.mesh.vertices[:, 2].max()) - float(
        reference.mesh.vertices[:, 2].min()
    )
    finger_lines = uniform_scale_skeleton(
        finger_lines,
        target_length_m=settings.anthropometry.reference_hand_length_m,
        measured_length_m=measured_len,
    )
    thumb_len = float(np.linalg.norm(thumb_line[-1] - thumb_line[0]))
    thumb_line = uniform_scale_skeleton(
        [thumb_line],
        target_length_m=settings.anthropometry.reference_hand_length_m * 0.44,
        measured_length_m=thumb_len,
    )[0]
    anthro = settings.anthropometry
    base_r = anthro.finger_radius_base_m
    tip_r = anthro.finger_radius_tip_m
    shells: list[Mesh] = []
    knuckles: list[np.ndarray] = []
    for line in finger_lines:
        knuckles.append(line[0].copy())
        shells.append(
            _continuous_finger_tube(
                line,
                base_radius_m=base_r,
                tip_radius_m=tip_r,
                sides=settings.procedural_skin_sides,
            )
        )
    knuckle_arr = np.vstack(knuckles)
    shells.append(
        _palm_cup_mesh(
            knuckle_arr,
            grip_radius_m=float(grip_radius_m),
            z_lo=z_lo,
            joint_angles=joint_angles,
            anthro=anthro,
            settings=settings,
        )
    )
    shells.append(
        _thenar_pad(
            thumb_line[0],
            hand,
            settings=settings,
            scale=scale,
        )
    )
    shells.append(
        _thumb_tube(
            thumb_line,
            base_r=base_r * 1.06,
            tip_r=tip_r * 0.94,
            sides=settings.procedural_skin_sides,
        )
    )
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
        material=settings.skin_material_for_role(side),
        palm_arch_angle_deg=arch,
    )


__all__ = ["BACKEND_ID", "build_hand_visual"]
