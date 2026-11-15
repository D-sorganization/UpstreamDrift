"""Lofted procedural skin backend (epic #15150 track — distinct from SDF)."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.shared.python.grip_spec.types import GripSpec
from src.shared.python.hand_size import hand_mesh_scale
from src.shared.python.model_appearance.geometry import (
    Mesh,
    concat_meshes,
    ellipsoid_mesh,
)
from src.shared.python.model_appearance.grip_hand_mesh import FINGER_NAMES
from src.shared.python.model_appearance.grip_pose import (
    NEUTRAL_V_AZIMUTH_DEG,
    GripStyle,
    Hand,
)
from src.shared.python.model_appearance.hand_backends.contract import (
    HandJointAngles,
    HandRole,
    HandVisualResult,
)
from src.shared.python.model_appearance.hand_backends.metrics import palm_arch_angle_deg
from src.shared.python.model_appearance.hand_backends.procedural_loft import (
    lofted_anatomical_digit,
    lofted_palm_slab,
)
from src.shared.python.model_appearance.hand_backends.settings import (
    DEFAULT_HAND_VISUAL_SETTINGS,
    DEFAULT_VISUAL_GRIP_RADIUS_M,
    HandVisualSettings,
)

Array = NDArray[np.float64]
BACKEND_ID = "procedural_skin"

_ORDERED_FINGERS: tuple[str, ...] = ("little", "ring", "middle", "index")
_KNUCKLE_U_FRACTIONS: tuple[float, ...] = (-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0)
_PHALANX_RATIOS: dict[str, tuple[float, float, float]] = {
    "index": (0.4875, 0.2875, 0.2250),
    "middle": (0.4944, 0.2921, 0.2135),
    "ring": (0.4881, 0.2976, 0.2143),
    "little": (0.4714, 0.2857, 0.2429),
}


@dataclass(frozen=True)
class _BuiltProceduralHand:
    mesh: Mesh
    finger_meshes: dict[str, Mesh]
    palm_mesh: Mesh
    mcp_anchors: dict[str, Array]
    palm_mcp_targets: dict[str, Array]
    finger_joints: dict[str, Array]
    finger_radii: dict[str, tuple[float, float]]
    proximal_dirs: dict[str, Array]
    palm_length_m: float
    palm_width_m: float
    cup_depth_m: float


def _unit(vec: Array) -> Array:
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 1e-12 else np.array([1.0, 0.0, 0.0], dtype=float)


def _radial(theta: float) -> Array:
    return np.array([math.cos(theta), math.sin(theta), 0.0], dtype=float)


def _tangential(theta: float) -> Array:
    return np.array([-math.sin(theta), math.cos(theta), 0.0], dtype=float)


def _rodrigues(vec: Array, axis: Array, angle_rad: float) -> Array:
    k = _unit(axis)
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return vec * c + np.cross(k, vec) * s + k * (float(np.dot(k, vec)) * (1.0 - c))


def _segment_skin_clearance(
    p0: Array,
    p1: Array,
    r0: float,
    r1: float,
    grip_radius_m: float,
    palmar_scale: float,
) -> float:
    ts = np.linspace(0.15, 1.0, 9)
    pts = p0[None, :] + ts[:, None] * (p1 - p0)[None, :]
    radii = r0 * (1.0 - ts) + r1 * ts
    rho = np.hypot(pts[:, 0], pts[:, 1])
    return float(np.min(rho - palmar_scale * radii - grip_radius_m))


def _solve_flex_angle(
    p_start: Array,
    u_in: Array,
    flex_axis: Array,
    length: float,
    r0: float,
    r1: float,
    grip_radius_m: float,
    target_clearance_m: float,
    palmar_scale: float,
) -> float:
    """Solve joint flexion (rad) bringing phalanx palmar skin onto ``target_clearance``."""
    thetas = np.linspace(0.05, 1.65, 24)
    gaps = np.array(
        [
            _segment_skin_clearance(
                p_start,
                p_start + length * _rodrigues(u_in, flex_axis, float(th)),
                r0,
                r1,
                grip_radius_m,
                palmar_scale,
            )
            for th in thetas
        ],
        dtype=float,
    )
    # Find the earliest crossing down to target_clearance_m, or closest safe point.
    crossings = np.where(gaps <= target_clearance_m)[0]
    if len(crossings) > 0:
        idx = int(crossings[0])
        if idx == 0:
            return float(thetas[0])
        lo, hi = float(thetas[idx - 1]), float(thetas[idx])
        for _ in range(14):
            mid = 0.5 * (lo + hi)
            g = _segment_skin_clearance(
                p_start,
                p_start + length * _rodrigues(u_in, flex_axis, mid),
                r0,
                r1,
                grip_radius_m,
                palmar_scale,
            )
            if g > target_clearance_m:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)
    valid = np.where(gaps >= -0.0002)[0]
    if len(valid) > 0:
        best = valid[int(np.argmin(np.abs(gaps[valid] - target_clearance_m)))]
        return float(thetas[best])
    return float(thetas[int(np.argmin(np.abs(gaps - target_clearance_m)))])


def _solve_finger_chain(
    mcp: Array,
    u_flat: Array,
    dorsal: Array,
    lengths: tuple[float, float, float],
    base_r: float,
    tip_r: float,
    grip_radius_m: float,
    target_clearance_m: float,
    depth_factor: float,
    palmar_scale: float,
) -> tuple[Array, Array]:
    total_len = sum(lengths)
    cumulative = (
        0.0,
        lengths[0] / total_len,
        (lengths[0] + lengths[1]) / total_len,
        1.0,
    )
    # Keep flexion axis consistent across all phalanges so lead never hyperextends.
    flex_axis = _unit(np.cross(u_flat, -dorsal))
    pts: list[Array] = [mcp]
    p_curr = mcp.copy()
    u_curr = u_flat.copy()
    prox_dir = u_flat.copy()
    for k, seg_len in enumerate(lengths):
        r0 = base_r * (1.0 - cumulative[k]) + tip_r * cumulative[k]
        r1 = base_r * (1.0 - cumulative[k + 1]) + tip_r * cumulative[k + 1]
        th_closed = _solve_flex_angle(
            p_curr,
            u_curr,
            flex_axis,
            seg_len,
            r0,
            r1,
            grip_radius_m,
            target_clearance_m,
            palmar_scale,
        )
        th_open = 0.05 if k == 0 else 0.04
        th = (1.0 - depth_factor) * th_open + depth_factor * th_closed
        u_next = _rodrigues(u_curr, flex_axis, th)
        if k == 0:
            prox_dir = u_next.copy()
        p_curr = p_curr + seg_len * u_next
        pts.append(p_curr.copy())
        u_curr = u_next
    return np.asarray(pts, dtype=float), prox_dir


def _mirror_mesh_y(mesh: Mesh) -> Mesh:
    verts = mesh.vertices.copy()
    verts[:, 1] = -verts[:, 1]
    faces = mesh.faces[:, [0, 2, 1]].copy()
    return Mesh(verts, faces)


def _mirror_point_y(pt: Array) -> Array:
    out = np.asarray(pt, dtype=float).copy()
    out[1] = -out[1]
    return out


def _build_canonical_hand(
    grip_spec: GripSpec,
    side: HandRole,
    joint_angles: HandJointAngles,
    *,
    grip_radius_m: float,
    settings: HandVisualSettings,
) -> _BuiltProceduralHand:
    anthro = settings.anthropometry
    canon_spec = (
        replace(grip_spec, handedness="right")
        if grip_spec.handedness == "left"
        else grip_spec
    )
    hand = Hand.LEAD if side == "lead" else Hand.TRAIL
    curl = -1.0 if side == "lead" else 1.0
    pose = canon_spec.to_grip_pose()
    placement = pose.placement(hand)

    length_scale = hand_mesh_scale(grip_spec.hand_size.length_m)
    breadth_m = float(grip_spec.hand_size.breadth_m)
    palm_length_m = anthro.reference_palm_length_m * length_scale

    v_rad = math.radians(placement.v_azimuth_deg)
    mcp_az = v_rad + curl * math.radians(settings.mcp_from_v_deg)
    depth_factor = min(
        1.0,
        max(
            0.0,
            joint_angles.palm_finger_depth / settings.full_contact_palm_finger_depth,
        ),
    )
    open_lift = (1.0 - depth_factor) * settings.open_hand_lift_m * length_scale

    palm_thickness = settings.palm_thickness_m * length_scale
    wrist_thickness = settings.wrist_thickness_m * length_scale
    mid_base_r = anthro.base_radius_for("middle") * length_scale
    palmar_scale = 0.96 * settings.palmar_pad_flatness
    inner_floor = max(
        0.001, float(grip_radius_m) - settings.shaft_penetration_tolerance_m + 1e-6
    )

    cross_rad = math.radians(
        float(np.clip(joint_angles.club_cross_angle_deg, 50.0, 90.0))
    )
    width_axis = np.array([0.0, 0.0, 1.0], dtype=float)
    dorsal_mcp = _radial(mcp_az)
    curl_tan = curl * _tangential(mcp_az)
    palm_back_az = mcp_az - curl * math.radians(72.0)
    wrist_az = mcp_az - curl * math.radians(102.0)

    knuckle_rho = float(grip_radius_m) + 0.0012 + mid_base_r + open_lift
    knuckle_center = knuckle_rho * dorsal_mcp + np.array(
        [0.0, 0.0, 0.008 * length_scale], dtype=float
    )
    # Curve palm snugly around the V-side of the grip toward proximal wrist heel.
    mid_rho = float(grip_radius_m) + 0.0055 + 0.72 * palm_thickness + 0.7 * open_lift
    mid_raw = mid_rho * _radial(palm_back_az) + np.array(
        [0.0, 0.0, (0.008 - 0.007 * math.cos(cross_rad) / 0.309) * length_scale],
        dtype=float,
    )
    wrist_rho = float(grip_radius_m) + 0.038 * length_scale + 0.6 * open_lift
    wrist_raw = wrist_rho * _radial(wrist_az) + np.array(
        [0.0, 0.0, (-0.042 - 0.010 * (math.cos(cross_rad) - 0.309)) * length_scale],
        dtype=float,
    )
    ts_arc = np.linspace(0.0, 1.0, 18)
    raw_centers = np.array(
        [
            ((1.0 - t) ** 2) * wrist_raw
            + 2.0 * (1.0 - t) * t * mid_raw
            + (t**2) * knuckle_center
            for t in ts_arc
        ],
        dtype=float,
    )
    raw_arc = float(np.sum(np.linalg.norm(np.diff(raw_centers, axis=0), axis=1)))
    arc_scale = palm_length_m / max(raw_arc, 1e-9)
    mid_ctrl = knuckle_center + arc_scale * (mid_raw - knuckle_center)
    wrist_center = knuckle_center + arc_scale * (wrist_raw - knuckle_center)

    cup_depth = (
        settings.palmar_cup_base_m + settings.palmar_cup_curl_gain_m * depth_factor
    ) * length_scale

    palm_mesh, mcp_surface_arr, measured_cup, palm_arc_length_m = lofted_palm_slab(
        wrist_center=wrist_center,
        mid_ctrl=mid_ctrl,
        knuckle_center=knuckle_center,
        width_axis=width_axis,
        dorsal_axis=_radial(mcp_az - curl * math.radians(18.0)),
        wrist_width_m=breadth_m * settings.wrist_width_fraction,
        knuckle_width_m=breadth_m,
        wrist_thickness_m=wrist_thickness,
        palm_thickness_m=palm_thickness,
        knuckle_dome_height_m=settings.knuckle_dome_height_m * length_scale,
        cup_depth_m=cup_depth,
        knuckle_u_fractions=_KNUCKLE_U_FRACTIONS,
        sides=settings.procedural_skin_sides,
        grip_floor_m=inner_floor,
        mcp_span_fraction=settings.mcp_span_fraction,
    )

    mid_target = anthro.middle_finger_length_m * length_scale * 1.11
    finger_meshes: dict[str, Mesh] = {}
    mcp_anchors: dict[str, Array] = {}
    palm_mcp_targets: dict[str, Array] = {}
    finger_joints: dict[str, Array] = {}
    finger_radii: dict[str, tuple[float, float]] = {}
    proximal_dirs: dict[str, Array] = {}

    spread_deg = {"little": -4.0, "ring": -1.4, "middle": 1.2, "index": 4.0}
    for idx, finger in enumerate(_ORDERED_FINGERS):
        base_r = anthro.base_radius_for(finger) * length_scale
        tip_r = anthro.tip_radius_for(finger) * length_scale
        ratios = _PHALANX_RATIOS[finger]
        lengths = (
            mid_target * ratios[0],
            mid_target * ratios[1],
            mid_target * ratios[2],
        )
        anchor = mcp_surface_arr[idx].copy()
        palm_mcp_targets[finger] = anchor.copy()
        # Keep MCP anchor snug against the distal knuckle collar & outside grip.
        mcp_rho = float(np.hypot(anchor[0], anchor[1]))
        min_mcp_rho = float(grip_radius_m) + 0.0010 + base_r + open_lift
        if mcp_rho < min_mcp_rho:
            anchor[0] *= min_mcp_rho / max(mcp_rho, 1e-12)
            anchor[1] *= min_mcp_rho / max(mcp_rho, 1e-12)
        mcp_anchors[finger] = anchor

        splay = math.radians(spread_deg[finger])
        on_grip = not (
            pose.style is not GripStyle.TEN_FINGER
            and side == "trail"
            and finger == "little"
        )
        u_flat = _unit(
            curl_tan * 0.88
            + dorsal_mcp * 0.44
            + width_axis * (math.sin(splay) + (0.0 if on_grip else 0.14))
        )
        solve_radius_m = (
            float(grip_radius_m)
            if on_grip
            else float(grip_radius_m)
            + settings.overlap_finger_clearance_m * length_scale
        )
        joints, p_dir = _solve_finger_chain(
            anchor,
            u_flat,
            dorsal_mcp,
            lengths,
            base_r,
            tip_r,
            solve_radius_m,
            0.0009,
            depth_factor,
            palmar_scale,
        )
        finger_joints[finger] = joints
        finger_radii[finger] = (base_r, tip_r)
        proximal_dirs[finger] = p_dir

        socket_pt = anchor - u_flat * (settings.knuckle_socket_inset_m * length_scale)
        socket_rho = float(np.hypot(socket_pt[0], socket_pt[1]))
        min_socket_rho = float(grip_radius_m) + palmar_scale * base_r + 0.0008
        if socket_rho < min_socket_rho:
            socket_pt[:2] *= min_socket_rho / max(socket_rho, 1e-12)
        spine = np.vstack([socket_pt[None, :], joints])
        finger_meshes[finger] = lofted_anatomical_digit(
            spine,
            base_radius_m=base_r,
            tip_radius_m=tip_r,
            initial_width_axis=width_axis,
            dorsal_hint=dorsal_mcp,
            sides=settings.procedural_skin_sides,
            palmar_flatness=settings.palmar_pad_flatness,
            grip_floor_m=inner_floor,
        )

    # Thumb + thenar eminence emerging from the radial side of the palm.
    roll_delta = placement.v_azimuth_deg - NEUTRAL_V_AZIMUTH_DEG
    thumb_az = math.radians(placement.thumb_offset_deg + roll_delta)
    th_base_r = 0.5 * anthro.thumb_base_diameter_m * length_scale
    th_tip_r = 0.5 * anthro.thumb_tip_diameter_m * length_scale
    th_out = _radial(thumb_az)
    th_tan = curl * _tangential(thumb_az)
    th_rho = float(grip_radius_m) + 0.0011 + th_base_r * 0.90 + 0.6 * open_lift
    radial_palm = 0.50 * mid_ctrl + 0.25 * wrist_center + 0.25 * mcp_anchors["index"]
    th_cmc_target = (th_rho + 0.006 * length_scale) * th_out + np.array(
        [0.0, 0.0, float(radial_palm[2]) - 0.008 * length_scale], float
    )
    th_cmc = 0.70 * radial_palm + 0.30 * th_cmc_target
    cmc_rho = float(np.hypot(th_cmc[0], th_cmc[1]))
    min_cmc_rho = float(grip_radius_m) + 0.0015 + th_base_r
    if cmc_rho < min_cmc_rho:
        th_cmc[:2] *= min_cmc_rho / max(cmc_rho, 1e-12)
    z_base = float(mcp_anchors["index"][2])
    th_mcp = th_rho * th_out + np.array(
        [0.0, 0.0, z_base - 0.016 * length_scale], float
    )
    straight = min(1.0, max(0.0, joint_angles.thumb_straightness))
    th_ip = (
        th_rho * th_out
        + th_tan * ((1.0 - straight) * 0.012 * length_scale)
        + np.array(
            [0.0, 0.0, z_base + (0.006 + 0.005 * straight) * length_scale], float
        )
    )
    th_tip = (
        (th_rho - 0.0005 * length_scale)
        * _radial(thumb_az + curl * math.radians(12.0 * (1.0 - straight)))
        + th_tan * ((1.0 - straight) * 0.020 * length_scale)
        + np.array(
            [0.0, 0.0, z_base + (0.018 + 0.012 * straight) * length_scale], float
        )
    )
    thumb_mesh = lofted_anatomical_digit(
        np.asarray([th_cmc, th_mcp, th_ip, th_tip], float),
        base_radius_m=th_base_r * 0.90,
        tip_radius_m=th_tip_r,
        initial_width_axis=th_tan,
        dorsal_hint=th_out,
        sides=settings.procedural_skin_sides,
        palmar_flatness=settings.palmar_pad_flatness,
        joint_fractions=(0.45, 0.74),
        grip_floor_m=inner_floor,
    )
    thenar_r = np.asarray(settings.thenar_half_extents_m, float) * length_scale
    thenar_c = 0.55 * th_cmc + 0.45 * th_mcp
    thenar_rho = float(np.hypot(thenar_c[0], thenar_c[1]))
    min_thenar_rho = float(grip_radius_m) + 0.002 + float(thenar_r[1])
    if thenar_rho < min_thenar_rho:
        thenar_c[:2] *= min_thenar_rho / max(thenar_rho, 1e-12)
    thenar_mesh = ellipsoid_mesh(
        thenar_c,
        np.array([thenar_r[2], thenar_r[0], thenar_r[1]], float),
        _unit(th_mcp - th_cmc),
        rings=12,
        sides=settings.procedural_skin_sides,
        width_hint=_unit(radial_palm - mid_ctrl),
    )

    shells = [
        palm_mesh,
        finger_meshes["little"],
        finger_meshes["ring"],
        finger_meshes["middle"],
        finger_meshes["index"],
        thumb_mesh,
        thenar_mesh,
    ]
    combined = concat_meshes(*shells)
    if grip_spec.handedness == "left":
        combined = _mirror_mesh_y(combined)
        palm_mesh = _mirror_mesh_y(palm_mesh)
        finger_meshes = {k: _mirror_mesh_y(v) for k, v in finger_meshes.items()}
        mcp_anchors = {k: _mirror_point_y(v) for k, v in mcp_anchors.items()}
        palm_mcp_targets = {k: _mirror_point_y(v) for k, v in palm_mcp_targets.items()}
        finger_joints = {
            k: np.asarray([_mirror_point_y(p) for p in v], float)
            for k, v in finger_joints.items()
        }
        proximal_dirs = {k: _unit(_mirror_point_y(v)) for k, v in proximal_dirs.items()}

    return _BuiltProceduralHand(
        mesh=combined,
        finger_meshes=finger_meshes,
        palm_mesh=palm_mesh,
        mcp_anchors=mcp_anchors,
        palm_mcp_targets=palm_mcp_targets,
        finger_joints=finger_joints,
        finger_radii=finger_radii,
        proximal_dirs=proximal_dirs,
        palm_length_m=palm_arc_length_m,
        palm_width_m=breadth_m,
        cup_depth_m=measured_cup,
    )


def measure_procedural_hand_anthropometry(
    grip_spec: GripSpec,
    side: HandRole,
    *,
    grip_radius_m: float = DEFAULT_VISUAL_GRIP_RADIUS_M,
    settings: HandVisualSettings = DEFAULT_HAND_VISUAL_SETTINGS,
) -> dict[str, Any]:
    """Measure true anatomical dimensions, knuckle attachment, cup depth, and pad gaps."""
    angles = HandJointAngles.from_grip_spec(grip_spec, side)
    built = _build_canonical_hand(
        grip_spec,
        side,
        angles,
        grip_radius_m=float(grip_radius_m),
        settings=settings,
    )
    base_d = {f: 2.0 * built.finger_radii[f][0] for f in FINGER_NAMES}
    tip_d = {f: 2.0 * built.finger_radii[f][1] for f in FINGER_NAMES}
    mid_pts = built.finger_joints["middle"]
    mid_len = float(np.sum(np.linalg.norm(np.diff(mid_pts, axis=0), axis=1)))
    knuckle_gaps = {
        f: float(np.linalg.norm(built.mcp_anchors[f] - built.palm_mcp_targets[f]))
        for f in FINGER_NAMES
    }
    inter_gaps: list[float] = []
    for f0, f1 in zip(_ORDERED_FINGERS[:-1], _ORDERED_FINGERS[1:], strict=True):
        v0 = built.finger_meshes[f0].vertices[::4]
        v1 = built.finger_meshes[f1].vertices[::4]
        d = np.linalg.norm(v0[:, None, :] - v1[None, :, :], axis=2)
        inter_gaps.append(float(d.min()))

    pad_clearances: dict[str, float] = {}
    for f in FINGER_NAMES:
        verts = built.finger_meshes[f].vertices
        rho = np.hypot(verts[:, 0], verts[:, 1])
        pad_clearances[f] = float(rho.min() - float(grip_radius_m))

    return {
        "finger_base_diameters_m": base_d,
        "finger_tip_diameters_m": tip_d,
        "palm_width_m": built.palm_width_m,
        "palm_length_m": built.palm_length_m,
        "middle_finger_length_m": mid_len,
        "hand_length_m": built.palm_length_m + mid_len,
        "mcp_positions_m": {f: built.mcp_anchors[f].tolist() for f in FINGER_NAMES},
        "max_knuckle_gap_m": max(knuckle_gaps.values()),
        "min_inter_finger_gap_m": min(inter_gaps),
        "cup_depth_m": built.cup_depth_m,
        "proximal_directions": {
            f: built.proximal_dirs[f].tolist() for f in FINGER_NAMES
        },
        "pad_clearances_m": pad_clearances,
    }


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
    built = _build_canonical_hand(
        grip_spec,
        side,
        joint_angles,
        grip_radius_m=float(grip_radius_m),
        settings=settings,
    )
    arch = palm_arch_angle_deg(built.mesh)
    return HandVisualResult(
        backend_id=BACKEND_ID,
        role=side,
        mesh=built.mesh,
        grip_radius_m=float(grip_radius_m),
        material=settings.skin_material_for_role(side),
        palm_arch_angle_deg=arch,
    )


__all__ = [
    "BACKEND_ID",
    "build_hand_visual",
    "measure_procedural_hand_anthropometry",
]
