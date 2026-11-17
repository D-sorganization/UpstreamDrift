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
    HandPlacement,
)
from src.shared.python.model_appearance.hand_backends.capsule import (
    rolled_about_shaft,
    skin_rgb_for,
)
from src.shared.python.model_appearance.hand_backends.contract import (
    GripArticulation,
)
from src.shared.python.model_appearance.hand_backends.joint_angles import (
    HandJointAngles,
)
from src.shared.python.model_appearance.hand_backends.registry import (
    HandVisualBackend,
    register_hand_visual_backend,
)
from src.shared.python.model_appearance.hand_backends.procedural_finger_solve import (
    FingerSolveSpec,
    radial,
    solve_finger_chain,
    tangential,
    unit,
)
from src.shared.python.model_appearance.hand_backends.procedural_loft import (
    PalmSlabSpec,
    lofted_anatomical_digit,
    lofted_palm_slab,
)
from src.shared.python.model_appearance.hand_backends.settings import (
    DEFAULT_HAND_VISUAL_SETTINGS,
    DEFAULT_VISUAL_GRIP_RADIUS_M,
    HandVisualSettings,
)
from src.shared.python.model_appearance.hand_backends.types import (
    HandSide,
    HandVisualResult,
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


def _mirror_mesh_y(mesh: Mesh) -> Mesh:
    verts = mesh.vertices.copy()
    verts[:, 1] = -verts[:, 1]
    faces = mesh.faces[:, [0, 2, 1]].copy()
    return Mesh(verts, faces)


def _mirror_point_y(pt: Array) -> Array:
    out = np.asarray(pt, dtype=float).copy()
    out[1] = -out[1]
    return out


@dataclass(frozen=True)
class _HandBuildContext:
    """Pose, scale and clearance terms shared by every part of one canonical hand."""

    settings: HandVisualSettings
    side: HandSide
    style: GripStyle
    placement: HandPlacement
    curl: float
    length_scale: float
    breadth_m: float
    grip_radius_m: float
    depth_factor: float
    open_lift: float
    palmar_scale: float
    inner_floor: float
    mcp_az: float
    width_axis: Array
    dorsal_mcp: Array
    curl_tan: Array


@dataclass(frozen=True)
class _PalmArc:
    """Quadratic Bezier wrist -> knuckle centre-line controls (arc-length scaled)."""

    wrist_center: Array
    mid_ctrl: Array
    knuckle_center: Array


@dataclass(frozen=True)
class _BuiltFinger:
    """One solved and lofted long finger."""

    mesh: Mesh
    anchor: Array
    palm_target: Array
    joints: Array
    radii: tuple[float, float]
    proximal_dir: Array


def _hand_build_context(
    grip_spec: GripSpec,
    side: HandSide,
    joint_angles: GripArticulation,
    *,
    grip_radius_m: float,
    settings: HandVisualSettings,
) -> _HandBuildContext:
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
    v_rad = math.radians(placement.v_azimuth_deg)
    mcp_az = v_rad + curl * math.radians(settings.mcp_from_v_deg)
    depth_factor = min(
        1.0,
        max(
            0.0,
            joint_angles.palm_finger_depth / settings.full_contact_palm_finger_depth,
        ),
    )
    return _HandBuildContext(
        settings=settings,
        side=side,
        style=pose.style,
        placement=placement,
        curl=curl,
        length_scale=length_scale,
        breadth_m=float(grip_spec.hand_size.breadth_m),
        grip_radius_m=float(grip_radius_m),
        depth_factor=depth_factor,
        open_lift=(1.0 - depth_factor) * settings.open_hand_lift_m * length_scale,
        palmar_scale=0.96 * settings.palmar_pad_flatness,
        inner_floor=max(
            0.001, float(grip_radius_m) - settings.shaft_penetration_tolerance_m + 1e-6
        ),
        mcp_az=mcp_az,
        width_axis=np.array([0.0, 0.0, 1.0], dtype=float),
        dorsal_mcp=radial(mcp_az),
        curl_tan=curl * tangential(mcp_az),
    )


def _palm_arc(ctx: _HandBuildContext, joint_angles: GripArticulation) -> _PalmArc:
    """Curve palm snugly around the V-side of the grip toward proximal wrist heel."""
    settings, ls, curl = ctx.settings, ctx.length_scale, ctx.curl
    palm_length_m = settings.anthropometry.reference_palm_length_m * ls
    palm_thickness = settings.palm_thickness_m * ls
    mid_base_r = settings.anthropometry.base_radius_for("middle") * ls
    cross_rad = math.radians(
        float(np.clip(joint_angles.club_cross_angle_deg, 50.0, 90.0))
    )
    palm_back_az = ctx.mcp_az - curl * math.radians(72.0)
    wrist_az = ctx.mcp_az - curl * math.radians(102.0)

    knuckle_rho = ctx.grip_radius_m + 0.0012 + mid_base_r + ctx.open_lift
    knuckle_center = knuckle_rho * ctx.dorsal_mcp + np.array(
        [0.0, 0.0, 0.008 * ls], dtype=float
    )
    mid_rho = ctx.grip_radius_m + 0.0055 + 0.72 * palm_thickness + 0.7 * ctx.open_lift
    mid_raw = mid_rho * radial(palm_back_az) + np.array(
        [0.0, 0.0, (0.008 - 0.007 * math.cos(cross_rad) / 0.309) * ls],
        dtype=float,
    )
    wrist_rho = ctx.grip_radius_m + 0.038 * ls + 0.6 * ctx.open_lift
    wrist_raw = wrist_rho * radial(wrist_az) + np.array(
        [0.0, 0.0, (-0.042 - 0.010 * (math.cos(cross_rad) - 0.309)) * ls],
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
    return _PalmArc(
        wrist_center=knuckle_center + arc_scale * (wrist_raw - knuckle_center),
        mid_ctrl=knuckle_center + arc_scale * (mid_raw - knuckle_center),
        knuckle_center=knuckle_center,
    )


def _palm_slab(
    ctx: _HandBuildContext, arc: _PalmArc
) -> tuple[Mesh, Array, float, float]:
    settings, ls = ctx.settings, ctx.length_scale
    cup_depth = (
        settings.palmar_cup_base_m + settings.palmar_cup_curl_gain_m * ctx.depth_factor
    ) * ls
    return lofted_palm_slab(
        PalmSlabSpec(
            wrist_center=arc.wrist_center,
            mid_ctrl=arc.mid_ctrl,
            knuckle_center=arc.knuckle_center,
            width_axis=ctx.width_axis,
            dorsal_axis=radial(ctx.mcp_az - ctx.curl * math.radians(18.0)),
            wrist_width_m=ctx.breadth_m * settings.wrist_width_fraction,
            knuckle_width_m=ctx.breadth_m,
            wrist_thickness_m=settings.wrist_thickness_m * ls,
            palm_thickness_m=settings.palm_thickness_m * ls,
            knuckle_dome_height_m=settings.knuckle_dome_height_m * ls,
            cup_depth_m=cup_depth,
            knuckle_u_fractions=_KNUCKLE_U_FRACTIONS,
            sides=settings.procedural_skin_sides,
            grip_floor_m=ctx.inner_floor,
            mcp_span_fraction=settings.mcp_span_fraction,
            interdigital_valley_m=settings.interdigital_valley_m * ls,
            distal_cap_shift_m=settings.distal_cap_shift_m * ls,
        )
    )


_FINGER_SPREAD_DEG: dict[str, float] = {
    "little": -4.0,
    "ring": -1.4,
    "middle": 1.2,
    "index": 4.0,
}


def _build_finger(
    ctx: _HandBuildContext,
    finger: str,
    palm_target: Array,
    mid_ctrl: Array,
) -> _BuiltFinger:
    """Seat, solve and loft one long finger from its palm MCP surface anchor."""
    settings, ls = ctx.settings, ctx.length_scale
    anthro = settings.anthropometry
    grip_radius_m = ctx.grip_radius_m
    mid_target = anthro.middle_finger_length_m * ls * 1.11
    base_r = anthro.base_radius_for(finger) * ls
    tip_r = anthro.tip_radius_for(finger) * ls
    ratios = _PHALANX_RATIOS[finger]
    lengths = (
        mid_target * ratios[0],
        mid_target * ratios[1],
        mid_target * ratios[2],
    )
    anchor = palm_target.copy()
    # Keep MCP anchor snug against the distal knuckle collar & outside grip.
    mcp_rho = float(np.hypot(anchor[0], anchor[1]))
    min_mcp_rho = grip_radius_m + 0.0010 + base_r + ctx.open_lift
    if mcp_rho < min_mcp_rho:
        anchor[0] *= min_mcp_rho / max(mcp_rho, 1e-12)
        anchor[1] *= min_mcp_rho / max(mcp_rho, 1e-12)

    splay = math.radians(_FINGER_SPREAD_DEG[finger])
    on_grip = not (
        ctx.style is not GripStyle.TEN_FINGER
        and ctx.side == "trail"
        and finger == "little"
    )
    u_flat = unit(
        ctx.curl_tan * 0.94
        + ctx.dorsal_mcp * 0.34
        + ctx.width_axis * (math.sin(splay) + (0.0 if on_grip else 0.14))
    )
    solve_radius_m = (
        grip_radius_m
        if on_grip
        else grip_radius_m + settings.overlap_finger_clearance_m * ls
    )
    joints, p_dir = solve_finger_chain(
        anchor,
        u_flat,
        ctx.dorsal_mcp,
        lengths,
        base_r,
        tip_r,
        FingerSolveSpec(
            grip_radius_m=solve_radius_m,
            target_clearance_m=0.0005 if on_grip else 0.0009,
            depth_factor=ctx.depth_factor,
            palmar_scale=ctx.palmar_scale,
            distal_pad_t_start=settings.distal_pad_t_start,
            stage_clearances_m=(
                (0.0018, 0.0016, 0.0005) if on_grip else (0.0009, 0.0009, 0.0009)
            ),
        ),
    )

    inward_dir = unit(0.60 * u_flat + 0.40 * unit(anchor - mid_ctrl))
    socket_pt = (
        anchor
        - inward_dir * (settings.knuckle_socket_inset_m * ls)
        - ctx.dorsal_mcp * (0.28 * base_r)
    )
    socket_rho = float(np.hypot(socket_pt[0], socket_pt[1]))
    min_socket_rho = grip_radius_m + ctx.palmar_scale * base_r + 0.0008
    if socket_rho < min_socket_rho:
        socket_pt[:2] *= min_socket_rho / max(socket_rho, 1e-12)
    spine = np.vstack([socket_pt[None, :], joints])
    mesh = lofted_anatomical_digit(
        spine,
        radii_m=(base_r, tip_r),
        initial_width_axis=ctx.width_axis,
        dorsal_hint=ctx.dorsal_mcp,
        sides=settings.procedural_skin_sides,
        palmar_flatness=settings.palmar_pad_flatness,
        grip_floor_m=ctx.inner_floor,
    )
    return _BuiltFinger(
        mesh=mesh,
        anchor=anchor,
        palm_target=palm_target.copy(),
        joints=joints,
        radii=(base_r, tip_r),
        proximal_dir=p_dir,
    )


def _thumb_joints(
    ctx: _HandBuildContext, joint_angles: GripArticulation, z_base: float
) -> Array:
    """CMC, MCP, IP and tip centres of the thumb wrapping the radial grip side."""
    settings, ls, curl = ctx.settings, ctx.length_scale, ctx.curl
    anthro = settings.anthropometry
    grip_radius_m, palmar_scale, open_lift = (
        ctx.grip_radius_m,
        ctx.palmar_scale,
        ctx.open_lift,
    )
    roll_delta = ctx.placement.v_azimuth_deg - NEUTRAL_V_AZIMUTH_DEG
    thumb_az = math.radians(ctx.placement.thumb_offset_deg + roll_delta)
    th_base_r = 0.5 * anthro.thumb_base_diameter_m * ls
    th_tip_r = 0.5 * anthro.thumb_tip_diameter_m * ls
    opp = settings.thumb_base_opposition_rad
    straight = min(1.0, max(0.0, joint_angles.thumb_straightness))
    th_cmc_az = thumb_az - curl * 0.65 * opp
    th_mcp_az = thumb_az - curl * 0.30 * opp
    th_ip_az = thumb_az + curl * (0.20 * opp + (1.0 - straight) * 0.16)
    th_tip_az = thumb_az + curl * (0.58 * opp + (1.0 - straight) * 0.24)
    th_mcp_rho = grip_radius_m + palmar_scale * th_base_r + 0.0022 + 0.6 * open_lift
    th_ip_rho = (
        grip_radius_m
        + palmar_scale * 0.5 * (th_base_r + th_tip_r)
        + 0.0010
        + 0.6 * open_lift
    )
    th_tip_rho = grip_radius_m + palmar_scale * th_tip_r + 0.0005 + 0.6 * open_lift
    th_cmc = (th_mcp_rho + 0.005 * ls) * radial(th_cmc_az) + np.array(
        [0.0, 0.0, z_base - 0.026 * ls], float
    )
    cmc_rho = float(np.hypot(th_cmc[0], th_cmc[1]))
    min_cmc_rho = grip_radius_m + 0.0015 + th_base_r
    if cmc_rho < min_cmc_rho:
        th_cmc[:2] *= min_cmc_rho / max(cmc_rho, 1e-12)
    th_mcp = th_mcp_rho * radial(th_mcp_az) + np.array(
        [0.0, 0.0, z_base - 0.012 * ls], float
    )
    th_ip = th_ip_rho * radial(th_ip_az) + np.array(
        [0.0, 0.0, z_base + (0.008 + 0.004 * straight) * ls], float
    )
    th_tip = th_tip_rho * radial(th_tip_az) + np.array(
        [0.0, 0.0, z_base + (0.021 + 0.011 * straight) * ls], float
    )
    return np.asarray([th_cmc, th_mcp, th_ip, th_tip], float)


def _thumb_and_thenar(
    ctx: _HandBuildContext,
    joint_angles: GripArticulation,
    arc: _PalmArc,
    index_anchor: Array,
) -> tuple[Mesh, Mesh]:
    """Thumb + thenar eminence emerging from the radial side of the palm."""
    settings, ls = ctx.settings, ctx.length_scale
    anthro = settings.anthropometry
    thumb_az = math.radians(
        ctx.placement.thumb_offset_deg
        + (ctx.placement.v_azimuth_deg - NEUTRAL_V_AZIMUTH_DEG)
    )
    th_joints = _thumb_joints(ctx, joint_angles, float(index_anchor[2]))
    th_cmc, th_mcp = th_joints[0], th_joints[1]
    thumb_mesh = lofted_anatomical_digit(
        th_joints,
        radii_m=(
            0.5 * anthro.thumb_base_diameter_m * ls * 0.90,
            0.5 * anthro.thumb_tip_diameter_m * ls,
        ),
        initial_width_axis=ctx.curl * tangential(thumb_az),
        dorsal_hint=radial(thumb_az),
        sides=settings.procedural_skin_sides,
        palmar_flatness=settings.palmar_pad_flatness,
        joint_fractions=(0.45, 0.74),
        grip_floor_m=ctx.inner_floor,
    )
    radial_palm = 0.50 * arc.mid_ctrl + 0.25 * arc.wrist_center + 0.25 * index_anchor
    flare = np.array(
        [
            settings.thenar_flare_scale * 0.90,
            0.95,
            settings.thenar_flare_scale,
        ],
        float,
    )
    thenar_r = np.asarray(settings.thenar_half_extents_m, float) * ls * flare
    thenar_c = 0.44 * th_cmc + 0.28 * th_mcp + 0.28 * radial_palm
    thenar_rho = float(np.hypot(thenar_c[0], thenar_c[1]))
    min_thenar_rho = ctx.grip_radius_m + 0.0015 + float(thenar_r[1])
    if thenar_rho < min_thenar_rho:
        thenar_c[:2] *= min_thenar_rho / max(thenar_rho, 1e-12)
    thenar_mesh = ellipsoid_mesh(
        thenar_c,
        np.array([thenar_r[2], thenar_r[0], thenar_r[1]], float),
        unit(th_mcp - arc.wrist_center),
        rings=12,
        sides=settings.procedural_skin_sides,
        width_hint=unit(radial_palm - arc.mid_ctrl),
    )
    return thumb_mesh, thenar_mesh


def _mirrored_left(built: _BuiltProceduralHand) -> _BuiltProceduralHand:
    """Mirror a canonical right hand across the XZ plane into a left hand."""
    return replace(
        built,
        mesh=_mirror_mesh_y(built.mesh),
        palm_mesh=_mirror_mesh_y(built.palm_mesh),
        finger_meshes={k: _mirror_mesh_y(v) for k, v in built.finger_meshes.items()},
        mcp_anchors={k: _mirror_point_y(v) for k, v in built.mcp_anchors.items()},
        palm_mcp_targets={
            k: _mirror_point_y(v) for k, v in built.palm_mcp_targets.items()
        },
        finger_joints={
            k: np.asarray([_mirror_point_y(p) for p in v], float)
            for k, v in built.finger_joints.items()
        },
        proximal_dirs={
            k: unit(_mirror_point_y(v)) for k, v in built.proximal_dirs.items()
        },
    )


def _build_canonical_hand(
    grip_spec: GripSpec,
    side: HandSide,
    joint_angles: GripArticulation,
    *,
    grip_radius_m: float,
    settings: HandVisualSettings,
) -> _BuiltProceduralHand:
    ctx = _hand_build_context(
        grip_spec,
        side,
        joint_angles,
        grip_radius_m=grip_radius_m,
        settings=settings,
    )
    arc = _palm_arc(ctx, joint_angles)
    palm_mesh, mcp_surface_arr, measured_cup, palm_arc_length_m = _palm_slab(ctx, arc)
    fingers = {
        finger: _build_finger(ctx, finger, mcp_surface_arr[idx], arc.mid_ctrl)
        for idx, finger in enumerate(_ORDERED_FINGERS)
    }
    finger_meshes = {f: b.mesh for f, b in fingers.items()}
    thumb_mesh, thenar_mesh = _thumb_and_thenar(
        ctx, joint_angles, arc, fingers["index"].anchor
    )
    finger_meshes["thumb"] = thumb_mesh
    combined = concat_meshes(
        palm_mesh,
        finger_meshes["little"],
        finger_meshes["ring"],
        finger_meshes["middle"],
        finger_meshes["index"],
        thumb_mesh,
        thenar_mesh,
    )
    built = _BuiltProceduralHand(
        mesh=combined,
        finger_meshes=finger_meshes,
        palm_mesh=palm_mesh,
        mcp_anchors={f: b.anchor for f, b in fingers.items()},
        palm_mcp_targets={f: b.palm_target for f, b in fingers.items()},
        finger_joints={f: b.joints for f, b in fingers.items()},
        finger_radii={f: b.radii for f, b in fingers.items()},
        proximal_dirs={f: b.proximal_dir for f, b in fingers.items()},
        palm_length_m=palm_arc_length_m,
        palm_width_m=ctx.breadth_m,
        cup_depth_m=measured_cup,
    )
    return _mirrored_left(built) if grip_spec.handedness == "left" else built


def measure_procedural_hand_anthropometry(
    grip_spec: GripSpec,
    side: HandSide,
    *,
    grip_radius_m: float = DEFAULT_VISUAL_GRIP_RADIUS_M,
    settings: HandVisualSettings = DEFAULT_HAND_VISUAL_SETTINGS,
) -> dict[str, Any]:
    """Measure true anatomical dimensions, knuckle attachment, cup depth, and pad gaps."""
    angles = GripArticulation.from_grip_spec(grip_spec, side)
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
    side: HandSide,
    joint_angles: HandJointAngles | None = None,
    *,
    grip_radius_m: float,
    skin_rgb: tuple[float, float, float] | None = None,
    settings: HandVisualSettings = DEFAULT_HAND_VISUAL_SETTINGS,
) -> HandVisualResult:
    """Lofted procedural skin in the hand-grip frame (ADR-0057 A2 contract).

    The hand poses itself from ``grip_spec``; explicit joint angles are not
    supported. The shaft roll is applied last, as a rigid rotation.
    """
    if joint_angles is not None:
        raise ValueError("procedural_skin poses itself; pass joint_angles=None")
    built = _build_canonical_hand(
        grip_spec,
        side,
        GripArticulation.from_grip_spec(grip_spec, side),
        grip_radius_m=float(grip_radius_m),
        settings=settings,
    )
    return HandVisualResult(
        backend=BACKEND_ID,
        side=side,
        mesh=rolled_about_shaft(built.mesh, grip_spec, side),
        skin_rgb=skin_rgb or skin_rgb_for(settings.skin_material_for_role(side)),
        grip_radius_m=float(grip_radius_m),
    )


def register() -> None:
    """Add the procedural skin backend to the shared registry (called lazily)."""
    register_hand_visual_backend(
        HandVisualBackend(
            name=BACKEND_ID,
            builder=build_hand_visual,
            description="Lofted procedural skin with an arched, cupped palm",
            capabilities=frozenset({"self_posing"}),
        )
    )


__all__ = [
    "BACKEND_ID",
    "build_hand_visual",
    "measure_procedural_hand_anthropometry",
    "register",
]
