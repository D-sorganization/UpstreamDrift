"""MuJoCo pose solver and mesh transform helpers for the Shadow visual backend."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from src.shared.python.contracts import require
from src.shared.python.core.handedness import Handedness
from src.shared.python.grip_spec.types import GripSpec, HandGripSpec, HandSize
from src.shared.python.model_appearance.grip_pose import Hand
from src.shared.python.model_appearance.grip_shadow_hand_assets import (
    shadow_hand_mjcf_path,
)
from src.shared.python.model_appearance.hand_backends.shadow_settings import (
    _CURL_STAGE_FINGERS,
    _FINGER_BODY_TAG,
    Array,
    ShadowHandPose,
    ShadowHandVisualSettings,
    visual_settings,
)
from src.shared.python.model_appearance.hand_backends.types import (
    HandJointAngles,
    HandSide,
)

#: Shadow-native names (``FFJ3``), canonical :class:`HandJointAngles`, or a
#: full ``qpos`` vector.
JointAngles = HandJointAngles | Mapping[str, float] | NDArray[np.float64] | None

_CANONICAL_FINGER_PREFIX = {"index": "FF", "middle": "MF", "ring": "RF", "little": "LF"}
_CANONICAL_SEGMENT_JOINT = {"prox": "J3", "mid": "J2", "dist": "J1"}
_CANONICAL_THUMB_JOINT = {
    "thumb_press": "THJ4",
    "thumb_prox": "THJ2",
    "thumb_dist": "THJ1",
}


def shadow_joint_name(canonical: str) -> str:
    """Shadow joint for a canonical hinge (``index_prox`` -> ``FFJ3``).

    Names that are not canonical pass through unchanged (Shadow-native keys).
    """
    if canonical in _CANONICAL_THUMB_JOINT:
        return _CANONICAL_THUMB_JOINT[canonical]
    finger, _, segment = canonical.partition("_")
    if finger in _CANONICAL_FINGER_PREFIX and segment in _CANONICAL_SEGMENT_JOINT:
        return _CANONICAL_FINGER_PREFIX[finger] + _CANONICAL_SEGMENT_JOINT[segment]
    if canonical == "meta_ring":
        raise ValueError("the Shadow hand has no ring metacarpal joint (meta_ring)")
    return canonical


def _require_mujoco() -> Any:
    import os

    os.environ.setdefault("MUJOCO_GL", "disable")
    try:
        import mujoco
    except ImportError as exc:  # pragma: no cover
        raise ImportError("mujoco is required for Shadow hand grip visuals") from exc
    return mujoco


@lru_cache(maxsize=4)
def _load_model(trail: bool, handedness: Handedness) -> Any:
    mujoco = _require_mujoco()
    path = shadow_hand_mjcf_path(trail=trail, handedness=handedness)
    return mujoco.MjModel.from_xml_path(str(path))


def _forward_model(
    hand: Hand,
    qpos: Array,
    handedness: Handedness,
) -> tuple[Any, Any]:
    mujoco = _require_mujoco()
    model = _load_model(hand == Hand.TRAIL, handedness)
    data = mujoco.MjData(model)
    require(len(qpos) == model.nq, "qpos length mismatch")
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    return model, data


def _hand_from_side(side: HandSide) -> Hand:
    if side not in ("lead", "trail"):
        raise ValueError(f"side must be 'lead' or 'trail', got {side!r}")
    return Hand.TRAIL if side == "trail" else Hand.LEAD


def _is_anatomical_right(hand: Hand, handedness: Handedness) -> bool:
    return (hand == Hand.TRAIL) ^ (handedness == "left")


def _prefix(hand: Hand, handedness: Handedness = "right") -> str:
    return "rh_" if _is_anatomical_right(hand, handedness) else "lh_"


def _validate_grip_radius(grip_radius_m: float, cfg: ShadowHandVisualSettings) -> float:
    if isinstance(grip_radius_m, bool) or not isinstance(grip_radius_m, int | float):
        raise TypeError(f"grip_radius_m must be finite real, got {grip_radius_m!r}")
    val = float(grip_radius_m)
    lo, hi = cfg.grip_radius_range_m
    if not math.isfinite(val) or not (lo <= val <= hi):
        raise ValueError(f"grip_radius_m must be in [{lo}, {hi}] m, got {val!r}")
    return val


def _joint_qpos_index(model: Any, joint_name: str) -> int:
    mujoco = _require_mujoco()
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        raise ValueError(f"joint {joint_name!r} not in model")
    return int(model.jnt_qposadr[jid])


def _baseline_qpos(model: Any, hand: Hand, handedness: Handedness) -> Array:
    _ = (hand, handedness)
    return np.zeros(model.nq, dtype=float)


def _apply_joint_angles(
    qpos: Array,
    model: Any,
    hand: Hand,
    joint_angles: JointAngles,
    *,
    handedness: Handedness = "right",
) -> Array:
    if joint_angles is None:
        return qpos
    if isinstance(joint_angles, np.ndarray):
        require(len(joint_angles) == model.nq, "joint_angles qpos length mismatch")
        return np.asarray(joint_angles, float)
    if isinstance(joint_angles, HandJointAngles):
        joint_angles = joint_angles.as_mapping()
    out = qpos.copy()
    prefix = _prefix(hand, handedness)
    for key, value in joint_angles.items():
        native = shadow_joint_name(key)
        jname = native if native.startswith(("lh_", "rh_")) else f"{prefix}{native}"
        out[_joint_qpos_index(model, jname)] = float(value)
    return out


def map_hand_grip_spec_to_wrist(
    spec_hand: HandGripSpec,
    *,
    handedness: Literal["left", "right"],
    settings: ShadowHandVisualSettings | None = None,
) -> dict[str, float]:
    """Map spec to wrist, palm cup, and thumb-base offsets."""
    cfg = visual_settings() if settings is None else settings
    g = cfg.wrist_gain
    cup = cfg.palm_cup_gain
    sign = -1.0 if handedness == "left" else 1.0
    palm_delta = spec_hand.palm_finger_depth - 0.5
    cross_delta = math.radians(spec_hand.club_cross_angle_deg - 40.0)
    return {
        "WRJ1": float(g["club_cross_to_wrj1_gain"]) * cross_delta,
        "WRJ2": float(g["palm_depth_to_wrj2_gain"]) * palm_delta,
        "LFJ5": float(cup.get("palm_depth_to_lfj5_gain", 0.0)) * palm_delta
        + float(cup.get("club_cross_to_lfj5_gain", 0.0)) * cross_delta,
        "FFJ4": float(
            np.clip(
                sign * float(g["v_azimuth_to_ffj4_gain"]) * spec_hand.v_azimuth_deg,
                -0.06,
                0.06,
            )
        ),
        "THJ5": sign
        * float(g["thumb_azimuth_to_thj5_gain"])
        * spec_hand.thumb_azimuth_deg
        + float(cup.get("palm_depth_to_thj5_gain", 0.0)) * palm_delta,
        "THJ4": float(g["thumb_straight_to_thj4_gain"])
        * (1.0 - spec_hand.thumb_straightness),
    }


def apply_wrist_offsets(
    qpos: Array,
    model: Any,
    hand: Hand,
    spec_hand: HandGripSpec,
    *,
    handedness: Literal["left", "right"],
    settings: ShadowHandVisualSettings | None = None,
) -> None:
    """Apply clamped wrist/palm-cup offsets without destabilizing the mounted palm."""
    cfg = visual_settings() if settings is None else settings
    arch_key = (
        "lfj5_lead_base_arch_rad" if hand == Hand.LEAD else "lfj5_trail_base_arch_rad"
    )
    base_arch = float(cfg.palm_cup_gain.get(arch_key, 0.0))
    offsets = dict(
        map_hand_grip_spec_to_wrist(spec_hand, handedness=handedness, settings=settings)
    )
    if base_arch:
        offsets["LFJ5"] = offsets.get("LFJ5", 0.0) + base_arch
    prefix = _prefix(hand, handedness)
    for short, delta in offsets.items():
        if short.startswith("WRJ"):
            continue
        try:
            idx = _joint_qpos_index(model, f"{prefix}{short}")
        except ValueError:
            continue
        qpos[idx] = qpos[idx] + delta


def _contour_palm_world(verts: Array, cfg: ShadowHandVisualSettings) -> Array:
    """Humanize Shadow's bulky palm block: compress wrist ears, tame dorsal fin, cup arch."""
    out = verts.copy()
    x_raw, y, z = out[:, 0], out[:, 1], out[:, 2]
    x = np.where(x_raw < 0.270, 0.270 - 0.012 * np.tanh((0.270 - x_raw) / 0.012), x_raw)
    out[:, 0] = x
    cap = cfg.palm_palmar_cap_z_m
    soft = cfg.palm_palmar_softness_m
    z1 = np.where(z > cap, cap + soft * np.tanh((z - cap) / soft), z)
    dcap = cfg.palm_dorsal_cap_z_m
    dsoft = cfg.palm_dorsal_softness_m
    z2 = np.where(z1 < dcap, dcap + dsoft * np.tanh((z1 - dcap) / dsoft), z1)
    # Taper square robot wrist ears smoothly toward human wrist proportions.
    u_w = np.clip((0.285 - x) / 0.045, 0.0, 1.0)
    wrist_s = 1.0 - (1.0 - cfg.wrist_taper_min_scale) * (u_w * u_w * (3.0 - 2.0 * u_w))
    out[:, 1] = y * wrist_s
    z2 = z2 * (1.0 - 0.22 * (u_w * u_w))
    # Curve the proximal palm heel palmarward so it wraps the grip instead of sticking straight out.
    u_heel = np.clip((0.315 - x) / 0.057, 0.0, 1.0)
    z2 = z2 + cfg.palm_heel_curve_m * (u_heel * u_heel)
    arch_y = np.clip(1.0 - (y / cfg.palm_arch_half_breadth_m) ** 2, 0.0, 1.0)
    arch_x = np.clip((x - 0.258) / 0.025, 0.0, 1.0) * np.clip(
        (0.358 - x) / 0.025, 0.0, 1.0
    )
    palmar_gate = 1.0 / (1.0 + np.exp(-(z - 0.006) / 0.003))
    out[:, 2] = z2 - cfg.palm_transverse_arch_m * arch_y * arch_x * palmar_gate
    return out


def _iter_visible_geoms(
    model: Any,
    data: Any,
    cfg: ShadowHandVisualSettings,
) -> Iterator[tuple[str, str, Array, NDArray[np.int64]]]:
    """Single source of truth yielding visible group-2 geoms (world verts, faces)."""
    mujoco = _require_mujoco()
    for gid in range(model.ngeom):
        if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        if int(model.geom_group[gid]) != 2:
            continue
        body_id = int(model.geom_bodyid[gid])
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if any(tag in bname for tag in cfg.hidden_body_substrings):
            continue
        mesh_id = int(model.geom_dataid[gid])
        if mesh_id < 0:
            continue
        mname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id) or ""
        if any(tag in mname for tag in cfg.hidden_mesh_substrings):
            continue
        adr_v = int(model.mesh_vertadr[mesh_id])
        count_v = int(model.mesh_vertnum[mesh_id])
        adr_f = int(model.mesh_faceadr[mesh_id])
        count_f = int(model.mesh_facenum[mesh_id])
        local = np.asarray(model.mesh_vert[adr_v : adr_v + count_v], float)
        rot = np.asarray(data.geom_xmat[gid], float).reshape(3, 3)
        world = local @ rot.T + np.asarray(data.geom_xpos[gid], float)
        if "palm" in mname or "metacarpal" in mname:
            world = _contour_palm_world(world, cfg)
        faces = np.asarray(model.mesh_face[adr_f : adr_f + count_f], np.int64).reshape(
            -1, 3
        )
        yield bname, mname, world, faces


def _canonical_part_key(bname: str, mname: str) -> str:
    if "metacarpal" in mname or "metacarpal" in bname:
        return "lf_metacarpal"
    if "palm" in mname or "palm" in bname:
        return "palm"
    for fname, tag in _FINGER_BODY_TAG.items():
        if tag in bname:
            for seg in ("proximal", "middle", "distal"):
                if seg in bname:
                    return f"{fname}_{seg}"
            return fname
    return mname or bname


def _select_part_vertices(
    model: Any,
    data: Any,
    cfg: ShadowHandVisualSettings,
    *,
    finger: str | None = None,
    segments: tuple[str, ...] | None = None,
) -> Array:
    tag = None if finger is None else _FINGER_BODY_TAG[finger]
    chunks: list[Array] = []
    for bname, _mname, verts, _faces in _iter_visible_geoms(model, data, cfg):
        if tag is not None and tag not in bname:
            continue
        if segments is not None and not any(seg in bname for seg in segments):
            continue
        chunks.append(verts)
    if not chunks:
        raise ValueError(f"no geoms found for finger={finger!r}, segments={segments!r}")
    return np.vstack(chunks)


def _scale_vector(hand_size: HandSize, cfg: ShadowHandVisualSettings) -> Array:
    s_len = (hand_size.length_m / cfg.native_hand_length_m) * cfg.length_scale
    s_brd = (hand_size.breadth_m / cfg.native_hand_breadth_m) * cfg.breadth_scale
    s_dep = min(s_len, s_brd)
    return np.array([s_len, s_brd, s_dep], dtype=float)


def _compute_mount_matrix(
    model: Any,
    data: Any,
    *,
    hand: Hand,
    spec_hand: HandGripSpec,
    grip_spec: GripSpec,
    grip_radius_m: float,
    cfg: ShadowHandVisualSettings,
) -> Array:
    """Compute the 4x4 affine transform mapping scaled MuJoCo world into grip_hand_frame."""
    curl = 1.0 if _is_anatomical_right(hand, grip_spec.handedness) else -1.0
    mirror = -1.0 if grip_spec.handedness == "left" else 1.0
    theta = math.radians(
        mirror * spec_hand.v_azimuth_deg + curl * cfg.palm_center_offset_deg
    )
    u_r = np.array([math.cos(theta), math.sin(theta), 0.0], dtype=float)
    u_t = np.array([-math.sin(theta), math.cos(theta), 0.0], dtype=float)
    u_z = np.array([0.0, 0.0, 1.0], dtype=float)

    tilt_deg = float(
        np.clip(
            spec_hand.club_cross_angle_deg * cfg.palm_diagonal_tilt_gain,
            0.0,
            cfg.palm_diagonal_max_deg,
        )
    )
    alpha = math.radians(tilt_deg)
    col2 = -u_r
    col0 = math.cos(alpha) * (curl * u_t) + math.sin(alpha) * u_z
    col1 = np.cross(col2, col0)
    col1 /= np.linalg.norm(col1)
    rot_grip = np.column_stack([col0, col1, col2])

    scale = _scale_vector(grip_spec.hand_size, cfg)
    linear = rot_grip @ np.diag(scale)

    palm_pts = _select_part_vertices(
        model, data, cfg, segments=("palm", "metacarpal", "proximal")
    )
    ref_world = np.array([cfg.palm_tangency_x_m, 0.0, 0.012], dtype=float)
    base_trans = -linear @ ref_world

    knuckle_x = [
        float(
            (
                _select_part_vertices(
                    model, data, cfg, finger=fn, segments=("proximal",)
                )
                @ linear.T
                + base_trans
            )[:, 2].mean()
        )
        for fn in ("index", "middle", "ring", "little")
    ]
    base_trans = base_trans - float(np.mean(knuckle_x)) * u_z

    palm_rot = palm_pts @ linear.T + base_trans
    target_r = grip_radius_m + cfg.pad_clearance_m
    lo_r, hi_r = -0.02, 0.08
    for _ in range(28):
        mid = 0.5 * (lo_r + hi_r)
        shifted = palm_rot + mid * u_r
        if float(np.hypot(shifted[:, 0], shifted[:, 1]).min()) < target_r:
            lo_r = mid
        else:
            hi_r = mid

    mount = np.eye(4)
    mount[:3, :3] = linear
    mount[:3, 3] = base_trans + hi_r * u_r
    return mount


def _transform_to_grip_frame(points: Array, mount: Array) -> Array:
    return points @ mount[:3, :3].T + mount[:3, 3]


def _radial_xy(points: Array) -> Array:
    return np.hypot(points[:, 0], points[:, 1])


def _part_min_radial(
    model: Any,
    data: Any,
    cfg: ShadowHandVisualSettings,
    mount: Array,
    *,
    finger: str,
    segments: tuple[str, ...],
) -> float:
    pts = _transform_to_grip_frame(
        _select_part_vertices(model, data, cfg, finger=finger, segments=segments),
        mount,
    )
    return float(_radial_xy(pts).min())


@dataclass(frozen=True)
class _SolveCtx:
    """Bundled MuJoCo state and mount parameters for per-joint contact solving."""

    model: Any
    data: Any
    cfg: ShadowHandVisualSettings
    mount: Array
    grip_radius_m: float


def _solve_joint_contact(
    qpos: Array,
    ctx: _SolveCtx,
    *,
    jname: str,
    finger: str,
    eval_segments: tuple[str, ...],
    guard_segments: tuple[str, ...],
    min_q: float | None = None,
) -> None:
    mujoco = _require_mujoco()
    model, data, cfg = ctx.model, ctx.data, ctx.cfg
    mount, grip_radius_m = ctx.mount, ctx.grip_radius_m
    try:
        qidx = _joint_qpos_index(model, jname)
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
    except ValueError:
        return
    base_lo = float(qpos[qidx]) if min_q is None else float(min_q)
    lo = max(float(model.jnt_range[jid, 0]), base_lo)
    hi_lim = float(model.jnt_range[jid, 1])
    target_r = grip_radius_m + cfg.pad_clearance_m
    init_guard_r = _part_min_radial(
        model, data, cfg, mount, finger=finger, segments=guard_segments
    )
    min_safe = min(grip_radius_m + 0.0003, init_guard_r - 0.0001)
    samples = np.linspace(lo, hi_lim, 24)
    best_q, best_err = lo, math.inf
    bracket: tuple[float, float] | None = None

    for idx_s, cand in enumerate(samples):
        qpos[qidx] = float(cand)
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        guard_r = _part_min_radial(
            model, data, cfg, mount, finger=finger, segments=guard_segments
        )
        if idx_s > 0 and guard_r < min_safe:
            bracket = (float(samples[idx_s - 1]), float(cand))
            break
        seg_r = _part_min_radial(
            model, data, cfg, mount, finger=finger, segments=eval_segments
        )
        if seg_r <= target_r:
            bracket = (float(samples[max(0, idx_s - 1)]), float(cand))
            best_q = float(cand)
            break
        err = abs(seg_r - target_r)
        if err < best_err:
            best_err, best_q = err, float(cand)

    if bracket is not None:
        a, b = bracket
        for _ in range(18):
            mid = 0.5 * (a + b)
            qpos[qidx] = mid
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            guard_r = _part_min_radial(
                model, data, cfg, mount, finger=finger, segments=guard_segments
            )
            seg_r = _part_min_radial(
                model, data, cfg, mount, finger=finger, segments=eval_segments
            )
            if guard_r < min_safe or seg_r < target_r:
                b = mid
            else:
                a = mid
        best_q = a

    qpos[qidx] = best_q
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)


def _curl_digit_cascade(
    qpos: Array,
    ctx: _SolveCtx,
    hand: Hand,
    finger: str,
    *,
    handedness: Handedness,
) -> None:
    prefix = _prefix(hand, handedness)
    tag = _FINGER_BODY_TAG[finger].upper()
    mcp_floor = max(0.46, ctx.cfg.mcp_preflex_rad - 0.04)
    stages = (
        (
            f"{prefix}{tag}J3",
            ("proximal",),
            ("proximal", "middle", "distal"),
            mcp_floor,
        ),
        (f"{prefix}{tag}J2", ("middle",), ("middle", "distal"), None),
        (f"{prefix}{tag}J1", ("distal",), ("distal",), None),
    )
    for jname, eval_segs, guard_segs, min_q in stages:
        _solve_joint_contact(
            qpos,
            ctx,
            jname=jname,
            finger=finger,
            eval_segments=eval_segs,
            guard_segments=guard_segs,
            min_q=min_q,
        )


def _score_thumb_state(
    ctx: _SolveCtx,
    idx_tree: cKDTree,
    *,
    target_az: float,
    target_r: float,
    safe_r: float,
    z_target: float,
) -> float:
    th_all = _transform_to_grip_frame(
        _select_part_vertices(ctx.model, ctx.data, ctx.cfg, finger="thumb"),
        ctx.mount,
    )
    min_r = float(_radial_xy(th_all).min())
    pen_pen = 1e5 * max(0.0, safe_r - min_r) ** 2
    idx_gap = float(np.asarray(idx_tree.query(th_all[::4], k=1)[0]).min())
    idx_pen = 500.0 * max(0.0, 0.0015 - idx_gap) ** 2
    distal = _transform_to_grip_frame(
        _select_part_vertices(
            ctx.model, ctx.data, ctx.cfg, finger="thumb", segments=("distal",)
        ),
        ctx.mount,
    )
    d_mean = distal.mean(axis=0)
    az = math.atan2(d_mean[1], d_mean[0])
    az_err = math.atan2(math.sin(az - target_az), math.cos(az - target_az))
    r_err = float(_radial_xy(distal).min()) - target_r
    z_cost = ((float(d_mean[2]) - z_target) / 0.022) ** 2
    return (r_err / 0.012) ** 2 + (az_err / 0.35) ** 2 + z_cost + pen_pen + idx_pen


def _solve_thumb_pose(
    qpos: Array,
    model: Any,
    data: Any,
    hand: Hand,
    grip_spec: GripSpec,
    grip_radius_m: float,
    cfg: ShadowHandVisualSettings,
    *,
    mount: Array,
) -> None:
    mujoco = _require_mujoco()
    ctx = _SolveCtx(
        model=model,
        data=data,
        cfg=cfg,
        mount=mount,
        grip_radius_m=grip_radius_m,
    )
    handedness = grip_spec.handedness
    prefix = _prefix(hand, handedness)
    spec_hand = grip_spec.trail if hand == Hand.TRAIL else grip_spec.lead
    role: Literal["lead", "trail"] = "trail" if hand == Hand.TRAIL else "lead"
    target_az = math.radians(grip_spec.resolved_thumb_azimuth_deg(role))
    target_r = grip_radius_m + cfg.pad_clearance_m
    safe_r = grip_radius_m + cfg.thumb_min_radial_clearance_m
    z_target = 0.036 + 0.015 * spec_hand.thumb_straightness

    idx_pts = _transform_to_grip_frame(
        _select_part_vertices(
            model, data, cfg, finger="index", segments=("proximal", "middle", "distal")
        ),
        mount,
    )
    idx_tree = cKDTree(idx_pts[::3])
    j5 = _joint_qpos_index(model, f"{prefix}THJ5")
    j4 = _joint_qpos_index(model, f"{prefix}THJ4")
    j2 = _joint_qpos_index(model, f"{prefix}THJ2")
    j1 = _joint_qpos_index(model, f"{prefix}THJ1")
    id5 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}THJ5")
    id4 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}THJ4")

    q5_vals = np.linspace(model.jnt_range[id5, 0], model.jnt_range[id5, 1], 11)
    q4_vals = np.linspace(model.jnt_range[id4, 0], model.jnt_range[id4, 1], 9)

    coarse: list[tuple[float, float, float]] = []
    qpos[j2], qpos[j1] = 0.25, 0.0
    for v5 in q5_vals:
        for v4 in q4_vals:
            qpos[j5], qpos[j4] = float(v5), float(v4)
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            score = _score_thumb_state(
                ctx,
                idx_tree,
                target_az=target_az,
                target_r=target_r,
                safe_r=safe_r,
                z_target=z_target,
            )
            coarse.append((score, float(v5), float(v4)))
    coarse.sort(key=lambda item: item[0])

    best_score = math.inf
    best_qpos = qpos.copy()
    for _cscore, v5, v4 in coarse[:5]:
        cand_q = qpos.copy()
        cand_q[j5], cand_q[j4], cand_q[j2], cand_q[j1] = v5, v4, 0.0, 0.0
        data.qpos[:] = cand_q
        mujoco.mj_forward(model, data)
        for jshort, segs in (
            ("THJ2", ("middle", "distal")),
            ("THJ1", ("distal",)),
        ):
            _solve_joint_contact(
                cand_q,
                ctx,
                jname=f"{prefix}{jshort}",
                finger="thumb",
                eval_segments=segs,
                guard_segments=("proximal", "middle", "distal"),
            )
        fscore = _score_thumb_state(
            ctx,
            idx_tree,
            target_az=target_az,
            target_r=target_r,
            safe_r=safe_r,
            z_target=z_target,
        )
        if fscore < best_score:
            best_score = fscore
            best_qpos = cand_q.copy()

    qpos[:] = best_qpos
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)


def _finger_curl_sequence(hand: Hand, grip_spec: GripSpec) -> tuple[str, ...]:
    prefix = "lead_" if hand == Hand.LEAD else "trail_"
    digits: list[str] = []
    for stage in grip_spec.curl_order:
        if stage.startswith(prefix):
            digits.extend(_CURL_STAGE_FINGERS.get(stage, ()))
    return tuple(digits) if digits else ("index", "middle", "ring", "little")


def _relax_club_penetration(
    qpos: Array,
    model: Any,
    data: Any,
    hand: Hand,
    radius: float,
    cfg: ShadowHandVisualSettings,
    mount: Array,
    *,
    handedness: Handedness,
) -> Array:
    mujoco = _require_mujoco()
    prefix = _prefix(hand, handedness)
    for _ in range(20):
        stepped = False
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        for finger in ("index", "middle", "ring", "little", "thumb"):
            pts = _transform_to_grip_frame(
                _select_part_vertices(
                    model,
                    data,
                    cfg,
                    finger=finger,
                    segments=("proximal", "middle", "distal"),
                ),
                mount,
            )
            deficit = float(radius - float(_radial_xy(pts).min()))
            if deficit <= cfg.club_penetration_tolerance_m * 0.5:
                continue
            tag = _FINGER_BODY_TAG[finger].upper()
            for j in ("J3", "J2", "J1", "J4"):
                jname = f"{prefix}{tag}{j}"
                try:
                    qidx = _joint_qpos_index(model, jname)
                except ValueError:
                    continue
                if qpos[qidx] <= 0.0:
                    continue
                qpos[qidx] = max(0.0, qpos[qidx] - cfg.curl_joint_step_rad)
                stepped = True
        if not stepped:
            return qpos
    return qpos


def _forward_and_mount(
    model: Any,
    data: Any,
    qpos: Array,
    hand: Hand,
    grip_spec: GripSpec,
    grip_radius_m: float,
    cfg: ShadowHandVisualSettings,
) -> Array:
    """Apply qpos, advance MuJoCo kinematics, and compute the grip mount matrix."""
    mujoco = _require_mujoco()
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    spec_hand = grip_spec.trail if hand == Hand.TRAIL else grip_spec.lead
    return _compute_mount_matrix(
        model,
        data,
        hand=hand,
        spec_hand=spec_hand,
        grip_spec=grip_spec,
        grip_radius_m=grip_radius_m,
        cfg=cfg,
    )


def solve_shadow_hand_qpos(
    hand: Hand,
    spec_hand: HandGripSpec,
    *,
    grip_spec: GripSpec,
    grip_radius_m: float,
    settings: ShadowHandVisualSettings | None = None,
) -> ShadowHandPose:
    """Solve Shadow qpos + affine mount matrix for one hand wrapping a grip radius."""
    mujoco = _require_mujoco()
    cfg = visual_settings() if settings is None else settings
    radius = _validate_grip_radius(grip_radius_m, cfg)
    handedness = grip_spec.handedness
    model = _load_model(hand == Hand.TRAIL, handedness)
    data = mujoco.MjData(model)
    qpos = _baseline_qpos(model, hand, handedness)
    apply_wrist_offsets(
        qpos, model, hand, spec_hand, handedness=handedness, settings=cfg
    )
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    prefix = _prefix(hand, handedness)
    for finger in ("index", "middle", "ring", "little"):
        tag = _FINGER_BODY_TAG[finger].upper()
        try:
            qpos[_joint_qpos_index(model, f"{prefix}{tag}J3")] = cfg.mcp_preflex_rad
        except ValueError:
            continue
    mount = _forward_and_mount(model, data, qpos, hand, grip_spec, radius, cfg)
    ctx = _SolveCtx(
        model=model,
        data=data,
        cfg=cfg,
        mount=mount,
        grip_radius_m=radius,
    )
    for finger in _finger_curl_sequence(hand, grip_spec):
        _curl_digit_cascade(
            qpos,
            ctx,
            hand,
            finger,
            handedness=handedness,
        )
    _solve_thumb_pose(qpos, model, data, hand, grip_spec, radius, cfg, mount=mount)
    qpos = _relax_club_penetration(
        qpos, model, data, hand, radius, cfg, mount, handedness=handedness
    )
    return ShadowHandPose(qpos=qpos, mount_matrix=mount, handedness=handedness)
