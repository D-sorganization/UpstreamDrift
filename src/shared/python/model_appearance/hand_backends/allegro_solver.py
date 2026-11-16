"""MuJoCo pose solver and mount transform helpers for the Wonik Allegro backend."""

from __future__ import annotations

import math
import os
from collections.abc import Iterator, Mapping, Sequence
from functools import lru_cache
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from src.shared.python.contracts import ensure, require
from src.shared.python.core.handedness import Handedness
from src.shared.python.grip_spec.shaft_roll import (
    APPEARANCE_GRIP_SHAFT_AXIS,
    apply_shaft_roll,
    resolved_shaft_roll_deg,
)
from src.shared.python.grip_spec.types import GripSpec, HandGripSpec, HandSize
from src.shared.python.model_appearance.grip_pose import Hand
from src.shared.python.model_appearance.hand_backends.allegro_settings import (
    _DIGIT_TAG,
    Array,
    AllegroHandPose,
    AllegroHandVisualSettings,
    allegro_mjcf_path,
    visual_settings,
)
from src.shared.python.model_appearance.hand_backends.types import (
    HandJointAngles,
    HandSide,
)

JointAngles = HandJointAngles | Mapping[str, float] | Array | Sequence[float] | None

# Canonical HandJointAngles hinge -> Allegro MJCF joint. Allegro j0 is finger
# abduction (thumb: rotation); j1/j2/j3 are the proximal, middle and distal
# flexion hinges. The hand has no little finger and no ring metacarpal.
_CANONICAL_SEGMENT_JOINT: dict[str, str] = {"prox": "j1", "mid": "j2", "dist": "j3"}
_CANONICAL_THUMB_JOINT: dict[str, str] = {
    "thumb_press": "thj1",
    "thumb_prox": "thj2",
    "thumb_dist": "thj3",
}


def allegro_joint_name(name: str) -> str:
    """Map a canonical ``HandJointAngles`` hinge to its Allegro MJCF joint.

    Native Allegro names (``ffj1``, ``thj0``...) pass through unchanged.
    """
    if name in _CANONICAL_THUMB_JOINT:
        return _CANONICAL_THUMB_JOINT[name]
    finger, _, segment = name.partition("_")
    if finger == "little" or name == "meta_ring":
        raise ValueError(f"Allegro has no joint for canonical hinge {name!r}")
    if finger in _DIGIT_TAG and segment in _CANONICAL_SEGMENT_JOINT:
        return f"{_DIGIT_TAG[finger]}{_CANONICAL_SEGMENT_JOINT[segment]}"
    return name


def _require_mujoco() -> Any:
    os.environ.setdefault("MUJOCO_GL", "disable")
    import mujoco

    return mujoco


@lru_cache(maxsize=4)
def _load_model(trail: bool, handedness: Handedness) -> Any:
    mujoco = _require_mujoco()
    return mujoco.MjModel.from_xml_path(
        str(allegro_mjcf_path(trail=trail, handedness=handedness))
    )


def _forward_model(hand: Hand, qpos: Array, handedness: Handedness) -> tuple[Any, Any]:
    mujoco = _require_mujoco()
    model = _load_model(hand == Hand.TRAIL, handedness)
    data = mujoco.MjData(model)
    require(len(qpos) == model.nq, f"qpos length {len(qpos)} != {model.nq}")
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    return model, data


def _hand_from_side(side: HandSide) -> Hand:
    if side not in ("lead", "trail"):
        raise ValueError(f"side must be 'lead' or 'trail', got {side!r}")
    return Hand.TRAIL if side == "trail" else Hand.LEAD


def _is_anatomical_right(hand: Hand, handedness: Handedness) -> bool:
    return (hand == Hand.TRAIL) ^ (handedness == "left")


def _validate_grip_radius(
    grip_radius_m: float, cfg: AllegroHandVisualSettings
) -> float:
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
        raise ValueError(f"joint {joint_name!r} not in Allegro model")
    return int(model.jnt_qposadr[jid])


def _baseline_qpos(model: Any) -> Array:
    q = np.zeros(model.nq, dtype=float)
    for jid in range(model.njnt):
        qidx = int(model.jnt_qposadr[jid])
        lo = float(model.jnt_range[jid, 0])
        hi = float(model.jnt_range[jid, 1])
        q[qidx] = float(np.clip(0.0, lo, hi))
    return q


def _iter_visible_geoms(
    model: Any,
    data: Any,
    cfg: AllegroHandVisualSettings,
    *,
    vertex_stride: int = 1,
) -> Iterator[tuple[str, str, Array, NDArray[np.int64]]]:
    """Yield (body_name, mesh_name, world_vertices, faces) for group-2 visual meshes."""
    mujoco = _require_mujoco()
    step = max(1, int(vertex_stride))
    for gid in range(model.ngeom):
        if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        if int(model.geom_group[gid]) != cfg.visible_geom_group:
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
        local = np.asarray(model.mesh_vert[adr_v : adr_v + count_v : step], float)
        rot = np.asarray(data.geom_xmat[gid], float).reshape(3, 3)
        world = local @ rot.T + np.asarray(data.geom_xpos[gid], float)
        faces = (
            np.empty((0, 3), dtype=np.int64)
            if step > 1
            else np.asarray(model.mesh_face[adr_f : adr_f + count_f], np.int64).reshape(
                -1, 3
            )
        )
        yield bname, mname, world, faces


def _canonical_part_key(bname: str, mname: str) -> str:
    if bname == "palm":
        return "palm"
    for fname, tag in _DIGIT_TAG.items():
        if bname.startswith(f"{tag}_"):
            suffix = bname[len(tag) + 1 :]
            return f"{fname}_{suffix}"
    return bname or mname


def _select_part_vertices(
    model: Any,
    data: Any,
    cfg: AllegroHandVisualSettings,
    *,
    finger: str | None = None,
    segments: tuple[str, ...] | None = None,
    vertex_stride: int = 1,
) -> Array:
    tag = None if finger is None else _DIGIT_TAG[finger]
    chunks: list[Array] = []
    for bname, _mname, verts, _faces in _iter_visible_geoms(
        model, data, cfg, vertex_stride=vertex_stride
    ):
        if tag is not None and not bname.startswith(f"{tag}_"):
            continue
        if segments is not None:
            if bname == "palm" and "palm" in segments:
                chunks.append(verts)
                continue
            if not any(bname.endswith(f"_{seg}") for seg in segments):
                continue
        chunks.append(verts)
    if not chunks:
        raise ValueError(f"no geoms found for finger={finger!r}, segments={segments!r}")
    return np.vstack(chunks)


def _scale_vector(hand_size: HandSize, cfg: AllegroHandVisualSettings) -> Array:
    s_len = (hand_size.length_m / cfg.native_hand_length_m) * cfg.length_scale
    s_brd = (hand_size.breadth_m / cfg.native_hand_breadth_m) * cfg.breadth_scale
    s_dep = min(s_len, s_brd)
    return np.array([s_len, s_brd, s_dep], dtype=float)


def _transform_to_grip_frame(points: Array, mount: Array) -> Array:
    return points @ mount[:3, :3].T + mount[:3, 3]


def _radial_xy(points: Array) -> Array:
    return np.hypot(points[:, 0], points[:, 1])


def _compute_mount_matrix(
    model: Any,
    data: Any,
    *,
    hand: Hand,
    spec_hand: HandGripSpec,
    grip_spec: GripSpec,
    grip_radius_m: float,
    cfg: AllegroHandVisualSettings,
) -> Array:
    """Compute proper 4x4 affine transform mapping MuJoCo world into grip_hand_frame."""
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
    col1 = np.asarray(np.cross(col2, col0), dtype=np.float64)
    col1 = col1 / float(np.linalg.norm(col1))
    rot_grip = np.column_stack([col0, col1, col2])

    scale = _scale_vector(grip_spec.hand_size, cfg)
    linear = rot_grip @ np.diag(scale)

    palm_pts = _select_part_vertices(
        model, data, cfg, segments=("palm", "base", "proximal")
    )
    ref_world = np.array([cfg.palm_tangency_x_m, 0.0, cfg.palm_tangency_z_m], float)
    base_trans = -linear @ ref_world

    mf_base_id = _require_mujoco().mj_name2id(
        model, _require_mujoco().mjtObj.mjOBJ_BODY, "mf_base"
    )
    mf_knuckle_grip = linear @ np.asarray(data.xpos[mf_base_id], float) + base_trans
    base_trans = base_trans - float(mf_knuckle_grip[2]) * u_z

    palm_rot = palm_pts @ linear.T + base_trans
    target_r = grip_radius_m + cfg.pad_clearance_m
    lo_r, hi_r = cfg.mount_radial_search_range_m
    for _ in range(cfg.mount_radial_bisect_iterations):
        mid = 0.5 * (lo_r + hi_r)
        shifted = palm_rot + mid * u_r
        if float(np.hypot(shifted[:, 0], shifted[:, 1]).min()) < target_r:
            lo_r = mid
        else:
            hi_r = mid

    mount = np.eye(4)
    mount[:3, :3] = linear
    mount[:3, 3] = base_trans + hi_r * u_r
    ensure(
        float(np.linalg.det(mount[:3, :3])) > 0.0,
        "Allegro mount must preserve right-handed orientation",
    )
    return mount


def _part_min_radial(
    model: Any,
    data: Any,
    cfg: AllegroHandVisualSettings,
    mount: Array,
    *,
    finger: str,
    segments: tuple[str, ...],
) -> float:
    pts = _transform_to_grip_frame(
        _select_part_vertices(
            model,
            data,
            cfg,
            finger=finger,
            segments=segments,
            vertex_stride=cfg.solver_vertex_stride,
        ),
        mount,
    )
    return float(_radial_xy(pts).min())


def _solve_joint_contact(
    qpos: Array,
    model: Any,
    data: Any,
    cfg: AllegroHandVisualSettings,
    *,
    jname: str,
    finger: str,
    eval_segments: tuple[str, ...],
    guard_segments: tuple[str, ...],
    grip_radius_m: float,
    mount: Array,
    target_offset_m: float = 0.0,
) -> None:
    mujoco = _require_mujoco()
    qidx = _joint_qpos_index(model, jname)
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
    lo = max(float(model.jnt_range[jid, 0]), float(qpos[qidx]))
    hi_lim = float(model.jnt_range[jid, 1])
    target_r = grip_radius_m + cfg.pad_clearance_m + target_offset_m
    init_guard_r = _part_min_radial(
        model, data, cfg, mount, finger=finger, segments=guard_segments
    )
    min_safe = min(grip_radius_m + 0.0002, init_guard_r - 0.0001)
    samples = np.linspace(lo, hi_lim, cfg.curl_coarse_samples)
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
        for _ in range(cfg.curl_bisect_iterations):
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


def _align_finger_abduction(
    qpos: Array, model: Any, data: Any, cfg: AllegroHandVisualSettings
) -> None:
    """Nudge index/ring base abduction slightly inward so fingers stay aligned."""
    mujoco = _require_mujoco()
    if cfg.finger_abduction_inward_rad <= 0.0:
        return
    mf_y = float(
        _select_part_vertices(model, data, cfg, finger="middle", segments=("tip",))[
            :, 1
        ].mean()
    )
    for finger in ("index", "ring"):
        tag = _DIGIT_TAG[finger]
        qidx = _joint_qpos_index(model, f"{tag}j0")
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{tag}j0")
        lo, hi = float(model.jnt_range[jid, 0]), float(model.jnt_range[jid, 1])
        base_val = float(qpos[qidx])
        best_val = base_val
        best_dist = math.inf
        for sign in (-1.0, 0.0, 1.0):
            cand = float(
                np.clip(base_val + sign * cfg.finger_abduction_inward_rad, lo, hi)
            )
            qpos[qidx] = cand
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            ty = float(
                _select_part_vertices(
                    model, data, cfg, finger=finger, segments=("tip",)
                )[:, 1].mean()
            )
            dist = abs(ty - mf_y)
            if dist < best_dist:
                best_dist = dist
                best_val = cand
        qpos[qidx] = best_val
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)


def _score_thumb_state(
    model: Any,
    data: Any,
    cfg: AllegroHandVisualSettings,
    mount: Array,
    idx_tree: cKDTree,
    *,
    target_az: float,
    target_r: float,
    safe_r: float,
    z_target: float,
    max_z_m: float,
) -> float:
    th_all = _transform_to_grip_frame(
        _select_part_vertices(
            model, data, cfg, finger="thumb", vertex_stride=cfg.solver_vertex_stride
        ),
        mount,
    )
    min_r = float(_radial_xy(th_all).min())
    pen_pen = 1e5 * max(0.0, safe_r - min_r) ** 2
    idx_gap = float(idx_tree.query(th_all, k=1)[0].min())
    idx_pen = 500.0 * max(0.0, cfg.thumb_index_min_clearance_m - idx_gap) ** 2
    max_z_pen = 2e4 * max(0.0, float(th_all[:, 2].max()) - max_z_m) ** 2
    distal = _transform_to_grip_frame(
        _select_part_vertices(
            model,
            data,
            cfg,
            finger="thumb",
            segments=("tip",),
            vertex_stride=cfg.solver_vertex_stride,
        ),
        mount,
    )
    d_mean = distal.mean(axis=0)
    az = math.atan2(float(d_mean[1]), float(d_mean[0]))
    az_err = math.atan2(math.sin(az - target_az), math.cos(az - target_az))
    r_err = float(_radial_xy(distal).min()) - target_r
    z_cost = ((float(d_mean[2]) - z_target) / 0.020) ** 2
    return (
        (r_err / 0.006) ** 2
        + (az_err / 0.45) ** 2
        + z_cost
        + pen_pen
        + idx_pen
        + max_z_pen
    )


def _solve_thumb_pose(
    qpos: Array,
    model: Any,
    data: Any,
    hand: Hand,
    grip_spec: GripSpec,
    grip_radius_m: float,
    cfg: AllegroHandVisualSettings,
    *,
    mount: Array,
) -> None:
    mujoco = _require_mujoco()
    spec_hand = grip_spec.trail if hand == Hand.TRAIL else grip_spec.lead
    role: Literal["lead", "trail"] = "trail" if hand == Hand.TRAIL else "lead"
    target_az = math.radians(grip_spec.resolved_thumb_azimuth_deg(role))
    target_r = grip_radius_m + cfg.pad_clearance_m
    safe_r = grip_radius_m + cfg.thumb_min_radial_clearance_m
    z_target = (
        cfg.thumb_z_base_m + cfg.thumb_z_straight_gain_m * spec_hand.thumb_straightness
    )

    idx_pts = _transform_to_grip_frame(
        _select_part_vertices(
            model,
            data,
            cfg,
            finger="index",
            segments=("proximal", "medial", "distal", "tip"),
        ),
        mount,
    )
    idx_tree = cKDTree(idx_pts[::3])
    q_indices = tuple(_joint_qpos_index(model, f"thj{k}") for k in range(4))
    j_limits = tuple(
        (
            float(
                model.jnt_range[
                    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"thj{k}"), 0
                ]
            ),
            float(
                model.jnt_range[
                    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"thj{k}"), 1
                ]
            ),
        )
        for k in range(4)
    )

    q0_vals = np.linspace(j_limits[0][0], j_limits[0][1], cfg.thumb_coarse_grid_n)
    q1_vals = np.linspace(j_limits[1][0], j_limits[1][1], cfg.thumb_coarse_grid_n)

    def eval_thumb(vals: Sequence[float]) -> float:
        for idx_q, val in zip(q_indices, vals, strict=True):
            qpos[idx_q] = float(val)
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        return _score_thumb_state(
            model,
            data,
            cfg,
            mount,
            idx_tree,
            target_az=target_az,
            target_r=target_r,
            safe_r=safe_r,
            z_target=z_target,
            max_z_m=cfg.thumb_max_z_m,
        )

    coarse: list[tuple[float, tuple[float, float, float, float]]] = []
    for v0 in q0_vals:
        for v1 in q1_vals:
            for v2, v3 in cfg.thumb_curl_presets_rad:
                q4 = (float(v0), float(v1), float(v2), float(v3))
                coarse.append((eval_thumb(q4), q4))
    coarse.sort(key=lambda item: item[0])

    best_score = math.inf
    best_q4 = coarse[0][1]
    for _cscore, q4 in coarse[:4]:
        cur = list(q4)
        for step in cfg.thumb_refine_steps_rad:
            for k in range(4):
                lo, hi = j_limits[k]
                best_k = cur[k]
                best_k_score = eval_thumb(cur)
                for delta in (-step, step):
                    cur[k] = float(np.clip(best_k + delta, lo, hi))
                    score = eval_thumb(cur)
                    if score < best_k_score:
                        best_k_score = score
                        best_k = cur[k]
                cur[k] = best_k
        fscore = eval_thumb(cur)
        if fscore < best_score:
            best_score = fscore
            best_q4 = (cur[0], cur[1], cur[2], cur[3])

    eval_thumb(best_q4)


def _relax_club_penetration(
    qpos: Array,
    model: Any,
    data: Any,
    radius: float,
    cfg: AllegroHandVisualSettings,
    mount: Array,
) -> Array:
    mujoco = _require_mujoco()
    thresh = cfg.club_penetration_tolerance_m * 0.25
    for _ in range(cfg.relax_max_iterations):
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        stepped = False
        for finger in ("index", "middle", "ring", "thumb"):
            f_pts = _transform_to_grip_frame(
                _select_part_vertices(
                    model,
                    data,
                    cfg,
                    finger=finger,
                    segments=("proximal", "medial", "distal", "tip"),
                ),
                mount,
            )
            if float(radius - float(_radial_xy(f_pts).min())) <= thresh:
                continue
            tag = _DIGIT_TAG[finger]
            for j in ("j1", "j2", "j3"):
                qidx = _joint_qpos_index(model, f"{tag}{j}")
                if qpos[qidx] <= 0.0:
                    continue
                qpos[qidx] = max(0.0, float(qpos[qidx] - cfg.curl_joint_step_rad))
                stepped = True
        if not stepped:
            return qpos
    return qpos


def apply_joint_angles(qpos: Array, model: Any, joint_angles: JointAngles) -> Array:
    """Apply optional joint angle dict or 16-DOF array, clamped to MJCF limits."""
    out = qpos.copy()
    if joint_angles is None:
        return out
    if isinstance(joint_angles, HandJointAngles):
        joint_angles = joint_angles.as_mapping()
    if isinstance(joint_angles, Mapping):
        for raw_name, val in joint_angles.items():
            name = allegro_joint_name(str(raw_name))
            qidx = _joint_qpos_index(model, str(name))
            jid = _require_mujoco().mj_name2id(
                model, _require_mujoco().mjtObj.mjOBJ_JOINT, str(name)
            )
            lo = float(model.jnt_range[jid, 0])
            hi = float(model.jnt_range[jid, 1])
            out[qidx] = float(np.clip(float(val), lo, hi))
        return out
    if isinstance(joint_angles, np.ndarray | Sequence):
        arr = np.asarray(joint_angles, dtype=float)
        require(len(arr) == model.nq, f"joint_angles length {len(arr)} != {model.nq}")
        for jid in range(model.njnt):
            qidx = int(model.jnt_qposadr[jid])
            lo = float(model.jnt_range[jid, 0])
            hi = float(model.jnt_range[jid, 1])
            out[qidx] = float(np.clip(float(arr[qidx]), lo, hi))
        return out
    raise TypeError(f"unsupported joint_angles type: {type(joint_angles).__name__}")


def _apply_role_shaft_roll(mount: Array, grip_spec: GripSpec, hand: Hand) -> Array:
    role: Literal["lead", "trail"] = "trail" if hand == Hand.TRAIL else "lead"
    roll_deg = resolved_shaft_roll_deg(grip_spec, role)
    r_mat = np.asarray(
        apply_shaft_roll(np.eye(3), APPEARANCE_GRIP_SHAFT_AXIS, roll_deg),
        dtype=np.float64,
    )
    rolled = mount.copy()
    rolled[:3, :3] = r_mat @ mount[:3, :3]
    rolled[:3, 3] = r_mat @ mount[:3, 3]
    return rolled


def solve_allegro_hand_qpos(
    hand: Hand,
    spec_hand: HandGripSpec,
    *,
    grip_spec: GripSpec,
    grip_radius_m: float,
    settings: AllegroHandVisualSettings | None = None,
) -> AllegroHandPose:
    """Solve curled 16-DOF Allegro qpos and proper grip-frame mount for one hand."""
    cfg = visual_settings() if settings is None else settings
    radius = _validate_grip_radius(grip_radius_m, cfg)
    mujoco = _require_mujoco()
    handedness = grip_spec.handedness
    model = _load_model(hand == Hand.TRAIL, handedness)
    data = mujoco.MjData(model)
    qpos = _baseline_qpos(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)

    _align_finger_abduction(qpos, model, data, cfg)
    base_mount = _compute_mount_matrix(
        model,
        data,
        hand=hand,
        spec_hand=spec_hand,
        grip_spec=grip_spec,
        grip_radius_m=radius,
        cfg=cfg,
    )

    for finger in ("index", "middle", "ring"):
        tag = _DIGIT_TAG[finger]
        stages = (
            (
                f"{tag}j1",
                ("proximal",),
                ("proximal", "medial", "distal", "tip"),
                cfg.proximal_stage_offset_m,
            ),
            (
                f"{tag}j2",
                ("medial",),
                ("medial", "distal", "tip"),
                cfg.proximal_stage_offset_m,
            ),
            (f"{tag}j3", ("tip",), ("distal", "tip"), 0.0),
        )
        for jname, eval_segs, guard_segs, offset_m in stages:
            _solve_joint_contact(
                qpos,
                model,
                data,
                cfg,
                jname=jname,
                finger=finger,
                eval_segments=eval_segs,
                guard_segments=guard_segs,
                grip_radius_m=radius,
                mount=base_mount,
                target_offset_m=offset_m,
            )

    _solve_thumb_pose(
        qpos,
        model,
        data,
        hand,
        grip_spec,
        radius,
        cfg,
        mount=base_mount,
    )
    qpos = _relax_club_penetration(qpos, model, data, radius, cfg, base_mount)
    rolled_mount = _apply_role_shaft_roll(base_mount, grip_spec, hand)
    return AllegroHandPose(qpos=qpos, mount_matrix=rolled_mount, handedness=handedness)


__all__ = [
    "_baseline_qpos",
    "_canonical_part_key",
    "_compute_mount_matrix",
    "_forward_model",
    "_hand_from_side",
    "_iter_visible_geoms",
    "_load_model",
    "_radial_xy",
    "_require_mujoco",
    "_scale_vector",
    "_select_part_vertices",
    "_transform_to_grip_frame",
    "_validate_grip_radius",
    "JointAngles",
    "allegro_joint_name",
    "apply_joint_angles",
    "solve_allegro_hand_qpos",
]
