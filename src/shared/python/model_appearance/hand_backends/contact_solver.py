"""Anatomical grip mounting and fast curl-to-contact solver for MyoHand bones."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.shared.python.contracts import require
from src.shared.python.grip_spec.types import GripSpec
from src.shared.python.model_appearance.geometry import Mesh
from src.shared.python.model_appearance.hand_backends.mujoco_extract import (
    apply_joint_targets,
    extract_hand_bone_meshes,
    load_myohand_model,
)
from src.shared.python.model_appearance.hand_backends.settings import (
    MyoHandVisualSettings,
)
from src.shared.python.model_appearance.hand_backends.types import HandVisualSide

Array = NDArray[np.float64]

_FINGER_INDICES: tuple[int, ...] = (2, 3, 4, 5)
_THUMB_BODIES: tuple[str, ...] = ("firstmc", "proximal_thumb", "distal_thumb")


@dataclass(frozen=True)
class _ModelCache:
    """Pre-indexed MuJoCo geom vertices (strided + full) and joint qpos metadata."""

    qpos_adr: dict[str, int]
    limits: dict[str, tuple[float, float]]
    sparse_geoms: dict[str, tuple[tuple[int, Array], ...]]
    full_geoms: dict[str, tuple[tuple[int, Array], ...]]


@lru_cache(maxsize=4)
def _cached_model_index(
    xml_path: str, mtime_ns: int, body_names: tuple[str, ...]
) -> _ModelCache:
    import mujoco
    from pathlib import Path

    _ = mtime_ns
    model = load_myohand_model(Path(xml_path))
    allowed = frozenset(body_names)
    qpos_adr: dict[str, int] = {}
    limits: dict[str, tuple[float, float]] = {}
    for j in range(model.njnt):
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if jname:
            qpos_adr[jname] = int(model.jnt_qposadr[j])
            limits[jname] = (float(model.jnt_range[j, 0]), float(model.jnt_range[j, 1]))

    sparse_acc: dict[str, list[tuple[int, Array]]] = {}
    full_acc: dict[str, list[tuple[int, Array]]] = {}
    for gid in range(model.ngeom):
        if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        bid = int(model.geom_bodyid[gid])
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        if bname not in allowed:
            continue
        mid = int(model.geom_dataid[gid])
        if mid < 0:
            continue
        vadr = int(model.mesh_vertadr[mid])
        vnum = int(model.mesh_vertnum[mid])
        local = np.asarray(model.mesh_vert[vadr : vadr + vnum], dtype=np.float64)
        full_acc.setdefault(bname, []).append((gid, local))
        sparse_acc.setdefault(bname, []).append((gid, local[::4].copy()))

    return _ModelCache(
        qpos_adr=qpos_adr,
        limits=limits,
        sparse_geoms={k: tuple(v) for k, v in sparse_acc.items()},
        full_geoms={k: tuple(v) for k, v in full_acc.items()},
    )


def _model_index(cfg: MyoHandVisualSettings) -> _ModelCache:
    resolved = cfg.model_xml_path.resolve()
    return _cached_model_index(
        str(resolved), resolved.stat().st_mtime_ns, cfg.hand_body_names
    )


def _forward_fast(
    model: Any,
    data: Any,
    cache: _ModelCache,
    q: dict[str, float],
) -> None:
    import mujoco

    data.qpos[:] = 0.0
    adr_map = cache.qpos_adr
    for name, val in q.items():
        adr = adr_map.get(name)
        if adr is not None:
            data.qpos[adr] = float(val)
    mujoco.mj_forward(model, data)


def _placed_from_cache(
    data: Any,
    geom_map: dict[str, tuple[tuple[int, Array], ...]],
    rot: Array,
    trans: Array,
    bodies: tuple[str, ...],
) -> dict[str, Array]:
    out: dict[str, Array] = {}
    rot_t = rot.T
    for bname in bodies:
        items = geom_map[bname]
        if len(items) == 1:
            gid, local = items[0]
            g_rot = np.asarray(data.geom_xmat[gid], dtype=np.float64).reshape(3, 3)
            g_pos = np.asarray(data.geom_xpos[gid], dtype=np.float64)
            world = local @ g_rot.T + g_pos
        else:
            chunks = []
            for gid, local in items:
                g_rot = np.asarray(data.geom_xmat[gid], dtype=np.float64).reshape(3, 3)
                g_pos = np.asarray(data.geom_xpos[gid], dtype=np.float64)
                chunks.append(local @ g_rot.T + g_pos)
            world = np.vstack(chunks)
        out[bname] = world @ rot_t + trans
    return out


def _unit(vec: Array) -> Array:
    norm = float(np.linalg.norm(vec))
    require(norm > 1e-12, "cannot normalize zero vector")
    return np.asarray(vec / norm, dtype=np.float64)


def _handed_curl_and_theta(
    spec_rh: GripSpec, side: HandVisualSide, cfg: MyoHandVisualSettings
) -> tuple[float, float, float]:
    hand_spec = spec_rh.lead if side == "lead" else spec_rh.trail
    curl = 1.0 if side == "trail" else -1.0
    theta = math.radians(hand_spec.v_azimuth_deg + curl * cfg.mcp_azimuth_offset_deg)
    tilt = math.radians(
        float(np.clip(hand_spec.club_cross_angle_deg * 0.45, 0.0, 10.0))
    )
    return curl, theta, tilt


def _model_triad_and_zero_bones(
    model: Any,
    data: Any,
    seed_targets: dict[str, float],
    cfg: MyoHandVisualSettings,
) -> tuple[dict[str, Mesh], Array]:
    q0 = dict(seed_targets)
    for jname in ("pro_sup", "deviation", "flexion"):
        q0[jname] = 0.0
    for idx in _FINGER_INDICES:
        q0[f"mcp{idx}_flexion"] = 0.0
        q0[f"pm{idx}_flexion"] = 0.0
        q0[f"md{idx}_flexion"] = 0.0
    allowed = frozenset(cfg.hand_body_names)
    apply_joint_targets(model, data, q0)
    raw0 = extract_hand_bone_meshes(model, data, allowed_bodies=allowed)

    q_flex = dict(q0)
    q_flex["mcp3_flexion"] = 0.6
    apply_joint_targets(model, data, q_flex)
    raw_flex = extract_hand_bone_meshes(
        model, data, allowed_bodies=frozenset({"distph3"})
    )

    mc2 = raw0["secondmc"].vertices.mean(axis=0)
    mc3 = raw0["thirdmc"].vertices.mean(axis=0)
    mc5 = raw0["fifthmc"].vertices.mean(axis=0)
    p3 = raw0["proxph3"].vertices.mean(axis=0)
    dp_delta = raw_flex["distph3"].vertices.mean(axis=0) - raw0[
        "distph3"
    ].vertices.mean(axis=0)

    e_dist = _unit(p3 - mc3)
    e_curl = _unit(dp_delta - float(np.dot(dp_delta, e_dist)) * e_dist)
    rad_raw = mc2 - mc5
    e_rad = _unit(
        rad_raw
        - float(np.dot(rad_raw, e_dist)) * e_dist
        - float(np.dot(rad_raw, e_curl)) * e_curl
    )
    r_model_t = np.column_stack((e_curl, e_dist, e_rad)).T
    return raw0, r_model_t


def anatomical_grip_mount_transform(
    spec_rh: GripSpec,
    side: HandVisualSide,
    raw0: dict[str, Mesh],
    r_model_t: Array,
    *,
    cfg: MyoHandVisualSettings,
    grip_radius_m: float,
) -> Array:
    """Place zero-posed MyoHand palm tangent & radially seated on the grip."""
    curl, theta, tilt = _handed_curl_and_theta(spec_rh, side, cfg)
    u_r = np.array([math.cos(theta), math.sin(theta), 0.0], dtype=np.float64)
    u_t = np.array([-math.sin(theta), math.cos(theta), 0.0], dtype=np.float64)
    u_z = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    tgt_dist = _unit(math.cos(tilt) * (curl * u_t) + math.sin(tilt) * u_z)
    tgt_curl = _unit(-u_r - float(np.dot(-u_r, tgt_dist)) * tgt_dist)
    tgt_rad = _unit(
        u_z
        - float(np.dot(u_z, tgt_dist)) * tgt_dist
        - float(np.dot(u_z, tgt_curl)) * tgt_curl
    )
    r_grip = np.column_stack((tgt_curl, tgt_dist, tgt_rad))

    s_len = (
        spec_rh.hand_size.length_m / cfg.anthropometric_reference_hand_length_m
    ) ** cfg.anthropometric_scale_exponent
    breadth_ref = cfg.anthropometric_reference_hand_length_m * 0.449
    s_brd = (
        (spec_rh.hand_size.breadth_m / breadth_ref) ** cfg.anthropometric_scale_exponent
        if cfg.width_scale_from_breadth
        else s_len
    )
    scale_anat = np.diag([min(s_len, s_brd), s_len, s_brd])
    linear = r_grip @ scale_anat @ r_model_t

    mc3_lin = raw0["thirdmc"].vertices @ linear.T
    p3_lin = raw0["proxph3"].vertices @ linear.T
    mcp3_anchor = 0.5 * (mc3_lin.mean(axis=0) + p3_lin.mean(axis=0))
    t_xy = (
        -float(mcp3_anchor @ u_r) * u_r
        - float(mcp3_anchor @ u_t) * u_t
        + (curl * cfg.mcp_tangential_bias_m) * u_t
    )
    mc_lin = np.vstack(
        [
            raw0[b].vertices @ linear.T
            for b in ("secondmc", "thirdmc", "fourthmc", "fifthmc")
        ]
    )
    t_z = -float(mc_lin[:, 2].mean()) + cfg.palm_axial_offset_m * s_len

    palm_verts = np.vstack(
        [raw0[b].vertices[::4] @ linear.T for b in cfg.palm_anchor_bodies]
    )
    palm_verts[:, 0] += t_xy[0]
    palm_verts[:, 1] += t_xy[1]
    palm_verts[:, 2] += t_z

    target_r = grip_radius_m + cfg.palm_seat_clearance_m
    lo, hi = -0.08, 0.08
    for _ in range(28):
        mid = 0.5 * (lo + hi)
        shifted = palm_verts + mid * u_r
        if float(np.hypot(shifted[:, 0], shifted[:, 1]).min()) < target_r:
            lo = mid
        else:
            hi = mid

    mount = np.eye(4, dtype=np.float64)
    mount[:3, :3] = linear
    mount[:3, 3] = np.array([t_xy[0], t_xy[1], t_z], dtype=np.float64) + hi * u_r
    return mount


def _finger_costs(
    placed: dict[str, Array],
    target_r: float,
    safe_r: float,
) -> dict[int, float]:
    costs: dict[int, float] = {}
    for idx in _FINGER_INDICES:
        rp = float(
            np.hypot(placed[f"proxph{idx}"][:, 0], placed[f"proxph{idx}"][:, 1]).min()
        )
        rm = float(
            np.hypot(placed[f"midph{idx}"][:, 0], placed[f"midph{idx}"][:, 1]).min()
        )
        rd = float(
            np.hypot(placed[f"distph{idx}"][:, 0], placed[f"distph{idx}"][:, 1]).min()
        )
        rmin = min(rp, rm, rd)
        costs[idx] = (
            1.0e6 * max(0.0, safe_r - rmin) ** 2
            + 0.7 * (rp - target_r) ** 2
            + 1.6 * (rm - target_r) ** 2
            + 2.0 * (rd - target_r) ** 2
        )
    return costs


def _solve_fingers(
    model: Any,
    data: Any,
    cache: _ModelCache,
    q: dict[str, float],
    rot: Array,
    trans: Array,
    locked: set[str],
    target_r: float,
    safe_r: float,
) -> None:
    finger_bodies = tuple(
        f"{p}{i}" for i in _FINGER_INDICES for p in ("proxph", "midph", "distph")
    )
    limits = cache.limits
    if "mcp5_abduction" not in locked and "mcp5_abduction" in limits:
        lo5, hi5 = limits["mcp5_abduction"]
        q["mcp5_abduction"] = float(np.clip(0.10, lo5, hi5))

    best_f: dict[int, tuple[float, float, float]] = {
        i: (q[f"mcp{i}_flexion"], q[f"pm{i}_flexion"], q[f"md{i}_flexion"])
        for i in _FINGER_INDICES
    }
    best_c: dict[int, float] = dict.fromkeys(_FINGER_INDICES, 1000000000000.0)

    def _clamp_val(jname: str, val: float) -> float:
        if jname in locked:
            return q[jname]
        lo, hi = limits[jname]
        return float(np.clip(val, lo, hi))

    for mf in np.linspace(0.35, 1.50, 7):
        for pf in np.linspace(0.55, 1.55, 7):
            df = min(1.57, 0.72 * float(pf))
            cand = dict(q)
            row: dict[int, tuple[float, float, float]] = {}
            for idx in _FINGER_INDICES:
                vals = (
                    _clamp_val(f"mcp{idx}_flexion", float(mf)),
                    _clamp_val(f"pm{idx}_flexion", float(pf)),
                    _clamp_val(f"md{idx}_flexion", df),
                )
                row[idx] = vals
                (
                    cand[f"mcp{idx}_flexion"],
                    cand[f"pm{idx}_flexion"],
                    cand[f"md{idx}_flexion"],
                ) = vals
            _forward_fast(model, data, cache, cand)
            placed = _placed_from_cache(
                data, cache.sparse_geoms, rot, trans, finger_bodies
            )
            costs = _finger_costs(placed, target_r, safe_r)
            for idx in _FINGER_INDICES:
                if costs[idx] < best_c[idx]:
                    best_c[idx] = costs[idx]
                    best_f[idx] = row[idx]

    for step in (0.16, 0.07, 0.03):
        for k_j, jstem in enumerate(("mcp", "pm", "md")):
            for sgn in (-1.0, 1.0):
                cand = dict(q)
                row = {}
                for idx in _FINGER_INDICES:
                    cur_vals = list(best_f[idx])
                    jname = f"{jstem}{idx}_flexion"
                    cur_vals[k_j] = _clamp_val(jname, cur_vals[k_j] + sgn * step)
                    row[idx] = (cur_vals[0], cur_vals[1], cur_vals[2])
                    (
                        cand[f"mcp{idx}_flexion"],
                        cand[f"pm{idx}_flexion"],
                        cand[f"md{idx}_flexion"],
                    ) = row[idx]
                _forward_fast(model, data, cache, cand)
                placed = _placed_from_cache(
                    data, cache.sparse_geoms, rot, trans, finger_bodies
                )
                costs = _finger_costs(placed, target_r, safe_r)
                for idx in _FINGER_INDICES:
                    if costs[idx] < best_c[idx]:
                        best_c[idx] = costs[idx]
                        best_f[idx] = row[idx]

    for idx in _FINGER_INDICES:
        q[f"mcp{idx}_flexion"], q[f"pm{idx}_flexion"], q[f"md{idx}_flexion"] = best_f[
            idx
        ]


@dataclass(frozen=True)
class _ThumbEvalCtx:
    """Immutable workspace context for evaluating candidate thumb joint poses."""

    model: Any
    data: Any
    cache: _ModelCache
    q_base: dict[str, float]
    rot: Array
    trans: Array
    tgt_az: float
    tgt_r: float
    safe_r: float
    chord_guard_r: float
    max_z_m: float


def _xy_segment_min_r(p0: Array, p1: Array) -> float:
    """Return minimum XY distance from the shaft axis (0, 0) to segment [p0, p1]."""
    a = p0[:2]
    d = p1[:2] - a
    denom = float(d[0] * d[0] + d[1] * d[1])
    t = (
        0.0
        if denom < 1e-12
        else float(np.clip(-(a[0] * d[0] + a[1] * d[1]) / denom, 0.0, 1.0))
    )
    closest = a + t * d
    return float(math.hypot(float(closest[0]), float(closest[1])))


def _thumb_pose_cost(
    ctx: _ThumbEvalCtx, angles: tuple[float, float, float, float]
) -> float:
    ca, cf, mp, ip = angles
    cand = dict(ctx.q_base)
    cand["cmc_abduction"] = ca
    cand["cmc_flexion"] = cf
    cand["mp_flexion"] = mp
    cand["ip_flexion"] = ip
    _forward_fast(ctx.model, ctx.data, ctx.cache, cand)
    pts = _placed_from_cache(
        ctx.data, ctx.cache.sparse_geoms, ctx.rot, ctx.trans, _THUMB_BODIES
    )
    th_all = np.vstack([pts[b] for b in _THUMB_BODIES])
    min_r = float(np.hypot(th_all[:, 0], th_all[:, 1]).min())
    max_z = float(th_all[:, 2].max())
    c_mc = pts["firstmc"].mean(axis=0)
    c_px = pts["proximal_thumb"].mean(axis=0)
    c_ds = pts["distal_thumb"].mean(axis=0)
    chord_r = min(_xy_segment_min_r(c_mc, c_px), _xy_segment_min_r(c_px, c_ds))

    dp = pts["distal_thumb"]
    dm = dp.mean(axis=0)
    r_dp = np.hypot(dp[:, 0], dp[:, 1])
    r_d = float(r_dp.min()) - ctx.tgt_r
    r_d_max = max(0.0, float(r_dp.max()) - (ctx.tgt_r + 0.0085))
    az = math.atan2(float(dm[1]), float(dm[0]))
    az_err = math.atan2(math.sin(az - ctx.tgt_az), math.cos(az - ctx.tgt_az))
    z_err = (float(dm[2]) - 0.028) / 0.020
    pen_cost = 4.0e5 * max(0.0, ctx.safe_r - min_r) ** 2
    chord_cost = 6.0e5 * max(0.0, ctx.chord_guard_r - chord_r) ** 2
    max_z_cost = 1.0e6 * max(0.0, max_z - ctx.max_z_m) ** 2
    ext_cost = 0.5 * (max(0.0, -mp - 0.35) ** 2 + max(0.0, -ip - 0.35) ** 2)
    return (
        (r_d / 0.0025) ** 2
        + (r_d_max / 0.003) ** 2
        + (az_err / 0.50) ** 2
        + z_err**2
        + pen_cost
        + chord_cost
        + max_z_cost
        + ext_cost
    )


def _solve_thumb(
    model: Any,
    data: Any,
    cache: _ModelCache,
    q: dict[str, float],
    spec_rh: GripSpec,
    side: HandVisualSide,
    rot: Array,
    trans: Array,
    locked: set[str],
    cfg: MyoHandVisualSettings,
    grip_radius_m: float,
) -> None:
    thumb_names = ("cmc_abduction", "cmc_flexion", "mp_flexion", "ip_flexion")
    if all(n in locked for n in thumb_names):
        return
    limits = cache.limits
    tgt_az = math.radians(
        spec_rh.resolved_thumb_azimuth_deg("trail" if side == "trail" else "lead")
    )
    ctx = _ThumbEvalCtx(
        model=model,
        data=data,
        cache=cache,
        q_base=q,
        rot=rot,
        trans=trans,
        tgt_az=tgt_az,
        tgt_r=grip_radius_m + cfg.bone_target_clearance_m,
        safe_r=grip_radius_m + cfg.bone_safe_clearance_m,
        chord_guard_r=grip_radius_m + 0.0005,
        max_z_m=cfg.thumb_max_z_m,
    )

    def _vals_for(name: str, grid: tuple[float, ...]) -> tuple[float, ...]:
        if name in locked:
            return (q[name],)
        lo, hi = limits[name]
        return tuple(float(np.clip(v, lo, hi)) for v in grid)

    best_th = (q["cmc_abduction"], q["cmc_flexion"], q["mp_flexion"], q["ip_flexion"])
    best_cost = 1.0e12
    for ca in _vals_for("cmc_abduction", (-0.4, 0.0, 0.4, 0.78)):
        for cf in _vals_for("cmc_flexion", (-0.6, -0.1, 0.3, 0.70)):
            for mp in _vals_for("mp_flexion", (-0.6, 0.0, 0.65)):
                for ip in _vals_for("ip_flexion", (-1.0, -0.5, -0.1, 0.25, 0.44)):
                    cost = _thumb_pose_cost(ctx, (ca, cf, mp, ip))
                    if cost < best_cost:
                        best_cost = cost
                        best_th = (ca, cf, mp, ip)

    cur = list(best_th)
    for step in (0.14, 0.05, 0.02):
        for k_idx, jname in enumerate(thumb_names):
            if jname in locked:
                continue
            lo, hi = limits[jname]
            for sgn in (-1.0, 1.0):
                probe = list(cur)
                probe[k_idx] = float(np.clip(cur[k_idx] + sgn * step, lo, hi))
                cost = _thumb_pose_cost(ctx, (probe[0], probe[1], probe[2], probe[3]))
                if cost < best_cost:
                    best_cost = cost
                    cur = probe

    q["cmc_abduction"], q["cmc_flexion"], q["mp_flexion"], q["ip_flexion"] = (
        cur[0],
        cur[1],
        cur[2],
        cur[3],
    )


def _relax_thumb_penetration(
    ctx: _ThumbEvalCtx,
    q: dict[str, float],
    locked: set[str],
    min_allowed: float,
) -> None:
    limits = ctx.cache.limits
    thumb_joints = ("ip_flexion", "mp_flexion", "cmc_flexion")
    for _ in range(14):
        _forward_fast(ctx.model, ctx.data, ctx.cache, q)
        pts = _placed_from_cache(
            ctx.data, ctx.cache.full_geoms, ctx.rot, ctx.trans, _THUMB_BODIES
        )
        r_verts = min(
            float(np.hypot(pts[b][:, 0], pts[b][:, 1]).min()) for b in _THUMB_BODIES
        )
        c_mc = pts["firstmc"].mean(axis=0)
        c_px = pts["proximal_thumb"].mean(axis=0)
        c_ds = pts["distal_thumb"].mean(axis=0)
        r_chord = min(_xy_segment_min_r(c_mc, c_px), _xy_segment_min_r(c_px, c_ds))
        if min(r_verts, r_chord) >= min_allowed:
            break
        changed = False
        for jname in thumb_joints:
            if jname in locked:
                continue
            lo, _ = limits[jname]
            if q[jname] > lo + 0.02:
                q[jname] = max(lo, q[jname] - 0.05)
                changed = True
                break
        if not changed:
            break


def _relax_club_penetration(
    model: Any,
    data: Any,
    cache: _ModelCache,
    q: dict[str, float],
    rot: Array,
    trans: Array,
    locked: set[str],
    cfg: MyoHandVisualSettings,
    grip_radius_m: float,
) -> None:
    min_allowed = grip_radius_m - 0.5 * cfg.penetration_tolerance_m
    limits = cache.limits
    for idx in _FINGER_INDICES:
        chain = (f"proxph{idx}", f"midph{idx}", f"distph{idx}")
        joints = (f"md{idx}_flexion", f"pm{idx}_flexion", f"mcp{idx}_flexion")
        for _ in range(8):
            _forward_fast(model, data, cache, q)
            pts = _placed_from_cache(data, cache.full_geoms, rot, trans, chain)
            rmin = min(float(np.hypot(pts[b][:, 0], pts[b][:, 1]).min()) for b in chain)
            if rmin >= min_allowed:
                break
            changed = False
            for jname in joints:
                if jname in locked:
                    continue
                lo, _ = limits[jname]
                if q[jname] > lo + 0.02:
                    q[jname] = max(lo, q[jname] - 0.05)
                    changed = True
                    break
            if not changed:
                break
    thumb_ctx = _ThumbEvalCtx(
        model=model,
        data=data,
        cache=cache,
        q_base=q,
        rot=rot,
        trans=trans,
        tgt_az=0.0,
        tgt_r=grip_radius_m,
        safe_r=grip_radius_m,
        chord_guard_r=grip_radius_m,
        max_z_m=cfg.thumb_max_z_m,
    )
    _relax_thumb_penetration(thumb_ctx, q, locked, min_allowed)
    apply_joint_targets(model, data, q)


def solve_myohand_contact_pose(
    model: Any,
    data: Any,
    spec_rh: GripSpec,
    side: HandVisualSide,
    seed_targets: dict[str, float],
    *,
    cfg: MyoHandVisualSettings,
    grip_radius_m: float,
    locked_joints: set[str],
) -> tuple[dict[str, float], Array]:
    """Mount MyoHand anatomically and curl unlocked digits to contact on grip."""
    cache = _model_index(cfg)
    raw0, r_model_t = _model_triad_and_zero_bones(model, data, seed_targets, cfg)
    mount = anatomical_grip_mount_transform(
        spec_rh,
        side,
        raw0,
        r_model_t,
        cfg=cfg,
        grip_radius_m=grip_radius_m,
    )
    rot, trans = mount[:3, :3], mount[:3, 3]
    q = dict(seed_targets)
    for wrist_j in ("pro_sup", "deviation", "flexion"):
        if wrist_j not in locked_joints:
            q[wrist_j] = 0.0

    target_r = grip_radius_m + cfg.bone_target_clearance_m
    safe_r = grip_radius_m + cfg.bone_safe_clearance_m
    _solve_fingers(model, data, cache, q, rot, trans, locked_joints, target_r, safe_r)
    _solve_thumb(
        model,
        data,
        cache,
        q,
        spec_rh,
        side,
        rot,
        trans,
        locked_joints,
        cfg,
        grip_radius_m,
    )
    _relax_club_penetration(
        model,
        data,
        cache,
        q,
        rot,
        trans,
        locked_joints,
        cfg,
        grip_radius_m,
    )
    return q, mount


__all__ = [
    "anatomical_grip_mount_transform",
    "solve_myohand_contact_pose",
]
