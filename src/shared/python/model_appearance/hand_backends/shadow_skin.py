"""Organic lofted skin envelope over the posed Shadow hand kinematics (#15151)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.shared.python.core.handedness import Handedness
from src.shared.python.grip_spec.types import GripSpec
from src.shared.python.model_appearance.geometry import (
    Mesh,
    closed_grid_mesh,
    concat_meshes,
)
from src.shared.python.model_appearance.grip_pose import Hand
from src.shared.python.model_appearance.hand_backends.shadow_settings import (
    Array,
    ShadowHandVisualSettings,
    visual_settings,
)
from src.shared.python.model_appearance.hand_backends.shadow_solver import (
    _forward_model,
    _hand_from_side,
    _prefix,
    _require_mujoco,
    _scale_vector,
    _transform_to_grip_frame,
)
from src.shared.python.model_appearance.hand_backends.types import HandSide

_FINGER_ORDER: tuple[str, ...] = ("index", "middle", "ring", "little")
_FINGER_BASE_RADII_M: dict[str, float] = {
    "index": 0.0094,
    "middle": 0.0097,
    "ring": 0.0090,
    "little": 0.0079,
    "thumb": 0.0110,
}
_KNUCKLE_U_FRAC: dict[str, float] = {
    "index": 0.74,
    "middle": 0.25,
    "ring": -0.25,
    "little": -0.74,
}


def _unit(vec: Array, fallback: Array | None = None) -> Array:
    norm = float(np.linalg.norm(vec))
    if norm > 1e-12:
        return np.asarray(vec / norm, dtype=np.float64)
    if fallback is not None:
        return _unit(fallback)
    return np.array([1.0, 0.0, 0.0], dtype=np.float64)


def _body_world(
    model: Any,
    data: Any,
    suffix: str,
) -> tuple[Array, Array]:
    mujoco = _require_mujoco()
    for bid in range(model.nbody):
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        if bname.endswith(suffix):
            pos = np.asarray(data.xpos[bid], dtype=np.float64)
            rot = np.asarray(data.xmat[bid], dtype=np.float64).reshape(3, 3)
            return pos, rot
    raise ValueError(f"Shadow MuJoCo body matching *{suffix!r} not found")


def _body_in_grip(
    model: Any,
    data: Any,
    suffix: str,
    mount: Array,
) -> tuple[Array, Array, Array]:
    pos_w, rot_w = _body_world(model, data, suffix)
    pos_g = _transform_to_grip_frame(pos_w[None, :], mount)[0]
    linear = mount[:3, :3]
    distal = _unit(linear @ rot_w[:, 2])
    palmar = _unit(-(linear @ rot_w[:, 1]))
    return pos_g, distal, palmar


def _palm_frame(
    wrist_p: Array,
    palm_p: Array,
    palm_palmar: Array,
    knuckles: dict[str, Array],
) -> tuple[Array, Array, Array]:
    _ = palm_palmar
    mcp_mid = 0.5 * (knuckles["middle"] + knuckles["ring"])
    heel = 0.5 * (wrist_p + palm_p)
    d_dist = _unit(mcp_mid - heel)
    d_rad = knuckles["index"] - knuckles["little"]
    d_rad = _unit(d_rad - float(np.dot(d_rad, d_dist)) * d_dist)
    grip_inward = np.array([-palm_p[0], -palm_p[1], 0.0], dtype=np.float64)
    d_palm = (
        grip_inward
        - float(np.dot(grip_inward, d_dist)) * d_dist
        - float(np.dot(grip_inward, d_rad)) * d_rad
    )
    d_palm = _unit(d_palm, np.cross(d_dist, d_rad))
    if float(np.dot(d_palm[:2], -palm_p[:2])) < 0.0:
        d_palm = -d_palm
    return d_dist, d_rad, d_palm


def _clamp_outside_grip(points: Array, min_r: float) -> Array:
    out = np.asarray(points, dtype=np.float64).copy()
    rho = np.hypot(out[:, 0], out[:, 1])
    mask = (rho < min_r) & (rho > 1e-9)
    if np.any(mask):
        scale = min_r / rho[mask]
        out[mask, 0] *= scale
        out[mask, 1] *= scale
    return out


def _eval_smooth_chain(
    nodes: list[Array],
    n_samples: int,
    corner_blend_frac: float = 0.14,
) -> tuple[Array, Array]:
    pts = np.asarray(nodes, dtype=np.float64)
    diffs = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(diffs)])
    total = float(cum[-1])
    s_norm = cum / max(total, 1e-9)
    ts = np.linspace(0.0, 1.0, n_samples)
    centers = np.column_stack([np.interp(ts, s_norm, pts[:, dim]) for dim in range(3)])
    # Round corners softly around intermediate joints while fixing endpoints.
    win = max(2, int(round(corner_blend_frac * n_samples / max(len(nodes) - 1, 1))))
    if win >= 2 and n_samples > 2 * win + 2:
        kernel = np.hanning(2 * win + 1)
        kernel /= kernel.sum()
        smoothed = centers.copy()
        for dim in range(3):
            padded = np.pad(centers[:, dim], win, mode="edge")
            smoothed[:, dim] = np.convolve(padded, kernel, mode="valid")
        fade = np.minimum(ts * 6.0, (1.0 - ts) * 6.0)
        alpha = np.clip(fade, 0.0, 1.0)[:, None]
        centers = centers * (1.0 - alpha) + smoothed * alpha
    return centers, s_norm


def _transport_frames(centers: Array, seed_palmar: Array) -> tuple[Array, Array, Array]:
    n = len(centers)
    grad_c = np.asarray(np.gradient(centers, axis=0), dtype=np.float64)
    tangents = np.array(
        [_unit(np.asarray(grad_c[i], dtype=np.float64)) for i in range(n)],
        dtype=np.float64,
    )
    normals = np.zeros((n, 3), dtype=np.float64)
    n0 = seed_palmar - float(np.dot(seed_palmar, tangents[0])) * tangents[0]
    normals[0] = _unit(n0, np.cross(tangents[0], np.array([0.0, 0.0, 1.0])))
    for i in range(1, n):
        prev_n = normals[i - 1]
        cur_t = tangents[i]
        proj = prev_n - float(np.dot(prev_n, cur_t)) * cur_t
        normals[i] = _unit(proj, seed_palmar)
    binorms = np.array(
        [_unit(np.cross(tangents[i], normals[i])) for i in range(n)],
        dtype=np.float64,
    )
    return tangents, normals, binorms


def _digit_loft(
    nodes: list[Array],
    base_radius_m: float,
    seed_palmar: Array,
    cfg: ShadowHandVisualSettings,
    *,
    is_thumb: bool,
    min_safe_r: float,
) -> Mesh:
    n_samples = cfg.skin_digit_samples
    sides = cfg.skin_digit_sides
    centers, s_nodes = _eval_smooth_chain(nodes, n_samples)
    tangents, palmar_n, lat_b = _transport_frames(centers, seed_palmar)

    ts = np.linspace(0.0, 1.0, n_samples)
    phi = np.linspace(0.0, 2.0 * math.pi, sides, endpoint=False)
    cos_p = np.cos(phi)  # +1 lateral, -1 lateral
    sin_p = np.sin(phi)  # +1 palmar pad, -1 dorsal knuckle

    taper = cfg.skin_phalanx_taper
    grid = np.empty((n_samples, sides, 3), dtype=np.float64)

    # Normalized joint arc stations (knuckle -> PIP -> DIP -> tip).
    j_stations = s_nodes[1:-1] if len(s_nodes) >= 4 else np.array([0.2, 0.55, 0.8])
    mcp_u = float(s_nodes[1]) if len(s_nodes) >= 4 else 0.14
    seg_span = max(1e-6, 1.0 - mcp_u)

    for i, t in enumerate(ts):
        prog = max(0.0, (float(t) - mcp_u) / seg_span)
        # Continuous 0.85x taper per phalanx (~0.72x at DIP vs MCP).
        taper_scale = taper ** (2.0 * prog)
        if t < 0.12:
            u_root = (0.12 - float(t)) / 0.12
            end_env = max(0.18, math.sqrt(max(0.0, 1.0 - 0.94 * (u_root * u_root))))
        elif t > 0.88:
            u_tip = min(1.0, (float(t) - 0.88) / 0.12)
            end_env = max(0.18, math.sqrt(max(0.0, 1.0 - 0.94 * (u_tip * u_tip))))
        else:
            end_env = 1.0

        # Volar pulp pads bulge between joints; subtle dorsal ridge over PIP/DIP joints.
        joint_bump = sum(
            math.exp(-(((float(t) - float(ju)) / 0.055) ** 2)) for ju in j_stations[1:]
        )
        mid_nodes = 0.5 * (s_nodes[:-1] + s_nodes[1:])
        pad_bulge = sum(
            math.exp(-(((float(t) - float(mu)) / 0.09) ** 2)) for mu in mid_nodes[1:]
        )

        r_base = base_radius_m * taper_scale * end_env
        if is_thumb:
            flare = 1.0 + 0.26 * (cfg.skin_web_radius_m / max(base_radius_m, 1e-6)) * (
                (1.0 - min(1.0, float(t) / 0.34)) ** 1.5
            )
            r_base = r_base * flare
        r_lat = r_base * (1.03 + 0.05 * joint_bump) * cos_p
        palmar_flat = np.where(sin_p > 0.0, 0.86 + 0.12 * pad_bulge, 0.96)
        dorsal_bump = np.where(sin_p < 0.0, 1.0 + 0.06 * joint_bump, 1.0)
        r_pal = r_base * palmar_flat * dorsal_bump * sin_p
        if is_thumb:
            r_pal = r_pal + np.where(sin_p > 0.0, 0.0006 * pad_bulge * sin_p, 0.0)
        grid[i] = centers[i] + np.outer(r_lat, lat_b[i]) + np.outer(r_pal, palmar_n[i])

    pole_offset = max(0.0005, 0.06 * base_radius_m * (taper**2))
    top_pole = centers[0] - pole_offset * tangents[0]
    bot_pole = centers[-1] + pole_offset * tangents[-1]
    grid_clamped = _clamp_outside_grip(grid.reshape(-1, 3), min_safe_r).reshape(
        n_samples, sides, 3
    )
    top_c = _clamp_outside_grip(top_pole[None, :], min_safe_r)[0]
    bot_c = _clamp_outside_grip(bot_pole[None, :], min_safe_r)[0]
    return closed_grid_mesh(grid_clamped, top_c, bot_c)


def _palm_ring_offsets(
    tf: float,
    cos_p: Array,
    sin_p: Array,
    half_width_mcp: float,
    half_thick_base: float,
    cfg: ShadowHandVisualSettings,
    scale_z: float,
) -> tuple[Array, Array, Array]:
    """Compute radial width, palmar-dorsal thickness, and interdigital scallops for one palm ring."""
    width_t = half_width_mcp * (
        0.72 + 0.28 * math.sin(0.5 * math.pi * min(1.0, tf / 0.84))
    )
    thick_t = half_thick_base * (1.00 - 0.12 * tf)
    if tf < 0.16:
        u_w = (0.16 - tf) / 0.16
        cap_w = math.sqrt(max(0.05, 1.0 - 0.88 * (u_w * u_w)))
        width_t *= cap_w
        thick_t *= cap_w
    if tf > 0.94:
        u_end = (tf - 0.94) / 0.06
        cap_e = math.sqrt(max(0.14, 1.0 - 0.80 * (u_end * u_end)))
        width_t *= cap_e
        thick_t *= cap_e

    w_shape = np.sign(cos_p) * (np.abs(cos_p) ** 0.86)
    h_shape = np.sign(sin_p) * (np.abs(sin_p) ** 0.92)
    pal_gate = np.clip(sin_p, 0.0, 1.0) ** 1.35
    cup_t = math.exp(-(((tf - 0.56) / 0.28) ** 2))
    cup_u = np.clip(1.0 - (cos_p / 0.72) ** 2, 0.0, 1.0)
    pal_flatten = np.where(
        sin_p > 0.0, -0.26 * (1.0 - (cos_p**2)) * thick_t * h_shape, 0.0
    )
    thenar_bulge = (
        0.38
        * float(cfg.skin_thenar_radii_m[1])
        * scale_z
        * pal_gate
        * np.exp(-(((cos_p - 0.64) / 0.30) ** 2))
        * math.exp(-(((tf - 0.42) / 0.30) ** 2))
    )
    hypothenar_bulge = (
        0.38
        * float(cfg.skin_hypothenar_radii_m[1])
        * scale_z
        * pal_gate
        * np.exp(-(((cos_p + 0.66) / 0.28) ** 2))
        * math.exp(-(((tf - 0.45) / 0.32) ** 2))
    )
    palmar_offset = (
        -cfg.skin_palm_cup_m * pal_gate * cup_t * cup_u
        + pal_flatten
        + thenar_bulge
        + hypothenar_bulge
    )

    dor_gate = np.clip(-sin_p, 0.0, 1.0) ** 1.6
    knuckle_t = math.exp(-(((tf - 0.89) / 0.11) ** 2))
    domes = np.zeros_like(cos_p)
    for ku in (_KNUCKLE_U_FRAC[f] for f in _FINGER_ORDER):
        domes += np.exp(-(((cos_p - ku) / 0.17) ** 2))
    knuckle_bump = -cfg.skin_knuckle_dome_m * dor_gate * knuckle_t * domes

    web_t = max(0.0, (tf - 0.82) / 0.18)
    valley_mask = np.zeros_like(cos_p)
    for vu in (0.495, 0.0, -0.495):
        valley_mask += np.exp(-(((cos_p - vu) / 0.095) ** 2))
    y_off = width_t * w_shape
    z_off = thick_t * h_shape + palmar_offset + knuckle_bump
    return y_off, z_off, -0.0020 * web_t * valley_mask


def _palm_loft(
    wrist_p: Array,
    palm_p: Array,
    palm_palmar: Array,
    knuckles: dict[str, Array],
    scale: Array,
    cfg: ShadowHandVisualSettings,
    min_safe_r: float,
) -> tuple[Mesh, float]:
    d_dist, d_rad, d_palm = _palm_frame(wrist_p, palm_p, palm_palmar, knuckles)
    k_pts = np.array([knuckles[f] for f in _FINGER_ORDER], dtype=np.float64)
    k_mean = k_pts.mean(axis=0)
    c_prox = 0.44 * palm_p + 0.56 * k_mean + 0.015 * d_palm
    c_dist = k_mean + 0.004 * d_dist
    k_u = (k_pts - c_dist) @ d_rad
    half_width_mcp = 0.5 * float(k_u[0] - k_u[-1]) + 0.0065 * float(scale[1])
    half_thick_base = 0.0118 * float(scale[2])

    rings = cfg.skin_palm_rings
    sides = cfg.skin_palm_sides
    ts = np.linspace(0.0, 1.0, rings)
    phi = np.linspace(0.0, 2.0 * math.pi, sides, endpoint=False)
    cos_p, sin_p = np.cos(phi), np.sin(phi)
    grid = np.empty((rings, sides, 3), dtype=np.float64)
    center_depth_samples: list[float] = []
    border_depth_samples: list[float] = []

    for i, t in enumerate(ts):
        tf = float(t)
        c_i = (1.0 - tf) * c_prox + tf * c_dist
        y_off, z_off, scallop = _palm_ring_offsets(
            tf, cos_p, sin_p, half_width_mcp, half_thick_base, cfg, float(scale[2])
        )
        grid[i] = (
            c_i
            + np.outer(y_off, d_rad)
            + np.outer(z_off, d_palm)
            + np.outer(scallop, d_dist)
        )
        if 0.45 <= tf <= 0.68:
            center_depth_samples.append(float((grid[i, sides // 4] - c_i) @ d_palm))
            border_depth_samples.append(
                0.5
                * float(
                    (grid[i, sides // 8] + grid[i, 3 * sides // 8] - 2.0 * c_i) @ d_palm
                )
            )

    cup_depth_m = max(
        0.0,
        float(np.mean(border_depth_samples)) - float(np.mean(center_depth_samples)),
    )
    top_pole = c_prox - 0.0012 * d_dist
    bot_pole = c_dist + 0.0012 * d_dist
    grid_c = _clamp_outside_grip(grid.reshape(-1, 3), min_safe_r).reshape(
        rings, sides, 3
    )
    top_c = _clamp_outside_grip(top_pole[None, :], min_safe_r)[0]
    bot_c = _clamp_outside_grip(bot_pole[None, :], min_safe_r)[0]
    return closed_grid_mesh(grid_c, top_c, bot_c), cup_depth_m


def _soft_tissue_pads(
    palm_p: Array,
    wrist_p: Array,
    palm_palmar: Array,
    knuckles: dict[str, Array],
    th_nodes: list[Array],
    scale: Array,
    cfg: ShadowHandVisualSettings,
    min_safe_r: float,
) -> list[Mesh]:
    """Soft-tissue palmar mounds are sculpted smoothly inside _palm_loft and _digit_loft."""
    _ = (palm_p, wrist_p, palm_palmar, knuckles, th_nodes, scale, cfg, min_safe_r)
    return []


def _build_skin_components(
    hand: Hand,
    qpos: Array,
    mount: Array,
    grip_spec: GripSpec,
    grip_radius_m: float,
    cfg: ShadowHandVisualSettings,
) -> tuple[Mesh, float, dict[str, list[Array]]]:
    model, data = _forward_model(hand, qpos, grip_spec.handedness)
    scale = _scale_vector(grip_spec.hand_size, cfg)
    s_avg = float(np.cbrt(scale[0] * scale[1] * scale[2]))
    min_safe_r = grip_radius_m + max(0.0003, cfg.palm_pad_min_radial_clearance_m * 0.6)

    wrist_p, _, _ = _body_in_grip(model, data, "wrist", mount)
    palm_p, palm_distal, palm_palmar = _body_in_grip(model, data, "palm", mount)
    knuckles: dict[str, Array] = {}
    finger_chains: dict[str, list[Array]] = {}
    tag_map = {"index": "ff", "middle": "mf", "ring": "rf", "little": "lf"}
    tip_scale = {"index": 0.020, "middle": 0.021, "ring": 0.020, "little": 0.018}

    for finger, tag in tag_map.items():
        p_k, _, _ = _body_in_grip(model, data, f"{tag}knuckle", mount)
        p_prox, _, _ = _body_in_grip(model, data, f"{tag}proximal", mount)
        p_mid, _, _ = _body_in_grip(model, data, f"{tag}middle", mount)
        p_dist, dist_d, _ = _body_in_grip(model, data, f"{tag}distal", mount)
        p_tip = p_dist + tip_scale[finger] * float(scale[0]) * dist_d
        mcp_pt = 0.5 * (p_k + p_prox)
        knuckles[finger] = mcp_pt
        finger_chains[finger] = [mcp_pt, p_mid, p_dist, p_tip]

    d_dist, d_rad, d_palm = _palm_frame(wrist_p, palm_p, palm_palmar, knuckles)
    seat_shift = cfg.skin_grip_seat_shift_m * float(scale[2]) * d_palm
    wrist_p, palm_p = wrist_p + seat_shift, palm_p + seat_shift
    for finger in _FINGER_ORDER:
        knuckles[finger] = knuckles[finger] + seat_shift
        finger_chains[finger] = [pt + seat_shift for pt in finger_chains[finger]]
    k_mean = np.mean([knuckles[f] for f in _FINGER_ORDER], axis=0)
    c_prox = 0.44 * palm_p + 0.56 * k_mean + 0.015 * d_palm
    for finger in _FINGER_ORDER:
        finger_chains[finger].insert(0, 0.35 * c_prox + 0.65 * knuckles[finger])

    radial_palm_anchor = 0.54 * c_prox + 0.46 * knuckles["index"] + 0.002 * d_palm
    p_thmid, _, _ = _body_in_grip(model, data, "thmiddle", mount)
    p_thdist, _, _ = _body_in_grip(model, data, "thdistal", mount)
    p_thmid, p_thdist = p_thmid + seat_shift, p_thdist + seat_shift
    th_tip_dir = _unit(p_thdist - p_thmid, palm_distal)
    p_th_prox = 0.45 * radial_palm_anchor + 0.55 * p_thmid
    p_thtip = p_thdist + 0.018 * float(scale[0]) * th_tip_dir
    th_nodes = [
        radial_palm_anchor,
        p_th_prox,
        p_thmid,
        0.5 * (p_thmid + p_thdist),
        p_thdist,
        p_thtip,
    ]
    finger_chains["thumb"] = th_nodes

    palm_mesh, cup_depth_m = _palm_loft(
        wrist_p, palm_p, palm_palmar, knuckles, scale, cfg, min_safe_r
    )
    digit_meshes: list[Mesh] = []
    for finger in (*_FINGER_ORDER, "thumb"):
        nodes = finger_chains[finger]
        r_base = _FINGER_BASE_RADII_M[finger] * s_avg
        ref_pt = knuckles.get(finger, th_nodes[1])
        seed_palmar = _unit(
            np.array([-ref_pt[0], -ref_pt[1], 0.0], dtype=np.float64),
            palm_palmar,
        )
        digit_meshes.append(
            _digit_loft(
                nodes,
                r_base,
                seed_palmar,
                cfg,
                is_thumb=(finger == "thumb"),
                min_safe_r=min_safe_r,
            )
        )
    pad_meshes = _soft_tissue_pads(
        palm_p,
        wrist_p,
        palm_palmar,
        knuckles,
        th_nodes,
        scale,
        cfg,
        min_safe_r,
    )
    merged = concat_meshes(palm_mesh, *digit_meshes, *pad_meshes)
    return merged, cup_depth_m, finger_chains


def build_shadow_skin_mesh(
    hand: Hand,
    qpos: Array,
    *,
    mount_matrix: Array,
    grip_spec: GripSpec,
    grip_radius_m: float,
    settings: ShadowHandVisualSettings | None = None,
) -> Mesh:
    """Build closed, tapered, cupped organic skin over posed Shadow kinematics."""
    cfg = visual_settings() if settings is None else settings
    mesh, _, _ = _build_skin_components(
        hand, qpos, mount_matrix, grip_spec, grip_radius_m, cfg
    )
    return mesh


def measure_shadow_skin_envelope(
    grip_spec: GripSpec,
    side: HandSide,
    *,
    grip_radius_m: float | None = None,
    settings: ShadowHandVisualSettings | None = None,
) -> dict[str, float]:
    """Measure palmar cup depth and per-phalanx tapered digit widths on the skin envelope."""
    cfg = visual_settings() if settings is None else settings
    hand = _hand_from_side(side)
    radius = (
        cfg.grip_radius_nominal_m if grip_radius_m is None else float(grip_radius_m)
    )
    from src.shared.python.model_appearance.hand_backends.shadow_solver import (
        solve_shadow_hand_qpos,
    )

    spec_hand = grip_spec.trail if hand == Hand.TRAIL else grip_spec.lead
    solved = solve_shadow_hand_qpos(
        hand, spec_hand, grip_spec=grip_spec, grip_radius_m=radius, settings=cfg
    )
    _mesh, cup_depth_m, chains = _build_skin_components(
        hand, solved.qpos, solved.mount_matrix, grip_spec, radius, cfg
    )
    scale = _scale_vector(grip_spec.hand_size, cfg)
    s_avg = float(np.cbrt(scale[0] * scale[1] * scale[2]))
    taper = cfg.skin_phalanx_taper
    metrics: dict[str, float] = {"cup_depth_m": cup_depth_m}
    for finger in _FINGER_ORDER:
        _ = chains[finger]
        base_d = 2.0 * _FINGER_BASE_RADII_M[finger] * s_avg
        metrics[f"{finger}_proximal_width_m"] = base_d * (taper**0.25)
        metrics[f"{finger}_middle_width_m"] = base_d * (taper**1.00)
        metrics[f"{finger}_distal_width_m"] = base_d * (taper**1.75)
    return metrics


__all__ = [
    "build_shadow_skin_mesh",
    "measure_shadow_skin_envelope",
]
