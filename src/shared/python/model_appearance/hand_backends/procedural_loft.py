"""Closed watertight lofted skin primitives for procedural_skin (#15145)."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray

from src.shared.python.model_appearance.geometry import Mesh, closed_grid_mesh
from src.shared.python.model_appearance.hand_backends.sdf_meshing import (
    catmull_rom_spline,
)

Array = NDArray[np.float64]


def _normalize(vec: Array) -> Array:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-12:
        raise ValueError("cannot normalise a near-zero vector")
    return vec / norm


def _clamp_radial_floor(verts: Array, floor_m: float) -> Array:
    """Push any vertex inside ``floor_m`` radially outward in XY."""
    out = np.asarray(verts, dtype=float).copy()
    rho = np.hypot(out[:, 0], out[:, 1])
    inside = rho < floor_m
    if np.any(inside):
        scale = floor_m / np.maximum(rho[inside], 1e-12)
        out[inside, 0] *= scale
        out[inside, 1] *= scale
    return out


def parallel_transport_frames(
    curve: Array,
    initial_width_axis: Array,
) -> tuple[Array, Array, Array]:
    """Compute twist-free ``(tangent, u_width, v_normal)`` along ``curve``."""
    n = len(curve)
    tangents = np.empty((n, 3), dtype=float)
    for i in range(n):
        if i == 0:
            t_vec = curve[1] - curve[0]
        elif i == n - 1:
            t_vec = curve[-1] - curve[-2]
        else:
            t_vec = curve[i + 1] - curve[i - 1]
        tangents[i] = _normalize(t_vec)

    u_axes = np.empty((n, 3), dtype=float)
    v_axes = np.empty((n, 3), dtype=float)
    w0 = np.asarray(initial_width_axis, dtype=float)
    w0 = w0 - float(np.dot(w0, tangents[0])) * tangents[0]
    if float(np.linalg.norm(w0)) < 1e-8:
        fallback = (
            np.array([0.0, 0.0, 1.0])
            if abs(float(tangents[0, 2])) < 0.9
            else np.array([1.0, 0.0, 0.0])
        )
        w0 = fallback - float(np.dot(fallback, tangents[0])) * tangents[0]
    u_axes[0] = _normalize(w0)
    v_axes[0] = _normalize(np.cross(tangents[0], u_axes[0]))

    for i in range(1, n):
        t_prev, t_curr = tangents[i - 1], tangents[i]
        rot_axis = np.cross(t_prev, t_curr)
        sin_a = float(np.linalg.norm(rot_axis))
        cos_a = float(np.clip(np.dot(t_prev, t_curr), -1.0, 1.0))
        if sin_a > 1e-10:
            k = rot_axis / sin_a
            u_prev = u_axes[i - 1]
            # Rodrigues rotation aligns t_prev -> t_curr without frame twist.
            u_next = (
                u_prev * cos_a
                + np.cross(k, u_prev) * sin_a
                + k * (float(np.dot(k, u_prev)) * (1.0 - cos_a))
            )
        else:
            u_next = u_axes[i - 1].copy()
        u_next = u_next - float(np.dot(u_next, t_curr)) * t_curr
        u_axes[i] = _normalize(u_next)
        v_axes[i] = _normalize(np.cross(t_curr, u_axes[i]))
    return tangents, u_axes, v_axes


def _dense_bone_polyline(joints: Array) -> Array:
    """Insert bone midpoints so Catmull-Rom hugs straight phalanges with soft joints."""
    refined: list[Array] = [joints[0]]
    for i in range(len(joints) - 1):
        p0, p1 = joints[i], joints[i + 1]
        if i > 0:
            refined.append(0.75 * p0 + 0.25 * p1)
        refined.append(0.5 * (p0 + p1))
        if i < len(joints) - 2:
            refined.append(0.25 * p0 + 0.75 * p1)
    refined.append(joints[-1])
    return np.asarray(refined, dtype=float)


def lofted_anatomical_digit(
    spine_joints: Array,
    *,
    radii_m: tuple[float, float],
    initial_width_axis: Array,
    dorsal_hint: Array,
    sides: int,
    palmar_flatness: float = 0.86,
    joint_fractions: tuple[float, float] = (0.48, 0.77),
    grip_floor_m: float = 0.0,
) -> Mesh:
    """Watertight lofted finger/thumb with ellipsoidal caps and joint contour.

    ``radii_m`` is ``(base_radius_m, tip_radius_m)``.
    """
    base_radius_m, tip_radius_m = radii_m
    control = _dense_bone_polyline(np.asarray(spine_joints, dtype=float))
    curve = catmull_rom_spline(control, max(44, 10 * len(control)))
    diffs = np.linalg.norm(np.diff(curve, axis=0), axis=1)
    arclen = np.concatenate([[0.0], np.cumsum(diffs)])
    total = float(arclen[-1]) if arclen[-1] > 1e-9 else 1.0
    tangents, u_axes, v_axes = parallel_transport_frames(curve, initial_width_axis)

    phi = np.linspace(0.0, 2.0 * math.pi, sides, endpoint=False)
    cos_p, sin_p = np.cos(phi), np.sin(phi)
    cap_rings = 3

    def _section_at(idx: int, cap_scale: float) -> Array:
        c = curve[idx]
        t_dir, u_dir, v_dir = tangents[idx], u_axes[idx], v_axes[idx]
        frac = float(arclen[idx] / total)
        r_base = base_radius_m * (1.0 - frac) + tip_radius_m * frac
        joint_bump = 0.0
        for jf in joint_fractions:
            joint_bump += 0.055 * math.exp(-(((frac - jf) / 0.055) ** 2))
        dorsal_proj = dorsal_hint - float(np.dot(dorsal_hint, t_dir)) * t_dir
        if float(np.linalg.norm(dorsal_proj)) > 1e-8:
            dorsal_u = _normalize(dorsal_proj)
        else:
            dorsal_u = v_dir
        dot_du = float(np.dot(v_dir, dorsal_u))
        sign_d = 1.0 if dot_du >= 0.0 else -1.0
        dorsal_angle = sin_p * sign_d
        radial_mod = 1.0 + joint_bump * np.maximum(dorsal_angle, 0.0) ** 2
        flat_mod = np.where(
            dorsal_angle < 0.0,
            1.0 - (1.0 - palmar_flatness) * (-dorsal_angle) ** 2,
            1.0,
        )
        rw = 1.04 * r_base * cap_scale * radial_mod
        rd = 0.96 * r_base * cap_scale * radial_mod * flat_mod
        ring = (
            c[None, :]
            + (rw * cos_p)[:, None] * u_dir[None, :]
            + (rd * sin_p)[:, None] * v_dir[None, :]
        )
        return ring

    rings: list[Array] = []
    # Proximal hemispherical dome cap
    t0 = tangents[0]
    for k in range(1, cap_rings):
        ang = (k / cap_rings) * (0.5 * math.pi)
        offset = -t0 * (0.85 * base_radius_m * math.cos(ang))
        rings.append(_section_at(0, math.sin(ang)) + offset[None, :])
    for i in range(len(curve)):
        rings.append(_section_at(i, 1.0))
    # Distal hemispherical fingertip dome cap
    tn = tangents[-1]
    for k in range(cap_rings - 1, 0, -1):
        ang = (k / cap_rings) * (0.5 * math.pi)
        offset = tn * (0.95 * tip_radius_m * math.cos(ang))
        rings.append(_section_at(len(curve) - 1, math.sin(ang)) + offset[None, :])

    grid = np.asarray(rings, dtype=float)
    if grip_floor_m > 0.0:
        grid = _clamp_radial_floor(grid.reshape(-1, 3), grip_floor_m).reshape(
            grid.shape
        )
    top_pole = curve[0] - t0 * (0.85 * base_radius_m)
    bot_pole = curve[-1] + tn * (0.95 * tip_radius_m)
    if grip_floor_m > 0.0:
        top_pole = _clamp_radial_floor(top_pole[None, :], grip_floor_m)[0]
        bot_pole = _clamp_radial_floor(bot_pole[None, :], grip_floor_m)[0]
    return closed_grid_mesh(grid, top_pole, bot_pole)


@dataclass(frozen=True)
class PalmSlabSpec:
    """Configuration for building a cupped anatomical metacarpal palm slab."""

    wrist_center: Array
    mid_ctrl: Array
    knuckle_center: Array
    width_axis: Array
    dorsal_axis: Array
    wrist_width_m: float
    knuckle_width_m: float
    wrist_thickness_m: float
    palm_thickness_m: float
    knuckle_dome_height_m: float
    cup_depth_m: float
    knuckle_u_fractions: tuple[float, ...]
    sides: int
    grip_floor_m: float
    mcp_span_fraction: float = 0.82
    interdigital_valley_m: float = 0.0028
    distal_cap_shift_m: float = 0.0045


@dataclass(frozen=True)
class _PalmSectionFrame:
    """Centre-line sample and orthonormal section axes for one palm ring."""

    t: float
    center: Array
    tangent: Array
    w_frame: Array
    d_frame: Array


def _palm_section_ring(
    spec: PalmSlabSpec,
    frame: _PalmSectionFrame,
    angle_trig: tuple[Array, Array],
    *,
    end_scale: float = 1.0,
    t_shift_m: float = 0.0,
) -> Array:
    """Compute one transverse ring of the metacarpal palm with knuckle & valley relief."""
    cos_p, sin_p = angle_trig
    t = frame.t
    shape_exp_w = 0.94 - 0.06 * math.sin(math.pi * t)
    shape_exp_d = 0.72 - 0.06 * math.sin(math.pi * t)
    super_w = np.sign(cos_p) * (np.abs(cos_p) ** shape_exp_w)
    super_d = np.sign(sin_p) * (np.abs(sin_p) ** shape_exp_d)
    shoulder_taper = 1.0 - 0.03 * (max(0.0, (t - 0.88) / 0.12) ** 2)
    hw = (
        0.5
        * shoulder_taper
        * (
            spec.wrist_width_m * (1.0 - t)
            + spec.knuckle_width_m * t
            + 0.005 * spec.knuckle_width_m * math.sin(math.pi * t)
        )
    )
    ht = 0.5 * (
        spec.wrist_thickness_m * (1.0 - t) ** 1.35
        + spec.palm_thickness_m * (1.0 - (1.0 - t) ** 1.35)
    )
    palmar_weight = np.maximum(-sin_p, 0.0)
    central_band = np.exp(-((cos_p / 0.58) ** 2))
    edge_eminences = (
        0.10
        * spec.palm_thickness_m
        * np.exp(-(((np.abs(cos_p) - 0.72) / 0.34) ** 2))
        * math.sin(math.pi * max(0.0, min(1.0, t / 0.85)))
    )
    cup_lift = spec.cup_depth_m * math.sin(math.pi * t) * central_band * palmar_weight
    dorsal_weight = np.maximum(sin_p, 0.0)
    arch = 0.04 * spec.palm_thickness_m * (1.0 - cos_p**2) * dorsal_weight
    u_knuckles = [uf * spec.mcp_span_fraction for uf in spec.knuckle_u_fractions]
    knuckle_bump = np.zeros_like(cos_p)
    for uk in u_knuckles:
        knuckle_bump += spec.knuckle_dome_height_m * np.exp(
            -(((super_w - uk) / 0.140) ** 2)
        )
    valley_dip = np.zeros_like(cos_p)
    for j in range(len(u_knuckles) - 1):
        uv = 0.5 * (u_knuckles[j] + u_knuckles[j + 1])
        valley_dip += spec.interdigital_valley_m * np.exp(
            -(((super_w - uv) / 0.115) ** 2)
        )
    relief = (knuckle_bump - valley_dip) * math.exp(-(((t - 0.92) / 0.16) ** 2))
    relief *= np.clip((sin_p + 0.05) / 0.35, 0.0, 1.0) * (end_scale**1.6)
    w_coords = end_scale * hw * super_w
    d_coords = (
        end_scale * ht * super_d
        + end_scale * (arch + cup_lift - edge_eminences * palmar_weight)
        + relief
    )
    c_pos = frame.center + frame.tangent * t_shift_m
    return (
        c_pos[None, :]
        + w_coords[:, None] * frame.w_frame[None, :]
        + d_coords[:, None] * frame.d_frame[None, :]
    )


def _palm_section_frames(
    spec: PalmSlabSpec, ts: Array, centers: Array, tangents: Array
) -> list[_PalmSectionFrame]:
    """Transport-aligned orthonormal section frames prevent wrist Gram-Schmidt twist."""
    n_samples = len(ts)
    w_frames: list[Array] = [
        _normalize(
            spec.width_axis - float(np.dot(spec.width_axis, tangents[i])) * tangents[i]
        )
        for i in range(n_samples)
    ]
    cross_end = np.cross(tangents[-1], w_frames[-1])
    sign_d = 1.0 if float(np.dot(cross_end, spec.dorsal_axis)) >= 0.0 else -1.0
    d_frames: list[Array] = [
        _normalize(sign_d * np.cross(tangents[i], w_frames[i]))
        for i in range(n_samples)
    ]
    return [
        _PalmSectionFrame(
            t=float(ts[i]),
            center=centers[i],
            tangent=tangents[i],
            w_frame=w_frames[i],
            d_frame=d_frames[i],
        )
        for i in range(n_samples)
    ]


def _palm_rings(
    spec: PalmSlabSpec,
    frames: list[_PalmSectionFrame],
    trig: tuple[Array, Array],
    cap_steps: int,
) -> list[Array]:
    """Wrist dome cap, body sections and distal knuckle cap rings, proximal first."""
    rings: list[Array] = []
    for k in range(1, cap_steps):
        ang = (k / cap_steps) * (0.5 * math.pi)
        rings.append(
            _palm_section_ring(
                spec,
                frames[0],
                trig,
                end_scale=math.sin(ang),
                t_shift_m=-0.011 * math.cos(ang),
            )
        )
    for frame in frames:
        rings.append(_palm_section_ring(spec, frame, trig))
    for k in range(cap_steps - 1, 0, -1):
        ang = (k / cap_steps) * (0.5 * math.pi)
        rings.append(
            _palm_section_ring(
                spec,
                frames[-1],
                trig,
                end_scale=math.sin(ang),
                t_shift_m=spec.distal_cap_shift_m * math.cos(ang),
            )
        )
    return rings


def _measured_cup_depth(
    mid_ring: Array, mid_frame: _PalmSectionFrame, trig: tuple[Array, Array]
) -> float:
    """Palmar centre-vs-border depth of the mid-palm ring along the dorsal axis."""
    cos_p, sin_p = trig
    palmar_idx = np.where(sin_p < -0.55)[0]
    center_idx = palmar_idx[np.argmin(np.abs(cos_p[palmar_idx]))]
    border_idx = palmar_idx[np.argmax(np.abs(cos_p[palmar_idx]))]
    rel_c = float(np.dot(mid_ring[center_idx] - mid_frame.center, mid_frame.d_frame))
    rel_b = float(np.dot(mid_ring[border_idx] - mid_frame.center, mid_frame.d_frame))
    return max(0.0, rel_c - rel_b)


def lofted_palm_slab(spec: PalmSlabSpec) -> tuple[Mesh, Array, float, float]:
    """Build a closed cupped metacarpal palm slab (wrist crease to knuckle line).

    Returns ``(palm_mesh, mcp_surface_anchors_world, measured_cup_depth_m, palm_arc_length_m)``.
    """
    n_samples = 24
    ts = np.linspace(0.0, 1.0, n_samples, dtype=np.float64)
    centers = np.array(
        [
            ((1.0 - t) ** 2) * spec.wrist_center
            + 2.0 * (1.0 - t) * t * spec.mid_ctrl
            + (t**2) * spec.knuckle_center
            for t in ts
        ],
        dtype=float,
    )
    tangents, _, _ = parallel_transport_frames(centers, spec.width_axis)

    phi = np.linspace(0.0, 2.0 * math.pi, spec.sides, endpoint=False)
    cos_p, sin_p = np.cos(phi), np.sin(phi)
    trig = (cos_p, sin_p)
    frames = _palm_section_frames(spec, ts, centers, tangents)

    cap_steps = 4
    grid = np.asarray(_palm_rings(spec, frames, trig, cap_steps), dtype=float)
    if spec.grip_floor_m > 0.0:
        grid = _clamp_radial_floor(grid.reshape(-1, 3), spec.grip_floor_m).reshape(
            grid.shape
        )
    top_pole = centers[0] - tangents[0] * 0.011
    bot_pole = centers[-1] + tangents[-1] * spec.distal_cap_shift_m
    if spec.grip_floor_m > 0.0:
        top_pole = _clamp_radial_floor(top_pole[None, :], spec.grip_floor_m)[0]
        bot_pole = _clamp_radial_floor(bot_pole[None, :], spec.grip_floor_m)[0]
    mesh = closed_grid_mesh(grid, top_pole, bot_pole)

    half_mcp_span = 0.5 * spec.mcp_span_fraction * spec.knuckle_width_m
    distal = frames[-1]
    mcp_anchors = np.array(
        [
            distal.center
            + uf * half_mcp_span * distal.w_frame
            + (0.12 * spec.palm_thickness_m * (1.0 - uf * uf)) * distal.d_frame
            for uf in spec.knuckle_u_fractions
        ],
        dtype=float,
    )
    palm_arc_length_m = float(np.sum(np.linalg.norm(np.diff(centers, axis=0), axis=1)))
    mid_ring = grid[(cap_steps - 1) + n_samples // 2]
    measured_cup = _measured_cup_depth(mid_ring, frames[n_samples // 2], trig)
    return mesh, mcp_anchors, measured_cup, palm_arc_length_m


__all__ = [
    "PalmSlabSpec",
    "lofted_anatomical_digit",
    "lofted_palm_slab",
    "parallel_transport_frames",
]
