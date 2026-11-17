"""Finger flexion solver for the procedural skin backend.

Curls each digit chain about its flexion axis until the palmar skin of every
segment reaches its target clearance from the grip.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def unit(vec: Array) -> Array:
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 1e-12 else np.array([1.0, 0.0, 0.0], dtype=float)


def radial(theta: float) -> Array:
    return np.array([math.cos(theta), math.sin(theta), 0.0], dtype=float)


def tangential(theta: float) -> Array:
    return np.array([-math.sin(theta), math.cos(theta), 0.0], dtype=float)


def _rodrigues(vec: Array, axis: Array, angle_rad: float) -> Array:
    k = unit(axis)
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return vec * c + np.cross(k, vec) * s + k * (float(np.dot(k, vec)) * (1.0 - c))


@dataclass(frozen=True)
class FingerSolveSpec:
    """Target grip wrap and distal-pad seating contract for a finger chain."""

    grip_radius_m: float
    target_clearance_m: float
    depth_factor: float
    palmar_scale: float
    distal_pad_t_start: float = 0.65
    min_full_clearance_m: float = -0.00015
    stage_clearances_m: tuple[float, float, float] = (0.0018, 0.0016, 0.0005)


def _segment_skin_clearance(
    p0: Array,
    p1: Array,
    r0: float,
    r1: float,
    solve: FingerSolveSpec,
    *,
    t_start: float = 0.15,
    n_samples: int = 9,
) -> float:
    ts = np.linspace(t_start, 1.0, n_samples)
    pts = p0[None, :] + ts[:, None] * (p1 - p0)[None, :]
    radii = r0 * (1.0 - ts) + r1 * ts
    rho = np.hypot(pts[:, 0], pts[:, 1])
    return float(np.min(rho - solve.palmar_scale * radii - solve.grip_radius_m))


def _solve_flex_angle(
    p_start: Array,
    u_in: Array,
    flex_axis: Array,
    length: float,
    radii: tuple[float, float],
    solve: FingerSolveSpec,
) -> float:
    """Solve joint flexion (rad) seating distal pad without mid-segment penetration."""
    r0, r1 = radii
    thetas = np.linspace(0.05, 1.75, 36)
    distal_gaps = np.empty(len(thetas), dtype=float)
    full_gaps = np.empty(len(thetas), dtype=float)
    for idx, th in enumerate(thetas):
        p_end = p_start + length * _rodrigues(u_in, flex_axis, float(th))
        distal_gaps[idx] = _segment_skin_clearance(
            p_start,
            p_end,
            r0,
            r1,
            solve,
            t_start=solve.distal_pad_t_start,
            n_samples=6,
        )
        full_gaps[idx] = _segment_skin_clearance(
            p_start, p_end, r0, r1, solve, t_start=0.15, n_samples=10
        )

    safe_mask = full_gaps >= solve.min_full_clearance_m
    safe_idx = np.where(safe_mask)[0]
    if len(safe_idx) == 0:
        return float(thetas[int(np.argmax(full_gaps))])

    # First angle where distal pad reaches target while full segment stays outside shaft.
    hits = [
        int(i) for i in safe_idx if i > 0 and distal_gaps[i] <= solve.target_clearance_m
    ]
    if hits:
        idx = hits[0]
        lo, hi = float(thetas[idx - 1]), float(thetas[idx])
        if distal_gaps[idx - 1] > solve.target_clearance_m:
            for _ in range(14):
                mid = 0.5 * (lo + hi)
                p_end = p_start + length * _rodrigues(u_in, flex_axis, mid)
                g_dist = _segment_skin_clearance(
                    p_start,
                    p_end,
                    r0,
                    r1,
                    solve,
                    t_start=solve.distal_pad_t_start,
                    n_samples=6,
                )
                g_full = _segment_skin_clearance(
                    p_start, p_end, r0, r1, solve, t_start=0.15, n_samples=10
                )
                if (
                    g_dist > solve.target_clearance_m
                    and g_full >= solve.min_full_clearance_m
                ):
                    lo = mid
                elif g_full < solve.min_full_clearance_m:
                    hi = mid
                else:
                    hi = mid
            return 0.5 * (lo + hi)
        return float(thetas[idx])

    best = safe_idx[
        int(np.argmin(np.abs(distal_gaps[safe_idx] - solve.target_clearance_m)))
    ]
    return float(thetas[best])


def solve_finger_chain(
    mcp: Array,
    u_flat: Array,
    dorsal: Array,
    lengths: tuple[float, float, float],
    base_r: float,
    tip_r: float,
    solve: FingerSolveSpec,
) -> tuple[Array, Array]:
    total_len = sum(lengths)
    cumulative = (
        0.0,
        lengths[0] / total_len,
        (lengths[0] + lengths[1]) / total_len,
        1.0,
    )
    # Keep flexion axis consistent across all phalanges so lead never hyperextends.
    flex_axis = unit(np.asarray(np.cross(u_flat, -dorsal), dtype=np.float64))
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
            (r0, r1),
            replace(solve, target_clearance_m=solve.stage_clearances_m[k]),
        )
        th_open = 0.05 if k == 0 else 0.04
        th = (1.0 - solve.depth_factor) * th_open + solve.depth_factor * th_closed
        u_next = _rodrigues(u_curr, flex_axis, th)
        if k == 0:
            prox_dir = u_next.copy()
        p_curr = p_curr + seg_len * u_next
        pts.append(p_curr.copy())
        u_curr = u_next
    return np.asarray(pts, dtype=float), prox_dir


__all__ = [
    "FingerSolveSpec",
    "radial",
    "solve_finger_chain",
    "tangential",
    "unit",
]
