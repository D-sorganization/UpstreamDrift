"""Sampled net-club-wrench equivalence for fixed-motion allocations (#15494).

This diagnostic reuses the canonical grasp map. It does not admit a motion,
solve an allocation, certify feasibility, or execute forward dynamics.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from src.shared.python.loop_nullspace.evidence import (
    GripFrame,
    check_shape,
    readonly_float,
)
from src.shared.python.loop_nullspace.grip_split import grasp_map

__all__ = ["compare_hand_wrench_histories"]


def compare_hand_wrench_histories(
    times_s: ArrayLike,
    grip_frames: Sequence[GripFrame],
    lead_points_world_m: ArrayLike,
    trail_points_world_m: ArrayLike,
    baseline: ArrayLike,
    candidate: ArrayLike,
    *,
    force_atol_n: float,
    moment_atol_nm: float,
) -> dict[str, Any]:
    """Compare net wrenches about each sample's shared grip midpoint.

    Preconditions: both histories use the same sampled motion/timebase and
    geometry. Each row is ``(f_lead, c_lead, f_trail, c_trail)`` in grip-frame
    axes, hand-on-club sign, N and N m; couples are free moments at the hand
    points. Points are world positions in metres. Callers must align histories
    and convert native frames/signs before calling; no interpolation is done.

    Postconditions: ``equivalent`` is true iff every sampled net-force and
    net-moment error norm meets its own absolute tolerance. The JSON-compatible
    report includes all residuals, times, tolerances and failed sample indices.
    No across-family invariant, between-sample bound, feasibility or replay
    claim follows from this sampled comparison.

    Raises:
        ValueError: for empty, non-finite, mis-shaped or unordered series,
            invalid geometry, or non-finite/negative tolerances.
        TypeError: if a frame is not a ``GripFrame``.
    """
    times = readonly_float(times_s, "times_s")
    if times.ndim != 1 or times.size == 0 or np.any(np.diff(times) <= 0):
        raise ValueError("times_s must be non-empty, 1-D and strictly increasing")
    frames = tuple(grip_frames)
    count = times.size
    if len(frames) != count:
        raise ValueError("grip_frames must have one frame per sample")
    if any(not isinstance(frame, GripFrame) for frame in frames):
        raise TypeError("grip_frames must contain only GripFrame instances")
    arrays = []
    for value, name, width in (
        (lead_points_world_m, "lead_points_world_m", 3),
        (trail_points_world_m, "trail_points_world_m", 3),
        (baseline, "baseline", 12),
        (candidate, "candidate", 12),
    ):
        array = readonly_float(value, name)
        check_shape(array, (count, width), name)
        arrays.append(array)
    lead, trail, base, proposed = arrays
    for name, value in (
        ("force_atol_n", force_atol_n),
        ("moment_atol_nm", moment_atol_nm),
    ):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")

    residuals = np.empty((count, 6))
    for k, frame in enumerate(frames):
        mapping = grasp_map(frame, lead[k], trail[k])
        residuals[k] = mapping @ (proposed[k] - base[k])
    if not np.all(np.isfinite(residuals)):
        raise ValueError("net wrench residuals overflowed")
    # hypot avoids squaring large finite residuals during norm evaluation.
    force_error = np.hypot.reduce(residuals[:, :3], axis=1)
    moment_error = np.hypot.reduce(residuals[:, 3:], axis=1)
    if not (np.all(np.isfinite(force_error)) and np.all(np.isfinite(moment_error))):
        raise ValueError("net wrench error norms overflowed")
    failed = np.flatnonzero(
        (force_error > force_atol_n) | (moment_error > moment_atol_nm)
    )
    return {
        "schema": "loop_nullspace/wrench_comparison/1",
        "evidence_scope": "sampled_net_wrench_equivalence",
        "reference_point": "shared_grip_midpoint",
        "component_axes": "grip",
        "wrench_sign": "hand_on_club",
        "times_s": times.tolist(),
        "force_atol_n": float(force_atol_n),
        "moment_atol_nm": float(moment_atol_nm),
        "force_error_n": force_error.tolist(),
        "moment_error_nm": moment_error.tolist(),
        "max_force_error_n": float(np.max(force_error)),
        "max_moment_error_nm": float(np.max(moment_error)),
        "failed_sample_indices": failed.tolist(),
        "equivalent": failed.size == 0,
        "feasibility_verified": False,
        "forward_replay_verified": False,
    }
