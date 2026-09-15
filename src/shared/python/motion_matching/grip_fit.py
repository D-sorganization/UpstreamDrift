"""Fit the constant hand-to-club rotation from a matched swing (MM-2, #10104).

The capture carries no hand markers, so the golfer's hand frame is observed
only through the club. The wrist coordinates of the anthropometric models
(forearm pronation ``Rz`` about the forearm, then wrist cock ``Rx`` and
flexion ``Rz`` about the wrist base) therefore depend on one unobserved
constant: the rotation of the hand relative to the wrist follower frame
(``child_to_follower`` of the wrist joint). If that constant is wrong, the
cock coordinate absorbs motion that belongs to flexion and pronation and the
matched swing leaves the human ranges although the golfer never did.

This module fits that constant. For every frame of an unbounded match the
rotation from the proximal forearm to the hand (or club) body is known;
``wrist_angles`` decomposes it exactly into the three coordinates, and
``fit_grip_rotation`` searches the constant rotation ``E`` that, composed on
the follower side of the wrist joint, minimises the coordinate excursions
beyond the human ranges over the whole swing. The fitted rotation is applied
to the document by the builder (``grip_rotation_deg``), so it never enters the
equations of motion as a constraint: the wrist joints keep their two free
coordinates and only the hand's resting orientation on them changes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

Array: TypeAlias = NDArray[np.float64]


#: Wrist chain after the pronation joint: ``Rz(pronation) @ base @ Rx(cock) @
#: Rz(flexion)``. With the anthropometric wrist base (``base.T @ z == -y``) the
#: chain is the intrinsic ``Y X Z`` Euler sequence with the first angle negated.
WRIST_BASE_AXIS = np.array([0.0, -1.0, 0.0])
_EULER: Literal["YXZ"] = "YXZ"
ANGLE_NAMES = ("pronation", "cock", "flexion")


def _matrices(relative: Array) -> Array:
    m = np.asarray(relative, dtype=float)
    if m.ndim != 3 or m.shape[1:] != (3, 3) or m.shape[0] == 0:
        raise ValueError("relative must be a non-empty stack of 3x3 rotations")
    if not np.isfinite(m).all():
        raise ValueError("relative rotations must be finite")
    return m


def _base(wrist_base: Array) -> Array:
    b = np.asarray(wrist_base, dtype=float)
    if b.shape != (3, 3) or not np.allclose(b @ b.T, np.eye(3), atol=1e-9):
        raise ValueError("wrist_base must be a 3x3 rotation")
    if not np.allclose(b.T @ np.array([0.0, 0.0, 1.0]), WRIST_BASE_AXIS, atol=1e-9):
        raise ValueError("wrist_base must carry the forearm axis to -y")
    return b


def grip_rotation(euler_deg: Sequence[float] | Array) -> Array:
    """Constant follower-side rotation from extrinsic x-y-z angles in degrees."""
    e = np.asarray(euler_deg, dtype=float)
    if e.shape != (3,) or not np.isfinite(e).all():
        raise ValueError("euler_deg must be three finite angles")
    return np.asarray(Rotation.from_euler("xyz", e, degrees=True).as_matrix())


def wrist_angles(relative: Array, wrist_base: Array) -> Array:
    """Exact (pronation, cock, flexion) in radians, shape (frames, 3), such
    that ``Rz(p) @ wrist_base @ Rx(c) @ Rz(f)`` equals each relative rotation.
    """
    m = _matrices(relative)
    b = _base(wrist_base)
    euler = Rotation.from_matrix(b.T @ m).as_euler(_EULER)
    angles = np.asarray(euler, dtype=float)
    angles[:, 0] = -angles[:, 0]
    return angles


def wrist_rotation(angles: Array, wrist_base: Array) -> Array:
    """Inverse of :func:`wrist_angles`: relative rotations from angles (rad)."""
    a = np.asarray(angles, dtype=float)
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError("angles must be (frames, 3)")
    b = _base(wrist_base)
    euler = a.copy()
    euler[:, 0] = -euler[:, 0]
    return np.asarray(b @ Rotation.from_euler(_EULER, euler).as_matrix())


def excess_deg(
    angles_deg: Array, ranges_deg: Sequence[Sequence[float]] | Array
) -> Array:
    """Degrees beyond each coordinate's (low, high) range, zero inside."""
    a = np.asarray(angles_deg, dtype=float)
    r = np.asarray(ranges_deg, dtype=float)
    if a.ndim != 2 or r.shape != (a.shape[1], 2) or (r[:, 0] >= r[:, 1]).any():
        raise ValueError("ranges must be (coordinates, 2) with low < high")
    return np.maximum(0.0, np.maximum(r[:, 0] - a, a - r[:, 1]))


@dataclass(frozen=True)
class GripFit:
    """Result of :func:`fit_grip_rotation`."""

    rotation_deg: tuple[float, float, float]
    angles_before_deg: Array
    angles_after_deg: Array
    cost_before: float
    cost_after: float
    max_excess_before_deg: tuple[float, float, float]
    max_excess_after_deg: tuple[float, float, float]

    def as_document(self) -> dict[str, object]:
        return {
            "rotation_deg": [round(v, 3) for v in self.rotation_deg],
            "cost_before_rms_excess_deg": round(self.cost_before, 3),
            "cost_after_rms_excess_deg": round(self.cost_after, 3),
            "max_excess_before_deg": dict(
                zip(
                    ANGLE_NAMES,
                    [round(v, 1) for v in self.max_excess_before_deg],
                    strict=True,
                )
            ),
            "max_excess_after_deg": dict(
                zip(
                    ANGLE_NAMES,
                    [round(v, 1) for v in self.max_excess_after_deg],
                    strict=True,
                )
            ),
            "angle_range_before_deg": {
                n: [round(float(v), 1) for v in (c.min(), c.max())]
                for n, c in zip(ANGLE_NAMES, self.angles_before_deg.T, strict=True)
            },
            "angle_range_after_deg": {
                n: [round(float(v), 1) for v in (c.min(), c.max())]
                for n, c in zip(ANGLE_NAMES, self.angles_after_deg.T, strict=True)
            },
        }


def fit_grip_rotation(
    relative: Array,
    wrist_base: Array,
    ranges_deg: Sequence[Sequence[float]],
    *,
    grid_half_deg: float = 90.0,
    grid_step_deg: float = 15.0,
    weights: Sequence[float] | None = None,
) -> GripFit:
    """Fit the constant follower-side rotation that keeps the wrist
    coordinates inside ``ranges_deg`` over the whole swing.

    ``relative`` holds, per frame, the rotation from the proximal forearm to
    the hand body with the current document; the fitted rotation ``E`` is
    the one under which ``relative @ E`` decomposes into the smallest
    root-mean-square excess beyond the ranges. A grid over extrinsic x-y-z
    angles seeds a Nelder-Mead refinement. Postcondition: ``cost_after <=
    cost_before`` (the identity is in the grid).
    """
    m = _matrices(relative)
    b = _base(wrist_base)
    r = np.asarray(ranges_deg, dtype=float)
    w = np.ones(3) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != (3,) or (w < 0).any():
        raise ValueError("weights must be three non-negative numbers")
    if grid_half_deg <= 0 or grid_step_deg <= 0:
        raise ValueError("grid extent and step must be positive")

    def angles_for(euler_deg: Array) -> Array:
        turned: Array = np.asarray(m @ grip_rotation(euler_deg))
        return np.degrees(wrist_angles(turned, b))

    def cost(euler_deg: Array) -> float:
        ex = excess_deg(angles_for(euler_deg), r) * w
        return float(np.sqrt(np.mean(ex**2)))

    ticks = np.arange(-grid_half_deg, grid_half_deg + 1e-9, grid_step_deg)
    ticks = np.union1d(ticks, [0.0])
    best: Array = np.zeros(3, dtype=float)
    best_cost = cost(best)
    for candidate in product(ticks, ticks, ticks):
        trial: Array = np.array(candidate, dtype=float)
        c = cost(trial)
        if c < best_cost - 1e-12:
            best, best_cost = trial, c
    refined = minimize(
        cost, best, method="Nelder-Mead", options={"xatol": 0.01, "fatol": 1e-6}
    )
    if refined.fun < best_cost:
        best = np.array(refined.x, dtype=float)
        best_cost = float(refined.fun)
    before = angles_for(np.zeros(3))
    after = angles_for(best)
    return GripFit(
        rotation_deg=(float(best[0]), float(best[1]), float(best[2])),
        angles_before_deg=before,
        angles_after_deg=after,
        cost_before=cost(np.zeros(3)),
        cost_after=best_cost,
        max_excess_before_deg=tuple(excess_deg(before, r).max(axis=0)),
        max_excess_after_deg=tuple(excess_deg(after, r).max(axis=0)),
    )
