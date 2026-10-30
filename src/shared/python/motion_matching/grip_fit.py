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

from src.shared.python.motion_matching._geodesic import (
    quaternion_geodesic_angles,
    rotmat_to_quaternion,
)

Array: TypeAlias = NDArray[np.float64]

#: Default per-hand neutral extrinsic x-y-z Euler triples (HRG-GR1 / HRG-HX1):
#: left/lead hand is ``(-75.0, 0.0, 0.0)`` (``M_{0,L} = R_x(-50 deg)``);
#: right/trail hand is the mirrored neutral ``(75.0, 0.0, 180.0)``
#: (``M_{0,R} = R_x(-50 deg) R_z(180 deg)``).
_DEFAULT_NEUTRAL_EULER_BY_SIDE_DEG: dict[str, tuple[float, float, float]] = {
    "L": (-75.0, 0.0, 0.0),
    "lead": (-75.0, 0.0, 0.0),
    "R": (75.0, 0.0, 180.0),
    "trail": (75.0, 0.0, 180.0),
}


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


def compose_rotation_deg(
    base_deg: Sequence[float] | Array, correction_deg: Sequence[float] | Array
) -> tuple[float, float, float]:
    """Extrinsic x-y-z degrees of ``G(base) @ G(correction)``.

    A fit made on a document that already carries ``base_deg`` yields a
    correction; this is the total to record in the document. Postcondition:
    ``grip_rotation(result)`` equals the matrix product to rounding.
    """
    total = grip_rotation(base_deg) @ grip_rotation(correction_deg)
    x, y, z = Rotation.from_matrix(total).as_euler("xyz", degrees=True)
    return (float(x), float(y), float(z))


def _count(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return int(value)


def weighted_frames(
    stack: Array,
    impact_frame: int | None = None,
    *,
    window_frames: int = 0,
    window_weight: int = 0,
    address_frames: int = 0,
    address_weight: int = 0,
) -> Array:
    """The fit input of one run: the whole stack once, plus the
    ``window_frames`` frames before ``impact_frame`` repeated ``window_weight``
    times, plus the first ``address_frames`` frames repeated
    ``address_weight`` times (concatenated along axis 0).

    Repetition weights the squared excess of those frames in the fit. With
    all weights zero the stack is returned unchanged. Raises ``ValueError``
    for negative or non-integer counts, a missing or out-of-range
    ``impact_frame`` when the window is active, or ``address_frames`` beyond
    the stack.
    """
    m = np.asarray(stack, dtype=float)
    n_window = _count(window_frames, "window_frames")
    w_window = _count(window_weight, "window_weight")
    n_address = _count(address_frames, "address_frames")
    w_address = _count(address_weight, "address_weight")
    parts = [m]
    if n_window * w_window > 0:
        if impact_frame is None or not n_window <= impact_frame <= len(m):
            raise ValueError(
                f"impact_frame must satisfy window_frames ({n_window}) <= "
                f"impact_frame <= {len(m)}, got {impact_frame!r}"
            )
        parts += [m[impact_frame - n_window : impact_frame]] * w_window
    if n_address > len(m):
        raise ValueError(
            f"address_frames ({n_address}) exceeds the {len(m)} frames given"
        )
    if n_address * w_address > 0:
        parts += [m[:n_address]] * w_address
    return np.concatenate(parts, axis=0)


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


def _prior_weight(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(
            f"neutral_prior_weight_deg2 must be a finite non-negative float, got {value!r}"
        )
    w = float(value)
    if not np.isfinite(w) or w < 0.0:
        raise ValueError(
            f"neutral_prior_weight_deg2 must be a finite non-negative float, got {value!r}"
        )
    return w


def _resolve_neutral_euler(
    neutral_rotation_deg: Sequence[float] | Array | None, side: str
) -> tuple[float, float, float]:
    if not isinstance(side, str) or side not in _DEFAULT_NEUTRAL_EULER_BY_SIDE_DEG:
        raise ValueError(
            f"side must be one of {sorted(_DEFAULT_NEUTRAL_EULER_BY_SIDE_DEG)}, got {side!r}"
        )
    if neutral_rotation_deg is None:
        return _DEFAULT_NEUTRAL_EULER_BY_SIDE_DEG[side]
    e = np.asarray(neutral_rotation_deg, dtype=float)
    if e.shape != (3,) or not np.isfinite(e).all():
        raise ValueError("neutral_rotation_deg must be three finite angles")
    return (float(e[0]), float(e[1]), float(e[2]))


class ExcessObjective:
    """The ROM-excess cost of a stack of relative rotations (shared by the
    legacy search and the HRG-GR3 modes, so the cost is defined once).

    ``cost(rotation)`` is the weighted root-mean-square excess, in degrees,
    of ``relative @ rotation`` beyond ``ranges_deg``, plus an optional
    geodesic neutral prior ``neutral_prior_weight_deg2 * d_SO3(rotation,
    neutral_rotation)^2``; ``angles_deg`` returns the decomposed coordinates.
    ``ValueError`` for bad shapes or weights.
    """

    def __init__(
        self,
        relative: Array,
        wrist_base: Array,
        ranges_deg: Sequence[Sequence[float]],
        weights: Sequence[float] | None = None,
        *,
        neutral_prior_weight_deg2: float = 0.0,
        neutral_rotation_deg: Sequence[float] | Array | None = None,
        side: str = "L",
    ) -> None:
        self.relative = _matrices(relative)
        self.wrist_base = _base(wrist_base)
        self.ranges = np.asarray(ranges_deg, dtype=float)
        self.weights = np.ones(3) if weights is None else np.asarray(weights, float)
        if (
            self.weights.shape != (3,)
            or not np.isfinite(self.weights).all()
            or (self.weights < 0).any()
        ):
            raise ValueError("weights must be three non-negative numbers")
        self.neutral_prior_weight_deg2 = _prior_weight(neutral_prior_weight_deg2)
        self.side = side
        self.neutral_rotation_deg = _resolve_neutral_euler(neutral_rotation_deg, side)
        self.neutral_rotation = grip_rotation(self.neutral_rotation_deg)
        self._neutral_q = rotmat_to_quaternion(self.neutral_rotation[np.newaxis, :, :])

    def angles_deg(self, rotation: Array) -> Array:
        turned: Array = np.asarray(self.relative @ rotation)
        return np.degrees(wrist_angles(turned, self.wrist_base))

    def neutral_deviation_deg(self, rotation: Array) -> float:
        r = np.asarray(rotation, dtype=float)
        if r.shape != (3, 3) or not np.isfinite(r).all():
            raise ValueError("rotation must be a finite 3x3 matrix")
        q = rotmat_to_quaternion(r[np.newaxis, :, :])
        angle_rad = float(quaternion_geodesic_angles(q, self._neutral_q)[0])
        return float(np.degrees(angle_rad))

    def cost(self, rotation: Array) -> float:
        ex = excess_deg(self.angles_deg(rotation), self.ranges) * self.weights
        rom_cost = float(np.sqrt(np.mean(ex**2)))
        if self.neutral_prior_weight_deg2 == 0.0:
            return rom_cost
        dev_deg = self.neutral_deviation_deg(rotation)
        return float(rom_cost + self.neutral_prior_weight_deg2 * (dev_deg**2))


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
    neutral_deviation_before_deg: float = 0.0
    neutral_deviation_after_deg: float = 0.0
    neutral_prior_weight_deg2: float = 0.0

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
    neutral_prior_weight_deg2: float = 0.0,
    side: str | Sequence[float] | Array = "L",
) -> GripFit:
    """Fit the constant follower-side rotation that keeps the wrist
    coordinates inside ``ranges_deg`` over the whole swing.

    ``relative`` holds, per frame, the rotation from the proximal forearm to
    the hand body with the current document; the fitted rotation ``E`` is
    the one under which ``relative @ E`` decomposes into the smallest
    root-mean-square excess beyond the ranges (plus an optional geodesic
    prior ``neutral_prior_weight_deg2 * d_SO3(E, G(neutral_rotation_deg))^2``,
    where ``side`` selects ``"L"``/``"R"`` or an explicit 3-angle target).
    A grid over extrinsic x-y-z angles seeds a Nelder-Mead refinement.
    Postcondition: ``cost_after <= cost_before`` (the identity is in the grid).
    """
    side_key = side if isinstance(side, str) else "L"
    neutral_deg = None if isinstance(side, str) else side
    objective = ExcessObjective(
        relative,
        wrist_base,
        ranges_deg,
        weights,
        neutral_prior_weight_deg2=neutral_prior_weight_deg2,
        neutral_rotation_deg=neutral_deg,
        side=side_key,
    )
    r = objective.ranges
    if grid_half_deg <= 0 or grid_step_deg <= 0:
        raise ValueError("grid extent and step must be positive")

    def angles_for(euler_deg: Array) -> Array:
        return objective.angles_deg(grip_rotation(euler_deg))

    def cost(euler_deg: Array) -> float:
        return objective.cost(grip_rotation(euler_deg))

    ticks = np.arange(-grid_half_deg, grid_half_deg + 1e-9, grid_step_deg)
    ticks = np.union1d(ticks, [0.0])
    best: Array = np.zeros(3, dtype=float)
    best_cost = cost(best)
    candidates: list[tuple[float, float, float]] = list(product(ticks, ticks, ticks))
    if objective.neutral_prior_weight_deg2 > 0.0:
        candidates.append(objective.neutral_rotation_deg)
    for candidate in candidates:
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
        neutral_deviation_before_deg=objective.neutral_deviation_deg(
            grip_rotation(np.zeros(3))
        ),
        neutral_deviation_after_deg=objective.neutral_deviation_deg(
            grip_rotation(best)
        ),
        neutral_prior_weight_deg2=objective.neutral_prior_weight_deg2,
    )
