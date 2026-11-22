"""Kinematic shaft-bend fit of the chain shaft to club markers (#15150 C10).

A rigid club fitted to head and shaft markers splits a real shaft bend between
them. Keeping the club (grip) body fixed, this solves over the whole trajectory
for the bend of the :mod:`club_shaft_chain` joints (``Rx`` then ``Rz`` about
club ``x`` and ``z`` at ``(0, y_i, HEAD_HEIGHT_M)``) that carries the head
markers onto their targets while the shaft markers stay on theirs.

Identifiability (:func:`observability`): per plane a rigid head triad sees the
tip slope and tip offset only, and a shaft marker past joint 0 adds that joint
weakly; torsion is not fitted (no chain twist coordinate; the head-triad roll
is confounded with the face-roll calibration). The bend is therefore a sum of
static beam shapes (:func:`shape_basis`) whose amplitudes are tip slopes,
bounded and smoothed by ``shaft_bend_fit_settings.json``. Kinematic only.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.shared.python.body_part_viz.fitters._kabsch import kabsch_rotation
from src.shared.python.motion_matching.club_models import HEAD_HEIGHT_M

Array = NDArray[np.float64]
SETTINGS_PATH = Path(__file__).with_name("shaft_bend_fit_settings.json")
#: Bending axes of each chain joint, in primitive order (club x, then z).
AXES = np.eye(3)[[0, 2]]
MAX_SHAPES = 2


@dataclass(frozen=True)
class BendSettings:
    """Validated ``shaft_bend_fit_settings.json`` (see its ``sources``)."""

    shapes_per_plane: int
    smoothness_weight_m_per_rad: float
    ridge_weight_m_per_rad: float
    max_tip_slope_deg: float
    rank_relative_tolerance: float
    solver_max_nfev: int
    solver_tolerance: float

    def __post_init__(self) -> None:
        weights = (self.smoothness_weight_m_per_rad, self.ridge_weight_m_per_rad)
        if not (
            self.shapes_per_plane in range(1, MAX_SHAPES + 1)
            and all(math.isfinite(w) and w >= 0 for w in weights)
            and 0 < self.max_tip_slope_deg < 90
            and 0 < self.rank_relative_tolerance < 1
            and self.solver_tolerance > 0
            and self.solver_max_nfev >= 1
        ):
            raise ValueError(f"invalid shaft-bend settings: {self}")


def load_bend_settings(path: Path | None = None) -> BendSettings:
    """Read the settings file; ``KeyError`` for a missing key, else as the class."""
    raw = json.loads((path or SETTINGS_PATH).read_text(encoding="utf-8"))
    return BendSettings(**{k: raw[k] for k in BendSettings.__dataclass_fields__})


@dataclass(frozen=True)
class ChainGeometry:
    """Chain joint points (club-body frame, tip at ``y = 0``) and stiffnesses."""

    joint_points_m: Array
    bend_stiffness: Array

    def __post_init__(self) -> None:
        if self.joint_points_m.ndim != 2 or self.joint_points_m.shape[1] != 3:
            raise ValueError("joint_points_m must be (N, 3)")
        if self.bend_stiffness.shape != (len(self.joint_points_m),):
            raise ValueError("one bending stiffness per joint")
        if not np.all(np.diff(self.joint_points_m[:, 1]) > 0):
            raise ValueError("joints must run from the grip toward the tip (+y)")

    @classmethod
    def from_chain(cls, chain: Any) -> ChainGeometry:
        """Geometry of a ``club_shaft_chain.ShaftChain``."""
        y = np.asarray(chain.joint_s_m, dtype=float) - chain.span_m
        points = np.column_stack([np.zeros_like(y), y, np.full_like(y, HEAD_HEIGHT_M)])
        return cls(points, np.asarray(chain.bend_stiffness, dtype=float))

    @property
    def n_joints(self) -> int:
        return len(self.joint_points_m)


def shape_basis(geometry: ChainGeometry, n_shapes: int) -> Array:
    """Joint-rotation weights ``(n_shapes, N)`` of the static beam shapes.

    Shape 0: transverse tip force (joint moment ``F (L - s_i)``, rotation
    ``(L - s_i) / k_i``); shape 1: tip moment (``1 / k_i``). Each row sums to
    one, so an amplitude is the tip slope in radians (small angles).
    """
    if n_shapes not in range(1, MAX_SHAPES + 1):
        raise ValueError(f"shapes must be 1..{MAX_SHAPES}, got {n_shapes}")
    lever = -geometry.joint_points_m[:, 1]
    rows = np.stack([lever, np.ones_like(lever)])[:n_shapes] / geometry.bend_stiffness
    return np.asarray(rows / rows.sum(axis=1, keepdims=True), dtype=np.float64)


def marker_links(geometry: ChainGeometry, offsets: Array) -> NDArray[np.int_]:
    """Number of chain joints between the club body and each marker."""
    return np.sum(geometry.joint_points_m[None, :, 1] < offsets[:, None, 1], axis=1)


def _rotations(bend: Array) -> Array:
    """``(..., N, 3, 3)`` joint rotations ``Rx(qx) @ Rz(qz)`` of ``(..., N, 2)``."""
    cx, sx = np.cos(bend[..., 0]), np.sin(bend[..., 0])
    cz, sz = np.cos(bend[..., 1]), np.sin(bend[..., 1])
    zero, one = np.zeros_like(cx), np.ones_like(cx)
    rx = np.stack([one, zero, zero, zero, cx, -sx, zero, sx, cx], axis=-1)
    rz = np.stack([cz, -sz, zero, sz, cz, zero, zero, zero, one], axis=-1)
    shape = (*bend.shape[:-1], 3, 3)
    return np.asarray(rx.reshape(shape) @ rz.reshape(shape), dtype=np.float64)


def _positions(geometry: ChainGeometry, bend: Array, offsets: Array) -> Array:
    """Markers ``(k, m, 3)`` for bends ``(k, N, 2)`` (club-body frame)."""
    rot = _rotations(bend)
    links = marker_links(geometry, offsets)
    out = np.broadcast_to(offsets, (len(bend), *offsets.shape)).copy()
    for m, count in enumerate(links):
        for i in range(count - 1, -1, -1):
            joint = geometry.joint_points_m[i]
            out[:, m] = np.einsum("kab,kb->ka", rot[:, i], out[:, m] - joint) + joint
    return out


def marker_positions(geometry: ChainGeometry, bend: Array, offsets: Array) -> Array:
    """Club-body-frame markers ``(m, 3)`` of the chain bent by ``bend (N, 2)``.

    Postcondition: ``bend == 0`` returns ``offsets``; markers on the club body
    (grip side of joint 0) never move.
    """
    if bend.shape != (geometry.n_joints, 2):
        raise ValueError(f"bend must be ({geometry.n_joints}, 2), got {bend.shape}")
    return _positions(geometry, bend[None], np.asarray(offsets, dtype=float))[0]


def tip_rotations(bend: Array) -> Array:
    """Rotation ``(k, 3, 3)`` of the tip link relative to the club body."""
    out = np.broadcast_to(np.eye(3), (len(bend), 3, 3)).copy()
    for rot in np.moveaxis(_rotations(bend), 1, 0):
        out = out @ rot
    return out


def club_frame_targets(
    captured: Array, reference_offsets: Array, reference: Sequence[int]
) -> Array:
    """Captured markers ``(k, m, 3)`` in the club frame of the reference set.

    Each frame's ``reference`` markers (the shaft markers) are registered
    rigidly onto ``reference_offsets`` (``(m, 3)`` or per frame ``(k, m, 3)``);
    every marker is mapped by that registration, so the result is free of the
    capture-to-model transform. Frames missing a reference marker are NaN.
    """
    idx = list(reference)
    if len(idx) < 3 or captured.ndim != 3 or captured.shape[2] != 3:
        raise ValueError("need (k, m, 3) markers and at least 3 reference markers")
    ref = np.broadcast_to(reference_offsets, captured.shape)
    out = np.full_like(captured, np.nan)
    for k, (points, local) in enumerate(zip(captured, ref, strict=True)):
        if not np.all(np.isfinite(points[idx])):
            continue
        a, b = local[idx], points[idx]
        rot = kabsch_rotation(a - a.mean(0), b - b.mean(0))
        out[k] = (points - b.mean(0)) @ rot + a.mean(0)
    return out


@dataclass(frozen=True)
class Observability:
    """Singular values of the marker Jacobian over all bend coordinates."""

    singular_values: Array
    n_coordinates: int
    strong_rank: int


def observability(
    geometry: ChainGeometry, offsets: Array, settings: BendSettings
) -> Observability:
    """Analytic Jacobian of ``offsets`` w.r.t. every bend coordinate at zero.

    Column ``(i, axis)`` is ``axis x (p - joint_i)`` for markers beyond joint
    ``i``. ``strong_rank`` counts singular values above
    ``rank_relative_tolerance`` times the largest.
    """
    links = marker_links(geometry, offsets)
    columns = []
    for i, joint in enumerate(geometry.joint_points_m):
        for axis in AXES:
            moved = np.cross(axis, offsets - joint) * (links > i)[:, None]
            columns.append(moved.reshape(-1))
    values = np.linalg.svd(np.column_stack(columns), compute_uv=False)
    strong = int(np.sum(values > settings.rank_relative_tolerance * values[0]))
    return Observability(values, len(columns), strong)


@dataclass(frozen=True)
class BendFit:
    """Fitted amplitudes ``(k, 2S)`` (rad, plane x then z), joint bends
    ``(k, N, 2)`` and residual RMS per frame ``(k,)`` and per marker ``(m,)``."""

    amplitudes: Array
    bend: Array
    rms_m: Array
    marker_rms_m: Array
    success: bool
    message: str


def _sparsity(frames: int, markers: int, n_par: int) -> Any:
    """Jacobian pattern: markers per frame, second differences, ridge."""
    from scipy import sparse

    blocks = sparse.kron(sparse.eye(frames), np.ones((markers * 3, n_par)))
    stencil = sparse.diags([1, 1, 1], [0, 1, 2], shape=(max(frames - 2, 0), frames))
    smooth = sparse.kron(stencil, sparse.eye(n_par))
    return sparse.vstack([blocks, smooth, sparse.eye(frames * n_par)]).tolil()


def fit_shaft_bend(
    geometry: ChainGeometry,
    targets: Array,
    offsets: Array,
    settings: BendSettings,
    breaks: Sequence[int] = (),
) -> BendFit:
    """Least-squares bend history carrying ``offsets`` onto ``targets``.

    Preconditions: ``targets`` is ``(k, m, 3)`` in the club-body frame with
    ``m == len(offsets)`` (NaN marks a missing marker); each ``breaks`` entry
    ``b`` in ``1..k-1`` cuts the smoothness term between frames ``b - 1`` and
    ``b`` (ball contact unloads the shaft abruptly). Postconditions: every
    amplitude lies within ``max_tip_slope_deg``; frames without targets follow
    the smoothness and ridge terms only.
    """
    offsets = np.asarray(offsets, dtype=float)
    if targets.ndim != 3 or targets.shape[1:] != offsets.shape:
        raise ValueError("targets must be (k, m, 3) with one column per markers row")
    if any(not 0 < b < len(targets) for b in breaks):
        raise ValueError("breaks must lie inside the trajectory")
    from scipy.optimize import least_squares

    frames, n_shapes = len(targets), settings.shapes_per_plane
    basis, n_par = shape_basis(geometry, n_shapes), 2 * n_shapes
    valid = np.all(np.isfinite(targets), axis=2)
    goal = np.where(valid[..., None], targets, 0.0)
    rows = np.arange(max(frames - 2, 0))
    keep = np.ones(len(rows))
    for b in breaks:
        keep[(rows < b) & (b <= rows + 2)] = 0.0

    def bend_of(x: Array) -> Array:
        amps = x.reshape(frames, 2, n_shapes)
        return np.asarray(np.einsum("kps,sn->knp", amps, basis), dtype=np.float64)

    def residuals(x: Array) -> Array:
        amps = x.reshape(frames, n_par)
        error = (_positions(geometry, bend_of(x), offsets) - goal) * valid[..., None]
        smooth = (amps[2:] - 2.0 * amps[1:-1] + amps[:-2]) * keep[:, None]
        smooth = settings.smoothness_weight_m_per_rad * smooth.reshape(-1)
        ridge = settings.ridge_weight_m_per_rad * x
        return np.concatenate([error.reshape(-1), smooth, ridge])

    bound = math.radians(settings.max_tip_slope_deg)
    result = least_squares(
        residuals,
        np.zeros(frames * n_par),
        jac_sparsity=_sparsity(frames, len(offsets), n_par),
        bounds=(-bound, bound),
        max_nfev=settings.solver_max_nfev,
        **dict.fromkeys(("ftol", "xtol", "gtol"), settings.solver_tolerance),
    )
    bend = bend_of(result.x)
    error = np.linalg.norm(_positions(geometry, bend, offsets) - targets, axis=2)
    squared = np.where(valid, error, 0.0) ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        rms = np.sqrt(squared.sum(1) / valid.sum(1))
        marker_rms = np.sqrt(squared.sum(0) / valid.sum(0))
    amplitudes = result.x.reshape(frames, n_par)
    return BendFit(amplitudes, bend, rms, marker_rms, result.success, result.message)
