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
bounded and smoothed by ``shaft_bend_fit_settings.json``. The bound is on the
combined tip slope of a plane (the sum of its amplitudes). Kinematic only.

Narrow acceptance scope and reproduction commands:
``docs/research/shaft_bend_kinematic_fit/shaft_bend_kinematic_fit.md``.
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
AXES = np.eye(3)[[0, 2]]  # joint bending axes in primitive order: club x, z
MAX_SHAPES = 2
_BOUND_SLACK = 1e-9  # relative rounding allowance of the post-fit bound check


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
        points = self.joint_points_m
        if not (
            points.ndim == 2
            and points.shape[1] == 3
            and self.bend_stiffness.shape == (len(points),)
            and np.all(np.diff(points[:, 1]) > 0)
        ):
            raise ValueError("need (N, 3) joints from grip to tip and N stiffnesses")

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
    """Club-frame markers ``(m, 3)`` bent by ``bend (N, 2)``; zero keeps them."""
    if bend.shape != (geometry.n_joints, 2):
        raise ValueError(f"bend must be ({geometry.n_joints}, 2), got {bend.shape}")
    return _positions(geometry, bend[None], np.asarray(offsets, dtype=float))[0]


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
    strong_rank: int


def observability(
    geometry: ChainGeometry, offsets: Array, settings: BendSettings
) -> Observability:
    """SVD of the analytic marker Jacobian at zero bend (column ``(i, axis)``:
    ``axis x (p - joint_i)`` beyond joint ``i``); ``strong_rank`` counts values
    above ``rank_relative_tolerance`` times the largest."""
    links = marker_links(geometry, offsets)
    columns = []
    for i, joint in enumerate(geometry.joint_points_m):
        for axis in AXES:
            moved = np.cross(axis, offsets - joint) * (links > i)[:, None]
            columns.append(moved.reshape(-1))
    values = np.linalg.svd(np.column_stack(columns), compute_uv=False)
    strong = int(np.sum(values > settings.rank_relative_tolerance * values[0]))
    return Observability(values, strong)


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


def _sparsity(frames: int, markers: int, n_par: int, coupled: bool = False) -> Any:
    """Jacobian pattern: markers per frame, second differences, ridge.

    ``coupled`` lets every amplitude row depend on all of a frame's variables
    (the two-shape total-slope parameterisation of :func:`fit_shaft_bend`).
    """
    from scipy import sparse

    link = np.ones((n_par, n_par)) if coupled else np.eye(n_par)
    blocks = sparse.kron(sparse.eye(frames), np.ones((markers * 3, n_par)))
    stencil = sparse.diags([1, 1, 1], [0, 1, 2], shape=(max(frames - 2, 0), frames))
    smooth = sparse.kron(stencil, link)
    stacked = sparse.vstack([blocks, smooth, sparse.kron(sparse.eye(frames), link)])
    return sparse.lil_matrix(stacked)


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
    plane's combined tip slope (sum of its amplitudes) lies within
    ``max_tip_slope_deg`` in every frame (checked, ``RuntimeError`` otherwise);
    frames without targets follow the smoothness and ridge terms only.

    With two shapes the solver variables per plane are the total tip slope
    ``t`` (box-bounded, so the bound is exact) and the first amplitude ``u``
    (unbounded); the amplitudes are ``(u, t - u)``.
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

    def amplitudes_of(x: Array) -> Array:
        """Amplitudes ``(k, 2, S)`` of solver variables (``(t, u)`` if S = 2)."""
        z = x.reshape(frames, 2, n_shapes)
        if n_shapes == 1:
            return z
        return np.stack([z[..., 1], z[..., 0] - z[..., 1]], axis=-1)

    def bend_of(x: Array) -> Array:
        return np.asarray(
            np.einsum("kps,sn->knp", amplitudes_of(x), basis), dtype=np.float64
        )

    def residuals(x: Array) -> Array:
        raw = amplitudes_of(x)
        amps = raw.reshape(frames, n_par)
        error = (_positions(geometry, bend_of(x), offsets) - goal) * valid[..., None]
        smooth = (amps[2:] - 2.0 * amps[1:-1] + amps[:-2]) * keep[:, None]
        smooth = settings.smoothness_weight_m_per_rad * smooth.reshape(-1)
        ridge = settings.ridge_weight_m_per_rad * raw.reshape(-1)
        return np.concatenate([error.reshape(-1), smooth, ridge])

    bound, tol = math.radians(settings.max_tip_slope_deg), settings.solver_tolerance
    lower, upper = np.full(frames * n_par, -bound), np.full(frames * n_par, bound)
    if n_shapes == 2:  # the split amplitude is free; only the total is bounded
        lower.reshape(frames, 2, 2)[..., 1], upper.reshape(frames, 2, 2)[..., 1] = (
            -np.inf,
            np.inf,
        )
    result = least_squares(
        residuals,
        np.zeros(frames * n_par),
        jac_sparsity=_sparsity(frames, len(offsets), n_par, n_shapes == 2),
        bounds=(lower, upper),
        max_nfev=settings.solver_max_nfev,
        ftol=tol,
        xtol=tol,
        gtol=tol,
    )
    bend = bend_of(result.x)
    error = np.linalg.norm(_positions(geometry, bend, offsets) - targets, axis=2)
    squared = np.where(valid, error, 0.0) ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        rms = np.sqrt(squared.sum(1) / valid.sum(1))
        marker_rms = np.sqrt(squared.sum(0) / valid.sum(0))
    amplitudes = amplitudes_of(result.x).reshape(frames, n_par)
    totals = amplitudes.reshape(frames, 2, n_shapes).sum(axis=2)
    if np.abs(totals).max() > bound * (1.0 + _BOUND_SLACK):
        raise RuntimeError("fitted combined tip slope exceeds max_tip_slope_deg")
    return BendFit(amplitudes, bend, rms, marker_rms, result.success, result.message)
