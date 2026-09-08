"""Fit an articulated model's joint angles to 3-D landmarks, continuously (#9712).

All frames are solved together. The unknowns are the joint-angle state
``q(t)`` (and optionally a few segment lengths); the residuals are

    (landmark_model - landmark_observed) / sigma_landmark        robust, weighted
    (q_{t-1} - 2 q_t + q_{t+1}) fps^2 / sigma_accel              per DOF: no jumps
    max(0, q - hi) / sigma_limit, max(0, lo - q) / sigma_limit   soft joint limits
    (L - L_measured) / sigma_length                              tape readings

The acceleration prior is what makes the fit a filter: a landmark the model
can only reach by a discontinuous change of some angle is out-voted by the
continuity of every other frame, its residual grows past the gate, and it is
rejected and listed, never followed. Rejected landmarks are refit without.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeAlias

import logging

import numpy as np
import numpy.typing as npt
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix, vstack

from src.shared.python.core.contracts import require

from .kinematics import ArticulatedModel

logger = logging.getLogger(__name__)

Array: TypeAlias = npt.NDArray[np.float64]


@dataclass(frozen=True)
class FitOptions:
    """Weights in physical units."""

    sigma_landmark_m: float = 0.01
    sigma_accel_rad_s2: float = 300.0  # rotational DOFs
    sigma_accel_m_s2: float = 30.0  # root translation
    sigma_limit_rad: float = 0.01
    sigma_length_m: float = 0.003
    # Weak pull of every rotation toward its rest angle: resolves the twist
    # about a segment axis that a single end-point landmark cannot see, so a
    # redundant DOF settles instead of wandering (a wander looks like motion).
    sigma_rest_rad: float = 3.0
    huber_delta: float = 2.5
    gate: float = 5.0  # sigma units; larger residuals are rejected after convergence
    max_velocity_rad_s: float | None = None  # reported, not clipped
    max_iterations: int = 60
    fit_lengths: tuple[str, ...] = ()
    # Image-space fits (fit2d): pixel noise and the single-view root depth.
    sigma_px: float = 4.0
    root_depth_m: float = 3.5

    def __post_init__(self) -> None:
        for name in (
            "sigma_landmark_m",
            "sigma_accel_rad_s2",
            "sigma_accel_m_s2",
            "sigma_limit_rad",
            "sigma_length_m",
            "sigma_rest_rad",
            "huber_delta",
            "gate",
            "sigma_px",
            "root_depth_m",
        ):
            require(getattr(self, name) > 0, f"{name} must be positive")
        require(self.max_iterations >= 1, "max_iterations must be >= 1")


@dataclass(frozen=True)
class RejectedLandmark:
    frame: int
    landmark: str
    residual_sigma: float


@dataclass(frozen=True)
class ModelFit:
    """The trajectory, what was rejected and how continuous it is."""

    q: Array  # (T, n_dof)
    lengths_m: dict[str, float]
    landmarks_m: Array  # (T, L, 3) model landmarks
    residual_m: Array  # (T, L) landmark distance, NaN where unobserved
    weights: Array  # (T, L) final weights, 0 for rejected or unobserved
    rejected: tuple[RejectedLandmark, ...]
    rms_m: float
    peak_velocity_rad_s: dict[str, float]
    velocity_violations: int
    iterations: int
    dof_names: tuple[str, ...] = field(default_factory=tuple)
    # Image-space fits (fit2d): per-view pixel residuals and their RMS.
    residual_px: Array | None = None  # (T, V, L), NaN where unobserved
    rms_px: float | None = None


class _Problem:
    def __init__(
        self,
        model: ArticulatedModel,
        observed: Array,
        weights: Array,
        fps: float,
        options: FitOptions,
        lengths: Mapping[str, float],
    ) -> None:
        self._init_common(model, observed.shape[0], fps, options, lengths)
        self.l = observed.shape[1]
        self.obs = np.nan_to_num(observed)
        self.mask = np.isfinite(observed).all(axis=2) & (weights > 0)
        self.base_w = np.sqrt(np.clip(weights, 0, 1)) * self.mask
        self.w = self.base_w.copy()

    def _init_common(
        self,
        model: ArticulatedModel,
        frames: int,
        fps: float,
        options: FitOptions,
        lengths: Mapping[str, float],
    ) -> None:
        """State shared by every observation model (3-D points, 2-D pixels)."""
        self.model, self.o, self.fps = model, options, fps
        self.t = frames
        self.n = model.n_dof
        self.lengths = dict(lengths)
        self.length_names = tuple(options.fit_lengths)
        self.lo, self.hi = model.limits()
        acc = np.full(self.n, options.sigma_accel_rad_s2)
        acc[:3] = options.sigma_accel_m_s2
        self.sigma_acc = acc
        self.n_params = self.t * self.n + len(self.length_names)

    # -- parameter vector: q flattened (T*n) then fitted lengths --------------
    def pack(self, q: Array, lengths: Mapping[str, float]) -> Array:
        return np.concatenate([q.ravel(), [lengths[k] for k in self.length_names]])

    def unpack(self, x: Array) -> tuple[Array, dict[str, float]]:
        q = x[: self.t * self.n].reshape(self.t, self.n)
        lengths = dict(self.lengths)
        for k, name in enumerate(self.length_names):
            lengths[name] = float(x[self.t * self.n + k])
        return q, lengths

    def residuals(self, x: Array) -> Array:
        q, lengths = self.unpack(x)
        return np.concatenate(
            [self.landmark_residuals(q, lengths), *self._prior_residuals(q, lengths)]
        )

    def landmark_residuals(self, q: Array, lengths: Mapping[str, float]) -> Array:
        """Flat, weighted, dimensionless landmark residuals (the data term)."""
        lm = self.model.landmarks(q, lengths)
        return ((lm - self.obs) * self.w[:, :, None] / self.o.sigma_landmark_m).ravel()

    def _prior_residuals(self, q: Array, lengths: Mapping[str, float]) -> list[Array]:
        parts: list[Array] = []
        if self.t >= 3:
            acc = (q[:-2] - 2 * q[1:-1] + q[2:]) * self.fps**2 / self.sigma_acc
            parts.append(acc.ravel())
        over = np.maximum(q - self.hi, 0.0) + np.minimum(q - self.lo, 0.0)
        parts.append((over / self.o.sigma_limit_rad).ravel())
        parts.append((q[:, 3:] / self.o.sigma_rest_rad).ravel())
        for name in self.length_names:
            parts.append(
                np.atleast_1d(
                    (lengths[name] - self.lengths[name]) / self.o.sigma_length_m
                )
            )
        return parts

    def jacobian(self, x: Array) -> csr_matrix:
        q, lengths = self.unpack(x)
        blocks = [self._landmark_block(q, lengths)]
        if self.t >= 3:
            blocks.append(self._accel_block())
        blocks.append(self._limit_block(q))
        blocks.append(self._rest_block())
        if self.length_names:
            blocks.append(self._length_block())
        return csr_matrix(vstack(blocks))

    def _landmark_block(self, q: Array, lengths: Mapping[str, float]) -> csr_matrix:
        jac = self.model.jacobian(q, lengths)  # (T, L*3, n)
        scale = np.repeat(self.w, 3, axis=1)[:, :, None] / self.o.sigma_landmark_m
        jac = jac * scale
        rows_per_t = self.l * 3
        rows: Array = np.repeat(np.arange(self.t * rows_per_t), self.n).astype(float)
        cols = (
            np.arange(self.t)[:, None, None] * self.n
            + np.arange(self.n)[None, None, :]
            + np.zeros((1, rows_per_t, 1), dtype=int)
        ).ravel()
        vals: Array = jac.ravel()
        if self.length_names:
            lj = self.model.length_jacobian(q, lengths, self.length_names) * scale
            lrows = np.repeat(np.arange(self.t * rows_per_t), len(self.length_names))
            lcols = np.tile(
                self.t * self.n + np.arange(len(self.length_names)), self.t * rows_per_t
            )
            rows = np.concatenate([rows, lrows.astype(float)])
            cols = np.concatenate([cols, lcols])
            vals = np.concatenate([vals, lj.ravel()])
        return csr_matrix(
            (vals, (rows.astype(int), cols)), shape=(self.t * rows_per_t, self.n_params)
        )

    def _accel_block(self) -> csr_matrix:
        n_rows = (self.t - 2) * self.n
        rows = np.repeat(np.arange(n_rows), 3)
        base = np.arange(n_rows)  # row r -> frame r // n, dof r % n
        frame, dof = base // self.n, base % self.n
        cols = np.stack(
            [
                frame * self.n + dof,
                (frame + 1) * self.n + dof,
                (frame + 2) * self.n + dof,
            ],
            1,
        ).ravel()
        coef = self.fps**2 / self.sigma_acc[dof]
        vals = np.stack([coef, -2 * coef, coef], 1).ravel()
        return csr_matrix((vals, (rows, cols)), shape=(n_rows, self.n_params))

    def _limit_block(self, q: Array) -> csr_matrix:
        active = ((q > self.hi) | (q < self.lo)).ravel()
        idx = np.arange(self.t * self.n)
        vals = active.astype(float) / self.o.sigma_limit_rad
        return csr_matrix((vals, (idx, idx)), shape=(self.t * self.n, self.n_params))

    def _rest_block(self) -> csr_matrix:
        rot = np.arange(self.t * self.n).reshape(self.t, self.n)[:, 3:].ravel()
        rows = np.arange(rot.size)
        vals = np.full(rot.size, 1.0 / self.o.sigma_rest_rad)
        return csr_matrix((vals, (rows, rot)), shape=(rot.size, self.n_params))

    def _length_block(self) -> csr_matrix:
        k = len(self.length_names)
        rows = np.arange(k)
        cols = self.t * self.n + rows
        return csr_matrix(
            (np.full(k, 1.0 / self.o.sigma_length_m), (rows, cols)),
            shape=(k, self.n_params),
        )

    def landmark_residual_m(self, x: Array) -> Array:
        q, lengths = self.unpack(x)
        d = np.linalg.norm(self.model.landmarks(q, lengths) - self.obs, axis=2)
        return np.where(self.mask, d, np.nan)

    # -- hooks the robust stages, the gate and the report go through ---------
    def residual_sigma(self, x: Array) -> Array:
        """Unweighted residual per observation in sigma units, NaN unobserved."""
        return self.landmark_residual_m(x) / self.o.sigma_landmark_m

    def reweight(self, huber: Array) -> None:
        self.w = self.base_w * np.sqrt(huber)

    def reject(self, index: tuple[int, ...]) -> RejectedLandmark:
        """Zero one observation's weight; returns its record."""
        self.base_w[index] = 0.0
        self.mask[index] = False
        self.w = self.base_w.copy()
        return RejectedLandmark(
            int(index[0]), self.model.landmark_names[index[-1]], 0.0
        )

    def report(self, q: Array, lengths: Mapping[str, float]) -> dict[str, Any]:
        """Per-landmark residuals and weights in the ``(T, L)`` layout."""
        residual = self.landmark_residual_m(self.pack(q, lengths))
        finite = residual[np.isfinite(residual)]
        return {
            "residual_m": residual,
            "weights": self.base_w**2,
            "rms_m": float(np.sqrt(np.mean(finite**2)))
            if finite.size
            else float("nan"),
            "residual_px": None,
            "rms_px": None,
        }


def _huber_weight(u: Array, delta: float) -> Array:
    a = np.abs(u)
    return np.where(a <= delta, 1.0, delta / np.maximum(a, 1e-12))


def _solve(problem: _Problem, x0: Array, options: FitOptions) -> tuple[Array, int]:
    fit = least_squares(
        problem.residuals,
        x0,
        jac=problem.jacobian,
        method="trf",
        max_nfev=options.max_iterations,
        ftol=1e-8,
        xtol=1e-8,
        x_scale="jac",
    )
    return np.asarray(fit.x, dtype=float), int(fit.nfev)


def _robust(problem: _Problem, x: Array, options: FitOptions) -> tuple[Array, int]:
    total = 0
    for delta in (None, 20 * options.huber_delta, options.huber_delta):
        if delta is not None:
            u = np.nan_to_num(problem.residual_sigma(x))
            problem.reweight(_huber_weight(u, delta))
        x, n = _solve(problem, x, options)
        total += n
    return x, total


def _reject(
    problem: _Problem, x: Array, options: FitOptions, model: ArticulatedModel
) -> list[RejectedLandmark]:
    u = np.nan_to_num(problem.residual_sigma(x))
    bad = (u > options.gate) & problem.mask
    if bad.sum() * 2 > problem.mask.sum():
        # Most observations beyond the gate means the model cannot represent
        # the motion, not that the data has outliers: keep everything so the
        # reported RMS says so (a pendulum on a bent-arm golfer, #9714).
        logger.warning(
            "gate would reject %d of %d observations; keeping all",
            int(bad.sum()),
            int(problem.mask.sum()),
        )
        return []
    out = []
    for index in np.argwhere(bad):
        key = tuple(int(i) for i in index)
        record = problem.reject(key)
        out.append(RejectedLandmark(record.frame, record.landmark, float(u[key])))
    return out


def fit_trajectory(
    model: ArticulatedModel,
    observed_m: Array,
    fps: float,
    *,
    weights: Array | None = None,
    q0: Array | None = None,
    lengths_m: Mapping[str, float] | None = None,
    options: FitOptions | None = None,
) -> ModelFit:
    """Continuous joint-angle trajectory through ``observed_m`` ``(T, L, 3)``.

    Preconditions: landmarks in the model's :attr:`landmark_names` order, NaN
    where unobserved; positive fps; at least one frame. Postcondition: every
    rejected landmark is listed with its residual and has weight 0.
    """
    o = options or FitOptions()
    obs = np.asarray(observed_m, dtype=float)
    require(
        obs.ndim == 3 and obs.shape[1:] == (len(model.landmark_names), 3),
        "observed must be (T, L, 3) in landmark order",
        obs.shape,
    )
    require(fps > 0 and obs.shape[0] >= 1, "positive fps and at least one frame")
    w = np.ones(obs.shape[:2]) if weights is None else np.asarray(weights, float)
    lengths = dict(model.spec.lengths_m if lengths_m is None else lengths_m)
    problem = _Problem(model, obs, w, fps, o, lengths)
    q_start = (
        np.zeros((obs.shape[0], model.n_dof)) if q0 is None else np.asarray(q0, float)
    )
    require(q_start.shape == (obs.shape[0], model.n_dof), "q0 shape")
    if q0 is None:
        root = np.nan_to_num(obs[:, 0])
        q_start[:, :3] = root  # first landmark is the root joint by convention
    return solve_problem(problem, q_start, lengths, fps, o)


def solve_problem(
    problem: _Problem,
    q_start: Array,
    lengths: Mapping[str, float],
    fps: float,
    o: FitOptions,
) -> ModelFit:
    """Robust stages, gate, refit and report for any observation model."""
    x, n1 = _robust(problem, problem.pack(q_start, lengths), o)
    rejected = _reject(problem, x, o, problem.model)
    n2 = 0
    if rejected:
        x, n2 = _solve(problem, x, o)
    q, fitted_lengths = problem.unpack(x)
    return _report(problem.model, problem, q, fitted_lengths, fps, o, rejected, n1 + n2)


def _report(
    model: ArticulatedModel,
    problem: _Problem,
    q: Array,
    lengths: dict[str, float],
    fps: float,
    o: FitOptions,
    rejected: Sequence[RejectedLandmark],
    iterations: int,
) -> ModelFit:
    lm = model.landmarks(q, lengths)
    summary = problem.report(q, lengths)
    vel = np.diff(q, axis=0) * fps if q.shape[0] > 1 else np.zeros((0, q.shape[1]))
    peak = {
        name: float(np.max(np.abs(vel[:, i]))) if vel.size else 0.0
        for i, name in enumerate(model.dof_names)
        if i >= 3
    }
    violations = 0
    if o.max_velocity_rad_s is not None and vel.size:
        violations = int(np.count_nonzero(np.abs(vel[:, 3:]) > o.max_velocity_rad_s))
    return ModelFit(
        q=q,
        lengths_m=lengths,
        landmarks_m=lm,
        residual_m=summary["residual_m"],
        weights=summary["weights"],
        rejected=tuple(rejected),
        rms_m=summary["rms_m"],
        peak_velocity_rad_s=peak,
        velocity_violations=violations,
        iterations=iterations,
        dof_names=tuple(model.dof_names),
        residual_px=summary["residual_px"],
        rms_px=summary["rms_px"],
    )


def fit_to_dict(fit: ModelFit, fps: float) -> dict[str, Any]:
    """JSON-ready record of a fit: series per DOF, lengths, rejections, summary."""
    return {
        "schema_version": "model-fit/1.0.0",
        "fps": fps,
        "dof_names": list(fit.dof_names),
        "q": fit.q.tolist(),
        "lengths_m": fit.lengths_m,
        "rms_m": fit.rms_m,
        "rejected": [
            {
                "frame": r.frame,
                "landmark": r.landmark,
                "residual_sigma": r.residual_sigma,
            }
            for r in fit.rejected
        ],
        "peak_velocity_rad_s": fit.peak_velocity_rad_s,
        "velocity_violations": fit.velocity_violations,
        "iterations": fit.iterations,
        "rms_px": fit.rms_px,
    }
