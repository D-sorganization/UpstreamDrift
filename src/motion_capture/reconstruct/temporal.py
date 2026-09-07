"""Dynamics-constrained temporal estimation with explicit outlier rejection.

Issue #9626: a golf swing is smooth at 60–200 fps, so a joint trajectory that
jumps between frames is a detector error, not motion. Each channel of a time
series is fitted by a penalised least-squares smoother in *physical units*:

    minimise  sum_t (c_t / sigma_z^2) rho(x_t - z_t)
            + sum_t ((x_{t-1} - 2 x_t + x_{t+1}) / dt^2)^2 / sigma_a^2

``z`` are measurements with detector confidences ``c``; ``sigma_z`` is the
measurement noise (estimated robustly from the data unless given);
``sigma_a`` is the prior standard deviation of acceleration in units per
second squared — the one tunable that encodes "how violently can this joint
move" and must be chosen per quantity (pixels, metres, radians); ``rho`` is
Huber. The system is banded, so thousands of frames cost milliseconds.

After convergence:

- a measurement whose residual exceeds ``gate_sigma`` times ``sigma_z`` is
  **rejected** and listed with its residual — never silently averaged in;
- velocity and acceleration of the fit are checked against explicit bounds
  and every violation is reported; the fit is evidence and the bound is the
  acceptance criterion, so violations are not hidden by clipping;
- per-frame uncertainty is the posterior standard deviation at the final
  weights.

Nothing here knows about skeletons: it works on any ``(T, D)`` series and is
applied per joint by the reconstruction stages.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict
from scipy import sparse
from scipy.sparse.linalg import splu

from src.shared.python.core.contracts import require

Array = npt.NDArray[np.float64]
_EXACT_COV_MAX_N = 2000


@dataclass(frozen=True)
class SmootherOptions:
    """Tunables in physical units; every value is validated."""

    acceleration_sigma: float  # units / s^2, prior spread of acceleration
    measurement_sigma: float | None = None  # units; estimated when None
    huber_delta: float = 2.5  # residual (in sigma_z units) where Huber bends
    # Outliers are judged under a prior this many times tighter (in sigma_a)
    # than the final fit's: with a loose prior a one-frame jump is simply
    # followed and its residual looks innocent; the tight prior makes it stand
    # out, the final fit then uses the physical prior without the point.
    detection_tightening: float = 10.0
    gate_sigma: float = 5.0  # rejection gate in sigma_z units
    iterations: int = 6
    min_weight: float = 1e-3
    max_velocity: float | None = None  # units / s, checked on the fit
    max_acceleration: float | None = None  # units / s^2

    def __post_init__(self) -> None:
        require(self.acceleration_sigma > 0, "acceleration_sigma must be positive")
        require(
            self.measurement_sigma is None or self.measurement_sigma > 0,
            "measurement_sigma must be positive when set",
        )
        require(self.huber_delta > 0, "huber_delta must be positive", self.huber_delta)
        require(self.gate_sigma > 0, "gate_sigma must be positive", self.gate_sigma)
        require(self.iterations >= 1, "iterations must be at least 1", self.iterations)
        require(self.detection_tightening >= 1, "detection_tightening must be >= 1")
        require(
            0 < self.min_weight < 1, "min_weight must be in (0, 1)", self.min_weight
        )
        for name in ("max_velocity", "max_acceleration"):
            v = getattr(self, name)
            require(v is None or v > 0, f"{name} must be positive when set", v)


class Rejection(BaseModel):
    model_config = ConfigDict(frozen=True)

    frame: int
    channel: int
    residual: float
    threshold: float


class BoundViolation(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: str  # velocity | acceleration
    frame: int
    channel: int
    value: float
    bound: float


@dataclass(frozen=True)
class SmoothResult:
    """The fit, what was rejected, what exceeded the bounds, and uncertainty."""

    values: Array  # (T, D)
    weights: Array  # (T, D) final weights, 0 for rejected or unobserved
    residuals: Array  # (T, D) measurement minus fit (NaN where unobserved)
    uncertainty: Array  # (T, D) posterior standard deviation
    measurement_sigma: Array  # (D,) noise scale used per channel
    rejected: tuple[Rejection, ...]
    violations: tuple[BoundViolation, ...]

    @property
    def ok(self) -> bool:
        """True when nothing exceeded the dynamics bounds."""
        return not self.violations


def second_difference(n: int) -> sparse.csr_matrix:
    """``(n-2, n)`` operator whose rows are ``x[t-1] - 2 x[t] + x[t+1]``."""
    require(n >= 3, "need at least 3 samples for a second difference", n)
    # dia_matrix indexes data by column, so each diagonal needs n entries.
    data = np.array([np.ones(n), -2 * np.ones(n), np.ones(n)])
    matrix = sparse.dia_matrix((data, np.array([0, 1, 2])), shape=(n - 2, n))
    return sparse.csr_matrix(matrix)


def robust_scale(values: Array) -> float:
    """1.4826 x MAD of the finite entries, floored above zero."""
    r = values[np.isfinite(values)]
    if r.size == 0:
        return 1.0
    mad = float(np.median(np.abs(r - np.median(r))))
    return max(1.4826 * mad, 1e-9)


def noise_estimate(z: Array) -> float:
    """Measurement noise from second differences of a smooth signal.

    For white noise of standard deviation ``s`` the second difference has
    standard deviation ``s * sqrt(6)``; the signal's own curvature per frame
    squared is negligible at capture rates, so ``MAD(diff2) * 1.4826 / sqrt(6)``
    estimates ``s`` without being fooled by velocity.
    """
    finite = z[np.isfinite(z)]
    if finite.size < 4:
        return 1.0
    estimate = robust_scale(np.diff(finite, n=2)) / np.sqrt(6.0)
    # A noise-free signal would give a scale near zero and weights near
    # infinity; floor at one part per million of the signal's range.
    span = float(np.ptp(finite)) if finite.size else 0.0
    return max(estimate, 1e-6 * span, 1e-9)


def _huber_weight(u: Array, delta: float) -> Array:
    a = np.abs(u)
    return np.where(a <= delta, 1.0, delta / np.maximum(a, 1e-12))


def _solve(z: Array, w: Array, prior: sparse.csc_matrix) -> tuple[Array, Array]:
    """Minimise ``sum w (x - z)^2 + x' P x``; return the fit and its posterior std."""
    n = z.size
    diagonal = sparse.dia_matrix((w[None, :], np.array([0])), shape=(n, n))
    a = sparse.csc_matrix(diagonal + prior)
    lu = splu(a)
    x = lu.solve(w * np.where(np.isfinite(z), z, 0.0))
    if n <= _EXACT_COV_MAX_N:
        cov_diag = np.array([lu.solve(e)[i] for i, e in enumerate(np.eye(n))])
    else:  # banded approximation for long takes
        cov_diag = 1.0 / np.maximum(a.diagonal(), 1e-12)
    return x, np.sqrt(np.maximum(cov_diag, 0.0))


def _fit_channel(
    z: Array,
    conf: Array,
    prior: sparse.csc_matrix,
    options: SmootherOptions,
) -> tuple[Array, Array, Array, float, list[tuple[int, float, float]]]:
    observed = np.isfinite(z)
    sigma = options.measurement_sigma or noise_estimate(z)
    base = np.where(observed, conf, 0.0) / sigma**2
    floor = options.min_weight * observed / sigma**2
    w = base.copy()
    x = np.zeros_like(z)
    detect_prior = prior * options.detection_tightening**2
    for _ in range(options.iterations):
        x, _ = _solve(z, np.maximum(w, floor), detect_prior)
        u = np.where(observed, (z - x) / sigma, 0.0)
        w = base * _huber_weight(u, options.huber_delta)
    gate = options.gate_sigma * sigma
    reject = np.zeros_like(observed)
    # Reject, refit without the rejected points, and re-judge: a gross point
    # drags the fit toward itself, so its neighbours can look wrong until it
    # is gone, and a second gross point can hide behind the first. Known
    # limits, measured on the synthetic harness: a gross point in the first
    # or last frames has little prior to contradict it, and offsets below
    # about two gates blend into the noise; both are reported by the metrics
    # rather than hidden by a looser gate.
    for _ in range(3):
        x, _ = _solve(z, np.maximum(w, floor * ~reject), detect_prior)
        residual = np.where(observed, z - x, np.nan)
        newly = observed & ~reject & (np.abs(residual) > gate)
        if not newly.any():
            break
        reject |= newly
        w[reject] = 0.0
    # Re-admit points that only looked wrong because of a neighbour that is
    # now gone: judged against the fit without them, they are inside the gate.
    x, _ = _solve(z, np.maximum(w, floor * ~reject), detect_prior)
    residual = np.where(observed, z - x, np.nan)
    readmit = reject & (np.abs(residual) <= gate)
    if readmit.any():
        reject &= ~readmit
        w[readmit] = base[readmit]
    x, std = _solve(z, np.maximum(w, floor * ~reject), prior)
    residual = np.where(observed, z - x, np.nan)
    rejections = [(int(t), float(residual[t]), gate) for t in np.flatnonzero(reject)]
    return x, w, std, sigma, rejections


def smooth(
    measurements: Array,
    confidence: Array | None,
    fps: float,
    options: SmootherOptions,
) -> SmoothResult:
    """Fit every channel of ``measurements`` ``(T, D)`` with the robust smoother.

    ``confidence`` ``(T,)`` or ``(T, D)`` in [0, 1] scales the data weights;
    NaN measurements are unobserved and are filled by the dynamics prior.
    Preconditions: at least 3 frames, positive fps. Postcondition: ``values``
    has no NaN and every rejected measurement has weight 0.
    """
    z = np.asarray(measurements, dtype=float)
    require(z.ndim == 2 and z.shape[0] >= 3, "measurements must be (T>=3, D)", z.shape)
    require(fps > 0, "fps must be positive", fps)
    t_n, d_n = z.shape
    if confidence is None:
        conf = np.ones_like(z)
    else:
        c = np.asarray(confidence, dtype=float)
        conf = np.broadcast_to(c[:, None] if c.ndim == 1 else c, z.shape).copy()
    require(bool(np.all((conf >= 0) & (conf <= 1))), "confidence must be within [0, 1]")
    dt = 1.0 / fps
    d2 = second_difference(t_n)
    prior = ((d2.T @ d2) / (options.acceleration_sigma**2 * dt**4)).tocsc()
    values = np.zeros_like(z)
    weights = np.zeros_like(z)
    unc = np.zeros_like(z)
    sigmas = np.zeros(d_n)
    rejected: list[Rejection] = []
    for j in range(d_n):
        x, w, std, sigma, rej = _fit_channel(z[:, j], conf[:, j], prior, options)
        values[:, j], weights[:, j], unc[:, j], sigmas[j] = x, w, std, sigma
        rejected.extend(
            Rejection(frame=t, channel=j, residual=r, threshold=g) for t, r, g in rej
        )
    residuals = np.where(np.isfinite(z), z - values, np.nan)
    return SmoothResult(
        values=values,
        weights=weights,
        residuals=residuals,
        uncertainty=unc,
        measurement_sigma=sigmas,
        rejected=tuple(rejected),
        violations=check_bounds(values, fps, options),
    )


def check_bounds(
    values: Array, fps: float, options: SmootherOptions
) -> tuple[BoundViolation, ...]:
    """Every frame/channel whose velocity or acceleration exceeds its bound."""
    x = np.asarray(values, dtype=float)
    require(x.ndim == 2 and x.shape[0] >= 3, "values must be (T>=3, D)", x.shape)
    require(fps > 0, "fps must be positive", fps)
    out: list[BoundViolation] = []
    checks = (
        ("velocity", np.diff(x, axis=0) * fps, options.max_velocity),
        ("acceleration", np.diff(x, n=2, axis=0) * fps * fps, options.max_acceleration),
    )
    for kind, arr, bound in checks:
        if bound is None:
            continue
        for t, j in zip(*np.nonzero(np.abs(arr) > bound), strict=True):
            out.append(
                BoundViolation(
                    kind=kind,
                    frame=int(t) + 1,
                    channel=int(j),
                    value=float(arr[t, j]),
                    bound=bound,
                )
            )
    return tuple(out)
