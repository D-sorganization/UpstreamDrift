"""ML-WS-7 (#13784): Inverse regressor kinematics->knots with fail-closed OOD guard."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.shared.python.motion_matching.proposal_initializer import validate_proposal

from .masked_proposal import MaskedObservation
from .proposal_shared import require_positive_training_hparams

__all__ = [
    "CorpusOodBounds",
    "InverseRegressorReceipt",
    "KinematicsKnotRegressor",
    "OodProposalOutcome",
    "fit_inverse_regressor_with_ood",
    "propose_knots_with_ood_guard",
]

_HIDDEN, _BATCH, _WD, _RANGE_MARGIN, _MAHA_MARGIN = 64, 16, 1e-5, 0.05, 1.5


def _check(cond: bool, message: str) -> None:
    """Raise ``ValueError(message)`` when ``cond`` is false."""
    if not cond:
        raise ValueError(message)


@dataclass(frozen=True)
class CorpusOodBounds:
    """Empirical corpus feature-range and Mahalanobis ellipsoid OOD bounds."""

    feature_min: tuple[float, ...]
    feature_max: tuple[float, ...]
    mean: tuple[float, ...]
    precision_flat: tuple[float, ...]
    max_mahalanobis: float
    seq_len: int
    trajectory_channels: int
    feature_dim: int

    def __post_init__(self) -> None:
        """Validate OOD bound shapes, finiteness, and positivity (DbC)."""
        d = self.feature_dim
        _check(
            self.seq_len > 0
            and self.trajectory_channels > 0
            and self.seq_len * self.trajectory_channels == d > 0,
            "bad dims",
        )
        vecs = (self.feature_min, self.feature_max, self.mean)
        _check(
            all(len(v) == d for v in vecs) and len(self.precision_flat) == d * d,
            "bound len mismatch",
        )
        _check(
            math.isfinite(self.max_mahalanobis) and self.max_mahalanobis > 0.0,
            "max_mahalanobis must be > 0",
        )

    def mahalanobis_distance(self, flat_features: ArrayLike) -> float:
        """Compute regularized Mahalanobis distance for a flattened feature vector."""
        x = np.asarray(flat_features, dtype=np.float64).reshape(-1)
        _check(x.size == self.feature_dim, f"expected {self.feature_dim} features")
        d = self.feature_dim
        delta = x - np.asarray(self.mean, dtype=np.float64)
        prec = np.asarray(self.precision_flat, dtype=np.float64).reshape(d, d)
        return float(math.sqrt(max(float(delta @ prec @ delta), 0.0)))

    def check(self, capture: ArrayLike) -> tuple[bool, str | None, float]:
        """Evaluate feature-range and Mahalanobis bounds on a ``(T, C)`` capture."""
        arr = _coerce_single_capture(capture, self.seq_len, self.trajectory_channels)
        flat = arr.astype(np.float64).reshape(-1)
        lo = np.asarray(self.feature_min, dtype=np.float64)
        hi = np.asarray(self.feature_max, dtype=np.float64)
        dist = self.mahalanobis_distance(flat)
        excess = np.maximum(np.maximum(lo - flat, 0.0), np.maximum(flat - hi, 0.0))
        n_bad = int(np.sum(excess > 0.0))
        if n_bad > 0:
            return False, f"feature_range_exceeded: {n_bad} out", dist
        if not math.isfinite(dist) or dist > self.max_mahalanobis:
            return False, f"mahalanobis_distance_exceeded: {dist:.4g}", dist
        return True, None, dist


@dataclass(frozen=True)
class InverseRegressorReceipt:
    """Training, held-out evaluation, and deterministic checkpoint receipt."""

    train_loss_curve: tuple[float, ...]
    val_loss_curve: tuple[float, ...]
    held_out_rmse: float
    mean_baseline_rmse: float
    checkpoint_sha256: str
    ood_bounds: CorpusOodBounds
    knot_shape: tuple[int, ...]
    checkpoint_path: Path | None = None

    def __post_init__(self) -> None:
        """Validate receipt curves, RMSEs, and checkpoint hash (DbC)."""
        _check(bool(self.train_loss_curve and self.val_loss_curve), "curves empty")
        curves = self.train_loss_curve + self.val_loss_curve
        _check(all(math.isfinite(v) for v in curves), "loss curves must be finite")
        _check(self.held_out_rmse >= 0.0 and self.mean_baseline_rmse >= 0.0, "rmse < 0")
        _check(len(self.checkpoint_sha256) == 64, "checkpoint_sha256 must be 64 hex")


@dataclass(frozen=True)
class OodProposalOutcome:
    """Fail-closed proposal outcome from :func:`propose_knots_with_ood_guard`."""

    status: str
    knots: NDArray[np.float32] | None
    fallback: str | None = None
    ood_reason: str | None = None
    mahalanobis_distance: float | None = None
    checkpoint_sha256: str | None = None

    def __post_init__(self) -> None:
        """Enforce fail-closed proposal vs no_proposal invariants (DbC)."""
        if self.status == "no_proposal":
            _check(self.knots is None, "no_proposal must not carry clipped knots")
            _check(self.fallback == "classical_initializer", "bad fallback")
            _check(bool(self.ood_reason and self.ood_reason.strip()), "ood_reason req")
            return
        _check(self.status == "proposal", f"invalid status {self.status!r}")
        ok_k = self.knots is not None and bool(np.all(np.isfinite(self.knots)))
        _check(ok_k, "finite knots required")
        _check(self.fallback is None and self.ood_reason is None, "unexpected fallback")


@dataclass(frozen=True)
class KinematicsKnotRegressor:
    """Deterministic two-layer tanh MLP mapping flattened kinematics to knots."""

    seq_len: int
    trajectory_channels: int
    knot_shape: tuple[int, ...]
    weights: tuple[NDArray[np.float64], ...]

    def predict_knots(self, capture: ArrayLike) -> NDArray[np.float32]:
        """Predict knots for a ``(T, C)`` or ``(B, T, C)`` kinematics array."""
        arr = np.asarray(capture, dtype=np.float64)
        arr = arr[np.newaxis, ...] if arr.ndim == 2 else arr
        exp = (self.seq_len, self.trajectory_channels)
        _check(arr.ndim == 3 and arr.shape[1:] == exp, f"expected {exp}")
        _check(bool(np.all(np.isfinite(arr))), "capture kinematics must be finite")
        w1, b1, w2, b2, xm, xs = self.weights
        x_norm = (arr.reshape(arr.shape[0], -1) - xm) / xs
        pred = np.tanh(x_norm @ w1 + b1) @ w2 + b2
        return pred.reshape((arr.shape[0], *self.knot_shape)).astype(np.float32)

    def checkpoint_bytes(self) -> bytes:
        """Serialize weights and metadata deterministically for SHA-256 hashing."""
        wl = [w.tolist() for w in self.weights]
        p = [self.seq_len, self.trajectory_channels, list(self.knot_shape), wl]
        return json.dumps(p, separators=(",", ":")).encode()


def fit_inverse_regressor_with_ood(
    kinematics: ArrayLike,
    knots: ArrayLike,
    *,
    val_fraction: float = 0.15,
    held_out_fraction: float = 0.15,
    epochs: int = 30,
    lr: float = 5e-3,
    seed: int = 0,
    output_dir: Path | str | None = None,
) -> tuple[KinematicsKnotRegressor, InverseRegressorReceipt]:
    """Train deterministic kinematics->knots regressor and calibrate OOD bounds."""
    require_positive_training_hparams(
        epochs=epochs, batch_size=_BATCH, lr=lr, control_regularization=_WD
    )
    _check(0.0 < val_fraction < 0.5 and 0.0 < held_out_fraction < 0.5, "bad split")
    kin, y_flat, k_shape, (tr, va, te) = _prepare_corpus(
        kinematics, knots, val_fraction, held_out_fraction, seed
    )
    x_flat = kin.reshape(kin.shape[0], -1)
    bounds = _fit_bounds(x_flat[tr], x_flat[va], kin.shape[1], kin.shape[2])
    weights, tr_c, va_c = _train_mlp(
        x_flat[tr], y_flat[tr], x_flat[va], y_flat[va], epochs, lr, seed
    )
    model = KinematicsKnotRegressor(kin.shape[1], kin.shape[2], k_shape, weights)
    err_h = model.predict_knots(kin[te]).reshape(te.size, -1) - y_flat[te]
    err_b = np.mean(y_flat[tr], axis=0, keepdims=True) - y_flat[te]
    h_rmse = float(np.sqrt(np.mean(err_h * err_h)))
    b_rmse = float(np.sqrt(np.mean(err_b * err_b)))
    ckpt_bytes, ckpt_path = model.checkpoint_bytes(), None
    if output_dir is not None:
        ckpt_path = Path(output_dir) / "checkpoint_best.json"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        ckpt_path.write_bytes(ckpt_bytes)
    sha = hashlib.sha256(ckpt_bytes).hexdigest()
    return model, InverseRegressorReceipt(
        tr_c, va_c, h_rmse, b_rmse, sha, bounds, k_shape, ckpt_path
    )


def propose_knots_with_ood_guard(
    capture_kinematics: ArrayLike | MaskedObservation,
    model: Any,
    bounds: CorpusOodBounds,
    *,
    knot_shape: tuple[int, ...] | None = None,
    checkpoint_sha256: str | None = None,
) -> OodProposalOutcome:
    """Propose knots when inside OOD bounds; otherwise fail closed without clipping."""
    cap = capture_kinematics
    raw: ArrayLike = cap.trajectory if isinstance(cap, MaskedObservation) else cap
    capture = _coerce_single_capture(raw, bounds.seq_len, bounds.trajectory_channels)
    in_dist, reason, dist = bounds.check(capture)
    if not in_dist:
        fb = "classical_initializer"
        return OodProposalOutcome(
            "no_proposal", None, fb, reason, dist, checkpoint_sha256
        )
    if isinstance(model, KinematicsKnotRegressor):
        exp = (bounds.seq_len, bounds.trajectory_channels)
        _check((model.seq_len, model.trajectory_channels) == exp, "bounds mismatch")
        pred = model.predict_knots(capture)[0]
        knots = pred.reshape(knot_shape) if knot_shape is not None else pred
    else:
        from .regressor_predict import predict_coefficients_regressor

        raw_p = predict_coefficients_regressor(model, capture)[0]
        shp = knot_shape if knot_shape is not None else (raw_p.size,)
        _check(math.prod(shp) <= raw_p.size, "knot_shape exceeds model output dim")
        knots = raw_p[: math.prod(shp)].reshape(shp).astype(np.float32, copy=False)
    val_arr = knots if knots.ndim == 2 else knots.reshape(1, -1)
    valid, val_reason = validate_proposal(val_arr, (val_arr.shape[0], val_arr.shape[1]))
    _check(valid, val_reason or "invalid predicted knots")
    return OodProposalOutcome("proposal", knots, None, None, dist, checkpoint_sha256)


def _coerce_single_capture(
    capture: ArrayLike, seq_len: int, channels: int
) -> NDArray[np.float32]:
    """Validate and coerce a single capture to finite ``(seq_len, channels)``."""
    arr = np.asarray(capture, dtype=np.float32)
    arr = arr[0] if arr.ndim == 3 and arr.shape[0] == 1 else arr
    _check(arr.ndim == 2, f"capture must be 2-D (T, C), got shape {arr.shape}")
    _check(bool(np.all(np.isfinite(arr))), "capture kinematics must be finite")
    _check(arr.shape == (seq_len, channels), f"capture shape mismatch: {arr.shape}")
    return arr


def _prepare_corpus(
    kinematics: ArrayLike,
    knots: ArrayLike,
    val_frac: float,
    held_frac: float,
    seed: int,
) -> tuple[Any, ...]:
    """Validate finite corpus arrays and split indices deterministically."""
    kin, knot_arr = (
        np.asarray(kinematics, dtype=np.float64),
        np.asarray(knots, dtype=np.float64),
    )
    _check(kin.ndim == 3 and kin.shape[1] >= 1 and kin.shape[2] >= 1, "bad kinematics")
    _check(knot_arr.ndim >= 2 and knot_arr.shape[0] == kin.shape[0], "sample mismatch")
    _check(kin.shape[0] >= 4, "need at least 4 samples")
    ok_fin = bool(np.all(np.isfinite(kin))) and bool(np.all(np.isfinite(knot_arr)))
    _check(ok_fin, "corpus kinematics and knots must be finite")
    k_shape = tuple(int(d) for d in knot_arr.shape[1:])
    y_flat = knot_arr.reshape(knot_arr.shape[0], -1)
    _check(y_flat.shape[1] >= 1, "knots must have >= 1 output dimension")
    perm = np.random.default_rng(seed).permutation(kin.shape[0])
    n_va = max(1, int(round(val_frac * kin.shape[0])))
    n_te = max(1, int(round(held_frac * kin.shape[0])))
    _check(kin.shape[0] - n_va - n_te >= 2, "corpus too small to split")
    splits = (perm[n_va + n_te :], perm[:n_va], perm[n_va : n_va + n_te])
    return kin, y_flat, k_shape, splits


def _fit_bounds(
    x_tr: NDArray[np.float64], x_va: NDArray[np.float64], seq_len: int, channels: int
) -> CorpusOodBounds:
    """Fit feature-range and regularized Mahalanobis bounds on training corpus."""
    lo, hi = np.min(x_tr, axis=0), np.max(x_tr, axis=0)
    span = np.maximum(hi - lo, 1e-6)
    mean = np.mean(x_tr, axis=0)
    centered = x_tr - mean
    cov = (centered.T @ centered) / max(1, x_tr.shape[0] - 1)
    reg = max(1e-6, 0.01 * float(np.mean(np.diag(cov))))
    prec = np.linalg.inv(cov + reg * np.eye(x_tr.shape[1], dtype=np.float64))
    deltas = np.concatenate([x_tr, x_va], axis=0) - mean
    dists = np.sqrt(np.maximum(np.sum((deltas @ prec) * deltas, axis=1), 0.0))
    f_min = tuple(map(float, lo - _RANGE_MARGIN * span))
    f_max = tuple(map(float, hi + _RANGE_MARGIN * span))
    max_m = float(np.max(dists)) * _MAHA_MARGIN + 1e-3
    mu_t, p_t = tuple(map(float, mean)), tuple(map(float, prec.reshape(-1)))
    d = x_tr.shape[1]
    return CorpusOodBounds(f_min, f_max, mu_t, p_t, max_m, seq_len, channels, d)


def _mse(x: Any, y: Any, t: list[Any]) -> float:
    """Compute scalar MSE loss for the 2-layer tanh MLP."""
    return float(np.mean((np.tanh(x @ t[0] + t[1]) @ t[2] + t[3] - y) ** 2))


def _train_mlp(
    x_tr: NDArray[np.float64],
    y_tr: NDArray[np.float64],
    x_va: NDArray[np.float64],
    y_va: NDArray[np.float64],
    epochs: int,
    lr: float,
    seed: int,
) -> tuple[tuple[NDArray[np.float64], ...], tuple[float, ...], tuple[float, ...]]:
    """Train a 2-layer tanh MLP with Adam and return best-val weights and curves."""
    rng = np.random.default_rng(seed)
    xm = np.mean(x_tr, axis=0, keepdims=True)
    xs = np.maximum(np.std(x_tr, axis=0, keepdims=True), 1e-6)
    xn_tr, xn_va = (x_tr - xm) / xs, (x_va - xm) / xs
    d_in, d_out = x_tr.shape[1], y_tr.shape[1]
    theta = [
        rng.normal(0.0, math.sqrt(1.0 / d_in), size=(d_in, _HIDDEN)),
        np.zeros((1, _HIDDEN), dtype=np.float64),
        rng.normal(0.0, math.sqrt(1.0 / _HIDDEN), size=(_HIDDEN, d_out)),
        np.mean(y_tr, axis=0, keepdims=True),
    ]
    m, v = [np.zeros_like(t) for t in theta], [np.zeros_like(t) for t in theta]
    best_va, best_t = float("inf"), [t.copy() for t in theta]
    tr_c, va_c, step = [], [], 0
    for _ in range(epochs):
        perm = rng.permutation(xn_tr.shape[0])
        for start in range(0, perm.size, _BATCH):
            step += 1
            idx = perm[start : start + _BATCH]
            h = np.tanh(xn_tr[idx] @ theta[0] + theta[1])
            dy = (2.0 / float(y_tr[idx].size)) * (h @ theta[2] + theta[3] - y_tr[idx])
            dh = (dy @ theta[2].T) * (1.0 - h * h)
            g0, g1 = xn_tr[idx].T @ dh + _WD * theta[0], dh.sum(0, keepdims=True)
            g2, g3 = h.T @ dy + _WD * theta[2], dy.sum(0, keepdims=True)
            for i, grad in enumerate((g0, g1, g2, g3)):
                m[i], v[i] = 0.9 * m[i] + 0.1 * grad, 0.999 * v[i] + 0.001 * grad * grad
                mh, vh = m[i] / (1.0 - 0.9**step), v[i] / (1.0 - 0.999**step)
                theta[i] -= lr * mh / (np.sqrt(vh) + 1e-8)
        tr_c.append(_mse(xn_tr, y_tr, theta))
        va_loss = _mse(xn_va, y_va, theta)
        va_c.append(va_loss)
        if va_loss < best_va:
            best_va, best_t = va_loss, [t.copy() for t in theta]
    return (*best_t, xm, xs), tuple(tr_c), tuple(va_c)
