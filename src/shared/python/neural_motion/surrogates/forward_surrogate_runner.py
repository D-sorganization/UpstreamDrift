"""ML-WS-6 (#13783): Forward residual surrogate training, held-out eval, and receipt."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from src.shared.python.neural_motion.json_io import SortedJsonWritableMixin

from .sparse_residual import CandidateLibrary, SparseResidualFit

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

__all__ = [
    "FORWARD_SURROGATE_RECEIPT_SCHEMA",
    "ForwardSurrogateRunResult",
    "SurrogateTrainingConfig",
    "SurrogateTrainingReceipt",
    "compare_against_mean_baseline",
    "load_surrogate_receipt",
    "serialize_receipt_json",
    "train_and_evaluate_forward_surrogate",
]

FORWARD_SURROGATE_RECEIPT_SCHEMA = "neural-forward-surrogate-receipt/1.0.0"


def _sha256_json(payload: Mapping[str, Any]) -> str:
    """Return deterministic SHA-256 hex digest of a JSON-serializable mapping."""
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True, slots=True)
class SurrogateTrainingConfig:
    """Validated training hyperparameters for ML-WS-6 forward residual surrogate."""

    model_id: str = "driven_double_pendulum"
    epochs: int = 50
    learning_rate: float = 0.1
    ridge_alpha: float = 1e-4
    polynomial_degree: int = 2
    include_trig: bool = True
    seed: int = 0

    def __post_init__(self) -> None:
        """Design by Contract validation for training hyperparameters."""
        if not self.model_id.strip() or self.epochs < 1 or self.polynomial_degree < 1:
            raise ValueError("model_id must be non-empty; epochs and degree >= 1")
        if not (np.isfinite(self.learning_rate) and self.learning_rate > 0.0):
            raise ValueError("learning_rate must be positive and finite")
        if not (np.isfinite(self.ridge_alpha) and self.ridge_alpha >= 0.0):
            raise ValueError("ridge_alpha must be non-negative and finite")


@dataclass(frozen=True, slots=True)
class SurrogateTrainingReceipt(SortedJsonWritableMixin):
    """Proposal-only training and held-out evaluation receipt for ML-WS-6."""

    model_id: str
    seed: int
    epochs: int
    train_sample_count: int
    val_sample_count: int
    train_episode_ids: tuple[str, ...]
    val_episode_ids: tuple[str, ...]
    train_loss_curve: tuple[float, ...]
    val_loss_curve: tuple[float, ...]
    held_out_rmse: float
    mean_baseline_rmse: float
    beats_mean_baseline: bool
    checkpoint_sha256: str
    proposal_only: bool = True
    claims_match: bool = False
    schema: str = FORWARD_SURROGATE_RECEIPT_SCHEMA
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        """Enforce proposal-only honesty invariants, disjoint splits, and digests."""
        if self.schema != FORWARD_SURROGATE_RECEIPT_SCHEMA or not self.model_id.strip():
            raise ValueError("invalid schema or empty model_id")
        if self.proposal_only is not True or self.claims_match is not False:
            raise ValueError("proposal_only=True and claims_match=False required")
        if self.train_sample_count < 1 or self.val_sample_count < 1 or self.epochs < 1:
            raise ValueError("sample counts and epochs must be >= 1")
        tr, va = self.train_loss_curve, self.val_loss_curve
        if len(tr) != self.epochs or len(va) != self.epochs:
            raise ValueError("loss curve lengths must match epochs")
        vals = (self.held_out_rmse, self.mean_baseline_rmse, *tr, *va)
        if not all(np.isfinite(v) for v in vals):
            raise ValueError("receipt loss curves and RMSE metrics must be finite")
        if len(self.checkpoint_sha256) != 64:
            raise ValueError("checkpoint_sha256 must be a 64-char SHA-256 hex digest")
        if set(self.train_episode_ids) & set(self.val_episode_ids):
            raise ValueError("overlapping train/val episode IDs refused")
        computed = _sha256_json(self._body_dict())
        if not self.receipt_sha256:
            object.__setattr__(self, "receipt_sha256", computed)
        elif self.receipt_sha256 != computed:
            raise ValueError("receipt_sha256 mismatch against receipt payload")

    def _body_dict(self) -> dict[str, Any]:
        """Return deterministic receipt payload excluding receipt_sha256."""
        data = asdict(self)
        data.pop("receipt_sha256", None)
        return data

    def as_dict(self) -> dict[str, Any]:
        """Return full JSON-serializable receipt including receipt_sha256."""
        return {**self._body_dict(), "receipt_sha256": self.receipt_sha256}


@dataclass(frozen=True, slots=True)
class ForwardSurrogateRunResult:
    """Trained surrogate fit, held-out predictions, and proposal-only receipt."""

    fit: SparseResidualFit
    receipt: SurrogateTrainingReceipt
    val_predictions: np.ndarray


def serialize_receipt_json(receipt: SurrogateTrainingReceipt) -> str:
    """Serialize a training receipt to deterministic sorted JSON with newline."""
    return json.dumps(receipt.as_dict(), indent=2, sort_keys=True) + "\n"


def load_surrogate_receipt(path: str | Path) -> SurrogateTrainingReceipt:
    """Load and digest-verify a proposal-only surrogate training receipt."""
    data = dict(json.loads(Path(path).read_text(encoding="utf-8")))
    for k in ("train_episode_ids", "val_episode_ids"):
        data[k] = tuple(str(ep) for ep in data[k])
    for k in ("train_loss_curve", "val_loss_curve"):
        data[k] = tuple(float(v) for v in data[k])
    return SurrogateTrainingReceipt(**data)


def _coerce_finite(values: ArrayLike, label: str) -> np.ndarray:
    """Coerce array-like input to non-empty finite 2-D float64 array."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{label} must be non-empty 2-D, got shape {arr.shape}")
    if not bool(np.all(np.isfinite(arr))):
        raise ValueError(f"{label} must contain only finite values")
    return arr


def _coerce_ids(
    ids_in: Sequence[str] | None, n_rows: int, label: str
) -> tuple[str, ...]:
    """Validate optional episode ID sequence length and non-empty labels."""
    if ids_in is None:
        return ()
    ids = tuple(str(ep) for ep in ids_in)
    if len(ids) != n_rows or any(not ep.strip() for ep in ids):
        raise ValueError(f"{label} must have {n_rows} non-empty episode IDs")
    return ids


def _split_episodes(
    x: np.ndarray, y: np.ndarray, tr_ids: tuple[str, ...], seed: int
) -> tuple[Any, ...]:
    """Split matrices deterministically by unique episode_id."""
    eps = sorted(set(tr_ids))
    if len(eps) < 2:
        raise ValueError("need >= 2 distinct episode_ids when val_features is None")
    perm = list(np.random.default_rng(int(seed)).permutation(len(eps)))
    n_val = min(len(eps) - 1, max(1, int(round(0.25 * len(eps)))))
    val_set = {eps[i] for i in perm[:n_val]}
    val_m = np.array([ep in val_set for ep in tr_ids], dtype=bool)
    tr_e = tuple(e for e, keep in zip(tr_ids, ~val_m, strict=True) if keep)
    va_e = tuple(e for e, keep in zip(tr_ids, val_m, strict=True) if keep)
    return x[~val_m], y[~val_m], tr_e, x[val_m], y[val_m], va_e


def compare_against_mean_baseline(
    train_targets: np.ndarray, val_targets: np.ndarray, val_predictions: np.ndarray
) -> tuple[float, float, bool]:
    """Return (held_out_rmse, mean_baseline_rmse, beats_mean_baseline)."""
    y_tr, y_va = (
        _coerce_finite(train_targets, "train_targets"),
        _coerce_finite(val_targets, "val_targets"),
    )
    preds = _coerce_finite(val_predictions, "val_predictions")
    if y_va.shape != preds.shape or y_va.shape[1] != y_tr.shape[1]:
        raise ValueError("target and prediction shapes are incompatible")
    mean_vec = np.mean(y_tr, axis=0, keepdims=True)
    base_rmse = float(
        np.sqrt(np.mean((y_va - np.broadcast_to(mean_vec, y_va.shape)) ** 2))
    )
    held_rmse = float(np.sqrt(np.mean((y_va - preds) ** 2)))
    return held_rmse, base_rmse, bool(held_rmse < base_rmse)


def _train_basis_ridge(
    phi_tr: np.ndarray,
    y_tr: np.ndarray,
    phi_va: np.ndarray,
    y_va: np.ndarray,
    cfg: SurrogateTrainingConfig,
) -> tuple[np.ndarray, tuple[float, ...], tuple[float, ...]]:
    """Run deterministic iterative ridge GD over CandidateLibrary features."""
    n_tr, n_feat = phi_tr.shape
    mu, raw_std = np.mean(phi_tr[:, 1:], axis=0), np.std(phi_tr[:, 1:], axis=0)
    std = np.where(raw_std > 1e-12, raw_std, 1.0)
    tr_n, va_n = phi_tr.copy(), phi_va.copy()
    tr_n[:, 1:], va_n[:, 1:] = (phi_tr[:, 1:] - mu) / std, (phi_va[:, 1:] - mu) / std
    reg = np.ones((n_feat, 1), dtype=np.float64)
    reg[0, 0] = 0.0
    max_eig = float(np.linalg.eigvalsh((tr_n.T @ tr_n) / n_tr)[-1]) + cfg.ridge_alpha
    step = min(cfg.learning_rate, 1.0 / max(max_eig, 1e-12))
    w_n, tr_curve, va_curve = (
        np.zeros((n_feat, y_tr.shape[1]), dtype=np.float64),
        [],
        [],
    )
    for _ in range(int(cfg.epochs)):
        grad = (tr_n.T @ (tr_n @ w_n - y_tr)) / n_tr + cfg.ridge_alpha * (reg * w_n)
        w_n = w_n - step * grad
        tr_curve.append(float(np.sqrt(np.mean((tr_n @ w_n - y_tr) ** 2))))
        va_curve.append(float(np.sqrt(np.mean((va_n @ w_n - y_va) ** 2))))
    weights = np.zeros_like(w_n)
    weights[1:, :] = w_n[1:, :] / std[:, None]
    weights[0, :] = w_n[0, :] - (mu @ weights[1:, :])
    return weights, tuple(tr_curve), tuple(va_curve)


def train_and_evaluate_forward_surrogate(
    features: ArrayLike,
    targets: ArrayLike,
    *,
    episode_ids: Sequence[str] | None = None,
    val_features: ArrayLike | None = None,
    val_targets: ArrayLike | None = None,
    val_episode_ids: Sequence[str] | None = None,
    config: SurrogateTrainingConfig | None = None,
    receipt_path: str | Path | None = None,
) -> ForwardSurrogateRunResult:
    """Train forward residual surrogate, evaluate held-out, and emit receipt."""
    cfg = config if config is not None else SurrogateTrainingConfig()
    x_tr, y_tr = (
        _coerce_finite(features, "features"),
        _coerce_finite(targets, "targets"),
    )
    if x_tr.shape[0] != y_tr.shape[0]:
        raise ValueError("features and targets row counts must match")
    tr_ids = _coerce_ids(episode_ids, x_tr.shape[0], "episode_ids")
    if (val_features is None) != (val_targets is None):
        raise ValueError("val_features and val_targets must both be provided or None")
    if val_features is None or val_targets is None:
        x_tr, y_tr, tr_ids, x_va, y_va, va_ids = _split_episodes(
            x_tr, y_tr, tr_ids, cfg.seed
        )
    else:
        x_va, y_va = (
            _coerce_finite(val_features, "val_features"),
            _coerce_finite(val_targets, "val_targets"),
        )
        if (
            x_va.shape[0] != y_va.shape[0]
            or x_va.shape[1] != x_tr.shape[1]
            or y_va.shape[1] != y_tr.shape[1]
        ):
            raise ValueError("val feature/target shapes must match training dimensions")
        va_ids = _coerce_ids(val_episode_ids, x_va.shape[0], "val_episode_ids")
        if set(tr_ids) & set(va_ids):
            raise ValueError("overlapping train/val episode IDs refused")
    lib = CandidateLibrary(
        degree=cfg.polynomial_degree, include_trig=cfg.include_trig, include_bias=True
    )
    names = lib.feature_names(tuple(f"x{i}" for i in range(x_tr.shape[1])))
    w, tr_c, va_c = _train_basis_ridge(
        lib.transform(x_tr), y_tr, lib.transform(x_va), y_va, cfg
    )
    fit = SparseResidualFit(
        w, np.ones_like(w, dtype=bool), names, lib, 0.0, cfg.epochs, True, tr_c[-1]
    )
    preds = fit.predict(x_va)
    held, base, beats = compare_against_mean_baseline(y_tr, y_va, preds)
    ckpt = _sha256_json(
        {
            "model_id": cfg.model_id,
            "seed": int(cfg.seed),
            "coeffs": np.asarray(w, dtype="<f8").tobytes().hex(),
        }
    )
    receipt = SurrogateTrainingReceipt(
        cfg.model_id,
        cfg.seed,
        cfg.epochs,
        int(x_tr.shape[0]),
        int(x_va.shape[0]),
        tuple(sorted(set(tr_ids))),
        tuple(sorted(set(va_ids))),
        tr_c,
        va_c,
        held,
        base,
        beats,
        ckpt,
    )
    if receipt_path is not None:
        receipt.write_json(receipt_path)
    return ForwardSurrogateRunResult(fit=fit, receipt=receipt, val_predictions=preds)
