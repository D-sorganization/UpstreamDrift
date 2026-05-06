"""Mode-coverage diagnostics for the trained inverse CVAE (issue #4004 / #035).

Three diagnostic primitives are exposed:

* :func:`latent_projection` -- encode validation kinematics, project the
  posterior means to 2-D via UMAP (preferred) or t-SNE (fallback). Used to
  visualise whether the latent space organises by swing characteristics.
* :func:`sample_diversity` -- draw N samples for a single conditioning
  trajectory and measure pairwise spread. The leading mode-collapse
  detector for a CVAE.
* :func:`dataset_coverage_map` -- per-trial round-trip RMSE across the
  validation split, flagging regions where the inverse model fails.

UMAP (``umap-learn``) is optional: if the import fails we transparently
fall back to scikit-learn t-SNE. PCA is always available via scikit-learn.

Plots live in :mod:`._plot_diagnostics` so this module stays headless and
matplotlib-independent.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
from numpy.typing import NDArray

from src.shared.python.core.contracts import postcondition, precondition
from src.shared.python.logging_pkg.logging_config import get_logger

from ._validate import ForwardFn, RoundTripOutput
from .cvae import SwingInverseCVAE

logger = get_logger(__name__)

ProjectionMethod = Literal["umap", "tsne", "pca"]
_VALID_METHODS: tuple[str, ...] = ("umap", "tsne", "pca")

# Below this mean pairwise L2 we consider a target "collapsed".
DEFAULT_COLLAPSE_THRESHOLD = 1e-3


__all__ = [
    "CoverageMap",
    "DEFAULT_COLLAPSE_THRESHOLD",
    "DiversityReport",
    "LatentProjection",
    "ProjectionMethod",
    "dataset_coverage_map",
    "latent_projection",
    "sample_diversity",
]


@dataclass(frozen=True)
class LatentProjection:
    """2-D projection of the encoder posterior means.

    Attributes:
        coords: ``(n_trials, 2)`` float64 ndarray.
        method: Which strategy was actually used (the request may be
            silently downgraded to ``"tsne"`` if UMAP is unavailable).
        seed: RNG seed propagated to the projector.
    """

    coords: NDArray[np.float64]
    method: ProjectionMethod
    seed: int


@dataclass(frozen=True)
class DiversityReport:
    """Pairwise-distance summary for samples drawn for one target.

    Attributes:
        samples: ``(n_samples, D)`` candidate coefficient ndarray.
        pairwise_distances: ``(n_samples * (n_samples-1) / 2,)`` flat
            vector of pairwise L2 distances.
        mean_distance: Mean pairwise L2.
        median_distance: Median pairwise L2.
        collapsed: True iff ``mean_distance < threshold``.
        threshold: Threshold used for the collapse decision.
    """

    samples: NDArray[np.float64]
    pairwise_distances: NDArray[np.float64]
    mean_distance: float
    median_distance: float
    collapsed: bool
    threshold: float


@dataclass(frozen=True)
class CoverageMap:
    """Per-trial round-trip RMSE map for a validation split.

    Attributes:
        trial_ids: ``(n_val,)`` int ndarray.
        rmses_m: ``(n_val,)`` float ndarray of per-trial round-trip RMSEs.
        flagged_mask: Boolean mask, True for trials whose RMSE exceeds
            ``flag_threshold_m``.
        flag_threshold_m: Threshold used to compute :attr:`flagged_mask`.
        mean_rmse_m: Mean RMSE across the split.
    """

    trial_ids: NDArray[np.int64]
    rmses_m: NDArray[np.float64]
    flagged_mask: NDArray[np.bool_]
    flag_threshold_m: float
    mean_rmse_m: float


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------
def _validate_method(method: str) -> ProjectionMethod:
    if method not in _VALID_METHODS:
        raise ValueError(f"method must be one of {_VALID_METHODS}; got {method!r}")
    return method  # type: ignore[return-value]


def _stack_kinematics(
    kinematics_iter: Iterable[torch.Tensor | NDArray[np.float64]],
) -> torch.Tensor:
    """Stack an iterable of ``(T, F)`` or ``(1, T, F)`` tensors into ``(N, T, F)``."""
    tensors: list[torch.Tensor] = []
    for item in kinematics_iter:
        t = (
            item.detach().cpu()
            if isinstance(item, torch.Tensor)
            else torch.as_tensor(np.asarray(item, dtype=np.float32))
        )
        if t.dim() == 3 and t.shape[0] == 1:
            t = t.squeeze(0)
        if t.dim() != 2:
            raise ValueError(
                f"each kinematics entry must be 2D (T, F) or (1, T, F); got {tuple(t.shape)}"
            )
        tensors.append(t.float())
    if not tensors:
        raise ValueError("kinematics iterable was empty")
    return torch.stack(tensors, dim=0)


def _encode_mu(
    model: SwingInverseCVAE, kinematics: torch.Tensor
) -> NDArray[np.float64]:
    """Encode a batch of kinematic sequences and return the posterior means."""
    model.eval()
    with torch.no_grad():
        out = model.encode(kinematics, sample=False)
    return out.mu.detach().cpu().numpy().astype(np.float64)


def _project_pca_numpy(mu: NDArray[np.float64], seed: int) -> NDArray[np.float64]:
    """Pure-numpy PCA fallback used when scikit-learn is not installed.

    A deterministic SVD-based projection -- ``seed`` is accepted for
    signature-parity with the sklearn variant but unused (SVD is
    deterministic).
    """
    del seed  # SVD is deterministic; seed accepted for parity only.
    centred = mu - mu.mean(axis=0, keepdims=True)
    n_components = min(2, mu.shape[1])
    if centred.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64)
    _u, _s, vh = np.linalg.svd(centred, full_matrices=False)
    components = vh[:n_components]
    proj = centred @ components.T
    if proj.shape[1] == 1:
        proj = np.column_stack([proj[:, 0], np.zeros_like(proj[:, 0])])
    return np.asarray(proj, dtype=np.float64)


def _project_pca(mu: NDArray[np.float64], seed: int) -> NDArray[np.float64]:
    """PCA projection. Prefers scikit-learn; falls back to numpy SVD."""
    try:
        from sklearn.decomposition import PCA  # local import; optional dep
    except ImportError:
        return _project_pca_numpy(mu, seed)
    n_components = min(2, mu.shape[1])
    proj = PCA(n_components=n_components, random_state=seed).fit_transform(mu)
    if proj.shape[1] == 1:
        proj = np.column_stack([proj[:, 0], np.zeros_like(proj[:, 0])])
    return np.asarray(proj, dtype=np.float64)


def _project_tsne(mu: NDArray[np.float64], seed: int) -> NDArray[np.float64]:
    """t-SNE projection. Falls back to PCA if scikit-learn is unavailable."""
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        logger.info("scikit-learn not installed; t-SNE falling back to PCA")
        return _project_pca_numpy(mu, seed)
    # t-SNE perplexity must be < n_samples; clamp for tiny test fixtures.
    perplexity = float(max(2, min(30, mu.shape[0] - 1)))
    proj = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        random_state=seed,
    ).fit_transform(mu)
    return np.asarray(proj, dtype=np.float64)


def _project_umap(mu: NDArray[np.float64], seed: int) -> NDArray[np.float64]:
    import umap  # type: ignore[import-untyped]

    n_neighbors = int(max(2, min(15, mu.shape[0] - 1)))
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        random_state=seed,
    )
    proj = reducer.fit_transform(mu)
    return np.asarray(proj, dtype=np.float64)


def _has_module(name: str) -> bool:
    try:
        __import__(name)
    except ImportError:
        return False
    return True


def _resolve_method(method: ProjectionMethod) -> tuple[ProjectionMethod, Callable]:
    """Pick a projector with optional-dependency fallbacks.

    Order: ``umap`` -> ``tsne`` -> ``pca``. UMAP requires ``umap-learn``;
    t-SNE requires ``scikit-learn``. The numpy-SVD PCA path always works.
    """
    if method == "pca":
        return "pca", _project_pca
    if method == "tsne":
        if _has_module("sklearn"):
            return "tsne", _project_tsne
        logger.info("scikit-learn not installed; falling back to PCA")
        return "pca", _project_pca
    # UMAP requested.
    if _has_module("umap"):
        return "umap", _project_umap
    if _has_module("sklearn"):
        logger.info("umap-learn not installed; falling back to t-SNE")
        return "tsne", _project_tsne
    logger.info("umap-learn and scikit-learn not installed; falling back to PCA")
    return "pca", _project_pca


# ---------------------------------------------------------------------------
# Public diagnostics
# ---------------------------------------------------------------------------
@precondition(
    lambda model, kinematics, *, method="umap", seed=0: isinstance(
        model, SwingInverseCVAE
    ),
    "model must be a SwingInverseCVAE",
)
@postcondition(
    lambda result: result.coords.ndim == 2 and result.coords.shape[1] == 2,
    "latent_projection must return a (N, 2) ndarray",
)
def latent_projection(
    model: SwingInverseCVAE,
    kinematics: Iterable[torch.Tensor | NDArray[np.float64]] | torch.Tensor,
    *,
    method: ProjectionMethod = "umap",
    seed: int = 0,
) -> LatentProjection:
    """Project encoder posterior means of ``kinematics`` into 2-D.

    Args:
        model: The trained inverse CVAE.
        kinematics: Either a single ``(N, T, F)`` tensor or an iterable of
            ``(T, F)`` / ``(1, T, F)`` tensors / ndarrays.
        method: ``"umap"`` (default), ``"tsne"``, or ``"pca"``. UMAP
            silently falls back to t-SNE if ``umap-learn`` is not
            installed.
        seed: RNG seed forwarded to the projector for reproducibility.

    Returns:
        :class:`LatentProjection` with ``(N, 2)`` coordinates.
    """
    requested = _validate_method(method)
    if isinstance(kinematics, torch.Tensor):
        if kinematics.dim() != 3:
            raise ValueError(
                f"kinematics tensor must be 3D (N, T, F); got {tuple(kinematics.shape)}"
            )
        batch = kinematics.float()
    else:
        batch = _stack_kinematics(kinematics)
    if batch.shape[0] < 2:
        raise ValueError(
            f"latent_projection needs >=2 samples to project to 2-D; got {batch.shape[0]}"
        )
    mu = _encode_mu(model, batch)
    used, fn = _resolve_method(requested)
    coords = fn(mu, seed)
    return LatentProjection(
        coords=np.asarray(coords, dtype=np.float64), method=used, seed=seed
    )


def _pairwise_l2(samples: NDArray[np.float64]) -> NDArray[np.float64]:
    """Flat upper-triangular pairwise L2 distance vector."""
    n = samples.shape[0]
    if n < 2:
        return np.zeros((0,), dtype=np.float64)
    diffs = samples[:, None, :] - samples[None, :, :]
    d = np.sqrt(np.sum(diffs * diffs, axis=-1))
    iu = np.triu_indices(n, k=1)
    return np.asarray(d[iu], dtype=np.float64)


@precondition(
    lambda model, kinematics, *, n_samples=16, threshold=DEFAULT_COLLAPSE_THRESHOLD: (
        n_samples >= 2
    ),
    "n_samples must be at least 2",
)
def sample_diversity(
    model: SwingInverseCVAE,
    kinematics: torch.Tensor | NDArray[np.float64],
    *,
    n_samples: int = 16,
    threshold: float = DEFAULT_COLLAPSE_THRESHOLD,
) -> DiversityReport:
    """Draw ``n_samples`` candidates for ``kinematics`` and score their spread.

    Args:
        model: The trained inverse CVAE.
        kinematics: ``(T, F)`` or ``(1, T, F)`` conditioning tensor.
        n_samples: Number of candidate samples (>=2).
        threshold: Mean-distance below which the target is flagged as a
            mode-collapse case.

    Returns:
        :class:`DiversityReport` carrying the samples and pairwise stats.
    """
    if not isinstance(model, SwingInverseCVAE):
        raise TypeError(f"model must be a SwingInverseCVAE; got {type(model).__name__}")
    if not (threshold >= 0.0):
        raise ValueError(f"threshold must be >= 0; got {threshold!r}")

    if isinstance(kinematics, np.ndarray):
        kin_t = torch.as_tensor(kinematics.astype(np.float32))
    else:
        kin_t = kinematics.detach().cpu().float()
    if kin_t.dim() == 2:
        kin_t = kin_t.unsqueeze(0)
    if kin_t.dim() != 3 or kin_t.shape[0] != 1:
        raise ValueError(
            f"kinematics must reduce to (1, T, F); got {tuple(kin_t.shape)}"
        )

    model.eval()
    with torch.no_grad():
        sampled = model.sample_coefficients(kin_t, n_samples=n_samples)
    samples = sampled.squeeze(0).detach().cpu().numpy().astype(np.float64)
    distances = _pairwise_l2(samples)
    mean_d = float(distances.mean()) if distances.size else 0.0
    median_d = float(np.median(distances)) if distances.size else 0.0
    return DiversityReport(
        samples=samples,
        pairwise_distances=distances,
        mean_distance=mean_d,
        median_distance=median_d,
        collapsed=bool(mean_d < threshold),
        threshold=float(threshold),
    )


def _select_validation_indices(
    n_total: int,
    val_split: float,
    seed: int,
) -> NDArray[np.int64]:
    """Deterministic validation-fold index selection (no leakage of train trials).

    The indices are drawn with a seeded RNG, so the same dataset + seed always
    yields the same validation set. Only the trailing ``val_split`` fraction
    is returned; train indices are explicitly *not* exposed by this helper to
    make accidental misuse harder.
    """
    if not (0.0 < val_split < 1.0):
        raise ValueError(f"val_split must be in (0, 1); got {val_split!r}")
    if n_total < 2:
        raise ValueError(f"need at least 2 trials to split; got {n_total}")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_total)
    n_val = max(1, int(round(n_total * val_split)))
    val = np.sort(perm[-n_val:])
    return val.astype(np.int64)


@dataclass(frozen=True)
class CoverageTrial:
    """One trial's contribution to the coverage map."""

    trial_id: int
    kinematics: torch.Tensor | NDArray[np.float64]
    target_butt: NDArray[np.float64]
    target_clubhead: NDArray[np.float64]


def _coerce_trial(item: Any, idx: int) -> CoverageTrial:
    if isinstance(item, CoverageTrial):
        return item
    if isinstance(item, dict):
        return CoverageTrial(
            trial_id=int(item.get("trial_id", idx)),
            kinematics=item["kinematics"],
            target_butt=np.asarray(item["target_butt"], dtype=np.float64),
            target_clubhead=np.asarray(item["target_clubhead"], dtype=np.float64),
        )
    raise TypeError(
        f"coverage trial must be CoverageTrial or dict; got {type(item).__name__}"
    )


def _trial_rmse(
    model: SwingInverseCVAE,
    trial: CoverageTrial,
    sim_fn: ForwardFn,
) -> float:
    """One-shot round-trip RMSE for a single validation trial."""
    if isinstance(trial.kinematics, np.ndarray):
        kin = torch.as_tensor(trial.kinematics.astype(np.float32))
    else:
        kin = trial.kinematics.detach().cpu().float()
    if kin.dim() == 2:
        kin = kin.unsqueeze(0)
    model.eval()
    with torch.no_grad():
        out = model.encode(kin, sample=False)
        coeffs = model.decode(out.z, context=None, kinematics=kin)
    coeffs_np = coeffs.squeeze(0).detach().cpu().numpy().astype(np.float64)
    butt_pred, clubhead_pred, _quat = sim_fn(coeffs_np)
    db = butt_pred - trial.target_butt
    dc = clubhead_pred - trial.target_clubhead
    return float(np.sqrt(np.mean(np.sum(db * db, axis=1) + np.sum(dc * dc, axis=1))))


def dataset_coverage_map(
    model: SwingInverseCVAE,
    trials: list[CoverageTrial] | list[dict[str, Any]],
    sim_fn: Callable[[NDArray[np.float64]], RoundTripOutput],
    *,
    val_split: float = 0.2,
    seed: int = 0,
    flag_threshold_m: float = 0.05,
) -> CoverageMap:
    """Compute round-trip RMSE per validation trial.

    Splits the supplied ``trials`` into a deterministic validation fold
    (the last ``val_split`` fraction of a seeded permutation) and runs a
    one-shot round trip (``encode -> decode -> sim_fn``) for each. The
    train portion of ``trials`` is never passed to ``sim_fn`` -- this is
    asserted in the unit tests as a leakage guard.

    Args:
        model: The trained inverse CVAE.
        trials: Sequence of :class:`CoverageTrial` or dict equivalents.
        sim_fn: Forward-model callable mapping coefficients to a
            ``(butt, clubhead, quat)`` ndarray triple.
        val_split: Fraction of trials reserved for validation in (0, 1).
        seed: RNG seed for the validation permutation.
        flag_threshold_m: Per-trial RMSE above which the trial is flagged.

    Returns:
        :class:`CoverageMap` with per-validation-trial RMSEs and a flag mask.
    """
    if not isinstance(model, SwingInverseCVAE):
        raise TypeError(f"model must be a SwingInverseCVAE; got {type(model).__name__}")
    if flag_threshold_m <= 0:
        raise ValueError(f"flag_threshold_m must be positive; got {flag_threshold_m!r}")
    if not trials:
        raise ValueError("trials must be non-empty")

    coerced = [_coerce_trial(t, i) for i, t in enumerate(trials)]
    val_idx = _select_validation_indices(len(coerced), val_split, seed)
    rmses = np.empty(val_idx.shape[0], dtype=np.float64)
    trial_ids = np.empty(val_idx.shape[0], dtype=np.int64)
    for out_i, idx in enumerate(val_idx):
        trial = coerced[int(idx)]
        rmses[out_i] = _trial_rmse(model, trial, sim_fn)
        trial_ids[out_i] = trial.trial_id
    flagged = rmses > flag_threshold_m
    return CoverageMap(
        trial_ids=trial_ids,
        rmses_m=rmses,
        flagged_mask=flagged,
        flag_threshold_m=float(flag_threshold_m),
        mean_rmse_m=float(rmses.mean()) if rmses.size else 0.0,
    )
