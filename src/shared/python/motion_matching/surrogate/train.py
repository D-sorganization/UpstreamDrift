"""End-to-end training loop for :class:`SwingSurrogate`.

The surrogate consumes a :class:`SweepDataset` (issue #019) directly: the
``trials`` frame supplies the per-trial ``coefficients`` and the per-timestep
``r_butt``/``r_clubhead``/``q_club``/``q`` arrays are pivoted into batched
tensors. Splits are 80/10/10 by ``trial_id`` stratified by clubhead-speed
quintile per APPROACH.md.

This module is intentionally CPU-friendly: mixed-precision is enabled only
when CUDA is available, and the loop runs in plain fp32 otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from src.shared.python.core.contracts import postcondition, precondition
from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.motion_matching.dataset import SweepDataset

from ._normalize import NormalizationStats, fit_stats, zscore_coeffs
from ._quaternion_loss import quaternion_loss
from .model import ClubTrajectory, SurrogateConfig, SwingSurrogate

logger = get_logger(__name__)


@dataclass(frozen=True)
class TrainConfig:
    """Hyperparameters for :func:`train_surrogate`.

    Attributes:
        n_epochs: Number of optimizer epochs.
        batch_size: Mini-batch size in trials.
        lr: Initial learning rate for AdamW.
        weight_decay: AdamW weight decay.
        grad_clip: Global gradient-norm clip (``<= 0`` disables).
        w_butt: Weight on butt-position MSE.
        w_clubhead: Weight on clubhead-position MSE.
        w_quat: Weight on the sign-invariant quaternion loss.
        w_aux: Weight on the auxiliary joint-angle MSE.
        seed: RNG seed for split + init reproducibility.
        val_fraction: Held-out validation fraction (by trial).
        test_fraction: Held-out test fraction (by trial).
        use_amp: Enable mixed-precision when CUDA is available.
        device: ``"cpu"``, ``"cuda"``, or ``"auto"``.
    """

    n_epochs: int = 50
    batch_size: int = 16
    lr: float = 3.0e-4
    weight_decay: float = 1.0e-4
    grad_clip: float = 1.0
    w_butt: float = 1.0
    w_clubhead: float = 1.0
    w_quat: float = 0.1
    w_aux: float = 0.1
    seed: int = 0xC0FFEE
    val_fraction: float = 0.1
    test_fraction: float = 0.1
    use_amp: bool = True
    device: str = "auto"


@dataclass
class TrainingCurves:
    """Per-epoch loss history for the train and val splits.

    Attributes:
        train_loss: Mean training loss per epoch.
        val_loss: Mean validation loss per epoch (``NaN`` if val empty).
        val_clubhead_rmse_m: Clubhead RMSE on val in metres per epoch.
    """

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_clubhead_rmse_m: list[float] = field(default_factory=list)


@dataclass
class TrainedSurrogate:
    """Bundle returned by :func:`train_surrogate`.

    Attributes:
        model: The trained :class:`SwingSurrogate`.
        config: Architectural config used to build the model.
        train_config: Hyperparameters used for training.
        norm_stats: Per-feature stats fitted on the train split.
        curves: Per-epoch training and validation loss history.
        joint_names: Joint ordering inherited from the dataset.
        seq_len: Number of timesteps the model was trained on.
        final_val_loss: Final-epoch validation loss (``NaN`` if no val).
    """

    model: SwingSurrogate
    config: SurrogateConfig
    train_config: TrainConfig
    norm_stats: NormalizationStats
    curves: TrainingCurves
    joint_names: list[str]
    seq_len: int
    final_val_loss: float


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@precondition(
    lambda dataset, config=None: dataset is not None,
    "dataset must be provided",
)
@postcondition(
    lambda result: result.final_val_loss == result.final_val_loss,
    "final_val_loss must not be NaN unless validation split is empty",
    enabled=False,  # disabled: small synthetic runs may legitimately have empty val
)
def train_surrogate(
    dataset: SweepDataset,
    config: TrainConfig | None = None,
) -> TrainedSurrogate:
    """Train a :class:`SwingSurrogate` on a :class:`SweepDataset`.

    Args:
        dataset: Loaded random-sweep dataset (real or synthetic).
        config: Optional :class:`TrainConfig`; defaults are used if omitted.

    Returns:
        A :class:`TrainedSurrogate` bundle including the model, fitted
        normalization stats, and per-epoch training curves.

    Raises:
        ValueError: If the dataset has no successful trials.
    """
    cfg = config or TrainConfig()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    coeffs, butt, head, q_club, joint_q, speeds = _materialize_tensors(dataset)
    n_trials, seq_len, _ = butt.shape

    splits = _stratified_split(speeds, cfg, n_trials)
    train_idx, val_idx, _ = splits

    stats = fit_stats(
        coeffs[train_idx],
        butt[train_idx],
        head[train_idx],
    )
    surrogate_cfg = SurrogateConfig(
        n_joints=len(dataset.joint_names),
        seq_len=seq_len,
    )
    device = _resolve_device(cfg.device)
    model = SwingSurrogate(surrogate_cfg).to(device)
    curves = _run_training_loop(
        model,
        cfg,
        stats,
        splits,
        coeffs,
        butt,
        head,
        q_club,
        joint_q,
        device,
    )

    final_val = curves.val_loss[-1] if curves.val_loss else float("nan")
    return TrainedSurrogate(
        model=model,
        config=surrogate_cfg,
        train_config=cfg,
        norm_stats=stats,
        curves=curves,
        joint_names=list(dataset.joint_names),
        seq_len=seq_len,
        final_val_loss=final_val,
    )


# ---------------------------------------------------------------------------
# Data materialization
# ---------------------------------------------------------------------------


def _materialize_tensors(
    dataset: SweepDataset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pivot the SweepDataset into per-trial numpy tensors.

    Returns ``(coeffs, butt, clubhead, q_club, joint_q, speeds)`` with shapes
    ``(N,D)``, ``(N,T,3)``, ``(N,T,3)``, ``(N,T,4)``, ``(N,T,J)``, ``(N,)``.
    Only ``solver_status == "success"`` trials are kept.
    """
    trials = dataset.trials
    success_mask = trials["solver_status"] == "success"
    trials = trials.loc[success_mask].reset_index(drop=True)
    if len(trials) == 0:
        raise ValueError("dataset has no successful trials to train on")

    timesteps = _timesteps_pandas(dataset.timesteps)
    timesteps = timesteps.sort_values(["trial_id", "t"]).reset_index(drop=True)

    by_trial = dict(list(timesteps.groupby("trial_id", sort=False)))
    sample_lengths = {tid: len(g) for tid, g in by_trial.items()}
    seq_len = min(sample_lengths.values())

    coeffs = np.stack(
        [np.asarray(c, dtype=np.float32) for c in trials["coefficients"]], axis=0
    )
    butt = _stack_vectors(trials, by_trial, seq_len, "r_butt", 3)
    head = _stack_vectors(trials, by_trial, seq_len, "r_clubhead", 3)
    q_club = _stack_vectors(trials, by_trial, seq_len, "q_club", 4)
    joint_q = _stack_vectors(trials, by_trial, seq_len, "q", len(dataset.joint_names))
    speeds = trials.get(
        "clubhead_speed_max_mph", trials["trial_id"].astype(float)
    ).to_numpy(dtype=np.float32)
    return coeffs, butt, head, q_club, joint_q, speeds


def _timesteps_pandas(timesteps: Any) -> Any:
    """Materialise a polars LazyFrame to pandas if needed."""
    if hasattr(timesteps, "collect"):
        return timesteps.collect().to_pandas()
    return timesteps


def _stack_vectors(
    trials: Any,
    by_trial: dict[int, Any],
    seq_len: int,
    column: str,
    width: int,
) -> np.ndarray:
    """Stack a list-valued timestep column into ``(N, T, width)``."""
    out = np.zeros((len(trials), seq_len, width), dtype=np.float32)
    for i, tid in enumerate(trials["trial_id"]):
        group = by_trial[tid]
        if column not in group.columns:
            continue
        values = np.asarray(list(group[column].iloc[:seq_len]), dtype=np.float32)
        if values.shape != (seq_len, width):
            raise ValueError(
                f"trial {tid}: column '{column}' has shape {values.shape}; "
                f"expected ({seq_len}, {width})"
            )
        out[i] = values
    return out


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------


def _stratified_split(
    speeds: np.ndarray,
    cfg: TrainConfig,
    n_trials: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratify by clubhead-speed quintile, then 80/10/10 by trial.

    Falls back to simple shuffled splits when there are too few trials
    to populate every quintile bucket.
    """
    rng = np.random.default_rng(cfg.seed)
    val_n = max(int(round(cfg.val_fraction * n_trials)), 0)
    test_n = max(int(round(cfg.test_fraction * n_trials)), 0)
    if val_n + test_n >= n_trials:
        val_n = max(n_trials // 10, 0)
        test_n = max(n_trials // 10, 0)
    if n_trials < 5:
        order = rng.permutation(n_trials)
        train_n = n_trials - val_n - test_n
        return (
            order[:train_n],
            order[train_n : train_n + val_n],
            order[train_n + val_n :],
        )

    bins = _quintile_bins(speeds)
    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []
    for bucket in range(5):
        bucket_idx = np.where(bins == bucket)[0]
        rng.shuffle(bucket_idx)
        n_bucket = len(bucket_idx)
        b_val = max(int(round(cfg.val_fraction * n_bucket)), 0)
        b_test = max(int(round(cfg.test_fraction * n_bucket)), 0)
        val_idx.extend(bucket_idx[:b_val])
        test_idx.extend(bucket_idx[b_val : b_val + b_test])
        train_idx.extend(bucket_idx[b_val + b_test :])
    return (
        np.array(train_idx, dtype=np.int64),
        np.array(val_idx, dtype=np.int64),
        np.array(test_idx, dtype=np.int64),
    )


def _quintile_bins(speeds: np.ndarray) -> np.ndarray:
    """Assign each trial to its clubhead-speed quintile (0..4)."""
    quantiles = np.quantile(speeds, [0.2, 0.4, 0.6, 0.8])
    return np.searchsorted(quantiles, speeds, side="right")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def _resolve_device(name: str) -> torch.device:
    """Resolve ``"auto"`` to CUDA when available, else CPU."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _run_training_loop(  # noqa: PLR0913 - splitting hurts readability here
    model: SwingSurrogate,
    cfg: TrainConfig,
    stats: NormalizationStats,
    splits: tuple[np.ndarray, np.ndarray, np.ndarray],
    coeffs: np.ndarray,
    butt: np.ndarray,
    head: np.ndarray,
    q_club: np.ndarray,
    joint_q: np.ndarray,
    device: torch.device,
) -> TrainingCurves:
    """Run the AdamW + cosine-LR loop and return per-epoch curves."""
    train_idx, val_idx, _ = splits
    train_loader = _make_loader(
        coeffs[train_idx],
        butt[train_idx],
        head[train_idx],
        q_club[train_idx],
        joint_q[train_idx],
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    val_loader = (
        _make_loader(
            coeffs[val_idx],
            butt[val_idx],
            head[val_idx],
            q_club[val_idx],
            joint_q[val_idx],
            batch_size=cfg.batch_size,
            shuffle=False,
        )
        if len(val_idx) > 0
        else None
    )

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(cfg.n_epochs, 1))
    use_amp = cfg.use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    curves = TrainingCurves()
    for epoch in range(cfg.n_epochs):
        train_loss = _run_one_epoch_train(
            model, train_loader, optimizer, scaler, cfg, stats, device, use_amp
        )
        val_loss, val_rmse = _evaluate(model, val_loader, cfg, stats, device)
        scheduler.step()
        curves.train_loss.append(train_loss)
        curves.val_loss.append(val_loss)
        curves.val_clubhead_rmse_m.append(val_rmse)
        logger.debug(
            "epoch %d/%d train=%.6f val=%.6f rmse_m=%.4f",
            epoch + 1,
            cfg.n_epochs,
            train_loss,
            val_loss,
            val_rmse,
        )
    return curves


def _make_loader(
    coeffs: np.ndarray,
    butt: np.ndarray,
    head: np.ndarray,
    q_club: np.ndarray,
    joint_q: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Wrap five aligned numpy arrays into a torch ``DataLoader``."""
    ds = TensorDataset(
        torch.from_numpy(coeffs).float(),
        torch.from_numpy(butt).float(),
        torch.from_numpy(head).float(),
        torch.from_numpy(q_club).float(),
        torch.from_numpy(joint_q).float(),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def _run_one_epoch_train(  # noqa: PLR0913
    model: SwingSurrogate,
    loader: DataLoader,
    optimizer: AdamW,
    scaler: torch.amp.GradScaler,
    cfg: TrainConfig,
    stats: NormalizationStats,
    device: torch.device,
    use_amp: bool,
) -> float:
    """Train ``model`` for one epoch; return mean batch loss."""
    model.train()
    total = 0.0
    count = 0
    for batch in loader:
        coeffs, butt, head, q_club, joint_q = (b.to(device) for b in batch)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            pred = model(zscore_coeffs(coeffs, stats))
            loss = _compute_loss(pred, butt, head, q_club, joint_q, cfg)
        if use_amp:
            scaler.scale(loss).backward()
            if cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
        total += float(loss.detach().cpu())
        count += 1
    return total / max(count, 1)


@torch.no_grad()
def _evaluate(
    model: SwingSurrogate,
    loader: DataLoader | None,
    cfg: TrainConfig,
    stats: NormalizationStats,
    device: torch.device,
) -> tuple[float, float]:
    """Compute mean val loss and clubhead RMSE in metres."""
    if loader is None:
        return float("nan"), float("nan")
    model.eval()
    total = 0.0
    sq_err = 0.0
    n_elements = 0
    count = 0
    for batch in loader:
        coeffs, butt, head, q_club, joint_q = (b.to(device) for b in batch)
        pred = model(zscore_coeffs(coeffs, stats))
        loss = _compute_loss(pred, butt, head, q_club, joint_q, cfg)
        total += float(loss.cpu())
        diff = (pred.clubhead - head).reshape(-1, 3)
        sq_err += float((diff * diff).sum().cpu())
        n_elements += diff.shape[0]
        count += 1
    rmse = (sq_err / max(n_elements * 3, 1)) ** 0.5
    return total / max(count, 1), rmse


def _compute_loss(
    pred: ClubTrajectory,
    butt: torch.Tensor,
    head: torch.Tensor,
    q_club: torch.Tensor,
    joint_q: torch.Tensor,
    cfg: TrainConfig,
) -> torch.Tensor:
    """Weighted MSE on positions + quaternion loss + auxiliary joint MSE."""
    mse = nn.functional.mse_loss
    return (
        cfg.w_butt * mse(pred.butt, butt)
        + cfg.w_clubhead * mse(pred.clubhead, head)
        + cfg.w_quat * quaternion_loss(pred.club_quat, q_club)
        + cfg.w_aux * mse(pred.joint_q, joint_q)
    )
