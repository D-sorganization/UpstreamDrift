"""Training loop for :class:`TimestepInverseDynamics` on the realistic-speed
filtered subset of the compact dataset.

Pipeline:
    1. Load compact dataset (eager pandas DataFrames).
    2. Split *by trial_id* (90/10) so val timesteps come from unseen trials.
    3. Filter both splits to ``||v_clubhead|| in [lo, hi]`` mph.
    4. Compute per-DOF (mean, std) from the train split (NaN-aware).
    5. Standardise inputs/targets, train an MLP with masked MSE on the
       standardised tau (mask zeros out gradients on NaN-target DOFs).
    6. Report per-epoch ``train_loss``, ``val_loss``, and ``val_tau_mae_nm``
       (de-standardised, masked, in physical Nm).
    7. Persist best checkpoint + ``metrics.json``.

Loss baseline: standardised MSE of the mean-prediction baseline is 1.0.
A working model should be < 0.1.
"""

from __future__ import annotations

import json
import logging
import time
import warnings
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from src.shared.python.dataset_tools.load_compact import (
    CompactSwingDataset,
    load_compact_swing_dataset,
)

from .filter import filter_timesteps_by_speed
from .model import (
    TimestepInverseConfig,
    TimestepInverseDynamics,
)

logger = logging.getLogger(__name__)


DEFAULT_OUTPUT_ROOT = Path("output/timestep_inverse")
DEFAULT_PATIENCE = 15
DEFAULT_SEED = 0xC0FFEE


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TimestepEpochMetrics:
    """Per-epoch metrics for the timestep inverse-dynamics trainer.

    Attributes:
        epoch: 0-based epoch index.
        train_loss: Mean masked MSE on standardised tau over train batches.
        val_loss: Mean masked MSE on standardised tau over val batches.
        val_tau_mae_nm: Mean absolute error in physical Nm over the val
            split, masked on NaN-target DOFs.
        duration_s: Wall-clock seconds for this epoch.
    """

    epoch: int
    train_loss: float
    val_loss: float
    val_tau_mae_nm: float
    duration_s: float


@dataclass(frozen=True)
class TimestepTrainingResult:
    """Outcome of :func:`train_timestep_inverse`."""

    history: tuple[TimestepEpochMetrics, ...]
    best_epoch: int
    best_val_loss: float
    final_epoch: int
    checkpoint_path: Path
    output_dir: Path
    n_train_trials: int
    n_val_trials: int
    n_train_timesteps: int
    n_val_timesteps: int
    parameter_count: int
    config: TimestepInverseConfig
    speed_lo_mph: float
    speed_hi_mph: float

    def to_summary(self) -> dict:
        """JSON-serialisable summary for ``metrics.json``."""
        return {
            "history": [asdict(h) for h in self.history],
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "final_epoch": self.final_epoch,
            "checkpoint_path": str(self.checkpoint_path),
            "output_dir": str(self.output_dir),
            "n_train_trials": self.n_train_trials,
            "n_val_trials": self.n_val_trials,
            "n_train_timesteps": self.n_train_timesteps,
            "n_val_timesteps": self.n_val_timesteps,
            "parameter_count": self.parameter_count,
            "speed_lo_mph": self.speed_lo_mph,
            "speed_hi_mph": self.speed_hi_mph,
        }


# ---------------------------------------------------------------------------
# Dataset adapter
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _PreparedTimestepBatch:
    state: torch.Tensor  # (N, input_dim)
    tau: torch.Tensor  # (N, output_dim)
    tau_mask: torch.Tensor  # (N, output_dim) float32 in {0, 1}


class _TimestepTensorDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        state: torch.Tensor,
        tau: torch.Tensor,
        tau_mask: torch.Tensor,
    ) -> None:
        if state.shape[0] != tau.shape[0] or tau.shape != tau_mask.shape:
            raise ValueError("state/tau/tau_mask first-dim mismatch")
        self._state = state
        self._tau = tau
        self._tau_mask = tau_mask

    def __len__(self) -> int:
        return int(self._state.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "state": self._state[idx],
            "tau": self._tau[idx],
            "tau_mask": self._tau_mask[idx],
        }


def _collate(items: Iterable[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    items_list = list(items)
    state = torch.stack([it["state"] for it in items_list], dim=0)
    tau = torch.stack([it["tau"] for it in items_list], dim=0)
    tau_mask = torch.stack([it["tau_mask"] for it in items_list], dim=0)
    return {"state": state, "tau": tau, "tau_mask": tau_mask}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def train_timestep_inverse(
    dataset_path: str | Path,
    *,
    speed_lo_mph: float = 50.0,
    speed_hi_mph: float = 150.0,
    epochs: int = 80,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    device: str | torch.device = "auto",
    seed: int = DEFAULT_SEED,
    patience: int = DEFAULT_PATIENCE,
    val_fraction: float = 0.1,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    grad_clip: float = 1.0,
    config: TimestepInverseConfig | None = None,
    dataset_loader: Any = None,
) -> TimestepTrainingResult:
    """Train a per-timestep inverse-dynamics MLP end-to-end.

    Parameters
    ----------
    dataset_path
        Folder containing ``trials.parquet`` and ``timesteps.parquet``.
    speed_lo_mph, speed_hi_mph
        Realistic clubhead-speed window (inclusive). Default 50-150 mph.
    epochs
        Maximum number of training epochs (must be > 0).
    batch_size
        DataLoader batch size (must be > 0).
    lr
        AdamW learning rate (must be > 0).
    device
        Torch device or ``"auto"`` (CUDA if available else CPU).
    seed
        Random seed for split + torch RNG.
    patience
        Early-stop after ``patience`` epochs without val_loss improvement.
    val_fraction
        Fraction of trials assigned to validation (0 < f < 1).
    output_root
        Root directory under which a timestamped output folder is created.
    config
        Optional :class:`TimestepInverseConfig`.
    dataset_loader
        Optional callable ``(path) -> CompactSwingDataset`` for testing.

    Returns
    -------
    TimestepTrainingResult
        History, best-epoch index, output directory, parameter count.

    Raises
    ------
    ValueError
        If any precondition is violated.
    """
    _validate_train_args(
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        val_fraction=val_fraction,
        patience=patience,
        speed_lo_mph=speed_lo_mph,
        speed_hi_mph=speed_hi_mph,
    )
    loader_fn = dataset_loader or _default_compact_loader
    dataset = loader_fn(Path(dataset_path))

    rng = np.random.default_rng(seed)
    train_ids, val_ids = _split_trial_ids(dataset, val_fraction, rng)

    train_ts = _select_trials(dataset, train_ids)
    val_ts = _select_trials(dataset, val_ids)
    train_ts = _filter_speed(train_ts, speed_lo_mph, speed_hi_mph)
    val_ts = _filter_speed(val_ts, speed_lo_mph, speed_hi_mph)
    if len(train_ts) == 0:
        raise ValueError("train split has 0 timesteps after speed filter")
    if len(val_ts) == 0:
        raise ValueError("val split has 0 timesteps after speed filter")

    cfg = config or TimestepInverseConfig()
    state_train, tau_train, mask_train = _materialise_arrays(train_ts, cfg)
    state_val, tau_val, mask_val = _materialise_arrays(val_ts, cfg)
    state_stats = _compute_stats(state_train)
    tau_stats = _compute_stats_with_mask(tau_train, mask_train)

    state_train_std = _standardise(state_train, state_stats)
    state_val_std = _standardise(state_val, state_stats)
    tau_train_std = _standardise(tau_train, tau_stats)
    tau_val_std = _standardise(tau_val, tau_stats)

    train_ds = _TimestepTensorDataset(
        torch.from_numpy(state_train_std),
        torch.from_numpy(tau_train_std),
        torch.from_numpy(mask_train),
    )
    val_ds = _TimestepTensorDataset(
        torch.from_numpy(state_val_std),
        torch.from_numpy(tau_val_std),
        torch.from_numpy(mask_val),
    )

    torch.manual_seed(seed)
    selected_device = _resolve_device(device)
    model = TimestepInverseDynamics(cfg).to(selected_device)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate,
    )

    output_dir = _make_output_dir(Path(output_root))
    history: list[TimestepEpochMetrics] = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_path = output_dir / "checkpoint_best.pt"
    plateau = 0
    tau_std_t = torch.from_numpy(np.asarray(tau_stats["std"], dtype=np.float32)).to(
        selected_device
    )

    for epoch in range(epochs):
        t0 = time.time()
        train_loss = _run_epoch(
            model,
            train_loader,
            selected_device,
            opt,
            train=True,
            grad_clip=grad_clip,
        )
        val_loss, val_tau_mae_nm = _eval_epoch(
            model,
            val_loader,
            selected_device,
            tau_std_t,
        )
        duration = time.time() - t0

        metrics = TimestepEpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            val_tau_mae_nm=val_tau_mae_nm,
            duration_s=duration,
        )
        history.append(metrics)
        logger.info(
            "epoch=%d train_loss=%.4g val_loss=%.4g val_tau_mae_nm=%.4g dt=%.2fs",
            epoch,
            train_loss,
            val_loss,
            val_tau_mae_nm,
            duration,
        )

        payload = _build_payload(
            model, state_stats, tau_stats, speed_lo_mph, speed_hi_mph
        )
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_epoch = epoch
            plateau = 0
            torch.save(payload, best_path)
        else:
            plateau += 1
            if plateau >= patience:
                logger.info("early stop: val_loss plateau at epoch %d", epoch)
                break

    final_epoch = history[-1].epoch
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    result = TimestepTrainingResult(
        history=tuple(history),
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        final_epoch=final_epoch,
        checkpoint_path=best_path,
        output_dir=output_dir,
        n_train_trials=len(train_ids),
        n_val_trials=len(val_ids),
        n_train_timesteps=int(state_train.shape[0]),
        n_val_timesteps=int(state_val.shape[0]),
        parameter_count=n_params,
        config=cfg,
        speed_lo_mph=float(speed_lo_mph),
        speed_hi_mph=float(speed_hi_mph),
    )
    summary_path = output_dir / "metrics.json"
    summary_path.write_text(json.dumps(result.to_summary(), indent=2))
    return result


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _validate_train_args(
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    val_fraction: float,
    patience: int,
    speed_lo_mph: float,
    speed_hi_mph: float,
) -> None:
    if epochs <= 0:
        raise ValueError(f"epochs must be positive, got {epochs}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if lr <= 0:
        raise ValueError(f"lr must be positive, got {lr}")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    if patience < 1:
        raise ValueError(f"patience must be >= 1, got {patience}")
    if speed_lo_mph < 0:
        raise ValueError(f"speed_lo_mph must be >= 0, got {speed_lo_mph}")
    if not speed_lo_mph < speed_hi_mph:
        raise ValueError(
            "must have speed_lo_mph < speed_hi_mph; got "
            f"lo={speed_lo_mph}, hi={speed_hi_mph}"
        )


def _resolve_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _make_output_dir(root: Path) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = root / stamp
    out.mkdir(parents=True, exist_ok=True)
    return out


def _split_trial_ids(
    dataset: CompactSwingDataset,
    val_fraction: float,
    rng: np.random.Generator,
) -> tuple[set[int], set[int]]:
    trials_df = _require_pandas(dataset.trials, "trials")
    trial_ids = np.asarray(
        trials_df["trial_id"].to_numpy(dtype=np.int64), dtype=np.int64
    )
    if trial_ids.size < 2:
        raise ValueError(f"need at least 2 trials to split; got {int(trial_ids.size)}")
    perm = rng.permutation(trial_ids)
    n_val = max(1, int(round(val_fraction * len(perm))))
    val = {int(t) for t in perm[:n_val]}
    train = {int(t) for t in perm[n_val:]}
    if not train:
        raise ValueError("split produced empty train set; reduce val_fraction")
    return train, val


def _select_trials(
    dataset: CompactSwingDataset,
    trial_ids: set[int],
) -> pd.DataFrame:
    timesteps_df = _require_pandas(dataset.timesteps, "timesteps")
    if not trial_ids:
        return timesteps_df.iloc[0:0].copy()
    mask = timesteps_df["trial_id"].isin(trial_ids)
    return timesteps_df.loc[mask].reset_index(drop=True)


def _filter_speed(timesteps_df: pd.DataFrame, lo: float, hi: float) -> pd.DataFrame:
    speeds = timesteps_df["clubhead_speed_mph"].to_numpy(dtype=np.float64)
    keep = (speeds >= lo) & (speeds <= hi)
    return timesteps_df.loc[keep].reset_index(drop=True)


def _materialise_arrays(
    timesteps_df: pd.DataFrame, cfg: TimestepInverseConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack list-columns into ``(N, 27)`` arrays for each of ``q, qd, qdd, tau``.

    Returns ``(state[N, 81], tau[N, 27], mask[N, 27])`` where ``state`` is
    ``concat(q, qd, qdd)``, ``tau`` has NaN replaced with 0.0, and
    ``mask`` is 1.0 where the original ``tau`` was finite, else 0.0.
    """
    n = len(timesteps_df)
    if n == 0:
        return (
            np.zeros((0, cfg.input_dim), dtype=np.float32),
            np.zeros((0, cfg.output_dim), dtype=np.float32),
            np.zeros((0, cfg.output_dim), dtype=np.float32),
        )
    q = _stack_list_col(timesteps_df["q"])
    qd = _stack_list_col(timesteps_df["qd"])
    qdd = _stack_list_col(timesteps_df["qdd"])
    tau = _stack_list_col(timesteps_df["tau"])
    state = np.concatenate([q, qd, qdd], axis=-1).astype(np.float32, copy=False)
    if state.shape[-1] != cfg.input_dim:
        raise ValueError(
            f"state input_dim mismatch: got {state.shape[-1]}, expected {cfg.input_dim}"
        )
    if tau.shape[-1] != cfg.output_dim:
        raise ValueError(
            f"tau output_dim mismatch: got {tau.shape[-1]}, expected {cfg.output_dim}"
        )
    mask = np.isfinite(tau).astype(np.float32)
    tau_clean = np.where(mask > 0, tau, np.float32(0.0)).astype(np.float32)
    return state.astype(np.float32, copy=False), tau_clean, mask


def _stack_list_col(series: pd.Series) -> np.ndarray:
    return np.stack([np.asarray(v, dtype=np.float32) for v in series], axis=0)


def _compute_stats(arr: np.ndarray) -> dict[str, np.ndarray]:
    """Compute per-column NaN-aware mean/std (NaN propagated via nanmean/nanstd).

    Std is floored at 1e-8 then any near-zero entries are replaced by 1.0
    so divide is safe and constant columns are not amplified.
    """
    mean = np.nanmean(arr, axis=0).astype(np.float32)
    std = np.nanstd(arr, axis=0).astype(np.float32)
    mean = np.nan_to_num(mean, nan=0.0).astype(np.float32)
    std = np.nan_to_num(std, nan=1.0).astype(np.float32)
    safe = np.where(std > 1e-8, std, np.float32(1.0)).astype(np.float32)
    return {"mean": mean, "std": safe}


def _compute_stats_with_mask(
    arr: np.ndarray, mask: np.ndarray
) -> dict[str, np.ndarray]:
    """Mean/std over only the masked-finite entries (per column).

    Columns with zero finite entries (e.g. an always-NaN tau index in this
    split) get mean=0, std=1.
    """
    masked = np.where(mask > 0, arr, np.nan)
    n_finite = mask.sum(axis=0)
    # nanmean/nanstd warn for all-NaN columns; we explicitly substitute
    # safe defaults via ``np.where`` after the call, so the warning is
    # uninteresting noise.
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        raw_mean = np.nanmean(masked, axis=0)
        raw_std = np.nanstd(masked, axis=0)
    mean = np.where(n_finite > 0, raw_mean, 0.0).astype(np.float32)
    std = np.where(n_finite > 1, raw_std, 1.0).astype(np.float32)
    safe = np.where(std > 1e-8, std, np.float32(1.0)).astype(np.float32)
    return {"mean": mean, "std": safe}


def _standardise(arr: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    return ((arr - stats["mean"]) / stats["std"]).astype(np.float32)


def _build_payload(
    model: TimestepInverseDynamics,
    state_stats: dict[str, np.ndarray],
    tau_stats: dict[str, np.ndarray],
    speed_lo_mph: float,
    speed_hi_mph: float,
) -> dict:
    base = model.state_payload()
    base["state_stats"] = {
        "mean": state_stats["mean"].tolist(),
        "std": state_stats["std"].tolist(),
    }
    base["tau_stats"] = {
        "mean": tau_stats["mean"].tolist(),
        "std": tau_stats["std"].tolist(),
    }
    base["speed_window_mph"] = [float(speed_lo_mph), float(speed_hi_mph)]
    return base


def _run_epoch(
    model: TimestepInverseDynamics,
    loader: DataLoader,
    device: torch.device,
    opt: AdamW,
    *,
    train: bool,
    grad_clip: float,
) -> float:
    """One pass over ``loader``. Returns the masked-mean standardised MSE."""
    if train:
        model.train()
    else:
        model.eval()
    total_loss = 0.0
    total_count = 0.0
    with torch.set_grad_enabled(train):
        for batch in loader:
            state = batch["state"].to(device, non_blocking=True)
            tau = batch["tau"].to(device, non_blocking=True)
            mask = batch["tau_mask"].to(device, non_blocking=True)
            pred = model(state)
            sq_err = (pred - tau) ** 2
            masked = sq_err * mask
            count = mask.sum()
            if float(count.detach().cpu()) <= 0:
                continue
            loss = masked.sum() / count.clamp_min(1.0)
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
            total_loss += float(masked.sum().detach().cpu())
            total_count += float(count.detach().cpu())
    if total_count <= 0:
        return 0.0
    return total_loss / total_count


def _eval_epoch(
    model: TimestepInverseDynamics,
    loader: DataLoader,
    device: torch.device,
    tau_std_t: torch.Tensor,
) -> tuple[float, float]:
    """Eval pass; returns ``(masked_mse_std, mae_physical_nm)``."""
    model.eval()
    total_sq = 0.0
    total_abs = 0.0
    total_count = 0.0
    with torch.no_grad():
        for batch in loader:
            state = batch["state"].to(device, non_blocking=True)
            tau = batch["tau"].to(device, non_blocking=True)
            mask = batch["tau_mask"].to(device, non_blocking=True)
            pred = model(state)
            sq_err = (pred - tau) ** 2
            masked_sq = sq_err * mask
            # MAE in physical units: undo standardisation by multiplying by
            # tau_std (broadcast over batch). Since pred and tau live in
            # standardised space, |pred - tau| * std == |pred_phys - tau_phys|.
            abs_err = (pred - tau).abs() * tau_std_t
            masked_abs = abs_err * mask
            total_sq += float(masked_sq.sum().detach().cpu())
            total_abs += float(masked_abs.sum().detach().cpu())
            total_count += float(mask.sum().detach().cpu())
    if total_count <= 0:
        return 0.0, 0.0
    return total_sq / total_count, total_abs / total_count


def _require_pandas(obj: Any, label: str) -> pd.DataFrame:
    if not isinstance(obj, pd.DataFrame):
        raise TypeError(
            f"timestep training requires eager pandas DataFrame for `{label}`; "
            f"got {type(obj).__name__} (load with lazy=False)"
        )
    return obj


def _default_compact_loader(path: Path) -> CompactSwingDataset:
    return load_compact_swing_dataset(path, lazy=False)


# Re-export for convenience.
__all__ = [
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_PATIENCE",
    "DEFAULT_SEED",
    "TimestepEpochMetrics",
    "TimestepTrainingResult",
    "filter_timesteps_by_speed",
    "train_timestep_inverse",
]
