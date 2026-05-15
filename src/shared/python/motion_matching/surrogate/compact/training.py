"""Training loop for the compact-schema Option-2 NN swing surrogate.

This module trains :class:`SwingSurrogate` against the compact swing
parquet dataset documented in
``src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/motion_matching/shared/COMPACT_DATASET_SCHEMA.md``.

The expected on-disk layout is::

    <dataset_path>/
        trials.parquet     # one row per simulation, includes 189-vec coefficients
        timesteps.parquet  # one row per (trial_id, t) sample

Public surface:
    train_surrogate -- run the training loop and return a TrainingResult.
    TrainingResult  -- frozen dataclass of artefacts produced by training.

Loss design (as required by #4075):
    * MSE per output channel.
    * An additional weighted term on clubhead-speed-at-impact (the channel
      everyone cares about most).

Checkpointing:
    Each epoch writes ``checkpoint_epoch_<N>.pt`` and an updated
    ``metrics.json`` to ``output/surrogate/<timestamp>/`` (or to a
    user-supplied directory).

Early stopping:
    Patience of 10 epochs on the standardised total ``val_loss`` (the same
    quantity the optimiser minimises). Earlier revisions stopped on
    ``val_grip_rmse_mm`` alone, which let "best" land on a checkpoint that
    had only learned the position channels — the clubhead-speed channel
    was still mid-convergence. Tracking the multi-channel objective avoids
    that asymmetry.
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .model import (
    CHANNEL_SLICES,
    CoeffNormalizer,
    SurrogateConfig,
    SwingSurrogate,
    TargetNormalizer,
)

_LOGGER = logging.getLogger(__name__)

# Conversion factor: name reads as "<numerator> per <denominator>", so
# ``metres * _MM_PER_M = millimetres``. (Inversion-free naming — the
# previous ``_M_PER_MM = 1e-3`` was easy to misread as the inverse.)
_MM_PER_M: float = 1.0e3


# --------------------------------------------------------------------------- #
# Result dataclass                                                            #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TrainingResult:
    """Frozen snapshot of a training run.

    Attributes:
        output_dir: Directory containing checkpoints + ``metrics.json``.
        best_checkpoint: Path to the best-``val_loss`` checkpoint (the
            standardised multi-channel objective; see
            :func:`_channel_standardised_mse`).
        last_checkpoint: Path to the final-epoch checkpoint.
        best_epoch: 1-indexed epoch at which best ``val_loss`` was hit.
        best_val_loss: Lowest val loss observed.
        best_val_grip_rmse_mm: Val grip-RMSE recorded at ``best_epoch`` (mm).
            (Bookkeeping only — not the early-stop criterion.)
        history: Per-epoch metrics dict (lists of floats, plus epoch index).
        config: Surrogate config that was trained.
        total_seconds: Wall-clock duration of the run.
        param_count: Trainable parameter count of the model.
    """

    output_dir: Path
    best_checkpoint: Path
    last_checkpoint: Path
    best_epoch: int
    best_val_loss: float
    best_val_grip_rmse_mm: float
    history: dict[str, list[float]] = field(default_factory=dict)
    config: SurrogateConfig | None = None
    total_seconds: float = 0.0
    param_count: int = 0


# --------------------------------------------------------------------------- #
# Dataset                                                                     #
# --------------------------------------------------------------------------- #


class _CompactSwingTorchDataset(Dataset):
    """In-memory PyTorch dataset built from a compact-schema parquet folder.

    Each sample is ``(coeffs_norm, traj_target)`` where ``traj_target``
    is the ``(seq_len, 12)`` ground-truth hand-path tensor assembled from
    the compact-schema columns.
    """

    def __init__(
        self,
        coeffs: np.ndarray,
        targets: np.ndarray,
        trial_ids: np.ndarray,
    ) -> None:
        if coeffs.shape[0] != targets.shape[0] or coeffs.shape[0] != trial_ids.shape[0]:
            raise ValueError(
                "coeffs / targets / trial_ids must have matching first dim; "
                f"got {coeffs.shape[0]}, {targets.shape[0]}, {trial_ids.shape[0]}"
            )
        if coeffs.ndim != 2 or targets.ndim != 3:
            raise ValueError(
                "coeffs must be 2-D, targets must be 3-D (N, T, 12); got "
                f"{coeffs.shape}, {targets.shape}"
            )
        if targets.shape[-1] != 12:
            raise ValueError(
                f"targets trailing dim must be 12; got {targets.shape[-1]}"
            )
        self._coeffs = torch.as_tensor(coeffs, dtype=torch.float32)
        self._targets = torch.as_tensor(targets, dtype=torch.float32)
        self._trial_ids = np.asarray(trial_ids)

    def __len__(self) -> int:
        return self._coeffs.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._coeffs[idx], self._targets[idx]

    @property
    def trial_ids(self) -> np.ndarray:
        """Trial ids parallel to dataset rows (for split bookkeeping)."""
        return self._trial_ids


# --------------------------------------------------------------------------- #
# Dataset loading                                                             #
# --------------------------------------------------------------------------- #


def _import_compact_loader() -> Callable[[Path], Any] | None:
    """Try to import the canonical loader (#4074) — return ``None`` if absent."""
    try:
        # Documented import path per the task spec.
        from src.shared.python.dataset_tools.load_compact import (  # type: ignore[import-not-found]
            load_compact_swing_dataset,
        )
    except (ImportError, ModuleNotFoundError):
        return None
    return load_compact_swing_dataset


def _build_targets_from_compact(
    trials_df: Any,
    timesteps_df: Any,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assemble ``(coeffs, targets, trial_ids)`` arrays from compact frames.

    Targets follow the 12-channel layout documented in :mod:`.model`.

    Args:
        trials_df: DataFrame indexed/keyed by ``trial_id`` with a
            ``coefficients`` ``list<float64>[189]`` column.
        timesteps_df: DataFrame with ``trial_id`` + per-timestep arrays
            (``r_clubhead``, ``v_clubhead``, ``r_grip``, ``clubhead_speed_mph``).
        seq_len: Expected timesteps per trial.

    Returns:
        ``(coeffs[N, 189], targets[N, T, 12], trial_ids[N])`` numpy arrays.
    """
    import pandas as pd  # local import — pandas is heavy

    if not isinstance(trials_df, pd.DataFrame):
        trials_df = trials_df.collect().to_pandas()  # polars LazyFrame fallback
    if not isinstance(timesteps_df, pd.DataFrame):
        timesteps_df = timesteps_df.collect().to_pandas()

    trial_ids = np.asarray(trials_df["trial_id"].to_numpy(), dtype=np.int64)
    coeff_rows = [np.asarray(c, dtype=np.float32) for c in trials_df["coefficients"]]
    coeffs = np.stack(coeff_rows, axis=0)
    if coeffs.ndim != 2 or coeffs.shape[1] == 0:
        raise ValueError(
            f"unexpected coefficients shape {coeffs.shape}; "
            "trials.coefficients must be list<float64>[D]"
        )

    targets = np.zeros((trial_ids.shape[0], seq_len, 12), dtype=np.float32)
    grouped = timesteps_df.groupby("trial_id", sort=False)
    for row_idx, tid in enumerate(trial_ids):
        try:
            block = grouped.get_group(tid)
        except KeyError as exc:
            raise ValueError(f"timesteps.parquet missing trial_id={tid}") from exc
        if len(block) != seq_len:
            raise ValueError(
                f"trial_id={tid} has {len(block)} timesteps; expected {seq_len}"
            )
        block = block.sort_values("t")
        r_ch = np.stack([np.asarray(v, dtype=np.float32) for v in block["r_clubhead"]])
        v_ch = np.stack([np.asarray(v, dtype=np.float32) for v in block["v_clubhead"]])
        r_gr = np.stack([np.asarray(v, dtype=np.float32) for v in block["r_grip"]])
        chs = np.asarray(block["clubhead_speed_mph"], dtype=np.float32).reshape(-1, 1)
        # shaft_axis -> az/polar
        shaft = r_ch - r_gr
        norm = np.sqrt(np.einsum("...i,...i->...", shaft, shaft))[..., np.newaxis]
        norm = np.maximum(norm, 1e-12)
        unit = shaft / norm
        azimuth = np.arctan2(unit[:, 1], unit[:, 0])
        polar = np.arccos(np.clip(unit[:, 2], -1.0, 1.0))
        az_pol = np.stack([azimuth, polar], axis=-1).astype(np.float32)
        targets[row_idx] = np.concatenate([r_ch, v_ch, r_gr, chs, az_pol], axis=-1)
    return coeffs, targets, trial_ids


def _load_compact_dataset(
    dataset_path: Path,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load coeffs/targets/trial-ids from the compact parquet folder.

    Falls back to a direct ``pyarrow`` read when the canonical loader
    (``load_compact_swing_dataset``, owned by PR #4074) is unavailable —
    so this trainer can still be exercised on a hand-built fixture.
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset_path does not exist: {dataset_path}")
    loader = _import_compact_loader()
    if loader is not None:
        ds = loader(dataset_path)
        return _build_targets_from_compact(ds.trials, ds.timesteps, seq_len=seq_len)

    # Fallback: direct parquet read.
    import pandas as pd

    trials_path = dataset_path / "trials.parquet"
    timesteps_path = dataset_path / "timesteps.parquet"
    if not (trials_path.exists() and timesteps_path.exists()):
        raise FileNotFoundError(
            "compact dataset must contain trials.parquet and timesteps.parquet"
        )
    trials_df = pd.read_parquet(trials_path)
    timesteps_df = pd.read_parquet(timesteps_path)
    return _build_targets_from_compact(trials_df, timesteps_df, seq_len=seq_len)


# --------------------------------------------------------------------------- #
# Loss + metrics                                                              #
# --------------------------------------------------------------------------- #


def _channel_standardised_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    target_normalizer: TargetNormalizer,
) -> torch.Tensor:
    """MSE computed on the per-channel-standardised ``(B, T, 12)`` tensor.

    Standardising each output channel to mean 0 / std 1 before the MSE is
    what stops the mph-scale clubhead-speed channel from dominating the
    metre-scale position channels (root-cause of the 17 mm grip-RMSE
    floor seen in the smoke run).
    """
    pred_n = target_normalizer.standardize(pred)
    target_n = target_normalizer.standardize(target)
    return torch.mean((pred_n - target_n) ** 2)


def _impact_speed_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    target_normalizer: TargetNormalizer,
) -> torch.Tensor:
    """Standardised MSE on the impact-time clubhead-speed channel.

    Impact is approximated as the timestep where ``target`` clubhead speed
    is maximal. The per-channel standardisation keeps the impact term on
    the same numerical scale as the trajectory MSE so the
    ``impact_weight`` hyper-parameter retains its semantics after the
    bug-1 fix.
    """
    chs_lo, chs_hi = CHANNEL_SLICES["clubhead_speed"]
    chs_mean = target_normalizer.mean[chs_lo].to(device=pred.device, dtype=pred.dtype)
    chs_std = target_normalizer.std[chs_lo].to(device=pred.device, dtype=pred.dtype)
    target_chs = target[..., chs_lo:chs_hi].squeeze(-1)
    impact_idx = torch.argmax(target_chs, dim=-1)
    batch_idx = torch.arange(target.shape[0], device=target.device)
    pred_at_impact = pred[batch_idx, impact_idx, chs_lo:chs_hi].squeeze(-1)
    target_at_impact = target_chs[batch_idx, impact_idx]
    pred_n = (pred_at_impact - chs_mean) / chs_std
    target_n = (target_at_impact - chs_mean) / chs_std
    return torch.mean((pred_n - target_n) ** 2)


def _grip_rmse_mm(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Validation metric: grip-position RMSE in millimetres."""
    lo, hi = CHANNEL_SLICES["r_grip"]
    diff = pred[..., lo:hi] - target[..., lo:hi]
    rmse_m = torch.sqrt(torch.mean(diff**2)).item()
    return rmse_m * _MM_PER_M  # m -> mm


def _clubhead_speed_mae_mph(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Validation metric: clubhead-speed MAE in mph (channel already in mph)."""
    lo, hi = CHANNEL_SLICES["clubhead_speed"]
    diff = (pred[..., lo:hi] - target[..., lo:hi]).abs()
    return float(diff.mean().item())


# --------------------------------------------------------------------------- #
# Training loop                                                               #
# --------------------------------------------------------------------------- #


def _split_indices_by_trial(
    trial_ids: np.ndarray,
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """90/10 split by ``trial_id`` so no trial appears in both subsets."""
    rng = np.random.default_rng(seed)
    unique_ids = np.unique(trial_ids)
    rng.shuffle(unique_ids)
    n_val = max(1, int(round(len(unique_ids) * val_fraction)))
    val_ids = set(unique_ids[:n_val].tolist())
    val_mask = np.array([t in val_ids for t in trial_ids], dtype=bool)
    val_idx = np.flatnonzero(val_mask)
    train_idx = np.flatnonzero(~val_mask)
    return train_idx, val_idx


def _run_epoch(
    model: SwingSurrogate,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    *,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    impact_weight: float,
    normalizer: CoeffNormalizer,
    target_normalizer: TargetNormalizer,
) -> dict[str, float]:
    """Run one pass over ``loader``. ``optimizer is None`` means eval mode."""
    is_train = optimizer is not None
    model.train(is_train)
    n = 0
    total_loss = 0.0
    total_mse = 0.0
    total_impact = 0.0
    grip_sq_sum = 0.0
    chs_abs_sum = 0.0
    chs_count = 0
    for coeffs_raw, target in loader:
        coeffs_raw = coeffs_raw.to(device, dtype=torch.float32)
        target = target.to(device, dtype=torch.float32)
        coeffs = normalizer.normalize(coeffs_raw)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            pred = model(coeffs)
            mse = _channel_standardised_mse(pred, target, target_normalizer)
            impact = _impact_speed_loss(pred, target, target_normalizer)
            loss = mse + impact_weight * impact
        if is_train:
            assert optimizer is not None  # narrow type
            loss.backward()
            optimizer.step()
        bs = coeffs.shape[0]
        n += bs
        total_loss += float(loss.item()) * bs
        total_mse += float(mse.item()) * bs
        total_impact += float(impact.item()) * bs
        # Per-batch validation metrics so we can report per-epoch averages.
        with torch.no_grad():
            lo, hi = CHANNEL_SLICES["r_grip"]
            diff = pred[..., lo:hi] - target[..., lo:hi]
            grip_sq_sum += float(torch.sum(diff**2).item())
            chs_lo, chs_hi = CHANNEL_SLICES["clubhead_speed"]
            chs_diff = (pred[..., chs_lo:chs_hi] - target[..., chs_lo:chs_hi]).abs()
            chs_abs_sum += float(torch.sum(chs_diff).item())
            chs_count += int(chs_diff.numel())
    if n == 0:
        return {
            "loss": float("nan"),
            "mse": float("nan"),
            "impact_mse": float("nan"),
            "grip_rmse_mm": float("nan"),
            "clubhead_speed_mae_mph": float("nan"),
        }
    grip_rmse_m = math.sqrt(grip_sq_sum / max(n * 3 * model.cfg.seq_len, 1))
    return {
        "loss": total_loss / n,
        "mse": total_mse / n,
        "impact_mse": total_impact / n,
        "grip_rmse_mm": grip_rmse_m * _MM_PER_M,
        "clubhead_speed_mae_mph": chs_abs_sum / max(chs_count, 1),
    }


def _save_checkpoint(
    path: Path,
    *,
    model: SwingSurrogate,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, Any],
    config: SurrogateConfig,
    normalizer: CoeffNormalizer,
    target_normalizer: TargetNormalizer,
) -> None:
    """Persist a resumable checkpoint to ``path``."""
    payload = {
        "schema_version": "swing-surrogate-1.1",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": int(epoch),
        "metrics": metrics,
        "config": asdict(config),
        "normalizer": {
            "n_joints": config.n_joints,
            "coeff_bounds": list(config.coeff_bounds),
        },
        "target_normalizer": target_normalizer.to_state_dict(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def train_surrogate(
    dataset_path: str | Path,
    *,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str | torch.device = "cpu",
    seed: int = 0,
    config: SurrogateConfig | None = None,
    impact_weight: float = 1.0,
    val_fraction: float = 0.1,
    early_stopping_patience: int = 10,
    output_dir: str | Path | None = None,
    resume_from: str | Path | None = None,
    progress_cb: Callable[[int, dict[str, float]], None] | None = None,
) -> TrainingResult:
    """Train :class:`SwingSurrogate` on a compact-schema parquet dataset.

    Args:
        dataset_path: Folder containing ``trials.parquet`` + ``timesteps.parquet``.
        epochs: Maximum number of training epochs.
        batch_size: Mini-batch size.
        lr: Adam learning rate.
        device: Torch device (string or ``torch.device``).
        seed: RNG seed for the train/val split and weight init.
        config: Surrogate config; defaults to :class:`SurrogateConfig()`.
        impact_weight: Multiplier on the impact-speed loss term.
        val_fraction: Fraction of trial_ids reserved for validation.
        early_stopping_patience: Epochs without ``val_loss`` improvement
            before training stops early. ``val_loss`` is the same
            standardised multi-channel objective that the optimiser
            minimises, so the saved "best" checkpoint reflects progress
            on every output channel — not just grip position. (Earlier
            revisions tracked ``val_grip_rmse_mm`` only and could pick a
            "best" checkpoint where the speed channel was still mid-
            convergence.)
        output_dir: Where to write checkpoints. Defaults to
            ``output/surrogate/<timestamp>``.
        resume_from: Optional path to a checkpoint produced by a previous
            run; loads model + optimizer state.
        progress_cb: Optional callback ``(epoch, metrics_dict) -> None``;
            called once per epoch for streaming UIs.

    Returns:
        A frozen :class:`TrainingResult`.

    Raises:
        ValueError: For invalid hyper-parameters or dataset shape.
        FileNotFoundError: If ``dataset_path`` does not exist.
    """
    _check_train_args(epochs, batch_size, lr, val_fraction, early_stopping_patience)
    cfg = config if config is not None else SurrogateConfig()
    cfg.validate()

    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)
    out_path = _resolve_output_dir(output_dir)
    _LOGGER.info("training surrogate -> %s", out_path)

    coeffs_np, targets_np, trial_ids = _load_compact_dataset(
        Path(dataset_path), seq_len=cfg.seq_len
    )
    if coeffs_np.shape[1] != cfg.coeff_dim:
        raise ValueError(
            f"dataset coeff dim {coeffs_np.shape[1]} != cfg.coeff_dim "
            f"({cfg.coeff_dim}); check schema vs SurrogateConfig"
        )
    train_idx, val_idx = _split_indices_by_trial(trial_ids, val_fraction, seed=seed)
    train_ds = _CompactSwingTorchDataset(
        coeffs_np[train_idx], targets_np[train_idx], trial_ids[train_idx]
    )
    val_ds = _CompactSwingTorchDataset(
        coeffs_np[val_idx], targets_np[val_idx], trial_ids[val_idx]
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=False
    )

    model = SwingSurrogate(cfg).to(dev)
    optimizer: torch.optim.Optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    normalizer = CoeffNormalizer(n_joints=cfg.n_joints, coeff_bounds=cfg.coeff_bounds)
    # Per-channel target stats (mean / std over the train split). This is the
    # core of the bug-1 fix: the loss is computed in standardised channel
    # space so every output channel contributes comparable gradient signal.
    train_targets_t = torch.as_tensor(targets_np[train_idx], dtype=torch.float32)
    target_normalizer = TargetNormalizer.from_targets(train_targets_t)
    _LOGGER.info(
        "target normalizer (train split): mean=%s std=%s",
        target_normalizer.mean.tolist(),
        target_normalizer.std.tolist(),
    )
    start_epoch = 0
    if resume_from is not None:
        start_epoch, restored_target_normalizer = _load_resume_state(
            Path(resume_from), model=model, optimizer=optimizer
        )
        if restored_target_normalizer is not None:
            # Reuse the original training run's target stats so the loss
            # definition is preserved across the resume boundary. The
            # optimizer moments carried in the checkpoint were computed
            # against this normaliser; recomputing fresh stats from a
            # different seed/split would silently change the loss.
            target_normalizer = restored_target_normalizer
            _LOGGER.info(
                "restored target_normalizer from checkpoint: mean=%s std=%s",
                target_normalizer.mean.tolist(),
                target_normalizer.std.tolist(),
            )
        else:
            _LOGGER.warning(
                "checkpoint %s does not embed target_normalizer; "
                "falling back to fresh stats from the current train split. "
                "Resume metrics may be non-comparable to the original run.",
                resume_from,
            )
        _LOGGER.info("resumed from %s at epoch %d", resume_from, start_epoch)

    return _train_loop(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        normalizer=normalizer,
        target_normalizer=target_normalizer,
        device=dev,
        epochs=epochs,
        start_epoch=start_epoch,
        impact_weight=impact_weight,
        early_stopping_patience=early_stopping_patience,
        out_path=out_path,
        progress_cb=progress_cb,
    )


def _train_loop(
    *,
    model: SwingSurrogate,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: SurrogateConfig,
    normalizer: CoeffNormalizer,
    target_normalizer: TargetNormalizer,
    device: torch.device,
    epochs: int,
    start_epoch: int,
    impact_weight: float,
    early_stopping_patience: int,
    out_path: Path,
    progress_cb: Callable[[int, dict[str, float]], None] | None,
) -> TrainingResult:
    """The actual epoch loop, factored out so it can be unit-tested narrowly."""
    history: dict[str, list[float]] = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_grip_rmse_mm": [],
        "val_clubhead_speed_mae_mph": [],
    }
    # Early stopping watches the standardised multi-channel ``val_loss`` —
    # the same quantity the optimiser minimises — so "best" reflects
    # progress on every output channel rather than only the grip-position
    # channel. ``best_val_grip_rmse`` is bookkept alongside for reporting.
    best_val_loss = float("inf")
    best_val_grip_rmse = float("inf")
    best_epoch = 0
    last_ckpt: Path | None = None
    best_ckpt: Path | None = None
    no_improve = 0
    t0 = time.perf_counter()
    for epoch_idx in range(start_epoch, epochs):
        epoch_human = epoch_idx + 1
        train_metrics = _run_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            device=device,
            impact_weight=impact_weight,
            normalizer=normalizer,
            target_normalizer=target_normalizer,
        )
        val_metrics = _run_epoch(
            model,
            val_loader,
            optimizer=None,
            device=device,
            impact_weight=impact_weight,
            normalizer=normalizer,
            target_normalizer=target_normalizer,
        )
        history["epoch"].append(float(epoch_human))
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_grip_rmse_mm"].append(val_metrics["grip_rmse_mm"])
        history["val_clubhead_speed_mae_mph"].append(
            val_metrics["clubhead_speed_mae_mph"]
        )
        ckpt_path = out_path / f"checkpoint_epoch_{epoch_human:03d}.pt"
        _save_checkpoint(
            ckpt_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch_human,
            metrics={"train": train_metrics, "val": val_metrics},
            config=cfg,
            normalizer=normalizer,
            target_normalizer=target_normalizer,
        )
        last_ckpt = ckpt_path
        _write_metrics_json(out_path, history)
        _LOGGER.info(
            "epoch %d/%d: train_loss=%.4g val_loss=%.4g grip_rmse=%.2f mm chs_mae=%.2f mph",
            epoch_human,
            epochs,
            train_metrics["loss"],
            val_metrics["loss"],
            val_metrics["grip_rmse_mm"],
            val_metrics["clubhead_speed_mae_mph"],
        )
        if progress_cb is not None:
            progress_cb(epoch_human, val_metrics)
        if val_metrics["loss"] < best_val_loss - 1e-6:
            best_val_loss = val_metrics["loss"]
            best_val_grip_rmse = val_metrics["grip_rmse_mm"]
            best_epoch = epoch_human
            best_ckpt = out_path / "checkpoint_best.pt"
            _save_checkpoint(
                best_ckpt,
                model=model,
                optimizer=optimizer,
                epoch=epoch_human,
                metrics={"train": train_metrics, "val": val_metrics},
                config=cfg,
                normalizer=normalizer,
                target_normalizer=target_normalizer,
            )
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= early_stopping_patience:
                _LOGGER.info(
                    "early stopping after %d epochs without val_loss improvement",
                    early_stopping_patience,
                )
                break

    elapsed = time.perf_counter() - t0
    if last_ckpt is None:
        raise RuntimeError("training produced no checkpoints (epochs == 0?)")
    if best_ckpt is None:
        best_ckpt = last_ckpt
    return TrainingResult(
        output_dir=out_path,
        best_checkpoint=best_ckpt,
        last_checkpoint=last_ckpt,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        best_val_grip_rmse_mm=best_val_grip_rmse,
        history=history,
        config=cfg,
        total_seconds=elapsed,
        param_count=model.parameter_count(),
    )


def _check_train_args(
    epochs: int,
    batch_size: int,
    lr: float,
    val_fraction: float,
    early_stopping_patience: int,
) -> None:
    """Validate scalar hyper-parameters with descriptive error messages."""
    if epochs <= 0:
        raise ValueError(f"epochs must be positive, got {epochs}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if lr <= 0:
        raise ValueError(f"lr must be positive, got {lr}")
    if not (0.0 < val_fraction < 1.0):
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    if early_stopping_patience <= 0:
        raise ValueError(
            f"early_stopping_patience must be positive, got {early_stopping_patience}"
        )


def _resolve_output_dir(output_dir: str | Path | None) -> Path:
    """Pick / create the output directory. Defaults to ``output/surrogate/<ts>``."""
    if output_dir is not None:
        out_path = Path(output_dir)
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = Path("output") / "surrogate" / ts
    out_path.mkdir(parents=True, exist_ok=True)
    return out_path


def _load_resume_state(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[int, TargetNormalizer | None]:
    """Load model + optimizer state from a previous checkpoint.

    Returns:
        ``(epoch, target_normalizer)`` where ``target_normalizer`` is the
        per-channel target stats stored alongside the model weights, or
        ``None`` for legacy checkpoints (pre-1.1 schema) that do not
        embed it. Callers should fall back to recomputing stats from the
        current split when ``None`` is returned, but should warn the user
        because the loss definition may then drift across the resume
        boundary.
    """
    if not path.exists():
        raise FileNotFoundError(f"resume_from checkpoint missing: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    target_normalizer: TargetNormalizer | None = None
    target_state = payload.get("target_normalizer")
    if isinstance(target_state, dict):
        try:
            target_normalizer = TargetNormalizer.from_state_dict(target_state)
        except (ValueError, KeyError) as exc:
            _LOGGER.warning(
                "checkpoint target_normalizer payload is invalid (%s); "
                "ignoring and falling back to fresh stats",
                exc,
            )
            target_normalizer = None
    return int(payload.get("epoch", 0)), target_normalizer


def _write_metrics_json(out_path: Path, history: dict[str, Sequence[float]]) -> None:
    """Persist the running per-epoch metrics dict to ``metrics.json``."""
    metrics_path = out_path / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump({k: list(v) for k, v in history.items()}, fh, indent=2)
