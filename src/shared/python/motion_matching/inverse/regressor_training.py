"""Training loop for :class:`InverseRegressor` (production inverse model).

Parallel surface to :func:`.training.train_inverse_cvae` but for the
deterministic regressor: no KL term, no beta annealing, no free-bits ΓÇö
just MSE on standardised coefficients (target/coefficient_bounds in
[-1, 1]) with AdamW + early-stopping.

NM-06 (#10621) adds :func:`train_masked_control_proposals` (re-exported
from :mod:`.proposal_training`) which trains on observation residual after
differentiable surrogate rollout plus control regularization ΓÇö not
coefficient MSE alone as the selection criterion.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from .regressor import (
    InverseRegressor,
    RegressorConfig,
    parameter_count,
)

logger = logging.getLogger(__name__)


DEFAULT_OUTPUT_ROOT = Path("output/inverse_regressor")
DEFAULT_PATIENCE = 20


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EpochMetrics:
    """Per-epoch loss summary (means over batches).

    Attributes:
        epoch: 0-based epoch index.
        train_loss: Mean MSE on standardised (``[-1, 1]``) targets ΓÇö train.
        val_loss: Mean MSE on standardised targets ΓÇö val.
        val_mse_physical: Mean MSE in physical units (Newton-metres squared)
            ΓÇö for human-readable reporting alongside the standardised loss.
        duration_s: Wall-clock seconds for this epoch (train + val pass).
    """

    epoch: int
    train_loss: float
    val_loss: float
    val_mse_physical: float
    duration_s: float


@dataclass(frozen=True)
class RegressorTrainingResult:
    """Outcome of :func:`train_inverse_regressor`.

    The ``checkpoint_path`` points at the best-val-loss epoch.
    """

    history: tuple[EpochMetrics, ...]
    best_epoch: int
    best_val_loss: float
    final_epoch: int
    checkpoint_path: Path
    output_dir: Path
    n_train_trials: int
    n_val_trials: int
    parameter_count: int
    config: RegressorConfig

    def to_summary(self) -> dict:
        """JSON-serialisable summary for the metrics file + reports."""
        return {
            "history": [asdict(h) for h in self.history],
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "final_epoch": self.final_epoch,
            "checkpoint_path": str(self.checkpoint_path),
            "output_dir": str(self.output_dir),
            "n_train_trials": self.n_train_trials,
            "n_val_trials": self.n_val_trials,
            "parameter_count": self.parameter_count,
        }


# ---------------------------------------------------------------------------
# Dataset adapter (mirrors training.py to avoid coupling to its internals)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _PreparedSample:
    trial_id: int
    trajectory: torch.Tensor  # (T, 12) float32
    coeffs: torch.Tensor  # (189,) float32


class _TrialTensorDataset(Dataset[_PreparedSample]):
    def __init__(self, samples: list[_PreparedSample]) -> None:
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> _PreparedSample:
        return self._samples[idx]


def _collate(samples: Iterable[_PreparedSample]) -> dict[str, torch.Tensor]:
    samples_list = list(samples)
    traj = torch.stack([s.trajectory for s in samples_list], dim=0)
    coeffs = torch.stack([s.coeffs for s in samples_list], dim=0)
    return {"trajectory": traj, "coeffs": coeffs}


def _stack_trajectory_channels(
    *vectors: np.ndarray, expected_channels: int = 12
) -> np.ndarray:
    arr = np.concatenate(vectors, axis=-1).astype(np.float32, copy=False)
    if arr.shape[-1] != expected_channels:
        raise ValueError(
            f"trajectory channels = {arr.shape[-1]}, expected {expected_channels}"
        )
    return arr


# ---------------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparameters for :func:`train_inverse_regressor`.

    Bundles the former long keyword list so the public entry stays within
    architecture parameter budgets while preserving typed call sites.
    """

    epochs: int = 80
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = DEFAULT_PATIENCE
    val_fraction: float = 0.1
    seed: int = 0xC0FFEE
    grad_clip: float = 1.0
    device: str | torch.device = "auto"
    output_root: Path = field(default_factory=lambda: Path(DEFAULT_OUTPUT_ROOT))
    regressor: RegressorConfig = field(default_factory=RegressorConfig)

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_root", Path(self.output_root))
        _validate_training_config(self)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def train_inverse_regressor(
    dataset_path: str | Path,
    training: TrainingConfig | None = None,
    *,
    dataset_loader=None,
) -> RegressorTrainingResult:
    """Train an :class:`InverseRegressor` end-to-end.

    Parameters
    ----------
    dataset_path
        Folder containing ``trials.parquet`` and ``timesteps.parquet``
        per ``COMPACT_DATASET_SCHEMA.md``.
    training
        Optional :class:`TrainingConfig`. Defaults yield a standard AdamW
        schedule with early stopping.
    dataset_loader
        Optional callable ``(path) -> CompactSwingDataset`` for testing.

    Returns
    -------
    RegressorTrainingResult
        History, best-epoch index, output directory, parameter count.

    Raises
    ------
    ValueError
        If hyperparameters are invalid or the dataset has fewer than 2 trials.
    """
    cfg = training or TrainingConfig()
    prepared = _prepare_regressor_run(
        Path(dataset_path), cfg, dataset_loader=dataset_loader
    )
    history, best_epoch, best_val_loss = _fit_regressor_epochs(prepared, cfg)
    result = RegressorTrainingResult(
        history=tuple(history),
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        final_epoch=history[-1].epoch,
        checkpoint_path=prepared.best_path,
        output_dir=prepared.output_dir,
        n_train_trials=len(prepared.train_samples),
        n_val_trials=len(prepared.val_samples),
        parameter_count=parameter_count(prepared.model),
        config=prepared.regressor_cfg,
    )
    (prepared.output_dir / "metrics.json").write_text(
        json.dumps(result.to_summary(), indent=2)
    )
    return result


@dataclass
class _PreparedRegressorRun:
    """Mutable training handles assembled once before the epoch loop."""

    model: InverseRegressor
    opt: AdamW
    coeff_scale: torch.Tensor
    train_loader: DataLoader
    val_loader: DataLoader
    selected_device: torch.device
    output_dir: Path
    best_path: Path
    train_samples: list[_PreparedSample]
    val_samples: list[_PreparedSample]
    regressor_cfg: RegressorConfig


def _validate_training_config(cfg: TrainingConfig) -> None:
    """DbC: reject non-positive schedule fields and out-of-range fractions."""
    if cfg.epochs <= 0:
        raise ValueError(f"epochs must be positive, got {cfg.epochs}")
    if cfg.batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {cfg.batch_size}")
    if cfg.lr <= 0:
        raise ValueError(f"lr must be positive, got {cfg.lr}")
    if not 0.0 < cfg.val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {cfg.val_fraction}")
    if cfg.patience < 1:
        raise ValueError(f"patience must be >= 1, got {cfg.patience}")


def _prepare_regressor_run(
    dataset_path: Path,
    cfg: TrainingConfig,
    *,
    dataset_loader,
) -> _PreparedRegressorRun:
    """Load data, split trials, and construct model + loaders."""
    loader_fn = dataset_loader or _default_compact_loader
    samples = _materialise_samples(loader_fn(dataset_path))
    if len(samples) < 2:
        raise ValueError(f"need at least 2 trials to train+val, got {len(samples)}")

    rng = np.random.default_rng(cfg.seed)
    train_samples, val_samples = _split_by_trial(samples, cfg.val_fraction, rng)
    torch.manual_seed(cfg.seed)
    selected_device = _resolve_device(cfg.device)
    regressor_cfg = cfg.regressor
    model = InverseRegressor(regressor_cfg).to(selected_device)
    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    # Use the model's coefficient_scale (per-letter bound * scale factor) as
    # the loss-standardisation divisor so val_loss is unitless and comparable
    # across runs even if the empirical coefficient range exceeds the nominal
    # per-letter bounds.
    coeff_scale: torch.Tensor = model.coefficient_scale.to(selected_device)  # type: ignore[assignment, attr-defined]
    train_loader = DataLoader(
        _TrialTensorDataset(train_samples),
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=_collate,
        drop_last=False,
    )
    val_loader = DataLoader(
        _TrialTensorDataset(val_samples),
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=_collate,
    )
    output_dir = _make_output_dir(Path(cfg.output_root))
    return _PreparedRegressorRun(
        model=model,
        opt=opt,
        coeff_scale=coeff_scale,
        train_loader=train_loader,
        val_loader=val_loader,
        selected_device=selected_device,
        output_dir=output_dir,
        best_path=output_dir / "checkpoint_best.pt",
        train_samples=train_samples,
        val_samples=val_samples,
        regressor_cfg=regressor_cfg,
    )


def _fit_regressor_epochs(
    prepared: _PreparedRegressorRun,
    cfg: TrainingConfig,
) -> tuple[list[EpochMetrics], int, float]:
    """Run AdamW epochs with early stopping; return history and best stats."""
    history: list[EpochMetrics] = []
    best_val_loss = float("inf")
    best_epoch = 0
    plateau = 0
    for epoch in range(cfg.epochs):
        t0 = time.time()
        train_loss = _run_epoch(
            prepared.model,
            prepared.train_loader,
            prepared.selected_device,
            prepared.opt,
            prepared.coeff_scale,
            train=True,
            grad_clip=cfg.grad_clip,
        )
        val_loss, val_mse_physical = _eval_epoch(
            prepared.model,
            prepared.val_loader,
            prepared.selected_device,
            prepared.coeff_scale,
        )
        metrics = EpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            val_mse_physical=val_mse_physical,
            duration_s=time.time() - t0,
        )
        history.append(metrics)
        logger.info(
            "epoch=%d train_loss=%.4g val_loss=%.4g val_mse_physical=%.4g dt=%.2fs",
            epoch,
            train_loss,
            val_loss,
            val_mse_physical,
            metrics.duration_s,
        )
        torch.save(
            prepared.model.state_payload(),
            prepared.output_dir / f"checkpoint_epoch_{epoch}.pt",
        )
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_epoch = epoch
            plateau = 0
            torch.save(prepared.model.state_payload(), prepared.best_path)
            continue
        plateau += 1
        if plateau >= cfg.patience:
            logger.info("early stop: val_loss plateau at epoch %d", epoch)
            break
    return history, best_epoch, best_val_loss


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


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


def _split_by_trial(
    samples: list[_PreparedSample],
    val_fraction: float,
    rng: np.random.Generator,
) -> tuple[list[_PreparedSample], list[_PreparedSample]]:
    indices = np.arange(len(samples))
    rng.shuffle(indices)
    n_val = max(1, int(round(val_fraction * len(samples))))
    val_idx = {int(i) for i in indices[:n_val]}
    train: list[_PreparedSample] = []
    val: list[_PreparedSample] = []
    for i, s in enumerate(samples):
        (val if i in val_idx else train).append(s)
    if not train:
        raise ValueError("split produced empty train set; reduce val_fraction")
    return train, val


def _run_epoch(
    model: InverseRegressor,
    loader: DataLoader,
    device: torch.device,
    opt: AdamW,
    coeff_scale: torch.Tensor,
    *,
    train: bool,
    grad_clip: float,
) -> float:
    """One training pass over ``loader``. Returns the mean standardised MSE."""
    if train:
        model.train()
    else:
        model.eval()
    total = 0.0
    n_batches = 0
    with torch.set_grad_enabled(train):
        for batch in loader:
            traj = batch["trajectory"].to(device, non_blocking=True)
            coeffs = batch["coeffs"].to(device, non_blocking=True)
            pred = model(traj)
            pred_std = pred / coeff_scale
            target_std = coeffs / coeff_scale
            loss = torch.mean((pred_std - target_std) ** 2)
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
            total += float(loss.detach().cpu())
            n_batches += 1
    if n_batches == 0:
        return 0.0
    return total / n_batches


def _eval_epoch(
    model: InverseRegressor,
    loader: DataLoader,
    device: torch.device,
    coeff_scale: torch.Tensor,
) -> tuple[float, float]:
    """One validation pass; returns ``(val_loss_std, val_mse_physical)``."""
    model.eval()
    total_std = 0.0
    total_phys = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in loader:
            traj = batch["trajectory"].to(device, non_blocking=True)
            coeffs = batch["coeffs"].to(device, non_blocking=True)
            pred = model(traj)
            pred_std = pred / coeff_scale
            target_std = coeffs / coeff_scale
            std_loss = torch.mean((pred_std - target_std) ** 2)
            phys_loss = torch.mean((pred - coeffs) ** 2)
            total_std += float(std_loss.detach().cpu())
            total_phys += float(phys_loss.detach().cpu())
            n_batches += 1
    if n_batches == 0:
        return 0.0, 0.0
    return total_std / n_batches, total_phys / n_batches


def _materialise_samples(dataset) -> list[_PreparedSample]:
    """Turn a CompactSwingDataset into a flat list of trial-level tensors."""
    trials_df = dataset.trials
    timesteps_df = dataset.timesteps
    samples: list[_PreparedSample] = []
    for row in trials_df.itertuples(index=False):
        trial_id = int(row.trial_id)
        coeffs = np.asarray(row.coefficients, dtype=np.float32).reshape(-1)
        ts = timesteps_df[timesteps_df["trial_id"] == trial_id]
        if ts.empty:
            continue
        ts_sorted = ts.sort_values("t")
        traj = _build_trajectory(ts_sorted)
        samples.append(
            _PreparedSample(
                trial_id=trial_id,
                trajectory=torch.from_numpy(traj),
                coeffs=torch.from_numpy(coeffs),
            )
        )
    return samples


def _build_trajectory(ts_sorted) -> np.ndarray:
    def _stack_col(name: str) -> np.ndarray:
        return np.stack([np.asarray(v, dtype=np.float32) for v in ts_sorted[name]])

    r_butt = _stack_col("r_buttend")
    r_club = _stack_col("r_clubhead")
    r_grip = _stack_col("r_grip")
    v_club = _stack_col("v_clubhead")
    return _stack_trajectory_channels(r_butt, r_club, r_grip, v_club)


def _default_compact_loader(path: Path):
    """Load a compact dataset; falls back to a minimal in-module loader."""
    try:
        from src.shared.python.motion_matching.dataset import (  # type: ignore
            load_compact_swing_dataset,
        )
    except ImportError:
        return _minimal_compact_loader(path)
    return load_compact_swing_dataset(path)


def _minimal_compact_loader(path: Path):
    """Fallback parquet loader matching ``COMPACT_DATASET_SCHEMA.md``."""
    import pandas as pd

    p = Path(path)
    trials = pd.read_parquet(p / "trials.parquet")
    timesteps = pd.read_parquet(p / "timesteps.parquet")

    @dataclass(frozen=True)
    class _MinimalCompact:
        trials: object
        timesteps: object
        joint_names: tuple
        coefficient_letters: tuple = ("A", "B", "C", "D", "E", "F", "G")
        schema_version: str = "compact-1.0"

    joint_names_raw = trials["joint_names"].iloc[0]
    return _MinimalCompact(
        trials=trials,
        timesteps=timesteps,
        joint_names=tuple(joint_names_raw),
    )
