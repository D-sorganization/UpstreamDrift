"""Training loop for :class:`SwingInverseCVAE` (Option 3, GH issue #4076).

Loads a compact swing dataset (per ``COMPACT_DATASET_SCHEMA.md``) and
fits the cVAE with a beta-annealed ELBO:

    L = MSE(theta_hat, theta_true) + beta(epoch) * KL(q(z|x,c) || p(z|c))

Where ``theta`` is the 189-dim coefficient vector. Train/val splits are
*by trial_id* (90/10 default) so a trial never appears in both. Early
stopping triggers when val recon plateaus for ``patience`` epochs.
Checkpoints + a ``metrics.json`` summary land under
``output/inverse_cvae/<timestamp>/`` by default.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from .cvae import (
    CVAEConfig,
    EncoderOutput,
    SwingInverseCVAE,
    kl_divergence_per_dim,
    parameter_count,
)

logger = logging.getLogger(__name__)


DEFAULT_OUTPUT_ROOT = Path("output/inverse_cvae")
DEFAULT_PATIENCE = 10


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EpochMetrics:
    """Per-epoch loss summary (means over batches)."""

    epoch: int
    train_recon: float
    train_kl: float
    train_loss: float
    val_recon: float
    val_kl: float
    val_loss: float
    beta: float
    duration_s: float


@dataclass(frozen=True)
class TrainingResult:
    """Outcome of :func:`train_inverse_cvae`.

    Frozen for safe sharing across callers (eg the matlab shim's
    ``pyrunfile`` boundary). The ``checkpoint_path`` points at the best
    val-recon epoch; ``best_epoch`` is its 0-based index.
    """

    history: tuple[EpochMetrics, ...]
    best_epoch: int
    final_epoch: int
    checkpoint_path: Path
    output_dir: Path
    n_train_trials: int
    n_val_trials: int
    parameter_count: int
    config: CVAEConfig

    def to_summary(self) -> dict:
        """JSON-serialisable summary used by the metrics file + reports."""
        return {
            "history": [asdict(h) for h in self.history],
            "best_epoch": self.best_epoch,
            "final_epoch": self.final_epoch,
            "checkpoint_path": str(self.checkpoint_path),
            "output_dir": str(self.output_dir),
            "n_train_trials": self.n_train_trials,
            "n_val_trials": self.n_val_trials,
            "parameter_count": self.parameter_count,
        }


# ---------------------------------------------------------------------------
# Dataset adapter
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _PreparedSample:
    """One trial's tensors ready for the dataloader."""

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
    """Concatenate per-timestep 1-D channel arrays along the channel axis.

    Each input is ``(T, k_i)`` with sum(k_i) == expected_channels.
    Returns ``(T, expected_channels)`` float32.
    """
    arr = np.concatenate(vectors, axis=-1).astype(np.float32, copy=False)
    if arr.shape[-1] != expected_channels:
        raise ValueError(
            f"trajectory channels = {arr.shape[-1]}, expected {expected_channels}"
        )
    return arr


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparameters for :func:`train_inverse_cvae`.

    Defaults updated to fix the bug-2 posterior collapse:

    * ``max_beta=0.1`` (was 1.0). With recon now computed in
      [-1, 1] coefficient space (O(1) magnitude) a max-β of 1.0 is
      far too aggressive — KL would dominate and crush the posterior.
    * ``kl_anneal_epochs=30`` (was 10). Slow anneal gives the encoder
      time to learn a useful posterior before KL pressure ramps in.
    * ``free_bits=0.5`` per latent dim. Standard cVAE technique: KL
      below this floor is not penalised, preventing the optimiser from
      collapsing the posterior to the prior when the encoder hasn't
      yet learned a useful latent code.
    """

    epochs: int = 30
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-5
    kl_anneal_epochs: int = 30
    max_beta: float = 0.1
    free_bits: float = 0.5
    patience: int = DEFAULT_PATIENCE
    val_fraction: float = 0.1
    seed: int = 0xC0FFEE
    grad_clip: float = 1.0
    device: str = "auto"
    cvae: CVAEConfig = field(default_factory=CVAEConfig)


def train_inverse_cvae(
    dataset_path: str | Path,
    *,
    epochs: int = TrainingConfig.epochs,
    batch_size: int = TrainingConfig.batch_size,
    lr: float = TrainingConfig.lr,
    device: str | torch.device = "auto",
    seed: int = TrainingConfig.seed,
    kl_anneal_epochs: int = TrainingConfig.kl_anneal_epochs,
    max_beta: float = TrainingConfig.max_beta,
    free_bits: float = TrainingConfig.free_bits,
    patience: int = DEFAULT_PATIENCE,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    val_fraction: float = TrainingConfig.val_fraction,
    cvae_config: CVAEConfig | None = None,
    dataset_loader=None,
    on_epoch_end: Callable[[EpochMetrics], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> TrainingResult:
    """Train :class:`SwingInverseCVAE` end-to-end.

    Parameters
    ----------
    dataset_path
        Folder containing ``trials.parquet`` and ``timesteps.parquet``
        per ``COMPACT_DATASET_SCHEMA.md``. The loader is dependency-injected
        for testing via ``dataset_loader``.
    dataset_loader
        Optional callable ``(path) -> CompactSwingDataset``. Defaults to
        :func:`load_compact_swing_dataset` (loaded lazily so this module
        can be imported without the dataset extras present).
    on_epoch_end
        Optional callback invoked after each completed epoch with that
        epoch's :class:`EpochMetrics`. Defaults to ``None`` (no-op).
        Provided so external observers (e.g. the training-controller
        ``PyTorchCVAERunner`` adapter) can stream metrics out without
        scraping the stdlib logger. Exceptions raised by the callback
        propagate to the caller — the loop does NOT swallow them.
    should_stop
        Optional callable polled once per epoch (immediately after the
        existing early-stop / plateau check). When it returns ``True``
        the loop ends gracefully and returns the best-so-far
        :class:`TrainingResult`. Defaults to ``None`` (no cooperative
        cancellation). Wired by the training-controller adapter to a
        :class:`CancelToken` so jobs can be cancelled mid-run.

    Returns
    -------
    TrainingResult
        History, best-epoch index, output directory and parameter count.

    Raises
    ------
    ValueError
        If the dataset has fewer than 2 trials (cannot split) or if
        ``epochs`` is non-positive.
    """
    if epochs <= 0:
        raise ValueError(f"epochs must be positive, got {epochs}")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    if free_bits < 0:
        raise ValueError(f"free_bits must be >= 0, got {free_bits}")

    loader_fn = dataset_loader or _default_compact_loader
    dataset = loader_fn(Path(dataset_path))
    samples = _materialise_samples(dataset)
    if len(samples) < 2:
        raise ValueError(f"need at least 2 trials to train+val, got {len(samples)}")

    rng = np.random.default_rng(seed)
    train_samples, val_samples = _split_by_trial(samples, val_fraction, rng)

    torch.manual_seed(seed)
    selected_device = _resolve_device(device)
    cfg = cvae_config or CVAEConfig()
    model = SwingInverseCVAE(cfg).to(selected_device)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    # Per-coefficient symmetric bound vector — used to standardise the
    # 189-dim recon target into [-1, 1] so reconstruction MSE is O(1)
    # and comparable in scale to the KL term. (Bug-2 fix.) Reusing the
    # model's buffer here means the standardiser tracks
    # ``coefficient_bound_strategy`` (spec vs empirical) automatically.
    coeff_bounds = cast(torch.Tensor, model.coefficient_bounds).to(selected_device)

    train_loader = DataLoader(
        _TrialTensorDataset(train_samples),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate,
        drop_last=False,
    )
    val_loader = DataLoader(
        _TrialTensorDataset(val_samples),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate,
    )

    output_dir = _make_output_dir(Path(output_root))
    history: list[EpochMetrics] = []
    best_val_recon = float("inf")
    best_epoch = 0
    best_path = output_dir / "checkpoint_best.pt"
    plateau = 0

    for epoch in range(epochs):
        beta = _beta_for_epoch(epoch, kl_anneal_epochs, max_beta)
        t0 = time.time()
        train_recon, train_kl = _run_epoch(
            model,
            train_loader,
            beta,
            selected_device,
            opt,
            train=True,
            coeff_bounds=coeff_bounds,
            free_bits=free_bits,
        )
        val_recon, val_kl = _run_epoch(
            model,
            val_loader,
            beta,
            selected_device,
            opt,
            train=False,
            coeff_bounds=coeff_bounds,
            free_bits=free_bits,
        )
        duration = time.time() - t0

        train_loss = train_recon + beta * train_kl
        val_loss = val_recon + beta * val_kl
        metrics = EpochMetrics(
            epoch=epoch,
            train_recon=train_recon,
            train_kl=train_kl,
            train_loss=train_loss,
            val_recon=val_recon,
            val_kl=val_kl,
            val_loss=val_loss,
            beta=beta,
            duration_s=duration,
        )
        history.append(metrics)
        logger.info(
            "epoch=%d train_recon=%.4g val_recon=%.4g beta=%.3f dt=%.2fs",
            epoch,
            train_recon,
            val_recon,
            beta,
            duration,
        )

        ckpt_path = output_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(model.state_payload(), ckpt_path)

        if val_recon < best_val_recon - 1e-6:
            best_val_recon = val_recon
            best_epoch = epoch
            plateau = 0
            torch.save(model.state_payload(), best_path)
        else:
            plateau += 1
            if plateau >= patience:
                logger.info("early stop: val recon plateau at epoch %d", epoch)
                if on_epoch_end is not None:
                    on_epoch_end(metrics)
                break

        if on_epoch_end is not None:
            on_epoch_end(metrics)

        if should_stop is not None and should_stop():
            logger.info("cooperative stop requested at epoch %d", epoch)
            break

    summary_path = output_dir / "metrics.json"
    final_epoch = history[-1].epoch
    result = TrainingResult(
        history=tuple(history),
        best_epoch=best_epoch,
        final_epoch=final_epoch,
        checkpoint_path=best_path,
        output_dir=output_dir,
        n_train_trials=len(train_samples),
        n_val_trials=len(val_samples),
        parameter_count=parameter_count(model),
        config=cfg,
    )
    summary_path.write_text(json.dumps(result.to_summary(), indent=2))
    return result


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


def _beta_for_epoch(epoch: int, anneal: int, max_beta: float) -> float:
    if anneal <= 0:
        return float(max_beta)
    return float(max_beta * min(1.0, (epoch + 1) / anneal))


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
    model: SwingInverseCVAE,
    loader: DataLoader,
    beta: float,
    device: torch.device,
    opt: AdamW,
    *,
    train: bool,
    coeff_bounds: torch.Tensor,
    free_bits: float,
) -> tuple[float, float]:
    """One pass over ``loader``. ``train=True`` flips the optimizer step on.

    The recon loss is computed on coefficients standardised to ``[-1, 1]``
    via ``coeff_bounds`` (the bug-2 fix). KL is the per-dim closed form,
    free-bits-clamped before summing so individual latent dims with
    near-zero KL don't drag the loss to a posterior collapse.
    """
    if train:
        model.train()
    else:
        model.eval()
    recon_total = 0.0
    kl_total = 0.0
    kl_uncapped_total = 0.0
    n_batches = 0
    with torch.set_grad_enabled(train):
        for batch in loader:
            traj = batch["trajectory"].to(device, non_blocking=True)
            coeffs = batch["coeffs"].to(device, non_blocking=True)
            coeff_pred, enc_out = model(traj, coeffs, sample=train)
            # Recon MSE on standardised coefficients: target in [-1, 1].
            pred_std = coeff_pred / coeff_bounds
            target_std = coeffs / coeff_bounds
            recon = torch.mean((pred_std - target_std) ** 2)
            kl, kl_uncapped = _kl_for_loss(enc_out, free_bits)
            loss = recon + beta * kl
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            recon_total += float(recon.detach().cpu())
            kl_total += float(kl.detach().cpu())
            kl_uncapped_total += float(kl_uncapped.detach().cpu())
            n_batches += 1
    if n_batches == 0:
        return 0.0, 0.0
    # Report the *uncapped* KL so plateau / collapse diagnostics aren't
    # masked by the free-bits floor.
    return recon_total / n_batches, kl_uncapped_total / n_batches


def _kl_for_loss(
    enc_out: EncoderOutput, free_bits: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(kl_for_loss, kl_uncapped)``.

    ``kl_for_loss`` applies free-bits: per-latent-dim KL is clamped to a
    minimum of ``free_bits`` nats *before* the sum across the latent
    axis, then averaged across the batch. This is the standard
    Kingma-Welling free-bits trick — without it, low-information latent
    dims collapse the posterior to the prior and the encoder stops
    learning.

    ``kl_uncapped`` is the raw mean of the per-batch KLs (sum-over-dim,
    mean-over-batch) and is reported in the metrics for diagnostics.
    """
    per_dim = kl_divergence_per_dim(
        enc_out.mu_q, enc_out.logvar_q, enc_out.mu_p, enc_out.logvar_p
    )  # (B, latent_dim)
    capped = torch.clamp(per_dim, min=free_bits) if free_bits > 0 else per_dim
    kl_for_loss = capped.sum(dim=-1).mean()
    kl_uncapped = per_dim.sum(dim=-1).mean()
    return kl_for_loss, kl_uncapped


def _materialise_samples(dataset) -> list[_PreparedSample]:
    """Turn a CompactSwingDataset into a flat list of trial-level tensors.

    Trajectory channels (12 total): r_buttend(3) + r_clubhead(3) + r_grip(3)
    + v_clubhead(3). This is the same per-trial 12-channel layout the
    surrogate consumes; the issue body does not constrain the exact split,
    only the channel count.
    """
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
    """Load a compact dataset without depending on unmerged compactor branch.

    Tries the canonical ``load_compact_swing_dataset`` first; falls back to
    a minimal in-module loader that reads the parquet pair directly.
    """
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
