"""Training pipeline for :class:`SwingInverseCVAE` (issue #033 / GH #4002).

Loss
----
``L = recon_mse(coeffs_pred, coeffs_true)
        + beta(t) * KL(q(z|x) || N(0, I))
        + lambda_W * | work_estimate(coeffs_pred) - work_estimate(coeffs_true) |``

``beta(t)`` ramps linearly from 0 to ``max_beta`` over the first
``kl_warmup_epochs`` epochs (see :mod:`._kl`). The work term uses the
closed-form polynomial estimator in :mod:`._work_estimator` and biases
the decoder toward modes whose total mechanical work matches the
demonstrated trial.

Splits
------
Train/val/test are split **by trial_id** so a trial never appears in
two splits, mirroring the surrogate trainer (#028).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
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

from ._kl import default_warmup_epochs, linear_kl_beta
from ._work_estimator import work_estimate_torch
from .cvae import COEFFICIENTS_PER_JOINT, CVAEConfig, SwingInverseCVAE

logger = get_logger(__name__)


@dataclass(frozen=True)
class TrainInverseConfig:
    """Hyperparameters for :func:`train_inverse_cvae`.

    Attributes:
        n_epochs: Total optimizer epochs.
        batch_size: Mini-batch size in trials.
        lr: AdamW learning rate.
        weight_decay: AdamW weight decay.
        grad_clip: Global gradient-norm clip (``<= 0`` disables).
        lambda_recon: MSE weight on coefficients (``λ_θ`` in APPROACH.md).
        lambda_work: Weight on the work-regularization term (``λ_W``).
        max_beta: Plateau value of the KL annealing schedule.
        kl_warmup_epochs: Length of the linear KL warmup in epochs.
            ``None`` -> 20% of ``n_epochs`` (default per APPROACH.md).
        val_fraction: Held-out validation fraction by trial.
        test_fraction: Held-out test fraction by trial.
        seed: RNG seed for split + init.
        device: ``"cpu"``, ``"cuda"``, or ``"auto"``.
        checkpoint_dir: If set, ``best.pt`` is saved here on val improvement.
        duration_s: Trial duration used by the work estimator.
    """

    n_epochs: int = 5
    batch_size: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    lambda_recon: float = 1.0
    lambda_work: float = 1e-3
    max_beta: float = 1.0
    kl_warmup_epochs: int | None = None
    val_fraction: float = 0.1
    test_fraction: float = 0.1
    seed: int = 0xC0FFEE
    device: str = "auto"
    checkpoint_dir: Path | None = None
    duration_s: float = 1.0

    def __post_init__(self) -> None:
        _validate_train_config(self)


@dataclass
class TrainingCurves:
    """Per-epoch loss curves for the inverse CVAE."""

    train_loss: list[float] = field(default_factory=list)
    train_recon: list[float] = field(default_factory=list)
    train_kl: list[float] = field(default_factory=list)
    train_work: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_kl: list[float] = field(default_factory=list)
    beta: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class TrainedInverseCVAE:
    """Bundle returned by :func:`train_inverse_cvae`.

    Mirrors INTERFACES.md but keeps the optional ``checkpoint_path`` /
    ``norm_stats`` slots free-form so the surrogate's stats type is not
    a hard import dependency.
    """

    model: SwingInverseCVAE
    config: CVAEConfig
    train_config: TrainInverseConfig
    joint_names: list[str]
    curves: TrainingCurves
    train_metrics: dict[str, float]
    checkpoint_path: Path | None
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@precondition(
    lambda dataset, config, train_config=None, **_: dataset is not None,
    "dataset must be provided",
)
@precondition(
    lambda dataset, config, train_config=None, **_: isinstance(config, CVAEConfig),
    "config must be a CVAEConfig instance",
)
@postcondition(
    lambda result: result.train_metrics.get("final_train_loss") is not None,
    "training must record a final-epoch training loss",
)
def train_inverse_cvae(
    dataset: SweepDataset,
    config: CVAEConfig,
    train_config: TrainInverseConfig | None = None,
    *,
    log_dir: Path | None = None,
) -> TrainedInverseCVAE:
    """Train a :class:`SwingInverseCVAE` on a :class:`SweepDataset`.

    Parameters
    ----------
    dataset
        Loaded random-sweep dataset (real or synthetic).
    config
        Architectural hyperparameters; ``n_joints`` must match the dataset.
    train_config
        Training hyperparameters; defaults are used when ``None``.
    log_dir
        Reserved for tensorboard support (#033 follow-up). Currently
        unused beyond logging.

    Returns
    -------
    TrainedInverseCVAE
        Bundle with the trained model in ``eval`` mode, per-epoch curves,
        final metrics, and split indices.
    """
    cfg = train_config or TrainInverseConfig()
    if config.n_joints != len(dataset.joint_names):
        raise ValueError(
            f"CVAEConfig.n_joints ({config.n_joints}) must equal the dataset's "
            f"joint count ({len(dataset.joint_names)})"
        )

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    coeffs, kinematics, trial_ids = _materialize_tensors(dataset, config)
    n_trials = coeffs.shape[0]

    train_idx, val_idx, test_idx = _split_by_trial_id(trial_ids, cfg, n_trials)
    device = _resolve_device(cfg.device)

    model = SwingInverseCVAE(config).to(device)
    curves, metrics, ckpt_path = _run_training_loop(
        model=model,
        cfg=cfg,
        coeffs=coeffs,
        kinematics=kinematics,
        train_idx=train_idx,
        val_idx=val_idx,
        device=device,
        log_dir=log_dir,
    )
    model.eval()

    return TrainedInverseCVAE(
        model=model,
        config=config,
        train_config=cfg,
        joint_names=list(dataset.joint_names),
        curves=curves,
        train_metrics=metrics,
        checkpoint_path=ckpt_path,
        train_indices=train_idx,
        val_indices=val_idx,
        test_indices=test_idx,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_train_config(cfg: TrainInverseConfig) -> None:
    _check_positive_ints(cfg)
    _check_non_negative_floats(cfg)
    _check_fractions(cfg)


def _check_positive_ints(cfg: TrainInverseConfig) -> None:
    if cfg.n_epochs <= 0:
        raise ValueError(f"n_epochs must be positive; got {cfg.n_epochs}")
    if cfg.batch_size <= 0:
        raise ValueError(f"batch_size must be positive; got {cfg.batch_size}")
    if cfg.kl_warmup_epochs is not None and cfg.kl_warmup_epochs < 0:
        raise ValueError(
            f"kl_warmup_epochs must be non-negative; got {cfg.kl_warmup_epochs}"
        )


def _check_non_negative_floats(cfg: TrainInverseConfig) -> None:
    if cfg.lr <= 0.0:
        raise ValueError(f"lr must be positive; got {cfg.lr}")
    if cfg.lambda_recon < 0.0:
        raise ValueError(f"lambda_recon must be non-negative; got {cfg.lambda_recon}")
    if cfg.lambda_work < 0.0:
        raise ValueError(f"lambda_work must be non-negative; got {cfg.lambda_work}")
    if cfg.max_beta < 0.0:
        raise ValueError(f"max_beta must be non-negative; got {cfg.max_beta}")
    if cfg.duration_s <= 0.0:
        raise ValueError(f"duration_s must be positive; got {cfg.duration_s}")


def _check_fractions(cfg: TrainInverseConfig) -> None:
    if not 0.0 <= cfg.val_fraction < 1.0:
        raise ValueError(f"val_fraction must lie in [0, 1); got {cfg.val_fraction}")
    if not 0.0 <= cfg.test_fraction < 1.0:
        raise ValueError(f"test_fraction must lie in [0, 1); got {cfg.test_fraction}")
    if cfg.val_fraction + cfg.test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must be < 1.0")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def _materialize_tensors(
    dataset: SweepDataset,
    config: CVAEConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build ``(coeffs, kinematics, trial_ids)`` numpy arrays.

    Kinematics layout matches ``CVAEConfig.n_kinematic_channels = 12``:
    ``r_butt(3) + r_clubhead(3) + q_club(4) + 2 reserved`` (zero-padded).
    """
    trials = dataset.trials
    success_mask = trials["solver_status"] == "success"
    trials = trials.loc[success_mask].reset_index(drop=True)
    if len(trials) == 0:
        raise ValueError("dataset has no successful trials to train on")

    timesteps = _timesteps_pandas(dataset.timesteps)
    timesteps = timesteps.sort_values(["trial_id", "t"]).reset_index(drop=True)

    by_trial = dict(list(timesteps.groupby("trial_id", sort=False)))
    seq_len = min(len(g) for g in by_trial.values())
    if seq_len > config.n_timesteps:
        seq_len = config.n_timesteps

    coeffs = np.stack(
        [np.asarray(c, dtype=np.float32) for c in trials["coefficients"]], axis=0
    )
    expected_dim = config.n_joints * COEFFICIENTS_PER_JOINT
    if coeffs.shape[1] != expected_dim:
        raise ValueError(
            f"dataset coefficient dim {coeffs.shape[1]} != "
            f"n_joints * 7 = {expected_dim}"
        )

    butt = _stack_vectors(trials, by_trial, seq_len, "r_butt", 3)
    head = _stack_vectors(trials, by_trial, seq_len, "r_clubhead", 3)
    q_club = _stack_vectors(trials, by_trial, seq_len, "q_club", 4)
    pad_width = max(config.n_kinematic_channels - (3 + 3 + 4), 0)
    pad = np.zeros((coeffs.shape[0], seq_len, pad_width), dtype=np.float32)
    kinematics = np.concatenate([butt, head, q_club, pad], axis=-1)
    if kinematics.shape[-1] != config.n_kinematic_channels:
        raise ValueError(
            f"assembled kinematics width {kinematics.shape[-1]} != "
            f"n_kinematic_channels {config.n_kinematic_channels}"
        )

    trial_ids = trials["trial_id"].to_numpy(dtype=np.int64)
    return coeffs, kinematics, trial_ids


def _timesteps_pandas(timesteps: Any) -> Any:
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


def _split_by_trial_id(
    trial_ids: np.ndarray,
    cfg: TrainInverseConfig,
    n_trials: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Disjoint splits keyed on ``trial_ids`` (no leakage)."""
    rng = np.random.default_rng(cfg.seed)
    order = rng.permutation(n_trials)
    val_n = max(int(round(cfg.val_fraction * n_trials)), 0)
    test_n = max(int(round(cfg.test_fraction * n_trials)), 0)
    if val_n + test_n >= n_trials:
        val_n = max(n_trials // 10, 0)
        test_n = max(n_trials // 10, 0)
    train_n = n_trials - val_n - test_n
    train_pos = order[:train_n]
    val_pos = order[train_n : train_n + val_n]
    test_pos = order[train_n + val_n :]
    # Returned indices index into the per-trial arrays; the ``trial_ids``
    # array is preserved so callers can verify no leakage at the trial-id
    # level.
    _ = trial_ids  # explicitly retained for the postcondition contract.
    return (
        np.asarray(train_pos, dtype=np.int64),
        np.asarray(val_pos, dtype=np.int64),
        np.asarray(test_pos, dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _make_loader(
    coeffs: np.ndarray,
    kinematics: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(coeffs).float(),
        torch.from_numpy(kinematics).float(),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def _run_training_loop(
    *,
    model: SwingInverseCVAE,
    cfg: TrainInverseConfig,
    coeffs: np.ndarray,
    kinematics: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    device: torch.device,
    log_dir: Path | None,
) -> tuple[TrainingCurves, dict[str, float], Path | None]:
    """Run AdamW + cosine-LR loop, return curves, metrics, checkpoint."""
    train_loader = _make_loader(
        coeffs[train_idx],
        kinematics[train_idx],
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    val_loader = (
        _make_loader(
            coeffs[val_idx],
            kinematics[val_idx],
            batch_size=cfg.batch_size,
            shuffle=False,
        )
        if len(val_idx) > 0
        else None
    )

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(cfg.n_epochs, 1))
    warmup = (
        cfg.kl_warmup_epochs
        if cfg.kl_warmup_epochs is not None
        else default_warmup_epochs(cfg.n_epochs)
    )

    curves = TrainingCurves()
    best_val = float("inf")
    ckpt_path: Path | None = None
    if cfg.checkpoint_dir is not None:
        cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = cfg.checkpoint_dir / "best.pt"

    for epoch in range(cfg.n_epochs):
        beta = linear_kl_beta(
            epoch,
            total_epochs=cfg.n_epochs,
            warmup_epochs=warmup,
            max_beta=cfg.max_beta,
        )
        train_stats = _run_one_epoch(
            model, train_loader, optimizer, cfg, device, beta=beta
        )
        val_stats = _evaluate(model, val_loader, cfg, device, beta=beta)
        scheduler.step()

        curves.train_loss.append(train_stats["loss"])
        curves.train_recon.append(train_stats["recon"])
        curves.train_kl.append(train_stats["kl"])
        curves.train_work.append(train_stats["work"])
        curves.val_loss.append(val_stats["loss"])
        curves.val_kl.append(val_stats["kl"])
        curves.beta.append(beta)

        if (
            ckpt_path is not None
            and val_stats["loss"] == val_stats["loss"]  # not NaN
            and val_stats["loss"] < best_val
        ):
            best_val = val_stats["loss"]
            torch.save({"model_state": model.state_dict(), "epoch": epoch}, ckpt_path)

        logger.debug(
            "epoch %d/%d beta=%.3f train=%.4f kl=%.4f val=%.4f",
            epoch + 1,
            cfg.n_epochs,
            beta,
            train_stats["loss"],
            train_stats["kl"],
            val_stats["loss"],
        )

    metrics = {
        "final_train_loss": curves.train_loss[-1],
        "final_train_kl": curves.train_kl[-1],
        "final_train_recon": curves.train_recon[-1],
        "final_val_loss": curves.val_loss[-1],
        "final_beta": curves.beta[-1],
        "val_round_trip_rmse_m": float("nan"),  # filled when sim_fn lands (#034)
    }
    _ = log_dir  # tensorboard hookup is a follow-up; arg kept for API parity.
    return curves, metrics, ckpt_path


def _run_one_epoch(
    model: SwingInverseCVAE,
    loader: DataLoader,
    optimizer: AdamW,
    cfg: TrainInverseConfig,
    device: torch.device,
    *,
    beta: float,
) -> dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "recon": 0.0, "kl": 0.0, "work": 0.0}
    n_batches = 0
    for coeffs_b, kin_b in loader:
        coeffs_b = coeffs_b.to(device)
        kin_b = kin_b.to(device)
        optimizer.zero_grad(set_to_none=True)
        coeffs_pred, enc = model(kin_b, sample=True)
        terms = _compute_loss(coeffs_pred, coeffs_b, enc, cfg, beta=beta)
        terms["loss"].backward()
        if cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        for key in totals:
            totals[key] += float(terms[key].detach().cpu())
        n_batches += 1
    return {k: v / max(n_batches, 1) for k, v in totals.items()}


@torch.no_grad()
def _evaluate(
    model: SwingInverseCVAE,
    loader: DataLoader | None,
    cfg: TrainInverseConfig,
    device: torch.device,
    *,
    beta: float,
) -> dict[str, float]:
    if loader is None:
        return {"loss": float("nan"), "kl": float("nan")}
    model.eval()
    total_loss = 0.0
    total_kl = 0.0
    n_batches = 0
    for coeffs_b, kin_b in loader:
        coeffs_b = coeffs_b.to(device)
        kin_b = kin_b.to(device)
        coeffs_pred, enc = model(kin_b, sample=False)
        terms = _compute_loss(coeffs_pred, coeffs_b, enc, cfg, beta=beta)
        total_loss += float(terms["loss"].cpu())
        total_kl += float(terms["kl"].cpu())
        n_batches += 1
    return {
        "loss": total_loss / max(n_batches, 1),
        "kl": total_kl / max(n_batches, 1),
    }


def _compute_loss(
    coeffs_pred: torch.Tensor,
    coeffs_true: torch.Tensor,
    encoder_out: Any,
    cfg: TrainInverseConfig,
    *,
    beta: float,
) -> dict[str, torch.Tensor]:
    """ELBO + work regularizer; returns dict of {loss, recon, kl, work}."""
    recon = nn.functional.mse_loss(coeffs_pred, coeffs_true)
    # Standard closed-form KL between diagonal Gaussian q and N(0, I).
    mu = encoder_out.mu
    log_var = encoder_out.log_var
    kl_per_dim = -0.5 * (1.0 + log_var - mu.pow(2) - log_var.exp())
    kl = kl_per_dim.sum(dim=-1).mean()

    work_pred = work_estimate_torch(coeffs_pred, duration_s=cfg.duration_s)
    work_true = work_estimate_torch(coeffs_true, duration_s=cfg.duration_s)
    work_term = (work_pred - work_true).abs().mean()

    loss = cfg.lambda_recon * recon + beta * kl + cfg.lambda_work * work_term
    return {"loss": loss, "recon": recon, "kl": kl, "work": work_term}
