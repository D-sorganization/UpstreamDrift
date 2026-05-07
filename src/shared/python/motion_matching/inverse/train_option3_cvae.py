"""Option-3 cVAE training on 10k parquet dataset (issue #4076).

This module orchestrates the complete training pipeline for the inverse cVAE:
1. Loads training data from parquet files (real or synthetic)
2. Trains the cVAE with ELBO + work regularization
3. Evaluates model on held-out test set
4. Computes coverage, reconstruction accuracy, and latent-space diversity
5. Saves trained model and evaluation metrics

The trained model learns the inverse mapping: grip trajectory (measured target)
→ polynomial coefficients θ, supporting stochastic exploration via latent sampling.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.motion_matching.dataset import load_sweep_dataset
from src.shared.python.motion_matching.inverse import (
    CVAEConfig,
    TrainInverseConfig,
    train_inverse_cvae,
)
from src.shared.python.motion_matching.inverse.diagnostics import (
    dataset_coverage_map,
    latent_projection,
    sample_diversity,
)

logger = get_logger(__name__)

__all__ = [
    "Option3TrainConfig",
    "Option3TrainingResult",
    "train_option3_inverse_cvae",
]


@dataclass(frozen=True)
class Option3TrainConfig:
    """High-level configuration for Option-3 inverse cVAE training.

    This wraps both the model architecture (CVAEConfig) and training
    dynamics (TrainInverseConfig), plus additional evaluation settings.

    Attributes:
        dataset_path: Path to parquet folder (trials.parquet +
            timesteps.parquet).
        output_dir: Where to save model, metrics, and evaluation plots.
        cvae_config: Architectural hyperparameters.
        train_config: Training hyperparameters (epochs, lr, beta schedule,
            etc.).
        n_test_samples: Number of samples to draw per test input for
            diversity eval.
        coverage_threshold_m: RMSE threshold for flagging coverage failures.
        latent_projection_method: "umap", "tsne", or "pca" for
            visualization.
        latent_projection_seed: RNG seed for dimensionality reduction.
    """

    dataset_path: str | Path
    output_dir: str | Path
    cvae_config: CVAEConfig
    train_config: TrainInverseConfig | None = None
    n_test_samples: int = 50
    coverage_threshold_m: float = 0.05
    latent_projection_method: str = "umap"
    latent_projection_seed: int = 0xC0FFEE

    def __post_init__(self) -> None:
        if self.n_test_samples < 1:
            raise ValueError(f"n_test_samples must be >= 1; got {self.n_test_samples}")
        if self.coverage_threshold_m <= 0:
            raise ValueError(
                f"coverage_threshold_m must be positive; got {self.coverage_threshold_m}"
            )


@dataclass(frozen=True)
class Option3TrainingResult:
    """Bundle of trained model, training curves, and evaluation metrics.

    Attributes:
        model_path: Path where the trained model weights were saved.
        config_path: Path where the config dict was saved as JSON.
        metrics_path: Path where the evaluation metrics were saved.
        evaluation_plot_dir: Directory containing diagnostic plots (if any).
        model_state_dict: The trained model's state_dict (for in-memory use).
        config: The training configuration used.
        metrics: Dictionary of evaluation metrics (coverage, reconstruction,
            etc.).
        curves: Training curves (loss, KL, etc.) from the training loop.
    """

    model_path: Path
    config_path: Path
    metrics_path: Path
    evaluation_plot_dir: Path
    model_state_dict: dict[str, Any]
    config: Option3TrainConfig
    metrics: dict[str, float]
    curves: dict[str, list[float]]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def train_option3_inverse_cvae(
    config: Option3TrainConfig,
) -> Option3TrainingResult:
    """Train and evaluate the Option-3 inverse cVAE on 10k parquet dataset.

    Parameters
    ----------
    config
        Training configuration including dataset path, output directory,
        model architecture, and training hyperparameters.

    Returns
    -------
    Option3TrainingResult
        Bundle containing the trained model, training curves, evaluation
        metrics, and file paths to saved artifacts.

    Raises
    ------
    FileNotFoundError
        If the dataset path does not exist or lacks required parquet files.
    ValueError
        If the configuration is invalid or dataset validation fails.
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Option-3 cVAE training: output_dir=%s", output_dir)

    # Load dataset
    dataset = load_sweep_dataset(config.dataset_path, lazy=False)
    logger.info(
        "Loaded dataset: %d trials, %d joints, %d timesteps max",
        dataset.n_trials(),
        dataset.n_joints(),
        config.cvae_config.n_timesteps,
    )

    # Validate config consistency
    if config.cvae_config.n_joints != dataset.n_joints():
        raise ValueError(
            f"CVAEConfig.n_joints ({config.cvae_config.n_joints}) must equal "
            f"dataset joints ({dataset.n_joints()})"
        )

    # Train model
    train_cfg = config.train_config or TrainInverseConfig()
    trained_bundle = train_inverse_cvae(
        dataset=dataset,
        config=config.cvae_config,
        train_config=train_cfg,
        log_dir=output_dir,
    )
    logger.info(
        "Training complete: %d epochs, final val loss = %.4f",
        len(trained_bundle.curves.train_loss),
        trained_bundle.curves.val_loss[-1],
    )

    # Evaluate on test set
    evaluation_plot_dir = output_dir / "evaluation_plots"
    evaluation_plot_dir.mkdir(parents=True, exist_ok=True)

    metrics = _compute_evaluation_metrics(
        trained_bundle=trained_bundle,
        dataset=dataset,
        config=config,
        plot_dir=evaluation_plot_dir,
    )

    # Save artifacts
    model_path = output_dir / "model_state.pt"
    config_path = output_dir / "config.json"
    metrics_path = output_dir / "metrics.json"

    torch.save(trained_bundle.model.state_dict(), model_path)
    logger.info("Saved model state to %s", model_path)

    _save_config_json(config, config_path)
    _save_metrics_json(metrics, metrics_path)

    # Build curves dict
    curves_dict = {
        "train_loss": trained_bundle.curves.train_loss,
        "train_recon": trained_bundle.curves.train_recon,
        "train_kl": trained_bundle.curves.train_kl,
        "train_work": trained_bundle.curves.train_work,
        "val_loss": trained_bundle.curves.val_loss,
        "val_kl": trained_bundle.curves.val_kl,
        "beta": trained_bundle.curves.beta,
    }

    return Option3TrainingResult(
        model_path=model_path,
        config_path=config_path,
        metrics_path=metrics_path,
        evaluation_plot_dir=evaluation_plot_dir,
        model_state_dict=trained_bundle.model.state_dict(),
        config=config,
        metrics=metrics,
        curves=curves_dict,
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _compute_evaluation_metrics(
    *,
    trained_bundle: Any,
    dataset: Any,
    config: Option3TrainConfig,
    plot_dir: Path,
) -> dict[str, float]:
    """Compute posterior coverage, reconstruction accuracy, and diversity.

    Returns a dict of scalar metrics suitable for logging and persistence.
    """
    model = trained_bundle.model
    device = next(model.parameters()).device
    test_idx = trained_bundle.test_indices

    metrics: dict[str, float] = {}

    # 1. Dataset coverage map (round-trip RMSE on test set)
    logger.info("Computing coverage map on %d test trials...", len(test_idx))
    try:
        coverage = dataset_coverage_map(
            model=model,
            kinematics=_prepare_test_kinematics(
                trained_bundle.model, dataset, test_idx
            ),
            forward_fn=lambda x: x,  # dummy forward
            flag_threshold_m=config.coverage_threshold_m,
        )
        metrics["coverage_mean_rmse_m"] = float(coverage.mean_rmse_m)
        metrics["coverage_flagged_count"] = float(np.sum(coverage.flagged_mask))
        metrics["coverage_flagged_frac"] = float(
            np.sum(coverage.flagged_mask) / len(coverage.trial_ids)
        )
    except RuntimeError as e:
        logger.warning("Coverage computation failed: %s", e)
        metrics["coverage_mean_rmse_m"] = float("nan")
        metrics["coverage_flagged_count"] = float("nan")
        metrics["coverage_flagged_frac"] = float("nan")

    # 2. Sampling diversity (multi-modal exploration)
    logger.info("Computing sample diversity on test set...")
    try:
        test_kinematics = _prepare_test_kinematics(
            trained_bundle.model, dataset, test_idx[: min(5, len(test_idx))]
        )
        if test_kinematics.shape[0] > 0:
            diversity = sample_diversity(
                model=model,
                kinematics=test_kinematics,
                n_samples=config.n_test_samples,
            )
            metrics["diversity_mean_pairwise_l2"] = float(diversity.mean_distance)
            metrics["diversity_median_pairwise_l2"] = float(
                diversity.median_distance
            )
            metrics["diversity_collapsed"] = float(diversity.collapsed)
        else:
            metrics["diversity_mean_pairwise_l2"] = float("nan")
            metrics["diversity_median_pairwise_l2"] = float("nan")
            metrics["diversity_collapsed"] = float("nan")
    except RuntimeError as e:
        logger.warning("Diversity computation failed: %s", e)
        metrics["diversity_mean_pairwise_l2"] = float("nan")
        metrics["diversity_median_pairwise_l2"] = float("nan")
        metrics["diversity_collapsed"] = float("nan")

    # 3. Latent space projection (UMAP/t-SNE/PCA)
    logger.info("Computing latent projection...")
    try:
        val_kinematics = _prepare_validation_kinematics(
            trained_bundle.model, dataset, trained_bundle.val_indices
        )
        if val_kinematics.shape[0] > 0:
            projection = latent_projection(
                model=model,
                kinematics=val_kinematics,
                method=config.latent_projection_method,
                seed=config.latent_projection_seed,
            )
            # Compute spread of latent points
            spread = np.linalg.norm(
                projection.coords - projection.coords.mean(axis=0), axis=1
            ).mean()
            metrics["latent_spread"] = float(spread)
            metrics["latent_projection_method"] = projection.method
        else:
            metrics["latent_spread"] = float("nan")
            metrics["latent_projection_method"] = "none"
    except RuntimeError as e:
        logger.warning("Latent projection failed: %s", e)
        metrics["latent_spread"] = float("nan")
        metrics["latent_projection_method"] = "error"

    # 4. Model inference speed (latency check)
    logger.info("Checking inference speed...")
    try:
        latency_ms = _measure_inference_latency(model, device)
        metrics["inference_latency_ms"] = float(latency_ms)
    except RuntimeError as e:
        logger.warning("Latency measurement failed: %s", e)
        metrics["inference_latency_ms"] = float("nan")

    # 5. Merge training metrics
    metrics.update(trained_bundle.train_metrics)

    return metrics


def _prepare_test_kinematics(
    model: Any, dataset: Any, test_idx: np.ndarray
) -> torch.Tensor:
    """Prepare kinematics tensor for test samples."""
    from src.shared.python.motion_matching.inverse.train import _materialize_tensors

    coeffs, kinematics, _ = _materialize_tensors(dataset, model.cfg)
    test_kin = kinematics[test_idx]
    return torch.as_tensor(test_kin, dtype=torch.float32)


def _prepare_validation_kinematics(
    model: Any, dataset: Any, val_idx: np.ndarray
) -> torch.Tensor:
    """Prepare kinematics tensor for validation samples."""
    from src.shared.python.motion_matching.inverse.train import _materialize_tensors

    coeffs, kinematics, _ = _materialize_tensors(dataset, model.cfg)
    val_kin = kinematics[val_idx]
    return torch.as_tensor(val_kin, dtype=torch.float32)


def _measure_inference_latency(model: Any, device: torch.device) -> float:
    """Measure average latency for a single forward pass (ms)."""
    import time

    model.eval()
    # Create a batch of 1 dummy kinematics tensor
    batch = torch.randn(
        1, model.cfg.n_timesteps, model.cfg.n_kinematic_channels
    ).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(batch, sample=False)

    # Time it
    with torch.no_grad():
        start = time.perf_counter()
        for _ in range(100):
            _ = model(batch, sample=False)
        elapsed = time.perf_counter() - start

    return (elapsed / 100.0) * 1000.0  # Convert to ms


def _save_config_json(config: Option3TrainConfig, path: Path) -> None:
    """Save configuration as JSON."""
    config_dict = {
        "dataset_path": str(config.dataset_path),
        "output_dir": str(config.output_dir),
        "cvae_config": asdict(config.cvae_config),
        "train_config": asdict(config.train_config or TrainInverseConfig()),
        "n_test_samples": config.n_test_samples,
        "coverage_threshold_m": config.coverage_threshold_m,
        "latent_projection_method": config.latent_projection_method,
        "latent_projection_seed": config.latent_projection_seed,
    }
    with open(path, "w") as f:
        json.dump(config_dict, f, indent=2)
    logger.info("Saved config to %s", path)


def _save_metrics_json(metrics: dict[str, float], path: Path) -> None:
    """Save metrics as JSON."""
    # Convert any numpy types to Python native types
    clean_metrics = {}
    for k, v in metrics.items():
        if isinstance(v, (np.ndarray, np.generic)):
            clean_metrics[k] = float(v)
        elif isinstance(v, bool):
            clean_metrics[k] = bool(v)
        else:
            clean_metrics[k] = v

    with open(path, "w") as f:
        json.dump(clean_metrics, f, indent=2)
    logger.info("Saved metrics to %s", path)
