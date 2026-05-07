#!/usr/bin/env python3
"""Example script for training Option-3 inverse cVAE on 10k parquet dataset.

This script demonstrates the complete workflow:
1. Load dataset from parquet files
2. Train the cVAE with ELBO + work regularization
3. Evaluate on held-out test set (coverage, reconstruction, diversity)
4. Save trained model and metrics

Usage (synthetic data):
    cd /path/to/UpstreamDrift
    python3 motion_matching/train_option3_example.py

Usage (real dataset):
    python3 motion_matching/train_option3_example.py \
        --dataset /path/to/10k_dataset \
        --output ./results/option3_cvae

Issue #4076: M3: Train Option-3 inverse cVAE on 10k parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.shared.python.logging_pkg.logging_config import (  # noqa: E402
    get_logger,
)
from src.shared.python.motion_matching.dataset.synthetic import (  # noqa: E402
    make_synthetic_sweep,
)
from src.shared.python.motion_matching.inverse import (  # noqa: E402
    CVAEConfig,
    Option3TrainConfig,
    TrainInverseConfig,
    train_option3_inverse_cvae,
)

logger = get_logger(__name__)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train Option-3 inverse cVAE on 10k parquet dataset."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to dataset folder (trials.parquet + timesteps.parquet). "
        "If None, generates synthetic data for demo.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./motion_matching/results/option3_cvae_trained",
        help="Output directory for model, metrics, and plots.",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=5,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size.",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=16,
        help="Latent space dimensionality.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Torch device.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0xC0FFEE,
        help="RNG seed.",
    )
    parser.add_argument(
        "--skip-synthetic",
        action="store_true",
        help="Do not generate synthetic data if dataset path is None.",
    )

    args = parser.parse_args()
    output_dir = Path(args.output)

    # Resolve dataset
    dataset_path = args.dataset
    if dataset_path is None:
        if args.skip_synthetic:
            logger.error("No dataset provided and --skip-synthetic is set.")
            return 1
        # Generate synthetic data for demonstration
        logger.info("No dataset provided; generating synthetic data for demo.")
        synthetic_path = output_dir.parent / "synthetic_dataset"
        dataset_path = make_synthetic_sweep(
            synthetic_path,
            n_trials=100,  # Small for demo; real = 10000
            n_joints=14,
            n_timesteps=300,
            seed=args.seed,
        )
        logger.info("Generated synthetic dataset: %s", dataset_path)
    else:
        dataset_path = Path(dataset_path)

    # Ensure dataset exists
    if not dataset_path.exists():
        logger.error("Dataset path does not exist: %s", dataset_path)
        return 1

    # Resolve device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logger.info("Using device: %s", device)

    # Build configs
    cvae_config = CVAEConfig(
        n_joints=14,
        n_timesteps=300,
        n_kinematic_channels=12,
        latent_dim=args.latent_dim,
        encoder_layers=4,
        encoder_heads=4,
        encoder_dim=128,
        decoder_hidden=256,
        dropout=0.1,
    )

    train_config = TrainInverseConfig(
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr=1e-3,
        weight_decay=1e-5,
        grad_clip=1.0,
        lambda_recon=1.0,
        lambda_work=1e-3,
        max_beta=1.0,
        kl_warmup_epochs=None,  # auto = 20% of n_epochs
        val_fraction=0.1,
        test_fraction=0.1,
        seed=args.seed,
        device=device,
    )

    option3_config = Option3TrainConfig(
        dataset_path=dataset_path,
        output_dir=output_dir,
        cvae_config=cvae_config,
        train_config=train_config,
        n_test_samples=50,
        coverage_threshold_m=0.05,
        latent_projection_method="umap",
        latent_projection_seed=args.seed,
    )

    logger.info(
        "Option-3 cVAE Training Config:\n"
        "  Dataset: %s (%d trials)\n"
        "  Latent dim: %d\n"
        "  Epochs: %d\n"
        "  Batch size: %d\n"
        "  Device: %s\n"
        "  Output: %s",
        dataset_path,
        "?",  # Will be printed during load
        args.latent_dim,
        args.n_epochs,
        args.batch_size,
        device,
        output_dir,
    )

    # Train
    try:
        result = train_option3_inverse_cvae(option3_config)
        logger.info(
            "\n" + "=" * 70 + "\nOption-3 cVAE Training Complete\n" + "=" * 70
        )
        logger.info("Model saved to: %s", result.model_path)
        logger.info("Config saved to: %s", result.config_path)
        logger.info("Metrics saved to: %s", result.metrics_path)
        logger.info("Evaluation plots: %s", result.evaluation_plot_dir)

        # Print key metrics
        logger.info("\nKey Evaluation Metrics:")
        logger.info("  Final train loss: %.4f", result.metrics.get("final_train_loss"))
        logger.info("  Final val loss: %.4f", result.metrics.get("final_val_loss"))
        logger.info(
            "  Coverage mean RMSE (m): %.4f",
            result.metrics.get("coverage_mean_rmse_m", float("nan")),
        )
        logger.info(
            "  Diversity mean pairwise L2: %.4f",
            result.metrics.get("diversity_mean_pairwise_l2", float("nan")),
        )
        logger.info(
            "  Inference latency (ms): %.3f",
            result.metrics.get("inference_latency_ms", float("nan")),
        )
        logger.info(
            "  Latent spread: %.4f", result.metrics.get("latent_spread", float("nan"))
        )

        logger.info("\nTraining curves saved (train_loss, val_loss, KL, etc.)")
        return 0

    except Exception as e:
        logger.exception("Training failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
