"""Train Option-2 NN surrogate on 10k parquet dataset (Issue #4075).

Entry point for training the FiLM-MLP surrogate on the full random-sweep
dataset. Handles:
- Loading the 10k parquet files from data/sweep/<run_id>/
- Training with stratified 80/10/10 split by trial_id
- Saving trained model weights, architecture, and normalization stats
- Reporting model size, training time, and inference speed
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.motion_matching.dataset import (
    load_sweep_dataset,
    make_synthetic_sweep,
)
from src.shared.python.motion_matching.surrogate import (
    TrainConfig,
    train_surrogate,
)

logger = get_logger(__name__)


def main() -> None:
    """Main entry point for surrogate training."""
    parser = argparse.ArgumentParser(
        description="Train Option-2 NN surrogate on 10k parquet dataset"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="data/sweep/20251030",
        help="Path to dataset folder (default: data/sweep/20251030)",
    )
    parser.add_argument(
        "--use-synthetic",
        action="store_true",
        help="Use synthetic dataset for testing",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/surrogates",
        help="Output directory for trained model",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size in trials",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3.0e-4,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["cpu", "cuda", "auto"],
        help="Device to use",
    )

    args = parser.parse_args()

    # Resolve dataset
    if args.use_synthetic:
        logger.info("Using synthetic dataset for testing")
        dataset_path = Path("data/sweep_synthetic")
        dataset_path.mkdir(parents=True, exist_ok=True)
        make_synthetic_sweep(dataset_path, n_trials=100, n_joints=14)
    else:
        dataset_path = Path(args.dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {dataset_path.resolve()}")

    logger.info("Loading dataset from %s", dataset_path)
    dataset = load_sweep_dataset(dataset_path, lazy=False)
    logger.info(
        "Loaded %d trials with %d joints",
        dataset.n_trials(),
        dataset.n_joints(),
    )

    # Configure training
    train_cfg = TrainConfig(
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        device=args.device,
    )

    # Train the surrogate
    logger.info("Training surrogate...")
    start_time = time.time()
    trained = train_surrogate(dataset, train_cfg)
    elapsed_s = time.time() - start_time
    logger.info("Training completed in %.1f seconds", elapsed_s)

    # Save the model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "best_model.pt"
    config_path = output_dir / "config.pt"
    stats_path = output_dir / "norm_stats.pt"

    torch.save(trained.model.state_dict(), model_path)
    torch.save(
        {
            "config": trained.config,
            "train_config": train_cfg,
        },
        config_path,
    )
    torch.save(
        {
            "coeff_mean": trained.norm_stats.coeffs_mean,
            "coeff_std": trained.norm_stats.coeffs_std,
            "r_butt_mean": trained.norm_stats.butt_mean,
            "r_butt_std": trained.norm_stats.butt_std,
            "r_head_mean": trained.norm_stats.clubhead_mean,
            "r_head_std": trained.norm_stats.clubhead_std,
        },
        stats_path,
    )
    logger.info("Model saved to %s", model_path)


if __name__ == "__main__":
    main()
