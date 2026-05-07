#!/usr/bin/env python3
"""Example script for training Option-3 inverse cVAE on the compact dataset.

This script demonstrates the workflow:
1. Load a compact swing dataset (per ``COMPACT_DATASET_SCHEMA.md``).
2. Train the cVAE end-to-end with beta-annealed ELBO.
3. Save the trained model + ``metrics.json`` under ``output/inverse_cvae/<timestamp>/``.

Usage::

    cd /path/to/UpstreamDrift
    python3 motion_matching/train_option3_example.py \\
        --dataset C:/Users/diete/Repositories/data/compact \\
        --epochs 50 \\
        --batch-size 32

GH issue #4076: M3 — Train Option-3 inverse cVAE on the compact 10k dataset.

Note: the legacy public API (``Option3TrainConfig``, ``TrainInverseConfig``,
``train_option3_inverse_cvae``) was replaced by the canonical
``TrainingConfig`` + ``train_inverse_cvae`` surface in PR #4240. This
example consumes the new API.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path so the ``src`` package resolves when run as a script.
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.shared.python.logging_pkg.logging_config import (  # noqa: E402
    get_logger,
)
from src.shared.python.motion_matching.inverse import (  # noqa: E402
    CVAEConfig,
    TrainingConfig,
    train_inverse_cvae,
)

logger = get_logger(__name__)


def _resolve_device(arg: str) -> str:
    if arg != "auto":
        return arg
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train Option-3 inverse cVAE on the compact swing dataset."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to compact dataset folder (trials.parquet + timesteps.parquet).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=TrainingConfig.epochs,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=TrainingConfig.batch_size,
        help="Training batch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=TrainingConfig.lr,
        help="AdamW learning rate.",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=CVAEConfig.latent_dim,
        help="Latent space dimensionality.",
    )
    parser.add_argument(
        "--kl-anneal-epochs",
        type=int,
        default=TrainingConfig.kl_anneal_epochs,
        help="Number of epochs over which to anneal KL beta from 0 to max.",
    )
    parser.add_argument(
        "--max-beta",
        type=float,
        default=TrainingConfig.max_beta,
        help="Maximum KL weight after annealing.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output/inverse_cvae"),
        help="Root output directory; a timestamped subdir is created inside it.",
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
        default=TrainingConfig.seed,
        help="RNG seed.",
    )

    args = parser.parse_args()

    if not args.dataset.exists():
        logger.error("Dataset path does not exist: %s", args.dataset)
        return 1

    device = _resolve_device(args.device)
    logger.info("Using device: %s", device)

    cvae_config = CVAEConfig(latent_dim=args.latent_dim)

    logger.info(
        "Option-3 cVAE training:\n"
        "  Dataset: %s\n"
        "  Latent dim: %d\n"
        "  Epochs: %d\n"
        "  Batch size: %d\n"
        "  Learning rate: %.2e\n"
        "  KL anneal epochs: %d (max beta %.3f)\n"
        "  Device: %s\n"
        "  Output root: %s",
        args.dataset,
        args.latent_dim,
        args.epochs,
        args.batch_size,
        args.lr,
        args.kl_anneal_epochs,
        args.max_beta,
        device,
        args.output_root,
    )

    try:
        result = train_inverse_cvae(
            dataset_path=args.dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
            seed=args.seed,
            kl_anneal_epochs=args.kl_anneal_epochs,
            max_beta=args.max_beta,
            output_root=args.output_root,
            cvae_config=cvae_config,
        )
    except (ValueError, FileNotFoundError) as exc:
        logger.error("Training rejected by precondition: %s", exc)
        return 1
    except Exception:
        logger.exception("Training failed")
        return 1

    logger.info("=" * 70)
    logger.info("Option-3 cVAE training complete")
    logger.info("=" * 70)
    logger.info("Output directory: %s", result.output_dir)
    logger.info("Best epoch: %d", result.best_epoch)
    logger.info("Total parameters: %d", result.parameter_count)

    last = result.history[-1]
    logger.info(
        "Final epoch %d: train_loss=%.4f val_recon=%.4f val_kl=%.4f",
        last.epoch,
        last.train_loss,
        last.val_recon,
        last.val_kl,
    )

    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    sys.exit(main())
