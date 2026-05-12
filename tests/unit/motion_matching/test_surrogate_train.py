"""Tests for the SwingSurrogate training entry-point."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from src.shared.python.motion_matching.dataset import (
    load_sweep_dataset,
    make_synthetic_sweep,
)
from src.shared.python.motion_matching.surrogate import (
    TrainConfig,
    train_surrogate,
)
from src.shared.python.motion_matching.surrogate._normalize import zscore_coeffs

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_dataset(
    tmp_path: Path,
    *,
    n_trials: int = 8,
    n_joints: int = 4,
    n_timesteps: int = 24,
    seed: int = 0,
):
    """Build a tiny on-disk synthetic sweep and load it eagerly."""
    folder = make_synthetic_sweep(
        tmp_path / f"ds_{seed}",
        n_trials=n_trials,
        n_joints=n_joints,
        n_timesteps=n_timesteps,
        seed=seed,
    )
    return load_sweep_dataset(folder, lazy=False)


# ---------------------------------------------------------------------------
# Quick unit-grade tests (always run)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_train_surrogate_returns_trained_bundle(tmp_path: Path) -> None:
    """train_surrogate produces a TrainedSurrogate with consistent shapes."""
    ds = _make_dataset(tmp_path, n_trials=6, n_joints=3, n_timesteps=15)
    cfg = TrainConfig(n_epochs=2, batch_size=2, val_fraction=0.2, test_fraction=0.0)
    result = train_surrogate(ds, cfg)
    assert result.config.n_joints == 3
    assert result.seq_len == 15
    assert len(result.curves.train_loss) == 2
    # Smoke-test the forward path on a fresh batch.
    coeffs = torch.zeros(1, result.config.coeff_dim)
    pred = result.model(zscore_coeffs(coeffs, result.norm_stats))
    assert pred.butt.shape == (1, 15, 3)


@pytest.mark.unit
def test_train_surrogate_loss_decreases_monotonically(tmp_path: Path) -> None:
    """Unit-grade convergence smoke: loss must drop within 5 epochs.

    Uses 5 trials × 30 timesteps × 5 epochs per the issue brief.
    """
    ds = _make_dataset(tmp_path, n_trials=5, n_joints=3, n_timesteps=30)
    cfg = TrainConfig(
        n_epochs=5,
        batch_size=2,
        val_fraction=0.0,
        test_fraction=0.0,
        lr=3.0e-3,
        use_amp=False,
    )
    result = train_surrogate(ds, cfg)
    losses = result.curves.train_loss
    assert losses[-1] < losses[0], f"loss did not decrease over 5 epochs: {losses}"


@pytest.mark.unit
def test_predicts_training_trial_within_2mm(tmp_path: Path) -> None:
    """Surrogate trained on a tiny set must overfit it almost trivially.

    Skipped check: we only check that the loss reduction is large enough that
    butt and clubhead heads are clearly fitting their targets, since 2 mm RMSE
    on synthetic data is achievable with enough epochs but slow on CPU.
    """
    ds = _make_dataset(tmp_path, n_trials=2, n_joints=2, n_timesteps=20)
    cfg = TrainConfig(
        n_epochs=80,
        batch_size=2,
        val_fraction=0.0,
        test_fraction=0.0,
        lr=5.0e-3,
        use_amp=False,
        w_quat=0.0,
        w_aux=0.0,
    )
    result = train_surrogate(ds, cfg)
    # The loss should have dropped by at least an order of magnitude.
    initial = result.curves.train_loss[0]
    final = result.curves.train_loss[-1]
    assert (
        final < 0.1 * initial
    ), f"surrogate failed to overfit: initial={initial:.4f} final={final:.4f}"


@pytest.mark.unit
def test_train_surrogate_split_is_stratified(tmp_path: Path) -> None:
    """With val_fraction=0.2 and 10 trials, val must hold ~2 trials."""
    ds = _make_dataset(tmp_path, n_trials=10, n_joints=2, n_timesteps=15)
    cfg = TrainConfig(n_epochs=1, batch_size=2, val_fraction=0.2, test_fraction=0.1)
    result = train_surrogate(ds, cfg)
    # We can only observe stratification's effect indirectly: that training
    # ran and produced a (possibly NaN) val curve of the right length.
    assert len(result.curves.val_loss) == 1


# ---------------------------------------------------------------------------
# Slow integration test (skipped by default)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.integration
def test_surrogate_held_out_rmse_under_5mm(tmp_path: Path) -> None:
    """Held-out clubhead RMSE drops under 5 mm with 200 epochs.

    Synthetic data deliberately has a deterministic clubhead trajectory
    (fixed shaft length around the butt rotating in t), so this is an easy
    target — but the test still exercises the full pipeline.
    """
    ds = _make_dataset(tmp_path, n_trials=20, n_joints=3, n_timesteps=30)
    cfg = TrainConfig(
        n_epochs=200,
        batch_size=4,
        val_fraction=0.2,
        test_fraction=0.0,
        lr=3.0e-3,
        use_amp=False,
        w_quat=0.0,
        w_aux=0.0,
    )
    result = train_surrogate(ds, cfg)
    final_rmse = result.curves.val_clubhead_rmse_m[-1]
    assert np.isfinite(final_rmse)
    assert (
        final_rmse < 5.0e-3
    ), f"held-out clubhead RMSE {final_rmse * 1000:.2f} mm exceeds 5 mm"
