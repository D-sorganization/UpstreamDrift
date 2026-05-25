"""Tests for the compact swing surrogate training loop."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from src.shared.python.motion_matching.surrogate.compact.model import SurrogateConfig
from src.shared.python.motion_matching.surrogate.compact.training import (
    _build_targets_from_compact,
    _CompactSwingTorchDataset,
    _split_indices_by_trial,
    train_surrogate,
)


@pytest.fixture
def dummy_compact_dataset(tmp_path: Path) -> Path:
    """Create a minimal synthetic compact dataset for testing."""
    n_trials = 10
    seq_len = 300
    coeff_dim = 189

    # 1. Create trials.parquet
    trials_data = []
    for tid in range(n_trials):
        coeffs = np.random.uniform(-1, 1, size=coeff_dim).tolist()
        trials_data.append({"trial_id": tid, "coefficients": coeffs})
    trials_df = pd.DataFrame(trials_data)
    trials_df.to_parquet(tmp_path / "trials.parquet")

    # 2. Create timesteps.parquet
    timesteps_data = []
    for tid in range(n_trials):
        for t in range(seq_len):
            r_clubhead = np.random.normal(0, 1, size=3).tolist()
            v_clubhead = np.random.normal(0, 1, size=3).tolist()
            # offset grip so shaft length > 0
            r_grip = [r_clubhead[0], r_clubhead[1], r_clubhead[2] - 1.0]
            timesteps_data.append(
                {
                    "trial_id": tid,
                    "t": float(t) * 0.001,
                    "r_clubhead": r_clubhead,
                    "v_clubhead": v_clubhead,
                    "r_grip": r_grip,
                    "clubhead_speed_mph": np.linalg.norm(v_clubhead) * 2.23694,
                }
            )
    timesteps_df = pd.DataFrame(timesteps_data)
    timesteps_df.to_parquet(tmp_path / "timesteps.parquet")

    return tmp_path


@pytest.mark.unit
def test_compact_torch_dataset():
    """Test the in-memory torch dataset wrapper."""
    coeffs = np.zeros((5, 189), dtype=np.float32)
    targets = np.ones((5, 300, 12), dtype=np.float32)
    trial_ids = np.arange(5, dtype=np.int64)

    ds = _CompactSwingTorchDataset(coeffs, targets, trial_ids)
    assert len(ds) == 5
    c, t = ds[0]
    assert c.shape == (189,)
    assert t.shape == (300, 12)
    assert isinstance(c, torch.Tensor)
    assert isinstance(t, torch.Tensor)

    # Check bounds
    with pytest.raises(ValueError):
        _CompactSwingTorchDataset(coeffs[:4], targets, trial_ids)

    with pytest.raises(ValueError):
        bad_targets = np.ones((5, 300, 10), dtype=np.float32)
        _CompactSwingTorchDataset(coeffs, bad_targets, trial_ids)


@pytest.mark.unit
def test_split_indices():
    """Test the random 90/10 trial split."""
    trial_ids = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    train_idx, val_idx = _split_indices_by_trial(trial_ids, val_fraction=0.2, seed=42)
    assert len(val_idx) == 2  # 20% of 5 unique trials = 1 trial = 2 rows
    assert len(train_idx) == 8

    # Ensure no overlap in underlying trial IDs
    train_tids = set(trial_ids[train_idx])
    val_tids = set(trial_ids[val_idx])
    assert not train_tids.intersection(val_tids)


@pytest.mark.unit
def test_build_targets(dummy_compact_dataset: Path):
    """Test assembling targets from dataframes."""
    trials_df = pd.read_parquet(dummy_compact_dataset / "trials.parquet")
    timesteps_df = pd.read_parquet(dummy_compact_dataset / "timesteps.parquet")

    coeffs, targets, trial_ids = _build_targets_from_compact(
        trials_df, timesteps_df, seq_len=300
    )
    assert coeffs.shape == (10, 189)
    assert targets.shape == (10, 300, 12)
    assert trial_ids.shape == (10,)


@pytest.mark.unit
def test_train_surrogate_happy_path(dummy_compact_dataset: Path, tmp_path: Path):
    """End-to-end test of the training loop."""
    out_dir = tmp_path / "output"

    cfg = SurrogateConfig(n_joints=27, seq_len=300, hidden_dim=64)

    # Mock progress callback
    progress_epochs = []

    def progress_cb(ep, metrics):
        progress_epochs.append(ep)

    from unittest.mock import patch

    with patch(
        "src.shared.python.motion_matching.surrogate.compact.training._import_compact_loader",
        return_value=None,
    ):
        res = train_surrogate(
            dataset_path=dummy_compact_dataset,
            epochs=2,
            batch_size=4,
            lr=1e-3,
            device="cpu",
            seed=42,
            config=cfg,
            val_fraction=0.2,
            output_dir=out_dir,
            progress_cb=progress_cb,
        )

    # Verify result properties
    assert res.output_dir == out_dir
    assert res.best_epoch > 0
    assert res.best_checkpoint.exists()
    assert res.last_checkpoint.exists()
    assert res.param_count > 0
    assert "train_loss" in res.history
    assert len(res.history["train_loss"]) == 2

    # Verify checkpoints and outputs were written
    assert (out_dir / "metrics.json").exists()

    # Check that callback was called
    assert len(progress_epochs) == 2
    assert progress_epochs == [1, 2]

    # Verify that we can resume from the checkpoint
    with patch(
        "src.shared.python.motion_matching.surrogate.compact.training._import_compact_loader",
        return_value=None,
    ):
        res2 = train_surrogate(
            dataset_path=dummy_compact_dataset,
            epochs=3,
            batch_size=4,
            lr=1e-3,
            device="cpu",
            config=cfg,
            output_dir=tmp_path / "output2",
            resume_from=res.best_checkpoint,
        )
    assert res2.best_epoch >= 2
    assert len(res2.history["train_loss"]) == 1  # ran epochs [2, 3) so 1 epoch


@pytest.mark.unit
def test_train_surrogate_early_stopping(dummy_compact_dataset: Path, tmp_path: Path):
    """Test early stopping triggers when validation loss doesn't improve."""
    cfg = SurrogateConfig(n_joints=27, seq_len=300, hidden_dim=32)

    from unittest.mock import patch

    with patch(
        "src.shared.python.motion_matching.surrogate.compact.training._import_compact_loader",
        return_value=None,
    ):
        res = train_surrogate(
            dataset_path=dummy_compact_dataset,
            epochs=20,
            batch_size=2,
            lr=1.0,  # high LR so it likely diverges and stops early
            device="cpu",
            seed=42,
            config=cfg,
            early_stopping_patience=2,
            output_dir=tmp_path / "early_stop",
        )

    # Should have stopped well before 20 epochs
    assert len(res.history["epoch"]) < 20
