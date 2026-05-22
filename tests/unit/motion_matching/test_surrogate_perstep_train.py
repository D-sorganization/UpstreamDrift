"""Tests for surrogate perstep training logic."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from src.shared.python.motion_matching.surrogate.perstep.train import (
    TrainConfig,
    _make_splits,
    _normalized_rmse_by_column,
    _r2_by_column,
    _rmse_by_column,
    _standardize,
    compute_phase_stratified_metrics,
    train,
)


@pytest.mark.unit
def test_standardize():
    """Test data standardization."""
    train_data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    values = np.array([[3.0, 4.0]], dtype=np.float32)

    scaled, mean, std = _standardize(train_data, values)
    assert np.allclose(mean, [3.0, 4.0])
    # std of [1, 3, 5] is sqrt(8/3) ~ 1.63299
    # std of [2, 4, 6] is sqrt(8/3) ~ 1.63299
    assert np.allclose(std, [1.63299316, 1.63299316])
    assert np.allclose(scaled, [0.0, 0.0])


@pytest.mark.unit
def test_make_splits():
    """Test train/val/test splits."""
    n = 100
    train_idx, val_idx, test_idx = _make_splits(
        n, validation_fraction=0.2, test_fraction=0.1, seed=42
    )
    assert len(test_idx) == 10
    assert len(val_idx) == 20
    assert len(train_idx) == 70

    # Check disjoint
    all_idx = set(train_idx) | set(val_idx) | set(test_idx)
    assert len(all_idx) == 100


@pytest.mark.unit
def test_rmse_by_column():
    """Test RMSE calculation."""
    pred = np.array([[1.0, 2.0], [3.0, 4.0]])
    target = np.array([[1.0, 0.0], [3.0, 4.0]])
    res = _rmse_by_column(pred, target, ["a", "b"])
    assert res["a"] == 0.0
    # b: (2-0)^2 = 4, (4-4)^2 = 0 -> mean = 2, sqrt = 1.414
    assert np.isclose(res["b"], 1.4142135)


@pytest.mark.unit
def test_r2_by_column():
    """Test R2 calculation."""
    pred = np.array([[1.0, 2.0], [3.0, 4.0]])
    target = np.array([[1.0, 2.0], [3.0, 4.0]])
    res = _r2_by_column(pred, target, ["a", "b"])
    assert np.isclose(res["a"], 1.0)
    assert np.isclose(res["b"], 1.0)


@pytest.mark.unit
def test_normalized_rmse_by_column():
    """Test normalized RMSE calculation."""
    pred = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    target = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    res = _normalized_rmse_by_column(pred, target, ["a", "b"])
    assert np.isclose(res["a"], 0.0)
    assert np.isclose(res["b"], 0.0)


@pytest.mark.unit
def test_compute_phase_stratified_metrics():
    """Test phase stratified metrics wrapper."""
    pred = np.zeros((0, 3))
    target = np.zeros((0, 3))
    time = np.zeros(0)

    assert compute_phase_stratified_metrics(pred, target, time) == {}

    time = np.array([1.0, 1.0])
    assert compute_phase_stratified_metrics(pred, target, time) == {}
    assert compute_phase_stratified_metrics(pred, target, time) == {}


@pytest.mark.unit
def test_train_loop(tmp_path: Path):
    """Test the full training loop with mocked data."""
    out_dir = tmp_path / "runs"
    out_dir.mkdir()

    cfg = TrainConfig(
        dataset="dummy.parquet",
        output_dir=str(out_dir),
        epochs=2,
        batch_size=4,
        learning_rate=1e-3,
        weight_decay=0.0,
        hidden_sizes=[16, 16],
        validation_fraction=0.2,
        test_fraction=0.2,
        seed=42,
        device="cpu",
        use_amp=False,
    )

    x_raw = np.random.randn(20, 5).astype(np.float32)
    y_raw = np.random.randn(20, 3).astype(np.float32)
    input_cols = [f"in_{i}" for i in range(5)]
    target_cols = [f"out_{i}" for i in range(3)]

    with patch(
        "src.shared.python.motion_matching.surrogate.perstep.train._load_arrays",
        return_value=(x_raw, y_raw, input_cols, target_cols),
    ):
        train(cfg)

    assert (out_dir / "best_model.pt").exists()
    assert (out_dir / "history.json").exists()
    assert (out_dir / "metrics.json").exists()

    metrics = json.loads((out_dir / "metrics.json").read_text())
    assert metrics["input_dim"] == 5
    assert metrics["target_dim"] == 3
    assert "best_val_loss_scaled" in metrics
    assert "test_rmse_mean_unscaled" in metrics
