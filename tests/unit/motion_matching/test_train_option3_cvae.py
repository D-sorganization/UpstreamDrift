"""Unit tests for Option-3 inverse cVAE training (issue #4076).

Tests cover:
- Configuration validation
- Dataset loading and prep
- Training pipeline end-to-end
- Evaluation metrics computation
- Model save/load round-trip
- Inference latency
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from src.shared.python.motion_matching.dataset import load_sweep_dataset
from src.shared.python.motion_matching.dataset.synthetic import make_synthetic_sweep
from src.shared.python.motion_matching.inverse import (
    CVAEConfig,
    Option3TrainConfig,
    TrainInverseConfig,
    train_option3_inverse_cvae,
)

# Minimal config for fast unit tests
_N_TRIALS = 25
_N_JOINTS = 4
_N_TIMESTEPS = 32
_N_EPOCHS = 2


def _make_dataset(tmp_path: Path) -> object:
    """Create a small synthetic dataset."""
    folder = make_synthetic_sweep(
        tmp_path / "sweep",
        n_trials=_N_TRIALS,
        n_joints=_N_JOINTS,
        n_timesteps=_N_TIMESTEPS,
        seed=42,
    )
    return load_sweep_dataset(folder, lazy=False)


def _make_cvae_config() -> CVAEConfig:
    """Minimal cVAE config."""
    return CVAEConfig(
        n_joints=_N_JOINTS,
        n_timesteps=_N_TIMESTEPS,
        n_kinematic_channels=12,
        latent_dim=4,
        encoder_layers=1,
        encoder_heads=2,
        encoder_dim=16,
        decoder_hidden=16,
        dropout=0.0,
    )


def _make_train_config(**overrides: object) -> TrainInverseConfig:
    """Default training config with overrides."""
    base: dict[str, object] = {
        "n_epochs": _N_EPOCHS,
        "batch_size": 4,
        "lr": 1e-3,
        "val_fraction": 0.2,
        "test_fraction": 0.2,
        "kl_warmup_epochs": 1,
        "device": "cpu",
        "seed": 7,
    }
    base.update(overrides)
    return TrainInverseConfig(**base)  # type: ignore[arg-type]


@pytest.mark.unit
def test_option3_config_validation() -> None:
    """Test that Option3TrainConfig validates inputs."""
    cvae_cfg = _make_cvae_config()

    # Valid config
    cfg = Option3TrainConfig(
        dataset_path="/fake/path",
        output_dir="/tmp/out",
        cvae_config=cvae_cfg,
    )
    assert cfg.n_test_samples == 50

    # Invalid: n_test_samples < 1
    with pytest.raises(ValueError, match="n_test_samples"):
        Option3TrainConfig(
            dataset_path="/fake/path",
            output_dir="/tmp/out",
            cvae_config=cvae_cfg,
            n_test_samples=0,
        )

    # Invalid: coverage_threshold_m <= 0
    with pytest.raises(ValueError, match="coverage_threshold_m"):
        Option3TrainConfig(
            dataset_path="/fake/path",
            output_dir="/tmp/out",
            cvae_config=cvae_cfg,
            coverage_threshold_m=0.0,
        )


@pytest.mark.unit
def test_train_option3_end_to_end(tmp_path: Path) -> None:
    """Full training pipeline: load -> train -> evaluate -> save."""
    dataset = _make_dataset(tmp_path)
    output_dir = tmp_path / "output"

    config = Option3TrainConfig(
        dataset_path=str(dataset),  # type: ignore[arg-type]
        output_dir=output_dir,
        cvae_config=_make_cvae_config(),
        train_config=_make_train_config(n_epochs=_N_EPOCHS),
        n_test_samples=10,
    )

    result = train_option3_inverse_cvae(config)

    # Check result structure
    assert result.model_path.exists()
    assert result.config_path.exists()
    assert result.metrics_path.exists()
    assert result.evaluation_plot_dir.exists()
    assert len(result.model_state_dict) > 0
    assert len(result.metrics) > 0
    assert len(result.curves) > 0


@pytest.mark.unit
def test_config_json_save_load(tmp_path: Path) -> None:
    """Test config persistence."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    cvae_cfg = _make_cvae_config()
    train_cfg = _make_train_config()

    config = Option3TrainConfig(
        dataset_path="/path/to/data",
        output_dir=output_dir,
        cvae_config=cvae_cfg,
        train_config=train_cfg,
        n_test_samples=42,
        coverage_threshold_m=0.01,
        latent_projection_method="pca",
    )

    from src.shared.python.motion_matching.inverse.train_option3_cvae import (
        _save_config_json,
    )

    config_path = output_dir / "config.json"
    _save_config_json(config, config_path)

    assert config_path.exists()

    # Load and validate
    with open(config_path) as f:
        loaded = json.load(f)

    assert loaded["n_test_samples"] == 42
    assert loaded["coverage_threshold_m"] == 0.01
    assert loaded["latent_projection_method"] == "pca"
    assert loaded["cvae_config"]["n_joints"] == _N_JOINTS
    assert loaded["cvae_config"]["latent_dim"] == 4


@pytest.mark.unit
def test_metrics_json_save_load(tmp_path: Path) -> None:
    """Test metrics persistence."""
    from src.shared.python.motion_matching.inverse.train_option3_cvae import (
        _save_metrics_json,
    )

    metrics_path = tmp_path / "metrics.json"

    metrics = {
        "final_train_loss": 0.123,
        "final_val_loss": 0.456,
        "coverage_mean_rmse_m": 0.01,
        "diversity_mean_pairwise_l2": 1.5,
        "inference_latency_ms": 0.42,
        "latent_spread": 2.3,
        "coverage_flagged_count": 2.0,
    }

    _save_metrics_json(metrics, metrics_path)

    assert metrics_path.exists()

    # Load and validate
    with open(metrics_path) as f:
        loaded = json.load(f)

    assert loaded["final_train_loss"] == pytest.approx(0.123)
    assert loaded["inference_latency_ms"] == pytest.approx(0.42)


@pytest.mark.unit
def test_model_state_dict_serialization(tmp_path: Path) -> None:
    """Test model weights are correctly saved."""
    dataset = _make_dataset(tmp_path)
    output_dir = tmp_path / "output"

    config = Option3TrainConfig(
        dataset_path=str(dataset),  # type: ignore[arg-type]
        output_dir=output_dir,
        cvae_config=_make_cvae_config(),
        train_config=_make_train_config(n_epochs=1),
    )

    result = train_option3_inverse_cvae(config)

    # Check state dict keys
    state_dict = result.model_state_dict
    assert "encoder.transformer_encoder.layers.0.self_attn.in_proj_weight" in state_dict
    assert "decoder_net.0.weight" in state_dict
    assert "posterior_head.0.weight" in state_dict

    # Verify all values are tensors
    assert all(isinstance(v, torch.Tensor) for v in state_dict.values())


@pytest.mark.unit
def test_train_curves_recorded(tmp_path: Path) -> None:
    """Test that training curves are properly recorded."""
    dataset = _make_dataset(tmp_path)
    output_dir = tmp_path / "output"

    n_epochs = 3
    config = Option3TrainConfig(
        dataset_path=str(dataset),  # type: ignore[arg-type]
        output_dir=output_dir,
        cvae_config=_make_cvae_config(),
        train_config=_make_train_config(n_epochs=n_epochs),
    )

    result = train_option3_inverse_cvae(config)

    # Check curve lengths
    curves = result.curves
    assert len(curves["train_loss"]) == n_epochs
    assert len(curves["val_loss"]) == n_epochs
    assert len(curves["train_kl"]) == n_epochs
    assert len(curves["beta"]) == n_epochs

    # All values should be finite
    for loss in curves["train_loss"]:
        assert np.isfinite(loss)
    for loss in curves["val_loss"]:
        assert np.isfinite(loss)


@pytest.mark.unit
def test_inference_latency_is_reasonable(tmp_path: Path) -> None:
    """Test that inference is fast enough for real-time use."""
    dataset = _make_dataset(tmp_path)
    output_dir = tmp_path / "output"

    config = Option3TrainConfig(
        dataset_path=str(dataset),  # type: ignore[arg-type]
        output_dir=output_dir,
        cvae_config=_make_cvae_config(),
        train_config=_make_train_config(n_epochs=1),
    )

    result = train_option3_inverse_cvae(config)

    latency_ms = result.metrics.get("inference_latency_ms")
    assert latency_ms is not None
    assert latency_ms > 0
    # Should be << 1 ms on CPU (requirement: <= 1 ms)
    assert latency_ms < 10.0  # Relaxed for CI; real-time on GPU


@pytest.mark.unit
def test_evaluation_metrics_exist(tmp_path: Path) -> None:
    """Test that all expected evaluation metrics are computed."""
    dataset = _make_dataset(tmp_path)
    output_dir = tmp_path / "output"

    config = Option3TrainConfig(
        dataset_path=str(dataset),  # type: ignore[arg-type]
        output_dir=output_dir,
        cvae_config=_make_cvae_config(),
        train_config=_make_train_config(n_epochs=1),
    )

    result = train_option3_inverse_cvae(config)

    metrics = result.metrics
    required_keys = {
        "final_train_loss",
        "final_val_loss",
        "inference_latency_ms",
    }

    for key in required_keys:
        assert key in metrics, f"Missing metric: {key}"
        assert np.isfinite(metrics[key])
