"""Inference-surface tests for the Option-3 inverse cVAE (GH issue #4076)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.shared.python.motion_matching.inverse import (  # noqa: E402
    DEFAULT_COEFFICIENT_DIM,
    DEFAULT_LATENT_DIM,
    DEFAULT_TRAJECTORY_CHANNELS,
    CoefficientPredictions,
    CVAEConfig,
    SwingInverseCVAE,
    build_coefficient_bound_vector,
    load_inverse_cvae,
    predict_coefficients,
    predict_coefficients_from_checkpoint,
)

pytestmark = [pytest.mark.unit, pytest.mark.requires_torch]


def _model() -> SwingInverseCVAE:
    cfg = CVAEConfig(encoder_channels=(32, 64), decoder_hidden=64)
    return SwingInverseCVAE(cfg)


def _trajectory_np(T: int = 16) -> np.ndarray:
    return np.zeros((T, DEFAULT_TRAJECTORY_CHANNELS), dtype=np.float32)


# ---- shape contract --------------------------------------------------------


def test_predict_returns_expected_shapes() -> None:
    model = _model()
    pred = predict_coefficients(model, _trajectory_np(), n_samples=5, seed=42)
    assert isinstance(pred, CoefficientPredictions)
    assert pred.samples.shape == (5, DEFAULT_COEFFICIENT_DIM)
    assert pred.mean.shape == (DEFAULT_COEFFICIENT_DIM,)
    assert pred.latent_mu.shape == (DEFAULT_LATENT_DIM,)
    assert pred.latent_logvar.shape == (DEFAULT_LATENT_DIM,)
    assert pred.samples.dtype == np.float32
    assert pred.mean.dtype == np.float32
    assert pred.n_samples == 5
    assert pred.coefficient_dim == DEFAULT_COEFFICIENT_DIM


def test_predict_accepts_torch_tensor_input() -> None:
    model = _model()
    traj = torch.zeros(16, DEFAULT_TRAJECTORY_CHANNELS, dtype=torch.float32)
    pred = predict_coefficients(model, traj, n_samples=3, seed=0)
    assert pred.samples.shape == (3, DEFAULT_COEFFICIENT_DIM)


def test_predict_accepts_3d_batch_takes_first_row() -> None:
    model = _model()
    traj = np.zeros((4, 16, DEFAULT_TRAJECTORY_CHANNELS), dtype=np.float32)
    pred = predict_coefficients(model, traj, n_samples=2, seed=1)
    assert pred.samples.shape == (2, DEFAULT_COEFFICIENT_DIM)


def test_predict_rejects_wrong_channel_count() -> None:
    model = _model()
    bad = np.zeros((16, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="trajectory channels"):
        predict_coefficients(model, bad)


def test_predict_rejects_wrong_rank() -> None:
    model = _model()
    bad = np.zeros((DEFAULT_TRAJECTORY_CHANNELS,), dtype=np.float32)
    with pytest.raises(ValueError, match="must be"):
        predict_coefficients(model, bad)


def test_predict_n_samples_must_be_positive() -> None:
    model = _model()
    with pytest.raises(ValueError, match="n_samples"):
        predict_coefficients(model, _trajectory_np(), n_samples=0)


# ---- determinism for the prior-mean point estimate ------------------------


def test_predict_mean_is_deterministic() -> None:
    model = _model()
    traj = _trajectory_np()
    pred_a = predict_coefficients(model, traj, n_samples=4, seed=11)
    pred_b = predict_coefficients(model, traj, n_samples=4, seed=999)
    # Mean is from the prior mean (no eps draw); identical regardless of seed.
    np.testing.assert_allclose(pred_a.mean, pred_b.mean, atol=1e-6)
    np.testing.assert_allclose(pred_a.latent_mu, pred_b.latent_mu, atol=1e-6)


def test_predict_seed_controls_sample_draws() -> None:
    model = _model()
    traj = _trajectory_np()
    pred_a = predict_coefficients(model, traj, n_samples=4, seed=11)
    pred_b = predict_coefficients(model, traj, n_samples=4, seed=11)
    np.testing.assert_allclose(pred_a.samples, pred_b.samples, atol=1e-6)


# ---- physical-bound respect (unit conversions) ----------------------------


def test_samples_respect_physical_bounds() -> None:
    model = _model()
    traj = (
        np.random.default_rng(0)
        .standard_normal((16, DEFAULT_TRAJECTORY_CHANNELS))
        .astype(np.float32)
    )
    pred = predict_coefficients(model, traj, n_samples=8, seed=0)
    bounds = build_coefficient_bound_vector(model.cfg.n_joints).numpy()
    assert np.all(pred.samples <= bounds + 1e-3)
    assert np.all(pred.samples >= -bounds - 1e-3)
    assert np.all(pred.mean <= bounds + 1e-3)
    assert np.all(pred.mean >= -bounds - 1e-3)


# ---- checkpoint round-trip via from_checkpoint + predict_from_checkpoint --


def test_predict_from_checkpoint_roundtrip(tmp_path: Path) -> None:
    model = _model()
    ckpt = tmp_path / "model.pt"
    torch.save(model.state_payload(), ckpt)

    restored = load_inverse_cvae(ckpt)
    assert isinstance(restored, SwingInverseCVAE)

    pred = predict_coefficients_from_checkpoint(
        ckpt, _trajectory_np(), n_samples=2, seed=7
    )
    assert pred.samples.shape == (2, DEFAULT_COEFFICIENT_DIM)


def test_swing_inverse_cvae_predict_from_checkpoint_missing_file_raises(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError):
        load_inverse_cvae(tmp_path / "does_not_exist.pt")
