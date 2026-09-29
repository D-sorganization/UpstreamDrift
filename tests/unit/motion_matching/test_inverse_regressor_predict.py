"""Inference-surface tests for :func:`predict_coefficients_regressor`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.shared.python.motion_matching.inverse import (  # noqa: E402
    DEFAULT_COEFFICIENT_DIM,
    DEFAULT_TRAJECTORY_CHANNELS,
    InverseRegressor,
    RegressorConfig,
    build_coefficient_bound_vector,
    load_inverse_regressor,
    predict_coefficients_regressor,
    predict_coefficients_regressor_from_checkpoint,
)

pytestmark = [pytest.mark.unit, pytest.mark.requires_torch]


def _model() -> InverseRegressor:
    cfg = RegressorConfig(embed_dim=32, mlp_hidden=64, n_blocks=2)
    return InverseRegressor(cfg)


def _trajectory_np(T: int = 31) -> np.ndarray:
    return np.zeros((T, DEFAULT_TRAJECTORY_CHANNELS), dtype=np.float32)


# ---- shape contract --------------------------------------------------------


def test_predict_returns_expected_shape() -> None:
    model = _model()
    out = predict_coefficients_regressor(model, _trajectory_np())
    assert out.shape == (1, DEFAULT_COEFFICIENT_DIM)
    assert out.dtype == np.float32


def test_predict_accepts_torch_tensor_input() -> None:
    model = _model()
    traj = torch.zeros(31, DEFAULT_TRAJECTORY_CHANNELS, dtype=torch.float32)
    out = predict_coefficients_regressor(model, traj)
    assert out.shape == (1, DEFAULT_COEFFICIENT_DIM)


def test_predict_accepts_3d_batch_keeps_all_rows() -> None:
    model = _model()
    traj = np.zeros((4, 31, DEFAULT_TRAJECTORY_CHANNELS), dtype=np.float32)
    out = predict_coefficients_regressor(model, traj)
    assert out.shape == (4, DEFAULT_COEFFICIENT_DIM)


def test_predict_rejects_wrong_channel_count() -> None:
    model = _model()
    bad = np.zeros((16, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="trajectory channels"):
        predict_coefficients_regressor(model, bad)


def test_predict_rejects_wrong_rank() -> None:
    model = _model()
    bad = np.zeros((DEFAULT_TRAJECTORY_CHANNELS,), dtype=np.float32)
    with pytest.raises(ValueError, match="must be"):
        predict_coefficients_regressor(model, bad)


# ---- determinism ----------------------------------------------------------


def test_predict_is_deterministic() -> None:
    model = _model()
    traj = _trajectory_np()
    out_a = predict_coefficients_regressor(model, traj)
    out_b = predict_coefficients_regressor(model, traj)
    np.testing.assert_allclose(out_a, out_b, atol=1e-6)


# ---- physical-bound respect (unit conversions) ---------------------------


def test_predictions_respect_coefficient_scale() -> None:
    """Predictions are hard-clamped to the model's ``coefficient_scale``
    (= per-letter bounds * ``coefficient_scale_factor``)."""
    model = _model()
    traj = (
        np.random.default_rng(0)
        .standard_normal((model.cfg.seq_len, DEFAULT_TRAJECTORY_CHANNELS))
        .astype(np.float32)
    )
    out = predict_coefficients_regressor(model, traj)
    scale = model.coefficient_scale.numpy()
    assert np.all(out <= scale + 1e-3)
    assert np.all(out >= -scale - 1e-3)


def test_predictions_with_unit_factor_respect_physical_bounds() -> None:
    """When ``coefficient_scale_factor=1.0`` predictions are clamped to the
    nominal per-letter bounds (matches the cVAE decoder)."""
    cfg = RegressorConfig(
        embed_dim=32, mlp_hidden=64, n_blocks=2, coefficient_scale_factor=1.0
    )
    model = InverseRegressor(cfg)
    traj = (
        np.random.default_rng(0)
        .standard_normal((model.cfg.seq_len, DEFAULT_TRAJECTORY_CHANNELS))
        .astype(np.float32)
    )
    out = predict_coefficients_regressor(model, traj)
    bounds = build_coefficient_bound_vector(model.cfg.n_joints).numpy()
    assert np.all(out <= bounds + 1e-3)
    assert np.all(out >= -bounds - 1e-3)


# ---- checkpoint round-trip ------------------------------------------------


def test_predict_from_checkpoint_roundtrip(tmp_path: Path) -> None:
    model = _model()
    ckpt = tmp_path / "model.pt"
    torch.save(model.state_payload(), ckpt)

    restored = load_inverse_regressor(ckpt)
    assert isinstance(restored, InverseRegressor)

    out = predict_coefficients_regressor_from_checkpoint(ckpt, _trajectory_np())
    assert out.shape == (1, DEFAULT_COEFFICIENT_DIM)


def test_inverse_regressor_predict_from_checkpoint_missing_file_raises(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError):
        load_inverse_regressor(tmp_path / "does_not_exist.pt")
