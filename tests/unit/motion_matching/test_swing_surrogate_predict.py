"""Inference tests for the compact-schema swing surrogate (#4075)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.shared.python.motion_matching.surrogate.compact import (  # noqa: E402
    CoeffNormalizer,
    SurrogateConfig,
    SwingSurrogate,
    predict_trajectory,
)
from src.shared.python.motion_matching.surrogate.compact.predict import (  # noqa: E402
    predict_clubhead_speed_ms,
)


@pytest.fixture
def trained_checkpoint(tmp_path: Path) -> Path:
    """Manufacture a tiny ``checkpoint_best.pt`` so we can test the loader.

    The "training" here is a single forward pass with random init, which
    is sufficient for the round-trip and shape-contract tests.
    """
    cfg = SurrogateConfig(
        n_joints=27,
        coeffs_per_joint=7,
        seq_len=31,
        hidden_dim=32,
        n_residual_blocks=2,
    )
    torch.manual_seed(0)
    model = SwingSurrogate(cfg)
    payload = {
        "schema_version": "swing-surrogate-1.0",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": {},
        "epoch": 1,
        "metrics": {},
        "config": {
            "n_joints": cfg.n_joints,
            "coeffs_per_joint": cfg.coeffs_per_joint,
            "seq_len": cfg.seq_len,
            "hidden_dim": cfg.hidden_dim,
            "n_residual_blocks": cfg.n_residual_blocks,
            "decoder_hidden": cfg.decoder_hidden,
            "dropout": cfg.dropout,
            "coeff_bounds": list(cfg.coeff_bounds),
        },
        "normalizer": {
            "n_joints": cfg.n_joints,
            "coeff_bounds": list(cfg.coeff_bounds),
        },
    }
    ckpt_path = tmp_path / "checkpoint_best.pt"
    torch.save(payload, ckpt_path)
    return ckpt_path


# --------------------------------------------------------------------------- #
# from_checkpoint                                                             #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
@pytest.mark.requires_torch
def test_from_checkpoint_round_trip(trained_checkpoint: Path) -> None:
    """``from_checkpoint`` reconstructs a model with matching weights/cfg."""
    model = SwingSurrogate.from_checkpoint(trained_checkpoint)
    assert isinstance(model, SwingSurrogate)
    assert model.cfg.coeff_dim == 189
    assert hasattr(model, "coeff_normalizer")
    assert isinstance(model.coeff_normalizer, CoeffNormalizer)


@pytest.mark.unit
@pytest.mark.requires_torch
def test_swing_surrogate_predict_from_checkpoint_missing_file_raises(
    tmp_path: Path,
) -> None:
    """A nonexistent path must raise ``FileNotFoundError``."""
    with pytest.raises(FileNotFoundError):
        SwingSurrogate.from_checkpoint(tmp_path / "no.pt")


@pytest.mark.unit
@pytest.mark.requires_torch
def test_from_checkpoint_bad_payload_raises(tmp_path: Path) -> None:
    """A payload missing ``model_state_dict`` raises ``ValueError``."""
    bad = tmp_path / "bad.pt"
    torch.save({"oops": True}, bad)
    with pytest.raises(ValueError, match="model_state_dict"):
        SwingSurrogate.from_checkpoint(bad)


# --------------------------------------------------------------------------- #
# predict_trajectory                                                          #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
@pytest.mark.requires_torch
def test_predict_trajectory_returns_documented_keys(trained_checkpoint: Path) -> None:
    """Output dict contains the documented physical-unit channel arrays."""
    model = SwingSurrogate.from_checkpoint(trained_checkpoint)
    theta = np.zeros(model.cfg.coeff_dim, dtype=np.float64)
    out = predict_trajectory(model, theta)
    assert set(out) >= {
        "r_clubhead",
        "v_clubhead",
        "r_grip",
        "clubhead_speed",
        "shaft_axis",
    }
    assert out["r_clubhead"].shape == (1, model.cfg.seq_len, 3)
    assert out["v_clubhead"].shape == (1, model.cfg.seq_len, 3)
    assert out["r_grip"].shape == (1, model.cfg.seq_len, 3)
    assert out["clubhead_speed"].shape == (1, model.cfg.seq_len)
    assert out["shaft_axis"].shape == (1, model.cfg.seq_len, 3)


@pytest.mark.unit
@pytest.mark.requires_torch
def test_predict_trajectory_batched(trained_checkpoint: Path) -> None:
    """Batch inputs (B>1) propagate cleanly through to (B, T, ·) outputs."""
    model = SwingSurrogate.from_checkpoint(trained_checkpoint)
    rng = np.random.default_rng(0)
    bounds = np.tile([1000, 1000, 500, 500, 100, 100, 25.0], 27)
    theta = (rng.uniform(-1.0, 1.0, size=(4, model.cfg.coeff_dim)) * bounds).astype(
        np.float32
    )
    out = predict_trajectory(model, theta)
    assert out["r_clubhead"].shape == (4, model.cfg.seq_len, 3)
    assert out["clubhead_speed"].shape == (4, model.cfg.seq_len)


@pytest.mark.unit
@pytest.mark.requires_torch
def test_predict_trajectory_shaft_axis_is_unit_norm(
    trained_checkpoint: Path,
) -> None:
    """Shaft-axis output is reconstructed as a unit 3-vec at every timestep."""
    model = SwingSurrogate.from_checkpoint(trained_checkpoint)
    theta = np.zeros(model.cfg.coeff_dim, dtype=np.float32)
    shaft = predict_trajectory(model, theta)["shaft_axis"]
    norms = np.linalg.norm(shaft, axis=-1)
    np.testing.assert_allclose(norms, np.ones_like(norms), atol=1e-5)


@pytest.mark.unit
@pytest.mark.requires_torch
def test_predict_trajectory_rejects_wrong_dim(trained_checkpoint: Path) -> None:
    """Wrong trailing dim raises ``ValueError``."""
    model = SwingSurrogate.from_checkpoint(trained_checkpoint)
    with pytest.raises(ValueError, match="trailing dim"):
        predict_trajectory(model, np.zeros(model.cfg.coeff_dim - 1))


@pytest.mark.unit
@pytest.mark.requires_torch
def test_predict_clubhead_speed_unit_conversion(trained_checkpoint: Path) -> None:
    """``predict_clubhead_speed_ms`` returns mph * (1/2.2369...) m/s."""
    model = SwingSurrogate.from_checkpoint(trained_checkpoint)
    theta = np.zeros(model.cfg.coeff_dim, dtype=np.float32)
    chs_mph = predict_trajectory(model, theta)["clubhead_speed"]
    chs_ms = predict_clubhead_speed_ms(model, theta)
    np.testing.assert_allclose(chs_ms, chs_mph / 2.2369362920544, rtol=1e-5)


@pytest.mark.unit
@pytest.mark.requires_torch
def test_predict_trajectory_missing_normalizer_raises() -> None:
    """A model built directly (no ``coeff_normalizer``) raises AttributeError."""
    model = SwingSurrogate()
    if hasattr(model, "coeff_normalizer"):
        del model.coeff_normalizer
    with pytest.raises(AttributeError, match="coeff_normalizer"):
        predict_trajectory(model, np.zeros(model.cfg.coeff_dim))
