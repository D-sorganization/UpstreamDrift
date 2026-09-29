"""Target-normaliser tests for the compact swing surrogate (Bug-1 fix).

The surrogate's training loss is computed on per-channel-standardised
trajectories so that the mph-scale clubhead-speed channel doesn't drown
out the metre-scale position channels (root cause of the 17 mm grip-RMSE
floor in the smoke run). These tests cover the new
:class:`TargetNormalizer` machinery.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.shared.python.motion_matching.surrogate.compact import (  # noqa: E402
    SurrogateConfig,
    SwingSurrogate,
    TargetNormalizer,
    predict_trajectory,
)


@pytest.mark.unit
@pytest.mark.requires_torch
def test_standardize_destandardize_round_trip() -> None:
    """``destandardize(standardize(x)) == x`` to float64 precision."""
    rng = np.random.default_rng(0)
    raw = rng.normal(0.0, 5.0, size=(40, 31, 12)).astype(np.float64)
    raw_t = torch.from_numpy(raw).to(torch.float32)
    norm = TargetNormalizer.from_targets(raw_t)
    standardised = norm.standardize(raw_t)
    recovered = norm.destandardize(standardised)
    # float32 round-trip — relax atol slightly from float64 idealisation.
    np.testing.assert_allclose(recovered.numpy(), raw_t.numpy(), atol=1e-4, rtol=1e-4)


@pytest.mark.unit
@pytest.mark.requires_torch
def test_per_channel_mean_zero_std_one_after_standardize() -> None:
    """Standardised targets are zero-mean / unit-std along the channel axis."""
    rng = np.random.default_rng(1)
    raw = torch.from_numpy(
        rng.normal(
            loc=[0, 0, 0, 0, 0, 0, 1.0, 0, 0, 80.0, 0.0, 0.0],
            scale=1.0,
            size=(64, 31, 12),
        ).astype(np.float32)
    )
    norm = TargetNormalizer.from_targets(raw)
    out = norm.standardize(raw).reshape(-1, 12)
    np.testing.assert_allclose(out.mean(dim=0).numpy(), 0.0, atol=1e-5)
    np.testing.assert_allclose(out.std(dim=0, unbiased=False).numpy(), 1.0, atol=1e-4)


@pytest.mark.unit
@pytest.mark.requires_torch
def test_state_dict_round_trip() -> None:
    """``from_state_dict(to_state_dict(x))`` reproduces the same stats."""
    raw = torch.randn(10, 5, 12) * 3.0 + 1.0
    norm = TargetNormalizer.from_targets(raw)
    restored = TargetNormalizer.from_state_dict(norm.to_state_dict())
    np.testing.assert_allclose(restored.mean.numpy(), norm.mean.numpy())
    np.testing.assert_allclose(restored.std.numpy(), norm.std.numpy())


@pytest.mark.unit
@pytest.mark.requires_torch
def test_eps_floor_protects_degenerate_channels() -> None:
    """A constant channel doesn't produce NaN / inf via std=0."""
    raw = torch.zeros(5, 3, 12)  # all channels constant -> std=0
    norm = TargetNormalizer.from_targets(raw, eps=1e-3)
    assert torch.all(norm.std >= 1e-3)
    out = norm.standardize(raw)
    assert torch.all(torch.isfinite(out))


@pytest.mark.unit
@pytest.mark.requires_torch
def test_predict_trajectory_still_returns_physical_units(tmp_path: Path) -> None:
    """The model output (and thus ``predict_trajectory``) is unchanged in
    physical units — the standardiser lives only inside the loss, not in
    the forward path. So velocities are still in m/s, speeds in mph, etc."""
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
        "schema_version": "swing-surrogate-1.1",
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
        "target_normalizer": TargetNormalizer(
            mean=torch.zeros(12), std=torch.ones(12)
        ).to_state_dict(),
    }
    ckpt_path = tmp_path / "checkpoint_best.pt"
    torch.save(payload, ckpt_path)
    model = SwingSurrogate.from_checkpoint(ckpt_path)

    theta = np.zeros(model.cfg.coeff_dim, dtype=np.float32)
    out = predict_trajectory(model, theta)
    # Documented physical-unit channels are present and have the right shapes.
    assert out["r_clubhead"].shape == (1, model.cfg.seq_len, 3)
    assert out["clubhead_speed"].shape == (1, model.cfg.seq_len)
    # Shaft axis is reconstructed as a unit 3-vec — i.e. physical units.
    norms = np.linalg.norm(out["shaft_axis"], axis=-1)
    np.testing.assert_allclose(norms, np.ones_like(norms), atol=1e-5)


@pytest.mark.unit
@pytest.mark.requires_torch
def test_rejects_wrong_channel_dim() -> None:
    """Inputs with wrong trailing dim raise ``ValueError``."""
    norm = TargetNormalizer(mean=torch.zeros(12), std=torch.ones(12))
    with pytest.raises(ValueError, match="num_channels"):
        norm.standardize(torch.randn(4, 5, 7))
    with pytest.raises(ValueError, match="num_channels"):
        norm.destandardize(torch.randn(4, 5, 11))
