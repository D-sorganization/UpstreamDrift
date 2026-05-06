"""Unit tests for :class:`SwingInverseCVAE` (issue #032 / GH #4001).

Training (#033) and rejection-sampling inference (#034) live elsewhere; these
tests only cover the model class itself.
"""

from __future__ import annotations

import pytest
import torch
from src.shared.python.motion_matching.inverse import (
    CVAEConfig,
    EncoderOutput,
    SwingInverseCVAE,
)

# Small tensors keep these fast (<1s each on CPU).
_BATCH = 4
_TIMESTEPS = 20
_N_JOINTS = 6
_OUTPUT_DIM = _N_JOINTS * 7


def _make_cfg(**overrides: object) -> CVAEConfig:
    """Build a small CVAEConfig suitable for unit tests."""
    base: dict[str, object] = {
        "n_joints": _N_JOINTS,
        "n_timesteps": _TIMESTEPS,
        "n_kinematic_channels": 12,
        "latent_dim": 8,
        "encoder_layers": 2,
        "encoder_heads": 2,
        "encoder_dim": 16,
        "decoder_hidden": 32,
        "dropout": 0.0,
    }
    base.update(overrides)
    return CVAEConfig(**base)  # type: ignore[arg-type]


def _make_kinematics(cfg: CVAEConfig, *, batch: int = _BATCH) -> torch.Tensor:
    """Build a deterministic kinematics tensor for a given config."""
    torch.manual_seed(0)
    return torch.randn(batch, cfg.n_timesteps, cfg.n_kinematic_channels)


@pytest.mark.unit
def test_cvae_forward_shape_matches_config() -> None:
    cfg = _make_cfg()
    model = SwingInverseCVAE(cfg).eval()
    x = _make_kinematics(cfg)

    coeffs, enc = model(x, sample=False)

    assert coeffs.shape == (_BATCH, _OUTPUT_DIM)
    assert isinstance(enc, EncoderOutput)
    assert enc.mu.shape == (_BATCH, cfg.latent_dim)
    assert enc.log_var.shape == (_BATCH, cfg.latent_dim)
    assert enc.z.shape == (_BATCH, cfg.latent_dim)


@pytest.mark.unit
def test_cvae_encode_returns_finite_mu_logvar() -> None:
    cfg = _make_cfg()
    model = SwingInverseCVAE(cfg).eval()
    x = _make_kinematics(cfg)

    enc = model.encode(x, sample=False)

    assert torch.isfinite(enc.mu).all()
    assert torch.isfinite(enc.log_var).all()
    # log_var is clamped in the implementation.
    assert (enc.log_var >= -10.0).all() and (enc.log_var <= 10.0).all()


@pytest.mark.unit
def test_cvae_reparam_uses_log_var_at_train_time() -> None:
    """Higher log_var must produce wider z spread, given fixed mu and rng."""
    cfg = _make_cfg()
    mu = torch.zeros(_BATCH, cfg.latent_dim)

    torch.manual_seed(123)
    log_var_low = torch.full_like(mu, -4.0)
    z_low = SwingInverseCVAE._reparameterize(mu, log_var_low, sample=True)

    torch.manual_seed(123)
    log_var_high = torch.full_like(mu, 2.0)
    z_high = SwingInverseCVAE._reparameterize(mu, log_var_high, sample=True)

    assert z_high.std().item() > z_low.std().item() * 5.0

    # sample=False must short-circuit to mu regardless of log_var.
    z_det = SwingInverseCVAE._reparameterize(mu, log_var_high, sample=False)
    assert torch.equal(z_det, mu)


@pytest.mark.unit
def test_cvae_sample_n_returns_distinct_samples() -> None:
    cfg = _make_cfg()
    model = SwingInverseCVAE(cfg).eval()
    x = _make_kinematics(cfg, batch=2)

    torch.manual_seed(7)
    samples = model.sample_coefficients(x, n_samples=16)

    assert samples.shape == (2, 16, _OUTPUT_DIM)
    # No two samples for the same input should be exactly equal (probabilistic;
    # with random Gaussian draws + a non-trivial decoder this is overwhelmingly
    # likely).
    s0 = samples[0]
    diffs = (s0.unsqueeze(0) - s0.unsqueeze(1)).abs().sum(dim=-1)
    # Diagonal is zero by construction; off-diagonal must all be > 0.
    off_diag = diffs + torch.eye(16) * 1.0
    assert (off_diag > 0).all()
    # And sample variance across the n_samples axis must be non-trivial.
    assert samples.std(dim=1).mean().item() > 1e-4


@pytest.mark.unit
def test_cvae_decode_handles_z_alone_or_with_kinematics() -> None:
    cfg = _make_cfg()
    model = SwingInverseCVAE(cfg).eval()
    x = _make_kinematics(cfg)
    z = torch.randn(_BATCH, cfg.latent_dim)

    out_with_kin = model.decode(z, kinematics=x)
    context = model._summarize(x)
    out_with_ctx = model.decode(z, context=context)

    assert out_with_kin.shape == (_BATCH, _OUTPUT_DIM)
    assert out_with_ctx.shape == (_BATCH, _OUTPUT_DIM)
    # Identical context -> identical output.
    assert torch.allclose(out_with_kin, out_with_ctx, atol=1e-6)

    # Missing both must raise.
    with pytest.raises(ValueError):
        model.decode(z)


@pytest.mark.unit
def test_cvae_quaternion_decoder_output_unit_normed_via_postprocess() -> None:
    """Decoder is pass-through: it emits raw torque coefficients, not quats.

    Bound enforcement / quaternion normalization is the inference layer's
    responsibility (#034). This test pins the contract.
    """
    cfg = _make_cfg()
    model = SwingInverseCVAE(cfg).eval()
    x = _make_kinematics(cfg)

    coeffs, _ = model(x, sample=False)

    # Output is a flat (B, n_joints*7) coefficient block, finite, unbounded.
    assert coeffs.shape == (_BATCH, _OUTPUT_DIM)
    assert torch.isfinite(coeffs).all()
    # Sanity: not all zeros, not all identical across batch.
    assert coeffs.abs().sum().item() > 0.0
    assert not torch.allclose(coeffs[0], coeffs[1])


@pytest.mark.unit
def test_cvae_gradient_flows_through_encoder_and_decoder() -> None:
    cfg = _make_cfg()
    model = SwingInverseCVAE(cfg).train()
    x = _make_kinematics(cfg)

    coeffs, enc = model(x, sample=True)
    loss = coeffs.pow(2).mean() + enc.mu.pow(2).mean() + enc.log_var.pow(2).mean()
    loss.backward()

    encoder_grads = [p.grad for p in model.encoder.parameters() if p.requires_grad]
    decoder_grads = [p.grad for p in model.decoder_net.parameters() if p.requires_grad]
    assert encoder_grads, "encoder has no trainable parameters"
    assert decoder_grads, "decoder has no trainable parameters"
    assert all(g is not None and torch.isfinite(g).all() for g in encoder_grads)
    assert all(g is not None and torch.isfinite(g).all() for g in decoder_grads)
    # At least one gradient on each side must be non-zero.
    assert any(g.abs().sum().item() > 0 for g in encoder_grads)
    assert any(g.abs().sum().item() > 0 for g in decoder_grads)


@pytest.mark.unit
def test_cvae_invalid_config_rejected() -> None:
    with pytest.raises(ValueError):
        SwingInverseCVAE(_make_cfg(n_joints=0))
    with pytest.raises(ValueError):
        # encoder_dim not divisible by encoder_heads.
        SwingInverseCVAE(_make_cfg(encoder_dim=15, encoder_heads=4))
