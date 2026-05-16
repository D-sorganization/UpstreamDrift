"""Contract tests for :class:`SwingInverseCVAE` (GH issue #4076).

Covers shape/dtype validation, deterministic forward under fixed seed,
gradient flow through the encoder + decoder, KL non-negativity for the
posterior/prior pair, and bound-respecting decoder output.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.shared.python.motion_matching.inverse import (  # noqa: E402
    COEFFICIENT_LETTER_BOUNDS,
    DEFAULT_COEFFICIENT_DIM,
    DEFAULT_LATENT_DIM,
    DEFAULT_TRAJECTORY_CHANNELS,
    CVAEConfig,
    EncoderOutput,
    SwingInverseCVAE,
    build_coefficient_bound_vector,
    kl_divergence,
    parameter_count,
)

pytestmark = [pytest.mark.unit, pytest.mark.requires_torch]


# ---- helpers ---------------------------------------------------------------


def _trajectory(batch: int = 4, T: int = 32) -> torch.Tensor:
    return torch.zeros(batch, T, DEFAULT_TRAJECTORY_CHANNELS, dtype=torch.float32)


def _coeffs(batch: int = 4) -> torch.Tensor:
    return torch.zeros(batch, DEFAULT_COEFFICIENT_DIM, dtype=torch.float32)


# ---- shape / dtype contract ------------------------------------------------


def test_default_config_dimensions() -> None:
    cfg = CVAEConfig()
    assert cfg.coefficient_dim == DEFAULT_COEFFICIENT_DIM == 189
    assert cfg.latent_dim == DEFAULT_LATENT_DIM == 32
    assert cfg.trajectory_channels == DEFAULT_TRAJECTORY_CHANNELS == 12


def test_forward_returns_expected_shapes() -> None:
    model = SwingInverseCVAE()
    coeff_pred, enc = model(_trajectory(), _coeffs())
    assert coeff_pred.shape == (4, DEFAULT_COEFFICIENT_DIM)
    assert coeff_pred.dtype == torch.float32
    assert isinstance(enc, EncoderOutput)
    for t in (enc.mu_q, enc.logvar_q, enc.mu_p, enc.logvar_p, enc.z):
        assert t.shape == (4, DEFAULT_LATENT_DIM)


def test_swing_inverse_cvae_model_forward_rejects_wrong_dtype() -> None:
    model = SwingInverseCVAE()
    bad = torch.zeros(2, 16, DEFAULT_TRAJECTORY_CHANNELS, dtype=torch.float64)
    with pytest.raises(TypeError, match="float32"):
        model(bad)


def test_forward_rejects_wrong_channel_count() -> None:
    model = SwingInverseCVAE()
    bad = torch.zeros(2, 16, 8, dtype=torch.float32)
    with pytest.raises(ValueError, match="trajectory last-dim"):
        model(bad)


def test_swing_inverse_cvae_model_forward_rejects_wrong_rank() -> None:
    model = SwingInverseCVAE()
    bad = torch.zeros(16, DEFAULT_TRAJECTORY_CHANNELS, dtype=torch.float32)
    with pytest.raises(ValueError, match="3-D"):
        model(bad)


def test_forward_rejects_non_tensor_trajectory() -> None:
    model = SwingInverseCVAE()
    with pytest.raises(TypeError, match="torch.Tensor"):
        model(np.zeros((1, 16, DEFAULT_TRAJECTORY_CHANNELS), dtype=np.float32))


def test_forward_rejects_coeffs_dimension_mismatch() -> None:
    model = SwingInverseCVAE()
    with pytest.raises(ValueError, match="coefficient_dim"):
        model(_trajectory(), torch.zeros(4, 100, dtype=torch.float32))


# ---- determinism / gradient flow ------------------------------------------


def test_forward_deterministic_under_fixed_seed() -> None:
    torch.manual_seed(123)
    model_a = SwingInverseCVAE()
    torch.manual_seed(123)
    model_b = SwingInverseCVAE()
    traj = _trajectory()
    coeffs = _coeffs()

    torch.manual_seed(7)
    coeff_a, _ = model_a(traj, coeffs, sample=True)
    torch.manual_seed(7)
    coeff_b, _ = model_b(traj, coeffs, sample=True)
    torch.testing.assert_close(coeff_a, coeff_b)


def test_gradients_flow_through_full_model() -> None:
    """Both posterior (training) and prior (inference) paths receive gradients
    when their respective forward calls are exercised."""
    torch.manual_seed(0)
    model = SwingInverseCVAE()
    traj = torch.randn(4, 32, DEFAULT_TRAJECTORY_CHANNELS, dtype=torch.float32)
    coeffs = torch.randn(4, DEFAULT_COEFFICIENT_DIM, dtype=torch.float32) * 10.0

    # Posterior path (coeffs given) + KL term to exercise the prior head too.
    coeff_pred, enc = model(traj, coeffs, sample=True)
    target = torch.randn_like(coeff_pred)
    recon = ((coeff_pred - target) ** 2).mean()
    from src.shared.python.motion_matching.inverse import kl_divergence

    kl = kl_divergence(enc.mu_q, enc.logvar_q, enc.mu_p, enc.logvar_p).mean()
    loss = recon + 0.1 * kl
    loss.backward()

    named_grads = [
        (name, p.grad) for name, p in model.named_parameters() if p.requires_grad
    ]
    missing = [n for n, g in named_grads if g is None]
    assert not missing, f"parameters with None grad: {missing}"
    nonzero = sum(1 for _, g in named_grads if float(g.abs().sum()) > 0)
    assert nonzero == len(named_grads), (
        f"only {nonzero}/{len(named_grads)} parameters got non-zero gradients"
    )


# ---- KL non-negativity -----------------------------------------------------


def test_kl_divergence_zero_for_identical_gaussians() -> None:
    mu = torch.randn(3, 32)
    logvar = torch.randn(3, 32) * 0.5
    kl = kl_divergence(mu, logvar, mu, logvar)
    torch.testing.assert_close(kl, torch.zeros_like(kl), atol=1e-6, rtol=0.0)


def test_kl_divergence_non_negative_for_random_pairs() -> None:
    torch.manual_seed(1)
    mu_q = torch.randn(8, 32)
    logvar_q = torch.randn(8, 32) * 0.5
    mu_p = torch.randn(8, 32)
    logvar_p = torch.randn(8, 32) * 0.5
    kl = kl_divergence(mu_q, logvar_q, mu_p, logvar_p)
    assert kl.shape == (8,)
    assert torch.all(kl >= -1e-5), f"KL has negative values: {kl}"


# ---- decoder respects coefficient bounds ----------------------------------


def test_decoder_output_within_physical_bounds() -> None:
    model = SwingInverseCVAE()
    bounds = build_coefficient_bound_vector(model.cfg.n_joints)
    # Multiple seeds so stochastic decoder draws are exercised.
    for seed in range(5):
        torch.manual_seed(seed)
        traj = torch.randn(2, 16, DEFAULT_TRAJECTORY_CHANNELS) * 5.0
        coeff_pred, _ = model(traj, sample=True)
        assert torch.all(coeff_pred <= bounds + 1e-3)
        assert torch.all(coeff_pred >= -bounds - 1e-3)


def test_bound_vector_layout() -> None:
    bounds = build_coefficient_bound_vector(n_joints=27)
    assert bounds.shape == (189,)
    # First 7 entries are the per-letter bounds.
    expected = torch.tensor(COEFFICIENT_LETTER_BOUNDS, dtype=torch.float32)
    torch.testing.assert_close(bounds[:7], expected)
    torch.testing.assert_close(bounds[7:14], expected)


# ---- parameter count is in the documented budget --------------------------


def test_parameter_count_in_documented_range() -> None:
    model = SwingInverseCVAE()
    n = parameter_count(model)
    assert 1_000_000 <= n <= 4_000_000, (
        f"parameter count {n:,} outside the 1-4 M budget"
    )


def test_invalid_config_rejected() -> None:
    with pytest.raises(ValueError, match="latent_dim"):
        SwingInverseCVAE(CVAEConfig(latent_dim=0))
    with pytest.raises(ValueError, match="trajectory_channels"):
        SwingInverseCVAE(CVAEConfig(trajectory_channels=0))
    with pytest.raises(ValueError, match="n_joints"):
        SwingInverseCVAE(CVAEConfig(n_joints=0))


# ---- coefficient_bound_strategy toggle ------------------------------------


def test_default_bound_strategy_matches_spec() -> None:
    """Default ``"spec"`` keeps the PROJECT_SPEC.md §4 nominal bounds."""
    cfg = CVAEConfig()
    assert cfg.coefficient_bound_strategy == "spec"
    assert cfg.coefficient_bound_scale == pytest.approx(1.0)
    model = SwingInverseCVAE(cfg)
    expected = torch.tensor(COEFFICIENT_LETTER_BOUNDS, dtype=torch.float32)
    torch.testing.assert_close(model.coefficient_bounds[:7], expected)


def test_empirical_bound_strategy_widens_bounds() -> None:
    """``"empirical"`` widens every per-letter bound by 50× (matches regressor)."""
    from src.shared.python.motion_matching.inverse.cvae import EMPIRICAL_BOUND_SCALE

    cfg = CVAEConfig(coefficient_bound_strategy="empirical")
    assert cfg.coefficient_bound_scale == pytest.approx(EMPIRICAL_BOUND_SCALE)
    model = SwingInverseCVAE(cfg)
    expected = (
        torch.tensor(COEFFICIENT_LETTER_BOUNDS, dtype=torch.float32)
        * EMPIRICAL_BOUND_SCALE
    )
    torch.testing.assert_close(model.coefficient_bounds[:7], expected)
    # Letter G (index 6 mod 7) under empirical bounds reaches at least 1000 N·m,
    # the user-reported actual coefficient magnitude on the compact dataset.
    assert float(model.coefficient_bounds[6]) >= 1000.0


def test_invalid_bound_strategy_rejected() -> None:
    cfg = CVAEConfig(coefficient_bound_strategy="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="coefficient_bound_strategy"):
        _ = cfg.coefficient_bound_scale


def test_bound_strategy_round_trips_in_checkpoint(tmp_path) -> None:
    """A checkpoint preserves ``coefficient_bound_strategy`` across reload."""
    cfg = CVAEConfig(coefficient_bound_strategy="empirical")
    model = SwingInverseCVAE(cfg)
    ckpt = tmp_path / "ckpt.pt"
    torch.save(model.state_payload(), ckpt)
    loaded = SwingInverseCVAE.from_checkpoint(ckpt)
    assert loaded.cfg.coefficient_bound_strategy == "empirical"
    torch.testing.assert_close(loaded.coefficient_bounds, model.coefficient_bounds)


def test_legacy_checkpoint_without_strategy_defaults_to_spec(tmp_path) -> None:
    """Old checkpoints without the new field load as ``"spec"`` for back-compat."""
    cfg = CVAEConfig()
    model = SwingInverseCVAE(cfg)
    payload = model.state_payload()
    # Simulate a pre-strategy checkpoint by stripping the new field.
    payload["config"].pop("coefficient_bound_strategy", None)
    ckpt = tmp_path / "legacy.pt"
    torch.save(payload, ckpt)
    loaded = SwingInverseCVAE.from_checkpoint(ckpt)
    assert loaded.cfg.coefficient_bound_strategy == "spec"
