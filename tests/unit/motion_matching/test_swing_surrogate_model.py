"""Tests for the compact-schema Option-2 SwingSurrogate model (#4075).

Covers shape/dtype contract, deterministic forward, gradient flow, and
the normalisation invariants on :class:`CoeffNormalizer`.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.shared.python.motion_matching.surrogate.compact.model import (  # noqa: E402
    CHANNEL_SLICES,
    COEFF_BOUNDS,
    CoeffNormalizer,
    SurrogateConfig,
    SwingSurrogate,
    az_pol_to_shaft_axis,
    shaft_axis_to_az_pol,
)

# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #


@pytest.fixture
def cfg() -> SurrogateConfig:
    """Default 27-joint compact-schema config (189 in, 31x12 out)."""
    return SurrogateConfig()


@pytest.fixture
def small_cfg() -> SurrogateConfig:
    """A trimmed config for fast unit tests (still uses real 12-channel head)."""
    return SurrogateConfig(
        n_joints=4,
        coeffs_per_joint=7,
        seq_len=8,
        hidden_dim=32,
        n_residual_blocks=2,
    )


@pytest.fixture
def small_model(small_cfg: SurrogateConfig) -> SwingSurrogate:
    """Deterministic small surrogate."""
    torch.manual_seed(0)
    return SwingSurrogate(small_cfg)


# --------------------------------------------------------------------------- #
# Config validation                                                           #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
@pytest.mark.requires_torch
def test_default_config_is_189_in_31x12_out() -> None:
    """The unparameterised default must match the issue #4075 contract."""
    cfg = SurrogateConfig()
    cfg.validate()
    assert cfg.coeff_dim == 189
    assert cfg.seq_len == 31
    assert cfg.out_channels == 12


@pytest.mark.unit
@pytest.mark.requires_torch
def test_invalid_config_raises() -> None:
    """Every malformed field surfaces as a ValueError with a useful message."""
    with pytest.raises(ValueError, match="n_joints"):
        SurrogateConfig(n_joints=0).validate()
    with pytest.raises(ValueError, match="seq_len"):
        SurrogateConfig(seq_len=1).validate()
    with pytest.raises(ValueError, match="hidden_dim"):
        SurrogateConfig(hidden_dim=0).validate()
    with pytest.raises(ValueError, match="dropout"):
        SurrogateConfig(dropout=1.5).validate()


# --------------------------------------------------------------------------- #
# Forward shape / dtype contract                                              #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
@pytest.mark.requires_torch
def test_forward_shape_dtype_contract(
    small_model: SwingSurrogate, small_cfg: SurrogateConfig
) -> None:
    """``forward`` must emit ``(B, T, 12)`` float32 trajectories."""
    batch = 5
    coeffs = torch.randn(batch, small_cfg.coeff_dim, dtype=torch.float32) * 0.5
    out = small_model(coeffs)
    assert out.shape == (batch, small_cfg.seq_len, 12)
    assert out.dtype == torch.float32


@pytest.mark.unit
@pytest.mark.requires_torch
def test_default_model_real_input_shape() -> None:
    """The default model accepts a real (B, 189) input and emits (B, 31, 12)."""
    torch.manual_seed(0)
    model = SwingSurrogate(SurrogateConfig())
    coeffs = torch.zeros(2, 189, dtype=torch.float32)
    out = model(coeffs)
    assert out.shape == (2, 31, 12)


@pytest.mark.unit
@pytest.mark.requires_torch
def test_swing_surrogate_model_forward_rejects_wrong_dtype(
    small_model: SwingSurrogate,
) -> None:
    """Float64 input must raise TypeError per the documented contract."""
    coeffs = torch.zeros(1, small_model.cfg.coeff_dim, dtype=torch.float64)
    with pytest.raises(TypeError, match="float32"):
        small_model(coeffs)


@pytest.mark.unit
@pytest.mark.requires_torch
def test_forward_rejects_wrong_shape(small_model: SwingSurrogate) -> None:
    """Wrong trailing dim raises ValueError; non-tensor raises TypeError."""
    bad_dim = torch.zeros(1, small_model.cfg.coeff_dim + 1, dtype=torch.float32)
    with pytest.raises(ValueError, match="trailing dim"):
        small_model(bad_dim)
    one_d = torch.zeros(small_model.cfg.coeff_dim, dtype=torch.float32)
    with pytest.raises(ValueError, match="2-D"):
        small_model(one_d)
    with pytest.raises(TypeError, match="torch.Tensor"):
        small_model([0.0] * small_model.cfg.coeff_dim)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Deterministic forward + gradient flow                                       #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
@pytest.mark.requires_torch
def test_forward_is_deterministic_with_fixed_seed(
    small_cfg: SurrogateConfig,
) -> None:
    """Same seed + same input -> bitwise-equal output."""
    torch.manual_seed(0)
    m1 = SwingSurrogate(small_cfg)
    torch.manual_seed(0)
    m2 = SwingSurrogate(small_cfg)
    x = torch.randn(2, small_cfg.coeff_dim, dtype=torch.float32) * 0.5
    out1 = m1(x)
    out2 = m2(x)
    torch.testing.assert_close(out1, out2, atol=0.0, rtol=0.0)


@pytest.mark.unit
@pytest.mark.requires_torch
def test_gradient_flow_to_input(
    small_model: SwingSurrogate, small_cfg: SurrogateConfig
) -> None:
    """Gradients w.r.t. the coefficient input are finite and non-zero."""
    coeffs = torch.randn(
        1, small_cfg.coeff_dim, dtype=torch.float32, requires_grad=True
    )
    out = small_model(coeffs)
    loss = (out**2).mean()
    loss.backward()
    assert coeffs.grad is not None
    assert torch.isfinite(coeffs.grad).all().item()
    assert coeffs.grad.abs().max().item() > 1e-12


@pytest.mark.unit
@pytest.mark.requires_torch
def test_parameter_count_within_documented_range(cfg: SurrogateConfig) -> None:
    """Default config falls in the 500k-2M parameter band from the spec."""
    torch.manual_seed(0)
    model = SwingSurrogate(cfg)
    n = model.parameter_count()
    assert 500_000 <= n <= 2_000_000, f"unexpected param count: {n}"


# --------------------------------------------------------------------------- #
# Normalisation invariants                                                    #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
@pytest.mark.requires_torch
def test_normalizer_round_trips_within_bounds() -> None:
    """``denormalize(normalize(x)) == x`` when ``x`` is inside the bounds."""
    norm = CoeffNormalizer()
    rng = torch.Generator().manual_seed(0)
    scale = torch.tensor(list(COEFF_BOUNDS) * 27)
    raw = (torch.rand(4, 189, generator=rng) * 2 - 1) * scale
    out = norm.denormalize(norm.normalize(raw))
    torch.testing.assert_close(out, raw, atol=1e-5, rtol=1e-5)


@pytest.mark.unit
@pytest.mark.requires_torch
def test_normalizer_clamps_outside_bounds() -> None:
    """Out-of-bounds raw coeffs land at exactly ``±1`` after normalize."""
    norm = CoeffNormalizer()
    raw = torch.full((1, 189), 1e6)
    raw_neg = torch.full((1, 189), -1e6)
    assert torch.all(norm.normalize(raw) == 1.0)
    assert torch.all(norm.normalize(raw_neg) == -1.0)


@pytest.mark.unit
@pytest.mark.requires_torch
def test_normalizer_rejects_bad_shape() -> None:
    """Wrong-dim input raises ValueError with a descriptive message."""
    norm = CoeffNormalizer()
    with pytest.raises(ValueError, match="2-D"):
        norm.normalize(torch.zeros(189))
    with pytest.raises(ValueError, match="trailing dim"):
        norm.normalize(torch.zeros(1, 100))


# --------------------------------------------------------------------------- #
# Channel layout helpers                                                      #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
@pytest.mark.requires_torch
def test_channel_slices_cover_12_channels() -> None:
    """The slice dict must partition the 12-channel output exactly."""
    seen: set[int] = set()
    for lo, hi in CHANNEL_SLICES.values():
        seen.update(range(lo, hi))
    assert seen == set(range(12))


@pytest.mark.unit
@pytest.mark.requires_torch
def test_shaft_axis_az_pol_round_trip() -> None:
    """``az_pol_to_shaft_axis(shaft_axis_to_az_pol(v))`` ≈ unit(v)."""
    rng = torch.Generator().manual_seed(0)
    v = torch.randn(8, 3, generator=rng)
    az_pol = shaft_axis_to_az_pol(v)
    v_unit_back = az_pol_to_shaft_axis(az_pol)
    v_unit = v / v.norm(dim=-1, keepdim=True)
    torch.testing.assert_close(v_unit_back, v_unit, atol=1e-5, rtol=1e-5)


@pytest.mark.unit
@pytest.mark.requires_torch
def test_shaft_axis_helpers_reject_wrong_dim() -> None:
    """Wrong-trailing-dim inputs raise ValueError."""
    with pytest.raises(ValueError, match="trailing dim 3"):
        shaft_axis_to_az_pol(torch.zeros(2, 4))
    with pytest.raises(ValueError, match="trailing dim 2"):
        az_pol_to_shaft_axis(torch.zeros(2, 3))
