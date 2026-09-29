"""Contract tests for :class:`InverseRegressor`.

Covers shape/dtype validation, deterministic forward under fixed seed,
gradient flow through the entire model, and bound-respecting output.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.shared.python.motion_matching.inverse import (  # noqa: E402
    DEFAULT_COEFFICIENT_DIM,
    DEFAULT_TRAJECTORY_CHANNELS,
    InverseRegressor,
    RegressorConfig,
    build_coefficient_bound_vector,
    parameter_count,
)

pytestmark = [pytest.mark.unit, pytest.mark.requires_torch]


# ---- helpers ---------------------------------------------------------------


def _trajectory(
    batch: int = 4, T: int | None = None, model: InverseRegressor | None = None
) -> torch.Tensor:
    """Build a zero trajectory with T matching the model's seq_len."""
    if T is None:
        T = model.cfg.seq_len if model is not None else RegressorConfig().seq_len
    return torch.zeros(batch, T, DEFAULT_TRAJECTORY_CHANNELS, dtype=torch.float32)


# ---- shape / dtype contract ------------------------------------------------


def test_default_config_dimensions() -> None:
    cfg = RegressorConfig()
    assert cfg.coefficient_dim == DEFAULT_COEFFICIENT_DIM == 189
    assert cfg.trajectory_channels == DEFAULT_TRAJECTORY_CHANNELS == 12
    assert cfg.embed_dim > 0 and cfg.mlp_hidden > 0
    assert cfg.n_blocks >= 1


def test_forward_returns_expected_shape_and_dtype() -> None:
    model = InverseRegressor()
    out = model(_trajectory())
    assert out.shape == (4, DEFAULT_COEFFICIENT_DIM)
    assert out.dtype == torch.float32


def test_inverse_regressor_model_forward_rejects_wrong_dtype() -> None:
    model = InverseRegressor()
    bad = torch.zeros(
        2, model.cfg.seq_len, DEFAULT_TRAJECTORY_CHANNELS, dtype=torch.float64
    )
    with pytest.raises(TypeError, match="float32"):
        model(bad)


def test_forward_rejects_wrong_channel_count() -> None:
    model = InverseRegressor()
    bad = torch.zeros(2, model.cfg.seq_len, 8, dtype=torch.float32)
    with pytest.raises(ValueError, match="trajectory last-dim"):
        model(bad)


def test_inverse_regressor_model_forward_rejects_wrong_rank() -> None:
    model = InverseRegressor()
    bad = torch.zeros(
        model.cfg.seq_len, DEFAULT_TRAJECTORY_CHANNELS, dtype=torch.float32
    )
    with pytest.raises(ValueError, match="3-D"):
        model(bad)


def test_forward_rejects_non_tensor_trajectory() -> None:
    model = InverseRegressor()
    with pytest.raises(TypeError, match="torch.Tensor"):
        model(
            np.zeros(
                (1, model.cfg.seq_len, DEFAULT_TRAJECTORY_CHANNELS), dtype=np.float32
            )
        )


def test_forward_rejects_wrong_seq_len_when_flatten() -> None:
    """Flatten aggregation expects T == cfg.seq_len."""
    model = InverseRegressor(RegressorConfig(temporal_aggregation="flatten"))
    bad_T = model.cfg.seq_len + 5
    bad = torch.zeros(2, bad_T, DEFAULT_TRAJECTORY_CHANNELS, dtype=torch.float32)
    with pytest.raises(ValueError, match="seq_len"):
        model(bad)


def test_meanmax_aggregation_is_seq_len_agnostic() -> None:
    """Meanmax aggregation accepts any T; flatten requires T == cfg.seq_len."""
    cfg = RegressorConfig(temporal_aggregation="meanmax")
    model = InverseRegressor(cfg)
    # Any T should work with meanmax.
    for T in (16, 31, 64):
        traj = torch.zeros(2, T, DEFAULT_TRAJECTORY_CHANNELS, dtype=torch.float32)
        with torch.no_grad():
            out = model(traj)
        assert out.shape == (2, model.cfg.coefficient_dim)


# ---- determinism / gradient flow ------------------------------------------


def test_forward_deterministic_under_fixed_seed() -> None:
    torch.manual_seed(123)
    model_a = InverseRegressor()
    torch.manual_seed(123)
    model_b = InverseRegressor()
    traj = _trajectory()
    model_a.eval()
    model_b.eval()
    with torch.no_grad():
        out_a = model_a(traj)
        out_b = model_b(traj)
    torch.testing.assert_close(out_a, out_b)


def test_forward_deterministic_in_eval_mode() -> None:
    """Same input, two passes in eval mode -> identical output."""
    torch.manual_seed(0)
    model = InverseRegressor()
    model.eval()
    traj = torch.randn(
        2, model.cfg.seq_len, DEFAULT_TRAJECTORY_CHANNELS, dtype=torch.float32
    )
    with torch.no_grad():
        out_a = model(traj)
        out_b = model(traj)
    torch.testing.assert_close(out_a, out_b)


def test_gradients_flow_through_full_model() -> None:
    """Every learnable parameter should receive a non-zero gradient."""
    torch.manual_seed(0)
    model = InverseRegressor()
    traj = torch.randn(
        4, model.cfg.seq_len, DEFAULT_TRAJECTORY_CHANNELS, dtype=torch.float32
    )
    pred = model(traj)
    target = torch.randn_like(pred)
    loss = ((pred - target) ** 2).mean()
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


# ---- output respects coefficient bounds -----------------------------------


def test_output_within_coefficient_scale() -> None:
    """The tanh-bound clamp uses ``coefficient_scale`` (= bounds * factor).

    The compact dataset's coefficients exceed the nominal per-letter bounds,
    so the model's tanh output is scaled by ``coefficient_scale_factor`` to
    give it enough range. The output must still be hard-clamped to that
    enlarged scale.
    """
    torch.manual_seed(1)
    model = InverseRegressor()
    scale = model.coefficient_scale
    for seed in range(5):
        torch.manual_seed(seed)
        traj = torch.randn(2, model.cfg.seq_len, DEFAULT_TRAJECTORY_CHANNELS) * 5.0
        with torch.no_grad():
            out = model(traj)
        assert torch.all(out <= scale + 1e-3)
        assert torch.all(out >= -scale - 1e-3)


def test_output_with_unit_factor_within_physical_bounds() -> None:
    """When ``coefficient_scale_factor=1.0`` the tanh clamp is the
    nominal per-letter bound vector (matches the cVAE decoder)."""
    cfg = RegressorConfig(coefficient_scale_factor=1.0)
    model = InverseRegressor(cfg)
    bounds = build_coefficient_bound_vector(model.cfg.n_joints)
    torch.manual_seed(0)
    traj = torch.randn(2, model.cfg.seq_len, DEFAULT_TRAJECTORY_CHANNELS) * 5.0
    with torch.no_grad():
        out = model(traj)
    assert torch.all(out <= bounds + 1e-3)
    assert torch.all(out >= -bounds - 1e-3)


# ---- parameter count is in the documented budget --------------------------


def test_parameter_count_in_documented_range() -> None:
    model = InverseRegressor()
    n = parameter_count(model)
    assert 1_000_000 <= n <= 4_000_000, (
        f"parameter count {n:,} outside the 1-4 M budget"
    )


# ---- config validation ----------------------------------------------------


def test_invalid_config_rejected() -> None:
    with pytest.raises(ValueError, match="embed_dim"):
        InverseRegressor(RegressorConfig(embed_dim=0))
    with pytest.raises(ValueError, match="trajectory_channels"):
        InverseRegressor(RegressorConfig(trajectory_channels=0))
    with pytest.raises(ValueError, match="n_joints"):
        InverseRegressor(RegressorConfig(n_joints=0))
    with pytest.raises(ValueError, match="mlp_hidden"):
        InverseRegressor(RegressorConfig(mlp_hidden=0))
    with pytest.raises(ValueError, match="n_blocks"):
        InverseRegressor(RegressorConfig(n_blocks=0))
    with pytest.raises(ValueError, match="dropout"):
        InverseRegressor(RegressorConfig(dropout=1.5))


def test_even_conv_kernel_rejected_for_flatten() -> None:
    """Even ``conv_kernel`` blows up the ``flatten`` path (issue #4269).

    With ``padding = kernel // 2`` SAME padding only preserves
    sequence length for odd kernels; an even kernel under
    ``flatten`` aggregation would silently produce a ``T+1`` conv
    output and crash the linear projection at the first forward pass.
    Validation must reject this up-front.
    """
    with pytest.raises(ValueError, match="conv_kernel must be odd"):
        InverseRegressor(RegressorConfig(temporal_aggregation="flatten", conv_kernel=4))
    # ``flatten_raw`` skips the conv stem entirely (forward path reshapes
    # raw input directly), so kernel parity is irrelevant there. Even
    # kernels must be accepted (issue #4294 — codex review on PR #4292
    # caught the over-broad rejection in the original #4269 fix).
    InverseRegressor(RegressorConfig(temporal_aggregation="flatten_raw", conv_kernel=2))
    InverseRegressor(RegressorConfig(temporal_aggregation="flatten_raw", conv_kernel=4))
    # ``meanmax`` is unaffected because aggregation collapses the time
    # axis before the linear projection, so an even kernel is still OK.
    InverseRegressor(RegressorConfig(temporal_aggregation="meanmax", conv_kernel=4))
