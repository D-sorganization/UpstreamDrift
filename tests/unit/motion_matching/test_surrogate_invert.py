"""Unit tests for ``fit_swing_via_surrogate`` (#029).

Covers the contract laid out in ``docs/issues/backlog/029_invert_via_surrogate.md``:
synthetic recovery, bound projection, restart diversity, history shape,
options validation, gradient-flow sanity, and the slow wall-clock budget.
"""

from __future__ import annotations

import time as _time

import numpy as np
import pytest

pytest.importorskip("torch")
import torch
from src.shared.python.motion_matching.surrogate import (
    FitResult,
    InvertOptions,
    SurrogateConfig,
    SwingSurrogate,
    fit_swing_via_surrogate,
)
from src.shared.python.motion_matching.surrogate._bounds import (
    clamp_,
    default_bounds,
    validate_bounds,
)

from ._fixtures import make_provenance

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tiny_surrogate(n_joints: int = 2, seq_len: int = 16) -> SwingSurrogate:
    """Build a tiny surrogate suitable for fast unit tests."""
    cfg = SurrogateConfig(
        n_joints=n_joints,
        coeffs_per_joint=7,
        seq_len=seq_len,
        hidden_dim=16,
        n_layers=2,
        time_embed_dim=8,
        encoder_layers=2,
    )
    return SwingSurrogate(cfg)


def _target_from_surrogate(
    surrogate: SwingSurrogate, theta_truth: np.ndarray
):  # pragma: no cover - trivial
    """Forward the surrogate at ``theta_truth`` and wrap as a ClubTarget."""
    from src.shared.python.motion_matching.club_target import ClubTarget

    surrogate.eval()
    with torch.no_grad():
        coeffs = torch.from_numpy(theta_truth).float().unsqueeze(0)
        pred = surrogate(coeffs)
    seq_len = surrogate.cfg.seq_len
    time = np.linspace(0.0, 0.3, seq_len)
    butt = pred.butt[0].cpu().numpy().astype(np.float64)
    head = pred.clubhead[0].cpu().numpy().astype(np.float64)
    quat = pred.club_quat[0].cpu().numpy().astype(np.float64)
    # Re-normalize quaternions to ClubTarget's strict 1e-6 tolerance.
    quat = quat / np.linalg.norm(quat, axis=1, keepdims=True)
    # Make sure positions are well within the 5 m sanity bound.
    butt = np.clip(butt, -1.5, 1.5)
    head = np.clip(head, -1.5, 1.5)
    return ClubTarget(
        time=time,
        butt=butt,
        clubhead=head,
        club_quat=quat,
        impact_idx=seq_len // 2,
        source=make_provenance(),
    )


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_invert_options_validation() -> None:
    """``InvertOptions`` rejects nonsensical hyperparameters."""
    with pytest.raises(ValueError, match="n_starts"):
        InvertOptions(n_starts=0)
    with pytest.raises(ValueError, match="n_iters_per_start"):
        InvertOptions(n_iters_per_start=0)
    with pytest.raises(ValueError, match="lr"):
        InvertOptions(lr=0.0)
    with pytest.raises(ValueError, match="schedule"):
        InvertOptions(schedule="weird")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="bound_strategy"):
        InvertOptions(bound_strategy="soft")  # type: ignore[arg-type]


@pytest.mark.unit
def test_clamp_projects_in_place() -> None:
    """``clamp_`` enforces per-dimension bounds and returns the same tensor."""
    coeffs = torch.tensor([[-5.0, 0.0, 5.0]], requires_grad=False)
    low = torch.tensor([-1.0, -1.0, -1.0])
    high = torch.tensor([1.0, 1.0, 1.0])
    out = clamp_(coeffs, low, high)
    assert out is coeffs
    assert torch.allclose(coeffs, torch.tensor([[-1.0, 0.0, 1.0]]))


@pytest.mark.unit
def test_validate_bounds_rejects_low_above_high() -> None:
    with pytest.raises(ValueError, match="bounds_low"):
        validate_bounds(np.array([1.0, 2.0]), np.array([0.0, 3.0]), 2)


# ---------------------------------------------------------------------------
# Inversion behaviour
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_inversion_recovers_known_coeffs_on_synthetic_target() -> None:
    """Adam-on-coefficients drives loss far below the random-start baseline.

    We don't require exact recovery (the surrogate is randomly initialised
    and the forward map is multimodal), only that optimisation makes
    substantial progress relative to the initial guesses.
    """
    torch.manual_seed(0)
    surrogate = _tiny_surrogate(n_joints=2, seq_len=12)
    theta_truth = np.zeros(surrogate.cfg.coeff_dim, dtype=np.float32)
    target = _target_from_surrogate(surrogate, theta_truth)

    opts = InvertOptions(
        n_starts=4,
        n_iters_per_start=80,
        lr=5.0e-2,
        schedule="cosine",
        seed=1,
    )
    result = fit_swing_via_surrogate(target, surrogate, opts)
    assert isinstance(result, FitResult)
    initial_losses = result.history["loss"][:, 0]
    final_losses = result.history["loss"][:, -1]
    # Optimisation should reduce loss substantially on at least the best restart.
    assert result.final_loss < float(initial_losses.min()) * 0.5
    assert float(final_losses.min()) == pytest.approx(result.final_loss)


@pytest.mark.unit
def test_inversion_respects_bounds() -> None:
    """Hard ``clamp`` strategy keeps every coefficient within the bounds."""
    torch.manual_seed(0)
    surrogate = _tiny_surrogate(n_joints=2, seq_len=10)
    theta_truth = np.zeros(surrogate.cfg.coeff_dim, dtype=np.float32)
    target = _target_from_surrogate(surrogate, theta_truth)

    coeff_dim = surrogate.cfg.coeff_dim
    low = -0.25 * np.ones(coeff_dim, dtype=np.float32)
    high = 0.25 * np.ones(coeff_dim, dtype=np.float32)
    opts = InvertOptions(n_starts=3, n_iters_per_start=20, lr=0.5, seed=0)

    result = fit_swing_via_surrogate(target, surrogate, opts, bounds=(low, high))
    assert np.all(result.theta_optimal >= low - 1.0e-6)
    assert np.all(result.theta_optimal <= high + 1.0e-6)


@pytest.mark.unit
def test_n_starts_distinct_initial_thetas() -> None:
    """The ``n_starts`` random restarts must be distinct vectors."""
    torch.manual_seed(0)
    surrogate = _tiny_surrogate(n_joints=2, seq_len=8)
    theta_truth = np.zeros(surrogate.cfg.coeff_dim, dtype=np.float32)
    target = _target_from_surrogate(surrogate, theta_truth)

    opts = InvertOptions(n_starts=5, n_iters_per_start=2, seed=42)
    result = fit_swing_via_surrogate(target, surrogate, opts)
    assert len(result.all_starts) == 5
    stacked = np.stack(result.all_starts, axis=0)
    # All five starts pairwise distinct.
    diffs = np.linalg.norm(stacked[:, None, :] - stacked[None, :, :], axis=-1)
    np.fill_diagonal(diffs, np.inf)
    assert float(diffs.min()) > 0.0


@pytest.mark.unit
def test_history_recorded_per_start() -> None:
    """``history['loss']`` has shape ``(n_starts, n_iters_per_start)``."""
    torch.manual_seed(0)
    surrogate = _tiny_surrogate(n_joints=2, seq_len=8)
    theta_truth = np.zeros(surrogate.cfg.coeff_dim, dtype=np.float32)
    target = _target_from_surrogate(surrogate, theta_truth)

    opts = InvertOptions(n_starts=3, n_iters_per_start=11, seed=0)
    result = fit_swing_via_surrogate(target, surrogate, opts)
    assert result.history["loss"].shape == (3, 11)
    assert np.all(np.isfinite(result.history["loss"]))


@pytest.mark.unit
def test_autograd_flows_to_coefficients() -> None:
    """Surrogate forward must produce non-zero gradients w.r.t. coefficients."""
    torch.manual_seed(0)
    surrogate = _tiny_surrogate(n_joints=2, seq_len=8)
    coeffs = torch.zeros(1, surrogate.cfg.coeff_dim, requires_grad=True)
    pred = surrogate(coeffs)
    loss = (
        pred.butt.pow(2).mean()
        + pred.clubhead.pow(2).mean()
        + pred.club_quat.pow(2).mean()
    )
    loss.backward()
    grad = coeffs.grad
    assert grad is not None
    grad_norm = float(grad.norm())
    assert np.isfinite(grad_norm)
    assert grad_norm > 0.0, "expected non-zero gradient to flow to coefficients"


@pytest.mark.unit
def test_default_bounds_shape() -> None:
    low, high = default_bounds(14)
    assert low.shape == (14,)
    assert high.shape == (14,)
    assert np.all(low < high)


@pytest.mark.slow
def test_fit_completes_under_30s_for_typical_target() -> None:
    """8 starts x 200 iters of the typical surrogate must fit under 30 s."""
    torch.manual_seed(0)
    cfg = SurrogateConfig(
        n_joints=4, coeffs_per_joint=7, seq_len=64, hidden_dim=64, n_layers=2
    )
    surrogate = SwingSurrogate(cfg)
    theta_truth = np.zeros(cfg.coeff_dim, dtype=np.float32)
    target = _target_from_surrogate(surrogate, theta_truth)

    opts = InvertOptions(n_starts=8, n_iters_per_start=200, seed=0)
    t0 = _time.perf_counter()
    result = fit_swing_via_surrogate(target, surrogate, opts)
    elapsed = _time.perf_counter() - t0
    assert elapsed < 30.0, f"inversion took {elapsed:.2f} s (budget 30 s)"
    assert np.isfinite(result.final_loss)
