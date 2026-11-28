"""ML-WS-7 (#13784): Inverse regressor kinematics->knots with fail-closed OOD rejection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.shared.python.motion_matching.inverse.masked_proposal import MaskedObservation
from src.shared.python.motion_matching.inverse.ood_rejection import (
    CorpusOodBounds,
    InverseRegressorReceipt,
    OodProposalOutcome,
    fit_inverse_regressor_with_ood,
    propose_knots_with_ood_guard,
)

pytestmark = pytest.mark.unit


def _make_corpus(
    n_samples: int = 36, seq_len: int = 16, channels: int = 6, seed: int = 13784
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a deterministic low-manifold kinematics->knots corpus."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, seq_len, dtype=np.float32)
    cols = [np.sin(2.0 * np.pi * t + 0.4 * c) for c in range(channels)]
    wave = np.stack(cols, axis=-1).astype(np.float32)
    proj = rng.normal(0.0, 1.5, size=(2, 12)).astype(np.float32)
    kin = np.empty((n_samples, seq_len, channels), dtype=np.float32)
    knots = np.empty((n_samples, 6, 2), dtype=np.float32)
    for i in range(n_samples):
        a1, a2 = float(rng.uniform(-1.0, 1.0)), float(rng.uniform(-0.5, 0.5))
        sample = a1 * wave + a2 * np.cos(wave)
        kin[i] = sample + rng.normal(0.0, 0.01, size=sample.shape).astype(np.float32)
        lat = np.array([a1, a2], dtype=np.float32)
        knots[i] = ((lat @ proj) + rng.normal(0.0, 0.02, 12)).reshape(6, 2)
    return kin, knots


def test_training_records_curves_hash_and_beats_mean_baseline(tmp_path: Path) -> None:
    """Training records loss curves, deterministic sha256, and beats mean baseline."""
    kin, knots = _make_corpus()
    model_a, rec_a = fit_inverse_regressor_with_ood(
        kin, knots, epochs=30, seed=7, output_dir=tmp_path / "a"
    )
    _, rec_b = fit_inverse_regressor_with_ood(
        kin, knots, epochs=30, seed=7, output_dir=tmp_path / "b"
    )
    assert isinstance(rec_a, InverseRegressorReceipt)
    assert isinstance(rec_a.ood_bounds, CorpusOodBounds)
    assert len(rec_a.train_loss_curve) == 30 and len(rec_a.val_loss_curve) == 30
    assert rec_a.train_loss_curve[-1] < rec_a.train_loss_curve[0]
    assert rec_a.held_out_rmse < rec_a.mean_baseline_rmse
    assert len(rec_a.checkpoint_sha256) == 64
    assert rec_a.checkpoint_sha256 == rec_b.checkpoint_sha256
    assert rec_a.checkpoint_path is not None and rec_a.checkpoint_path.is_file()

    in_dist = 0.25 * kin[0] + 0.75 * kin[1]
    out = propose_knots_with_ood_guard(
        in_dist, model_a, rec_a.ood_bounds, knot_shape=rec_a.knot_shape
    )
    assert out.status == "proposal" and out.knots is not None
    assert out.knots.shape == (6, 2) and np.all(np.isfinite(out.knots))
    assert out.fallback is None and out.ood_reason is None

    mask = tuple(True for _ in range(in_dist.shape[1]))
    times = np.linspace(0.0, 0.5, in_dist.shape[0])
    obs = MaskedObservation(
        in_dist, mask, times, 0.5, (0.0, 0.0), (0.0, 0.0), "g", "m", "torque", "nc"
    )
    obs_out = propose_knots_with_ood_guard(
        obs, model_a, rec_a.ood_bounds, knot_shape=rec_a.knot_shape
    )
    assert obs_out.status == "proposal" and obs_out.knots is not None
    np.testing.assert_allclose(obs_out.knots, out.knots, atol=1e-6)


def test_ood_rejection_fail_closed_range_and_mahalanobis() -> None:
    """Out-of-range and off-manifold captures return no_proposal without clipping."""
    kin, knots = _make_corpus()
    model, rec = fit_inverse_regressor_with_ood(kin, knots, epochs=15, seed=11)
    bounds = rec.ood_bounds

    r_out = propose_knots_with_ood_guard(
        kin[0] + 25.0, model, bounds, knot_shape=rec.knot_shape
    )
    exp = OodProposalOutcome(
        "no_proposal",
        None,
        "classical_initializer",
        r_out.ood_reason,
        r_out.mahalanobis_distance,
    )
    assert r_out == exp and r_out.ood_reason and "range" in r_out.ood_reason.lower()

    s, c = bounds.seq_len, bounds.trajectory_channels
    lo = np.asarray(bounds.feature_min, dtype=np.float32).reshape((s, c))
    hi = np.asarray(bounds.feature_max, dtype=np.float32).reshape((s, c))
    off = np.where(np.indices((s, c)).sum(axis=0) % 2 == 0, lo * 0.9, hi * 0.9).astype(
        np.float32
    )
    m_out = propose_knots_with_ood_guard(off, model, bounds, knot_shape=rec.knot_shape)
    assert m_out.status == "no_proposal" and m_out.knots is None
    assert m_out.fallback == "classical_initializer"
    assert m_out.ood_reason and "mahalanobis" in m_out.ood_reason.lower()
    assert (
        m_out.mahalanobis_distance is not None
        and m_out.mahalanobis_distance > bounds.max_mahalanobis
    )


def test_non_finite_and_mismatched_dimensions_fail_closed() -> None:
    """NaN/Inf inputs and mismatched feature shapes raise ValueError fail-closed."""
    kin, knots = _make_corpus()
    model, rec = fit_inverse_regressor_with_ood(kin, knots, epochs=5, seed=3)
    bounds = rec.ood_bounds

    for r, c, val in ((0, 0, np.nan), (2, 1, np.inf)):
        bad = kin[0].copy()
        bad[r, c] = val
        with pytest.raises(ValueError, match="finite"):
            propose_knots_with_ood_guard(bad, model, bounds)

    seq, ch = bounds.seq_len, bounds.trajectory_channels
    for bad in (np.zeros((seq, ch + 1)), np.zeros((seq + 2, ch)), np.zeros((ch,))):
        with pytest.raises(ValueError):
            propose_knots_with_ood_guard(bad, model, bounds)

    bad_kin = kin.copy()
    bad_kin[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        fit_inverse_regressor_with_ood(bad_kin, knots)
    with pytest.raises(ValueError, match="sample"):
        fit_inverse_regressor_with_ood(kin[:-1], knots)
