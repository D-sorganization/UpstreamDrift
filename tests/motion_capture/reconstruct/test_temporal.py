"""Robust temporal smoother: spikes rejected with reasons, peaks kept, bounds reported."""

from __future__ import annotations

import numpy as np
import pytest

from src.motion_capture.reconstruct.temporal import (
    SmootherOptions,
    check_bounds,
    noise_estimate,
    robust_scale,
    second_difference,
    smooth,
)

pytestmark = pytest.mark.unit

FPS = 120.0
NOISE = 0.005
# True accelerations of the test signal peak near 10 units/s^2; the prior is
# deliberately looser than that so it never fights real motion.
OPTS = SmootherOptions(acceleration_sigma=30.0)


def _swing_like(n: int = 600) -> np.ndarray:
    """A smooth signal with a sharp peak, like a joint speed profile."""
    t = np.arange(n) / FPS
    return np.column_stack(
        [np.sin(2 * np.pi * t / 2.5), 0.4 * np.cos(4 * np.pi * t / 2.5)]
    )


def test_operators_and_scale_estimates() -> None:
    d2 = second_difference(5).toarray()
    assert d2.shape == (3, 5) and d2[0].tolist() == [1, -2, 1, 0, 0]
    assert robust_scale(np.array([0.0, 0.0, 0.0])) == pytest.approx(1e-9)
    assert robust_scale(np.array([-1.0, 0.0, 1.0, np.nan])) == pytest.approx(1.4826)
    rng = np.random.default_rng(0)
    z = _swing_like()[:, 0] + rng.normal(0, NOISE, 600)
    assert 0.6 * NOISE < noise_estimate(z) < 1.6 * NOISE
    with pytest.raises(Exception, match="at least 3"):
        second_difference(2)


def test_clean_signal_is_reproduced_and_nothing_is_rejected() -> None:
    truth = _swing_like()
    noisy = truth + np.random.default_rng(1).normal(0, NOISE, truth.shape)
    result = smooth(noisy, None, FPS, OPTS)
    assert result.rejected == () and result.ok
    assert np.abs(result.values - truth).max() < 3 * NOISE
    assert not np.isnan(result.values).any()
    assert result.uncertainty.shape == truth.shape and (result.uncertainty > 0).all()
    assert (result.uncertainty < NOISE).all()  # the prior tightens the estimate


def test_single_frame_spikes_are_rejected_and_peak_preserved() -> None:
    truth = _swing_like()
    noisy = truth + np.random.default_rng(2).normal(0, NOISE, truth.shape)
    spikes = [(100, 0), (250, 1), (400, 0)]
    for frame, ch in spikes:
        noisy[frame, ch] += 0.8  # a "flashy jump" of 0.8 units in one frame
    result = smooth(noisy, None, FPS, OPTS)
    flagged = {(r.frame, r.channel) for r in result.rejected}
    assert flagged == set(spikes)
    for r in result.rejected:
        assert abs(r.residual) > r.threshold > 0
    for frame, ch in spikes:
        assert result.weights[frame, ch] == 0.0
        assert abs(result.values[frame, ch] - truth[frame, ch]) < 3 * NOISE
    peak_true = truth[:, 0].max()
    assert abs(result.values[:, 0].max() - peak_true) / peak_true < 0.02


def test_low_confidence_and_missing_measurements_lean_on_the_prior() -> None:
    truth = _swing_like()
    noisy = truth + np.random.default_rng(3).normal(0, NOISE, truth.shape)
    conf = np.ones(truth.shape[0])
    noisy[300:310, :] = np.nan  # a 10-frame gap
    noisy[50, 0] += 0.5
    conf[50] = 0.05  # the detector already doubted it
    result = smooth(noisy, conf, FPS, OPTS)
    assert not np.isnan(result.values).any()
    assert np.abs(result.values[300:310] - truth[300:310]).max() < 0.02
    assert np.isnan(result.residuals[305, 0]) and result.weights[305, 0] == 0.0
    assert abs(result.values[50, 0] - truth[50, 0]) < 0.02
    assert (result.uncertainty[305] > result.uncertainty[200]).all()


def test_bounds_are_reported_not_hidden() -> None:
    truth = _swing_like(200)
    opts = SmootherOptions(acceleration_sigma=30.0, max_velocity=1.0)
    violations = check_bounds(truth, FPS, opts)
    assert violations and all(v.kind == "velocity" for v in violations)
    assert all(abs(v.value) > 1.0 for v in violations)
    assert (
        check_bounds(
            truth, FPS, SmootherOptions(acceleration_sigma=30.0, max_velocity=10.0)
        )
        == ()
    )
    result = smooth(truth, None, FPS, opts)
    assert not result.ok and result.violations


def test_contracts() -> None:
    with pytest.raises(Exception, match="acceleration_sigma"):
        SmootherOptions(acceleration_sigma=0)
    with pytest.raises(Exception, match="max_velocity"):
        SmootherOptions(acceleration_sigma=1.0, max_velocity=-1)
    with pytest.raises(Exception, match="T>=3"):
        smooth(np.zeros((2, 1)), None, FPS, OPTS)
    with pytest.raises(Exception, match="confidence"):
        smooth(np.zeros((5, 1)), np.full(5, 2.0), FPS, OPTS)
