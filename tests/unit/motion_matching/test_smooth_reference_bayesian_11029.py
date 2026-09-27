"""Unit tests for smooth_reference_bayesian and bit-for-bit pinning (Issue #11029)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import butter, filtfilt

from src.shared.python.core.contracts import ContractViolationError
from src.shared.python.estimation.kinematic_smoother import KinematicSmoothingResult
from src.shared.python.motion_matching.pipeline.reference import (
    smooth_reference,
    smooth_reference_bayesian,
)

pytestmark = pytest.mark.unit


def test_smooth_reference_pinned_bit_for_bit() -> None:
    """Pin smooth_reference output bit-for-bit against butter(4) + filtfilt."""
    rng = np.random.default_rng(20260927)
    N = 120
    nq = 4
    rate_hz = 360.0
    cutoff_hz = 12.0

    t = np.linspace(0, 1, N)[:, None]
    q_base = np.sin(2.0 * np.pi * 3.0 * t) * np.arange(1, nq + 1)
    noise = rng.normal(0.0, 0.05, size=(N, nq))
    q = q_base + noise

    # Run smooth_reference
    smoothed = smooth_reference(q, rate_hz, cutoff_hz)

    # Recompute reference independently
    nyquist = 0.5 * rate_hz
    b, a = butter(4, cutoff_hz / nyquist)
    padlen = min(60, N - 1)
    expected = filtfilt(b, a, q, axis=0, padlen=padlen)

    # Bit-for-bit identical assertion
    np.testing.assert_array_equal(smoothed, expected)


def test_smooth_reference_bayesian_basic() -> None:
    """Verify smooth_reference_bayesian runs and returns KinematicSmoothingResult."""
    rng = np.random.default_rng(42)
    N = 100
    nq = 3
    rate_hz = 120.0

    t = np.linspace(0, 1, N)[:, None]
    q = np.sin(2.0 * np.pi * t) + rng.normal(0, 0.01, size=(N, nq))

    result = smooth_reference_bayesian(
        q,
        rate_hz,
        jerk_psd=5.0,
        measurement_var=1e-4,
    )

    assert isinstance(result, KinematicSmoothingResult)
    assert result.position.shape == (N, nq)
    assert result.velocity.shape == (N, nq)
    assert result.acceleration.shape == (N, nq)
    assert result.position_std.shape == (N, nq)
    assert result.velocity_std.shape == (N, nq)
    assert result.acceleration_std.shape == (N, nq)
    assert np.all(result.position_std > 0.0)
    assert np.all(result.velocity_std > 0.0)
    assert np.all(result.acceleration_std > 0.0)
    assert np.isfinite(result.log_marginal_likelihood)


def test_smooth_reference_bayesian_dbc() -> None:
    """Preconditions on smooth_reference_bayesian reject invalid inputs."""
    valid_q = np.ones((20, 2))

    with pytest.raises((ContractViolationError, ValueError)):
        smooth_reference_bayesian(valid_q, 0.0, jerk_psd=1.0, measurement_var=1.0)

    with pytest.raises((ContractViolationError, ValueError)):
        smooth_reference_bayesian(valid_q, 100.0, jerk_psd=-1.0, measurement_var=1.0)

    with pytest.raises((ContractViolationError, ValueError)):
        smooth_reference_bayesian(valid_q, 100.0, jerk_psd=1.0, measurement_var=0.0)

    with pytest.raises((ContractViolationError, ValueError)):
        smooth_reference_bayesian(np.ones(20), 100.0, jerk_psd=1.0, measurement_var=1.0)

    with pytest.raises((ContractViolationError, ValueError)):
        smooth_reference_bayesian(
            np.ones((2, 2)), 100.0, jerk_psd=1.0, measurement_var=1.0
        )
