"""``WrenchTrace.filter`` sampling and length preconditions (#8617, B25).

``filter`` derives the sample rate from ``np.mean(np.diff(time))``, which is
meaningless for non-uniform sampling, and ``scipy.signal.filtfilt`` needs more
than ``3 * (order + 1)`` samples -- short traces used to fail with an opaque
scipy padlen error. Both are now explicit, documented preconditions.
"""

from __future__ import annotations

import numpy as np
import pytest

from bunkershot3d.postproc.wrench_trace import WrenchTrace

pytestmark = pytest.mark.unit


def _trace(time: np.ndarray) -> WrenchTrace:
    n = time.size
    force = np.column_stack(
        [np.sin(2 * np.pi * 50 * time), np.zeros(n), np.ones(n) * 3.0]
    )
    torque = np.zeros((n, 3))
    return WrenchTrace(time, force, torque)


def test_uniform_trace_filters_and_preserves_shape() -> None:
    trace = _trace(np.linspace(0.0, 0.1, 200))
    out = trace.filter(cutoff_freq=200.0, order=4)
    assert out.force_world.shape == (200, 3)
    np.testing.assert_allclose(out.torque_world, trace.torque_world, atol=1e-9)


def test_non_uniform_sampling_raises_a_clear_error() -> None:
    time = np.array([0.0, 0.001, 0.002, 0.010, 0.011, 0.012, 0.013, 0.014, 0.02, 0.03])
    with pytest.raises(ValueError, match="uniformly sampled"):
        _trace(time).filter(cutoff_freq=100.0, order=2)


def test_non_monotonic_time_raises() -> None:
    time = np.array([0.0, 0.002, 0.001, 0.003])
    with pytest.raises(ValueError, match="increasing"):
        _trace(time).filter(cutoff_freq=100.0, order=1)


def test_single_sample_raises() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        _trace(np.array([0.0])).filter()


def test_short_trace_reports_the_required_length_not_a_scipy_padlen_error() -> None:
    """order=4 needs > 3*(4+1) = 15 samples for filtfilt's default padlen."""
    trace = _trace(np.linspace(0.0, 0.01, 10))
    with pytest.raises(ValueError, match="at least 16 samples"):
        trace.filter(cutoff_freq=100.0, order=4)


def test_short_trace_is_filterable_at_a_lower_order() -> None:
    trace = _trace(np.linspace(0.0, 0.01, 10))
    out = trace.filter(cutoff_freq=100.0, order=2)
    assert out.force_world.shape == (10, 3)


def test_cutoff_at_or_above_nyquist_returns_an_unfiltered_copy() -> None:
    """Pre-existing behaviour: no filtering, but a copy, and no length check."""
    time = np.linspace(0.0, 0.05, 6)
    trace = _trace(time)
    out = trace.filter(cutoff_freq=1e6, order=4)
    np.testing.assert_array_equal(out.force_world, trace.force_world)
    assert out.force_world is not trace.force_world


def test_resample_output_is_filterable() -> None:
    """resample() is the documented remedy for a non-uniform trace."""
    time = np.array([0.0, 0.001, 0.002, 0.010, 0.011, 0.012, 0.013, 0.014, 0.02, 0.03])
    uniform = np.linspace(0.0, 0.03, 64)
    out = _trace(time).resample(uniform).filter(cutoff_freq=100.0, order=4)
    assert out.force_world.shape == (64, 3)
