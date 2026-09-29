"""Parity tests for the vectorized Kalman smoother (issue #8923).

``_kalman_filter_python`` in ``filter.py`` used to loop
``for i in range(n_points): for j in range(n_dims):`` with a scalar-update
forward Kalman pass and a scalar-update backward RTS pass inside (750k pure-
Python iterations for a 1,250-frame x 50-marker x 3-axis clip). Because the
covariance is initialized at its steady-state (DARE fixed point), the Kalman
gain and RTS gain are the *same constant* at every timestep and for every
series, so both passes collapse to fixed-coefficient first-order IIR
filters applied to the whole array via two ``scipy.signal.lfilter`` calls.

This test compares the current, vectorized implementation against a private
reference reimplementation of the *old* per-(marker/keypoint) double-loop
logic. Keeping the reference inline means this suite stays permanent even
though the original loop-based production code was deleted (mirrors the
precedent in test_filter_resample_vectorize_parity.py for issue #8924).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.motion_pipeline.preprocessing.filter import (
    _kalman_filter_python,
)

pytestmark = pytest.mark.unit


def _make_data(num_frames=97, num_points=6, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(num_frames, num_points, 3))


# ---------------------------------------------------------------------------
# Reference (pre-vectorization) loop implementation
# ---------------------------------------------------------------------------


def _ref_kalman_filter(
    data: np.ndarray,
    process_noise: float = 0.01,
    measurement_noise: float = 0.1,
) -> np.ndarray:
    if data.size == 0:
        return data

    q = float(process_noise)
    r = float(measurement_noise)
    p_steady = 0.5 * (-q + np.sqrt(q**2 + 4.0 * q * r))

    filtered = np.zeros_like(data)
    n_frames = data.shape[0]

    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            series = data[:, i, j]

            x_fwd = np.empty(n_frames)
            p_fwd = np.empty(n_frames)

            p = p_steady
            x = float(series[0])

            for t in range(n_frames):
                p_pred = p + q
                k_gain = p_pred / (p_pred + r)
                x = x + k_gain * (series[t] - x)
                p = (1.0 - k_gain) * p_pred
                x_fwd[t] = x
                p_fwd[t] = p

            smoothed = np.empty(n_frames)
            smoothed[-1] = x_fwd[-1]
            p_s = p_fwd[-1]

            for t in range(n_frames - 2, -1, -1):
                p_pred = p_fwd[t] + q
                g_s = p_fwd[t] / p_pred
                smoothed[t] = x_fwd[t] + g_s * (smoothed[t + 1] - x_fwd[t])
                p_s = p_fwd[t] + g_s**2 * (p_s - p_pred)

            filtered[:, i, j] = smoothed

    return filtered


# ---------------------------------------------------------------------------
# Parity
# ---------------------------------------------------------------------------


def test_kalman_filter_parity_realistic_shape():
    """Realistic mocap-sized clip: 1250 frames x 50 markers x 3 axes."""
    data = _make_data(num_frames=1250, num_points=50, seed=1)
    got = _kalman_filter_python(data, process_noise=0.01, measurement_noise=0.1)
    want = _ref_kalman_filter(data, process_noise=0.01, measurement_noise=0.1)
    np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize(
    ("process_noise", "measurement_noise"),
    [
        (0.01, 0.1),
        (1.0, 1.0),
        (0.001, 5.0),
        (2.5, 0.01),
    ],
)
def test_kalman_filter_parity_various_noise_params(process_noise, measurement_noise):
    data = _make_data(num_frames=150, num_points=8, seed=2)
    got = _kalman_filter_python(data, process_noise, measurement_noise)
    want = _ref_kalman_filter(data, process_noise, measurement_noise)
    np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("num_frames", [1, 2, 3, 4])
def test_kalman_filter_parity_short_sequences(num_frames):
    data = _make_data(num_frames=num_frames, num_points=3, seed=3)
    got = _kalman_filter_python(data, process_noise=0.05, measurement_noise=0.2)
    want = _ref_kalman_filter(data, process_noise=0.05, measurement_noise=0.2)
    np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-10)


def test_kalman_filter_empty_array_is_noop():
    data = np.zeros((0, 4, 3))
    got = _kalman_filter_python(data)
    assert got.shape == data.shape


def test_kalman_filter_single_frame_returns_input():
    data = _make_data(num_frames=1, num_points=5, seed=4)
    got = _kalman_filter_python(data)
    np.testing.assert_allclose(got, data, rtol=0, atol=0)
