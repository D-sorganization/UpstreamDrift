"""Parity tests for the vectorized filter/resample code (issue #8924).

Each test compares the current, vectorized implementation against a
private reference reimplementation of the *old* per-(marker/keypoint,
axis) Python-loop logic that these modules used before vectorization
(one SciPy/numpy call per 1-D slice). Keeping the reference inline means
this suite stays permanent even though the original loop-based
production code was deleted.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import median_filter
from scipy.signal import butter, filtfilt, medfilt, savgol_filter

from src.shared.python.motion_pipeline.preprocessing._filter_pure_python import (
    _butterworth_filter,
    _median_filter,
    _moving_average,
    _savgol_filter,
)
from src.shared.python.motion_pipeline.preprocessing._frame_arrays import (
    vectorized_interp_axes,
)
from src.shared.python.motion_pipeline.preprocessing._resample_pure_python import (
    resample as resample_pure_python,
)
from src.shared.python.motion_pipeline.preprocessing.resample import (
    _rust_interp_axes,
)

from ._local_fixtures import make_marker_trajectory

pytestmark = pytest.mark.unit


def _make_data(num_frames=97, num_points=6, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(num_frames, num_points, 3))


# ---------------------------------------------------------------------------
# Reference (pre-vectorization) loop implementations
# ---------------------------------------------------------------------------


def _ref_butterworth(data, cutoff, order, fps):
    nyquist = fps / 2
    normalized_cutoff = cutoff / nyquist
    if normalized_cutoff >= 1.0:
        normalized_cutoff = 0.99
    b, a = butter(order, normalized_cutoff, btype="low")
    filtered = np.zeros_like(data)
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            filtered[:, i, j] = filtfilt(b, a, data[:, i, j])
    return filtered


def _ref_savgol(data, window_length, polyorder):
    if window_length % 2 == 0:
        window_length += 1
    if window_length <= polyorder:
        window_length = polyorder + 2
    filtered = np.zeros_like(data)
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            filtered[:, i, j] = savgol_filter(data[:, i, j], window_length, polyorder)
    return filtered


def _ref_median(data, kernel_size):
    filtered = np.zeros_like(data)
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            filtered[:, i, j] = medfilt(data[:, i, j], kernel_size=kernel_size)
    return filtered


def _ref_moving_average(data, window):
    if window < 2:
        return data
    filtered = np.zeros_like(data)
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            filtered[:, i, j] = np.convolve(
                data[:, i, j], np.ones(window) / window, mode="same"
            )
    return filtered


def _ref_interp_axes(target_ts, source_ts, data):
    out = np.zeros((target_ts.shape[0], data.shape[1], data.shape[2]))
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            out[:, i, j] = np.interp(target_ts, source_ts, data[:, i, j])
    return out


# ---------------------------------------------------------------------------
# Filter parity
# ---------------------------------------------------------------------------


def test_butterworth_filter_parity():
    data = _make_data(150, 20, seed=1)
    got = _butterworth_filter(data, cutoff=6.0, order=2, fps=100.0)
    want = _ref_butterworth(data, cutoff=6.0, order=2, fps=100.0)
    np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-10)


def test_savgol_filter_parity():
    data = _make_data(150, 20, seed=2)
    got = _savgol_filter(data, window_length=11, polyorder=2)
    want = _ref_savgol(data, window_length=11, polyorder=2)
    np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("kernel_size", [3, 5, 7])
def test_median_filter_parity(kernel_size):
    data = _make_data(80, 15, seed=3)
    got = _median_filter(data, kernel_size=kernel_size)
    want = _ref_median(data, kernel_size=kernel_size)
    np.testing.assert_allclose(got, want, rtol=0, atol=0)


def test_median_filter_ndimage_boundary_mismatch_documented():
    """Sanity check that motivated keeping medfilt: ndimage.median_filter's
    default 'reflect' boundary does NOT match medfilt's zero-padding, so
    it is not a safe drop-in for this vectorization.
    """
    data = _make_data(40, 5, seed=4)
    loop = _ref_median(data, kernel_size=5)
    nd_default = median_filter(data, size=(5, 1, 1))
    assert not np.allclose(loop, nd_default)
    nd_constant = median_filter(data, size=(5, 1, 1), mode="constant", cval=0.0)
    np.testing.assert_allclose(loop, nd_constant, rtol=0, atol=0)


@pytest.mark.parametrize("window", [3, 4, 5, 8])
def test_moving_average_parity(window):
    data = _make_data(90, 12, seed=5)
    got = _moving_average(data, window=window)
    want = _ref_moving_average(data, window=window)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)


def test_moving_average_window_below_two_is_noop():
    data = _make_data(10, 3, seed=6)
    got = _moving_average(data, window=1)
    np.testing.assert_array_equal(got, data)


# ---------------------------------------------------------------------------
# Resample interpolation parity
# ---------------------------------------------------------------------------


def test_vectorized_interp_axes_parity_in_range():
    rng = np.random.default_rng(10)
    source_ts = np.sort(rng.uniform(0, 10, size=25))
    data = _make_data(25, 8, seed=11)
    target_ts = np.sort(rng.uniform(source_ts[0], source_ts[-1], size=40))
    got = vectorized_interp_axes(target_ts, source_ts, data)
    want = _ref_interp_axes(target_ts, source_ts, data)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)


def test_vectorized_interp_axes_parity_out_of_range_clamps():
    rng = np.random.default_rng(12)
    source_ts = np.sort(rng.uniform(0, 10, size=15))
    data = _make_data(15, 5, seed=13)
    target_ts = np.array(
        [
            source_ts[0] - 5.0,
            source_ts[0] - 0.001,
            source_ts[-1] + 0.001,
            source_ts[-1] + 5.0,
        ]
    )
    got = vectorized_interp_axes(target_ts, source_ts, data)
    want = _ref_interp_axes(target_ts, source_ts, data)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)


def test_vectorized_interp_axes_parity_exact_matches():
    rng = np.random.default_rng(14)
    source_ts = np.sort(rng.uniform(0, 10, size=12))
    data = _make_data(12, 4, seed=15)
    # target timestamps exactly equal to source entries (tie-breaking)
    target_ts = source_ts.copy()
    got = vectorized_interp_axes(target_ts, source_ts, data)
    want = _ref_interp_axes(target_ts, source_ts, data)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(got, data, rtol=1e-12, atol=1e-12)


def test_vectorized_interp_axes_parity_duplicate_source_timestamps():
    source_ts = np.array([0.0, 1.0, 1.0, 1.0, 2.0, 3.0])
    data = _make_data(6, 3, seed=16)
    target_ts = np.array([-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0])
    got = vectorized_interp_axes(target_ts, source_ts, data)
    want = _ref_interp_axes(target_ts, source_ts, data)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)


def test_rust_interp_axes_fallback_matches_vectorized_helper():
    """resample.py's pure-Python fallback branch delegates to the shared
    vectorized_interp_axes helper (no Rust wheel available in this test
    env), so it must match np.interp per-slice output directly.
    """
    rng = np.random.default_rng(17)
    source_ts = np.sort(rng.uniform(0, 5, size=20))
    data = _make_data(20, 6, seed=18)
    target_ts = np.sort(rng.uniform(-1, 6, size=30))
    got = _rust_interp_axes(data, source_ts, target_ts)
    want = _ref_interp_axes(target_ts, source_ts, data)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)


def test_resample_pure_python_end_to_end_matches_reference_interp():
    """End-to-end: resampling a MarkerTrajectory through the pure-Python
    module produces coordinates matching a per-slice np.interp reference.
    """
    traj = make_marker_trajectory(
        num_frames=61, marker_names=[f"M{i}" for i in range(9)]
    )
    out = resample_pure_python(traj, target_fps=45.0)

    source_ts = np.array([f.timestamp for f in traj.frames])
    from src.shared.python.motion_pipeline.preprocessing._frame_arrays import (
        markers_to_array,
    )

    source_data = markers_to_array(traj.frames)
    target_ts = np.array([f.timestamp for f in out.frames])
    want = _ref_interp_axes(target_ts, source_ts, source_data)

    got = markers_to_array(out.frames)
    np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-10)
