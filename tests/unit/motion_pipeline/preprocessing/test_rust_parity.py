"""Rust-vs-Python parity tests for motion-pipeline preprocessing kernels.

Verifies the Rust wheel (``upstream_mocap_preproc``) matches SciPy / numpy
reference outputs to within 1e-9 RMSE on representative mocap-shaped data.
Skipped when the wheel is not installed (e.g. clean checkout, CI matrix
without the maturin step).

Acceptance for issue 1 of ``upstreamdrift_rust_opportunities.md``:
    RMSE < 1e-9 vs SciPy reference for every public filter / resample call.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.unit

ump = pytest.importorskip("upstream_mocap_preproc")


SHAPE = (300, 33, 3)  # 10 s at 30 fps × 33 keypoints × xyz — a typical mocap slab


def _make_data(seed: int = 0, shape: tuple[int, int, int] = SHAPE) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape).astype(np.float64)


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


# ── Butterworth ──────────────────────────────────────────────────────────────


def test_butterworth_matches_scipy() -> None:
    from scipy.signal import butter, filtfilt

    data = _make_data(seed=1)
    fps = 60.0
    cutoff = 6.0
    order = 2
    rust = ump.butterworth_filter(data, cutoff, order, fps)

    b, a = butter(order, cutoff / (fps / 2), btype="low")
    ref = np.zeros_like(data)
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            ref[:, i, j] = filtfilt(b, a, data[:, i, j])

    assert _rmse(rust, ref) < 1e-9


# ── Savitzky-Golay ───────────────────────────────────────────────────────────


def test_savgol_matches_scipy() -> None:
    from scipy.signal import savgol_filter as sp_savgol

    data = _make_data(seed=2)
    rust = ump.savgol_filter(data, 11, 2)

    ref = np.zeros_like(data)
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            ref[:, i, j] = sp_savgol(data[:, i, j], 11, 2)
    assert _rmse(rust, ref) < 1e-9


# ── Median ───────────────────────────────────────────────────────────────────


def test_median_filter_matches_scipy() -> None:
    from scipy.signal import medfilt

    data = _make_data(seed=3)
    rust = ump.median_filter(data, 3)
    ref = np.zeros_like(data)
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            ref[:, i, j] = medfilt(data[:, i, j], 3)
    assert _rmse(rust, ref) < 1e-9


# ── Gaussian ─────────────────────────────────────────────────────────────────


def test_gaussian_filter_matches_scipy() -> None:
    from scipy.ndimage import gaussian_filter1d

    data = _make_data(seed=4)
    rust = ump.gaussian_filter(data, 1.5)
    ref = gaussian_filter1d(data, sigma=1.5, axis=0)
    assert _rmse(rust, ref) < 1e-9


# ── Kalman (consistency, not SciPy parity — SciPy has no exact analogue) ─────


def test_kalman_matches_python_reference() -> None:
    """1D random-walk Kalman: compare against an inline Python equivalent."""
    data = _make_data(seed=5)
    q, r = 0.01, 0.1
    rust = ump.kalman_filter(data, q, r)

    ref = np.zeros_like(data)
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            x = data[:, i, j]
            state = x[0]
            p = 1.0
            for t, z in enumerate(x):
                p += q
                k = p / (p + r)
                state = state + k * (z - state)
                p = (1.0 - k) * p
                ref[t, i, j] = state
    assert _rmse(rust, ref) < 1e-9


# ── Resample ─────────────────────────────────────────────────────────────────


def test_resample_matches_numpy_interp() -> None:
    data = _make_data(seed=6, shape=(100, 5, 3))
    src = np.linspace(0.0, 1.0, 100)
    tgt = np.linspace(0.0, 1.0, 250)
    rust = ump.resample_fps(data, src, tgt)
    ref = np.zeros((250, 5, 3))
    for i in range(5):
        for j in range(3):
            ref[:, i, j] = np.interp(tgt, src, data[:, i, j])
    assert _rmse(rust, ref) < 1e-9


# ── Gap fill: linear ─────────────────────────────────────────────────────────


def test_linear_gap_fill_reproduces_python_logic() -> None:
    data = _make_data(seed=7, shape=(40, 4, 3))
    mask = np.zeros((40, 4), dtype=bool)
    mask[5:8, 1] = True  # short gap, fillable
    mask[20:35, 2] = True  # long gap, exceeds max_gap=10
    out, out_mask = ump.linear_gap_fill(data, mask, 10)
    out = np.asarray(out)
    out_mask = np.asarray(out_mask)
    # Verify short gap was filled and long gap was not.
    assert not out_mask[5:8, 1].any()
    assert out_mask[20:35, 2].all()
    # Verify the linear interpolation: known formula.
    v_before = data[4, 1]
    v_after = data[8, 1]
    for k, frame_idx in enumerate(range(5, 8), start=1):
        t = k / (8 - 5 + 1)
        expected = v_before + t * (v_after - v_before)
        np.testing.assert_allclose(out[frame_idx, 1], expected, atol=1e-12)
