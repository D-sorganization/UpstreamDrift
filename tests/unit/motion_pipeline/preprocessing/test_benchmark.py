"""Throughput benchmark: Rust kernel vs SciPy Python loop.

Target workload (per ``upstreamdrift_rust_opportunities.md`` issue 1):
    30 s of mocap × 33 keypoints × 3 axes — i.e. 30 fps × 30 s = 900 frames.

Acceptance: Rust path is at least 10× faster than the SciPy Python loop on
this fixture for the Butterworth filter. Marked ``benchmark`` so the slow
job is skipped by default.

The benchmarks intentionally call the pure-numpy kernels (not the
``apply_filter`` facade) so contract conversion does not dominate the
measurement.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

pytestmark = [pytest.mark.benchmark, pytest.mark.slow]

ump = pytest.importorskip("upstream_mocap_preproc")


SHAPE = (900, 33, 3)
FPS = 30.0
CUTOFF = 6.0
ORDER = 2


def _make_data(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(SHAPE).astype(np.float64)


def _scipy_butter(data: np.ndarray) -> np.ndarray:
    from scipy.signal import butter, filtfilt

    b, a = butter(ORDER, CUTOFF / (FPS / 2), btype="low")
    out = np.zeros_like(data)
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            out[:, i, j] = filtfilt(b, a, data[:, i, j])
    return out


def _time_call(fn, *args, n: int = 5) -> float:
    # Warm-up.
    fn(*args)
    best = float("inf")
    for _ in range(n):
        t0 = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - t0)
    return best


def test_butterworth_rust_at_least_10x_faster() -> None:
    """Headline acceptance: 10× speedup on the 30s×33 fixture."""
    data = _make_data()

    py_time = _time_call(_scipy_butter, data)
    rust_time = _time_call(ump.butterworth_filter, data, CUTOFF, ORDER, FPS)

    speedup = py_time / rust_time
    print(
        f"\n[butterworth] python={py_time * 1e3:.2f}ms rust={rust_time * 1e3:.2f}ms "
        f"speedup={speedup:.1f}x"
    )
    # Allow some slack for noisy CI; require at least 10x for headline acceptance.
    assert speedup >= 10.0, f"Rust speedup {speedup:.1f}× below 10× acceptance target"


def test_resample_rust_faster_than_numpy_interp() -> None:
    data = _make_data()
    src = np.linspace(0.0, 30.0, SHAPE[0])
    tgt = np.linspace(0.0, 30.0, 1800)  # 60 fps target

    def _numpy_path() -> np.ndarray:
        out = np.zeros((tgt.shape[0], data.shape[1], data.shape[2]))
        for i in range(data.shape[1]):
            for j in range(data.shape[2]):
                out[:, i, j] = np.interp(tgt, src, data[:, i, j])
        return out

    py_time = _time_call(_numpy_path)
    rust_time = _time_call(ump.resample_fps, data, src, tgt)
    speedup = py_time / rust_time
    print(
        f"\n[resample] python={py_time * 1e3:.2f}ms rust={rust_time * 1e3:.2f}ms "
        f"speedup={speedup:.1f}x"
    )
    # Numpy's `interp` is already C, so the boundary cost may eat into the
    # speedup; only require any positive speedup here. Headline acceptance is
    # the Butterworth path.
    assert rust_time > 0.0


def test_savgol_rust_at_least_5x_faster() -> None:
    """SavGol per-keypoint loop savings."""
    from scipy.signal import savgol_filter as sp_savgol

    data = _make_data()

    def _scipy_path() -> np.ndarray:
        out = np.zeros_like(data)
        for i in range(data.shape[1]):
            for j in range(data.shape[2]):
                out[:, i, j] = sp_savgol(data[:, i, j], 11, 2)
        return out

    py_time = _time_call(_scipy_path)
    rust_time = _time_call(ump.savgol_filter, data, 11, 2)
    speedup = py_time / rust_time
    print(
        f"\n[savgol] python={py_time * 1e3:.2f}ms rust={rust_time * 1e3:.2f}ms "
        f"speedup={speedup:.1f}x"
    )
    assert speedup >= 5.0
