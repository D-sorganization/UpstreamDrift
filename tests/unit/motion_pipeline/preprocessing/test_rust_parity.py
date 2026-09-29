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


# ── Gap fill: LINEAR marker dispatch, Rust vs Python (issue #8927) ──────────
#
# The default GapFillStep.strategy is LINEAR (pipeline.py:95); prior to
# #8927, gap_fill.py never called the Rust linear_gap_fill/cubic_gap_fill
# kernels even though they were exported. This compares the two code paths
# through the public gap_fill() -> _gap_fill_markers() dispatch (not the raw
# binding), using the same rtol/atol precedent as the other vectorize/Rust
# parity suites in this directory (test_filter_resample_vectorize_parity.py,
# test_kalman_vectorize_parity.py): rtol=1e-10, atol=1e-10.


def _make_gappy_marker_trajectory(
    seed: int, num_frames: int = 90, num_markers: int = 6
):
    """Build a MarkerTrajectory with well-separated per-marker occlusion windows.

    Each marker's occluded run is isolated in its own time slot (12 frames
    apart) rather than staggered/overlapping with other markers'. This keeps
    every gap's "combined window" (what ``_find_gaps_markers`` sees, unioned
    across all markers) identical to that single marker's own occluded run.

    That distinction matters for parity: the pre-Rust Python path
    (``_linear_interp_markers``) anchors its interpolation ``t`` fraction to
    the *combined* window's start/end, while the Rust kernel fills each
    marker column using its *own* contiguous run. When gaps from different
    markers overlap/touch, those two boundaries diverge and the two paths
    would legitimately compute different (both "correct", but non-identical)
    interpolated values for a pre-existing reason unrelated to #8927 dispatch
    wiring. Isolating each marker's gap sidesteps that so this test verifies
    the actual regression surface: that the Rust dispatch reproduces the
    Python fallback's values and its skip-when-oversized behavior.
    """
    from src.shared.python.motion_pipeline.contracts import (
        Marker,
        MarkerFrame,
        MarkerTrajectory,
    )

    rng = np.random.default_rng(seed)
    marker_names = [f"M{j}" for j in range(num_markers)]
    coords = rng.standard_normal((num_frames, num_markers, 3))

    # Carve one short (fillable) gap per marker, well separated so the
    # windows never merge, plus one long (unfillable) gap on marker 0 in its
    # own isolated slot too.
    occluded = np.zeros((num_frames, num_markers), dtype=bool)
    for j in range(num_markers):
        start = 5 + j * 12
        occluded[start : start + 3, j] = True  # short gap, within max_gap=10
    occluded[70:85, 0] = True  # long gap on marker 0, exceeds max_gap=10

    frames = []
    for i in range(num_frames):
        markers = {
            marker_names[j]: Marker(
                name=marker_names[j],
                x=float(coords[i, j, 0]),
                y=float(coords[i, j, 1]),
                z=float(coords[i, j, 2]),
                occluded=bool(occluded[i, j]),
            )
            for j in range(num_markers)
        }
        frames.append(MarkerFrame(timestamp=i / 30.0, markers=markers, frame_index=i))
    return MarkerTrajectory(id="gappy", frames=frames), marker_names


def test_gap_fill_markers_linear_rust_matches_python_fallback(monkeypatch) -> None:
    """Rust-dispatched LINEAR marker fill matches the pure-Python fallback."""
    from importlib import import_module

    from src.shared.python.motion_pipeline.preprocessing.gap_fill import (
        GapFillStrategy,
        gap_fill as gap_fill_fn,
    )

    # NOTE: `preprocessing/__init__.py` re-exports the `gap_fill` *function*
    # into the package namespace, shadowing the `gap_fill` *submodule*
    # attribute — `from ...preprocessing import gap_fill` would therefore
    # bind the function, not the module. Use import_module for the real
    # module object (same pattern as test_gap_fill.py).
    gap_fill_module = import_module(
        "src.shared.python.motion_pipeline.preprocessing.gap_fill"
    )

    traj, marker_names = _make_gappy_marker_trajectory(seed=11)

    # Rust path: the module already has _RUST_AVAILABLE=True since `ump`
    # imported successfully above (module-level import happens at gap_fill.py
    # import time, before this test runs).
    assert gap_fill_module._RUST_AVAILABLE is True
    rust_out = gap_fill_fn(traj, strategy=GapFillStrategy.LINEAR, max_gap=10)

    # Python fallback path, forced via monkeypatch.
    monkeypatch.setattr(gap_fill_module, "_RUST_AVAILABLE", False)
    python_out = gap_fill_fn(traj, strategy=GapFillStrategy.LINEAR, max_gap=10)

    for i in range(traj.num_frames):
        for name in marker_names:
            r = rust_out.frames[i].markers[name]
            p = python_out.frames[i].markers[name]
            assert r.occluded == p.occluded
            np.testing.assert_allclose(r.x, p.x, rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(r.y, p.y, rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(r.z, p.z, rtol=1e-10, atol=1e-10)

    # Sanity: the long gap on marker 0 stayed occluded on both paths.
    assert rust_out.frames[75].markers["M0"].occluded is True
    assert python_out.frames[75].markers["M0"].occluded is True
    # And a short, fillable gap actually got filled on both paths.
    assert rust_out.frames[6].markers["M0"].occluded is False
    assert python_out.frames[6].markers["M0"].occluded is False
