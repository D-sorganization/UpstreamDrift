"""Byte-equivalence tests for the Rust-backed C3D / BVH / TRC adapters.

Issue #5213 acceptance: native ``upstream_mocap_io`` outputs must match the
canonical pure-Python parser to within float32 epsilon on the checked-in
golden files. Skipped when the Rust wheel is not installed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.unit

GOLDEN = Path(__file__).resolve().parents[3] / "data" / "motion_pipeline" / "golden"

_rust = pytest.importorskip("upstream_mocap_io")


def _marker_traj_to_array(traj) -> tuple[np.ndarray, list[str]]:
    """Flatten a MarkerTrajectory to (n_frames, n_markers*3); NaN for occlusion."""
    names = traj.frames[0].marker_names
    arr = np.full((len(traj.frames), len(names) * 3), np.nan, dtype=np.float64)
    for fi, frame in enumerate(traj.frames):
        for mi, name in enumerate(names):
            m = frame.markers.get(name)
            if m is None:
                continue
            arr[fi, mi * 3] = m.x
            arr[fi, mi * 3 + 1] = m.y
            arr[fi, mi * 3 + 2] = m.z
    return arr, names


def test_c3d_rust_matches_ezc3d() -> None:
    """Rust C3D parser produces the same MarkerTrajectory as the ezc3d path."""
    ezc3d = pytest.importorskip("ezc3d")
    assert ezc3d  # silence unused warning

    from src.shared.python.motion_pipeline.sources import c3d_adapter as mod
    from src.shared.python.motion_pipeline.sources.c3d_adapter import C3DAdapter

    path = GOLDEN / "sample.c3d"
    if not path.exists():
        pytest.skip(f"golden file missing: {path}")
    adapter = C3DAdapter()

    # Force the Rust path.
    rust_traj = adapter._load_via_rust(path, None)
    # Force the ezc3d path.
    if not mod._HAS_EZC3D:
        pytest.skip("ezc3d not installed")
    py_traj = adapter._load_via_ezc3d(path, None)

    rust_arr, rust_names = _marker_traj_to_array(rust_traj)
    py_arr, py_names = _marker_traj_to_array(py_traj)
    assert rust_names == py_names
    # Float32 epsilon is acceptable; the Rust crate operates in f32 internally.
    np.testing.assert_allclose(rust_arr, py_arr, rtol=0, atol=1e-5, equal_nan=True)


def test_trc_rust_matches_python() -> None:
    """Rust TRC parser produces the same MarkerTrajectory as the Python parser."""
    from src.shared.python.motion_pipeline.sources import trc_adapter as mod
    from src.shared.python.motion_pipeline.sources.trc_adapter import TRCAdapter

    path = GOLDEN / "sample.trc"
    if not path.exists():
        pytest.skip(f"golden file missing: {path}")

    adapter = TRCAdapter()
    rust_traj = adapter._load_via_rust(path, None)
    # Temporarily disable Rust to exercise the pure-Python branch.
    saved = mod._HAS_RUST
    mod._HAS_RUST = False
    try:
        py_traj = adapter.load(path, None)
    finally:
        mod._HAS_RUST = saved

    rust_arr, rust_names = _marker_traj_to_array(rust_traj)
    py_arr, py_names = _marker_traj_to_array(py_traj)
    assert rust_names == py_names
    np.testing.assert_allclose(rust_arr, py_arr, rtol=0, atol=1e-5, equal_nan=True)


def test_bvh_rust_matches_python() -> None:
    """Rust BVH parser produces the same JointTrajectory as the Python parser."""
    from src.shared.python.motion_pipeline.sources import bvh_adapter as mod
    from src.shared.python.motion_pipeline.sources.bvh_adapter import BVHAdapter

    path = GOLDEN / "sample.bvh"
    if not path.exists():
        pytest.skip(f"golden file missing: {path}")

    adapter = BVHAdapter()
    rust_traj = adapter._load_via_rust(path)
    # Force pure-Python branch.
    saved = mod._HAS_RUST
    mod._HAS_RUST = False
    try:
        py_traj = adapter.load(path, None)
    finally:
        mod._HAS_RUST = saved

    assert rust_traj.skeleton.num_dofs == py_traj.skeleton.num_dofs
    assert len(rust_traj.frames) == len(py_traj.frames)

    rust_q = np.array([f.q for f in rust_traj.frames], dtype=np.float64)
    py_q = np.array([f.q for f in py_traj.frames], dtype=np.float64)
    # Rust uses f32 internally; deg->rad happens in Python in both paths.
    # 1e-5 rad ≈ 0.0006° — well below any real BVH angular precision.
    np.testing.assert_allclose(rust_q, py_q, rtol=0, atol=1e-5)
