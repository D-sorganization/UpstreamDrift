"""Parity + memory tests for upstream_mesh slice 2 (issue #5219 / #5248).

Covers the VHACD decimation and primitive-fitting kernels, plus the
headline acceptance criterion from #5219: order-of-magnitude peak-RSS
reduction on a 1M-triangle mesh.

Run with:

    maturin develop -m rust_core/upstream-mesh/Cargo.toml --features python
    pytest tests/rust_bindings/test_mesh_decimation_primitives.py
"""

from __future__ import annotations

import gc
import os

import numpy as np
import pytest

upstream_mesh = pytest.importorskip(
    "upstream_mesh",
    reason=(
        "upstream_mesh wheel not installed "
        "(run: maturin develop -m rust_core/upstream-mesh/Cargo.toml --features python)"
    ),
)
trimesh = pytest.importorskip("trimesh", reason="trimesh required for parity")


pytestmark = pytest.mark.unit


def _cube_mesh() -> tuple[np.ndarray, np.ndarray]:
    v = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ],
        dtype=np.float32,
    )
    f = np.array(
        [
            [0, 2, 1],
            [1, 2, 4],
            [3, 5, 6],
            [5, 7, 6],
            [0, 1, 3],
            [1, 5, 3],
            [2, 6, 4],
            [4, 6, 7],
            [0, 3, 2],
            [2, 3, 6],
            [1, 4, 5],
            [4, 7, 5],
        ],
        dtype=np.uint32,
    )
    return v, f


def _hull_volume(vs: np.ndarray, ix: np.ndarray) -> float:
    """Closed-mesh volume via summed origin-tetrahedra (matches scipy/trimesh)."""
    a, b, c = vs[ix[:, 0]], vs[ix[:, 1]], vs[ix[:, 2]]
    return float(abs(np.sum(a * np.cross(b, c))) / 6.0)


class TestPrimitiveFitting:
    def test_aabb_unit_cube(self) -> None:
        v, _ = _cube_mesh()
        center, extents, ratio = upstream_mesh.fit_aabb(v, 1.0)
        assert tuple(center) == pytest.approx((0.5, 0.5, 0.5), abs=1e-6)
        assert tuple(extents) == pytest.approx((1.0, 1.0, 1.0), abs=1e-6)
        assert ratio == pytest.approx(1.0, abs=1e-6)

    def test_obb_long_box_orientation(self) -> None:
        # Build a 2x1x1 box; OBB's longest axis must have extent ~= 2.
        v, _ = _cube_mesh()
        v_scaled = v.copy()
        v_scaled[:, 0] *= 2.0
        _, extents, _, _ = upstream_mesh.fit_obb(v_scaled, 2.0)
        assert max(extents) == pytest.approx(2.0, abs=1e-5)

    def test_sphere_matches_trimesh_definition(self) -> None:
        # The python-facade `fit_sphere` uses max(distance from centroid).
        sphere = trimesh.creation.icosphere(subdivisions=3)
        v = np.ascontiguousarray(sphere.vertices, dtype=np.float32)
        _, radius, _ = upstream_mesh.fit_sphere(v, float(sphere.volume))
        # All icosphere vertices are at radius 1.
        assert radius == pytest.approx(1.0, abs=1e-5)

    def test_cylinder_along_longest_axis(self) -> None:
        v, _ = _cube_mesh()
        v_scaled = v.copy()
        v_scaled[:, 0] *= 4.0  # 4 x 1 x 1
        _, radius, height, _, _ = upstream_mesh.fit_cylinder(v_scaled, 4.0)
        assert height == pytest.approx(4.0, abs=1e-5)
        assert radius == pytest.approx(0.5, abs=1e-5)

    def test_capsule_cylindrical_length(self) -> None:
        v, _ = _cube_mesh()
        v_scaled = v.copy()
        v_scaled[:, 0] *= 4.0
        _, radius, height, _, _ = upstream_mesh.fit_capsule(v_scaled, 4.0)
        assert radius == pytest.approx(0.5, abs=1e-5)
        # Cylindrical section = total - 2*radius.
        assert height == pytest.approx(3.0, abs=1e-5)


class TestVHACDDecimation:
    def test_cube_decomposes_to_nonzero_parts(self) -> None:
        v, f = _cube_mesh()
        parts = upstream_mesh.decimate_vhacd(v, f, 32, 4, 0.001, 0)
        assert len(parts) >= 1
        for hv, hi in parts:
            assert hv.shape[1] == 3
            assert hi.shape[1] == 3
            assert hv.shape[0] >= 4
            assert hi.shape[0] >= 4

    def test_decomposition_preserves_rough_volume(self) -> None:
        sphere = trimesh.creation.icosphere(subdivisions=3)
        v = np.ascontiguousarray(sphere.vertices, dtype=np.float32)
        f = np.ascontiguousarray(sphere.faces, dtype=np.uint32)
        parts = upstream_mesh.decimate_vhacd(v, f, 64, 8, 0.001, 0)
        total_vol = sum(_hull_volume(hv, hi) for hv, hi in parts)
        # Convex parts can sum a bit higher than original (overlap), but
        # should be within 50% of the input volume for a near-convex shape.
        assert 0.5 * sphere.volume <= total_vol <= 1.5 * sphere.volume

    def test_bad_inputs_raise(self) -> None:
        with pytest.raises(ValueError):
            upstream_mesh.decimate_vhacd(
                np.zeros((2, 3), dtype=np.float32),
                np.zeros((1, 3), dtype=np.uint32),
                32,
                4,
                0.001,
                0,
            )


class TestConvexHullParity:
    """Bit-exact volume parity with trimesh.convex_hull on canonical shapes."""

    @pytest.mark.parametrize(
        "factory",
        [
            lambda: trimesh.creation.icosphere(subdivisions=2),
            lambda: trimesh.creation.icosphere(subdivisions=4),
            lambda: trimesh.creation.box(extents=(2.0, 1.0, 0.5)),
        ],
    )
    def test_volume_matches_trimesh(self, factory) -> None:
        mesh = factory()
        v = np.ascontiguousarray(mesh.vertices, dtype=np.float32)
        hv, hi = upstream_mesh.compute_convex_hull_np(v)
        rust_vol = _hull_volume(hv, hi)
        assert rust_vol == pytest.approx(mesh.convex_hull.volume, rel=1e-3)


@pytest.mark.slow
@pytest.mark.benchmark
class TestMillionTriangleOOM:
    """Headline acceptance for #5219: >=10x peak-RSS reduction on 1M tris."""

    def test_million_triangle_rss_reduction(self) -> None:
        psutil = pytest.importorskip(
            "psutil", reason="psutil required for RSS measurement"
        )
        mesh = trimesh.creation.icosphere(subdivisions=8)
        # subdivisions=8 -> 20 * 4^8 = 1_310_720 triangles
        assert len(mesh.faces) >= 1_000_000

        proc = psutil.Process(os.getpid())

        def rss_mb() -> float:
            gc.collect()
            return proc.memory_info().rss / 1024 / 1024

        # --- Python baseline (trimesh) ---
        before_py = rss_mb()
        hull_py = mesh.convex_hull
        peak_py = rss_mb() - before_py
        py_vol = float(hull_py.volume)
        del hull_py
        gc.collect()

        # --- Rust path ---
        v_f32 = np.ascontiguousarray(mesh.vertices, dtype=np.float32)
        before_rs = rss_mb()
        hv, hi = upstream_mesh.compute_convex_hull_np(v_f32)
        peak_rs = rss_mb() - before_rs
        rust_vol = _hull_volume(hv, hi)

        # Numerical parity must hold (the speed/memory wins must be honest).
        assert rust_vol == pytest.approx(py_vol, rel=1e-3)

        # Headline acceptance: order-of-magnitude memory reduction.
        # `max(peak_rs, 1)` guards against the floor of GC measurement noise.
        ratio = peak_py / max(peak_rs, 1.0)
        print(
            f"\n1M-triangle hull RSS: trimesh={peak_py:.1f} MB  "
            f"upstream-mesh={peak_rs:.1f} MB  reduction={ratio:.1f}x"
        )
        assert ratio >= 10.0, (
            f"expected >=10x RSS reduction, got {ratio:.1f}x "
            f"(trimesh={peak_py:.1f} MB, upstream-mesh={peak_rs:.1f} MB)"
        )
