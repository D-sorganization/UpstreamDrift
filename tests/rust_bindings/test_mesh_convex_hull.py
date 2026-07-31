"""Parity tests for upstream_mesh.compute_convex_hull vs scipy.

First-slice deliverable for issue #5219. Run with:

    maturin develop -m rust_core/upstream-mesh/Cargo.toml --features python
    pytest tests/rust_bindings/test_mesh_convex_hull.py

Markers:
- ``unit``: deterministic, fast, no live simulation.
"""

from __future__ import annotations

import numpy as np
import pytest

upstream_mesh = pytest.importorskip(
    "upstream_mesh",
    reason=(
        "upstream_mesh wheel not installed "
        "(run: maturin develop -m rust_core/upstream-mesh/Cargo.toml --features python)"
    ),
)
spatial = pytest.importorskip(
    "scipy.spatial",
    reason="scipy required for parity check",
)


pytestmark = pytest.mark.unit


def _random_unit_cube_points(n: int, seed: int) -> np.ndarray:
    """Deterministic uniform points in [-1, 1]^3."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=(n, 3)).astype(np.float32)


class TestConvexHullParity:
    """compute_convex_hull(...) must match scipy within tolerance."""

    def test_unit_cube_corners(self) -> None:
        """Hull of cube corners is the cube itself."""
        corners = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 1.0, 0.0),
            (1.0, 0.0, 1.0),
            (0.0, 1.0, 1.0),
            (1.0, 1.0, 1.0),
        ]
        result = upstream_mesh.compute_convex_hull(corners)
        assert result.num_vertices == 8
        assert result.num_triangles == 12
        assert result.volume == pytest.approx(1.0, rel=1e-5)

    def test_parity_vs_scipy_100_random_points(self) -> None:
        """100 random points: vertex count within ±1 of scipy, volume within 1e-3 rel."""
        pts = _random_unit_cube_points(100, seed=0xCAFEBABE)
        scipy_hull = spatial.ConvexHull(pts)
        rust_result = upstream_mesh.compute_convex_hull(
            [tuple(p) for p in pts.tolist()]
        )
        scipy_vertex_count = len(scipy_hull.vertices)
        rust_vertex_count = rust_result.num_vertices

        # Coplanar / numerical edge cases can shift either implementation
        # by a single vertex. Allow ±1 slack as the issue spec requested.
        assert (
            abs(rust_vertex_count - scipy_vertex_count) <= 1
        ), f"vertex count mismatch: rust={rust_vertex_count} scipy={scipy_vertex_count}"

        assert rust_result.volume == pytest.approx(scipy_hull.volume, rel=1e-3)

    def test_too_few_points_raises(self) -> None:
        """Fewer than 4 input points must raise ValueError."""
        with pytest.raises(ValueError):
            upstream_mesh.compute_convex_hull([(0.0, 0.0, 0.0)] * 3)

    def test_repr_is_informative(self) -> None:
        """repr() should expose vertex/triangle count for debugging."""
        result = upstream_mesh.compute_convex_hull(
            [
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            ]
        )
        text = repr(result)
        assert "ConvexHullResult" in text
        assert "vertices=" in text
        assert "triangles=" in text
