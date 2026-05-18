"""Unit tests for ``fsp_renderer`` (Phase 3 of the FSP epic, issue #5504).

The renderer is tested against a minimal ``MockViewport`` that captures
``add_mesh`` / ``remove_mesh`` calls.  No GL or Qt context is required, so
these tests are fully headless-safe.

Coverage:
  - One mesh is added per ``render`` call.
  - Color picks match the deviation sign (green/orange/blue).
  - The mesh geometry has 4 vertices and 2 triangular faces.
  - The plane vertices satisfy ``(v - centroid) . normal == 0``.
  - ``clear`` removes the previously-added handle.
  - DbC preconditions are enforced (bad viewport, malformed result).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.visualization.fsp_renderer import (
    FspRenderConfig,
    FspRenderer,
)

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class MockViewport:
    """Minimal viewport that records every add_mesh / remove_mesh call."""

    def __init__(self) -> None:
        self.added: list[tuple] = []
        self.removed: list[object] = []

    def add_mesh(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        *,
        color: tuple,
        alpha: float,
    ) -> object:
        self.added.append((vertices, faces, color, alpha))
        return f"handle-{len(self.added)}"

    def remove_mesh(self, handle: object) -> None:
        self.removed.append(handle)


class _MockPlane:
    def __init__(
        self,
        normal: np.ndarray | None = None,
        centroid: np.ndarray | None = None,
    ) -> None:
        self.normal = normal if normal is not None else np.array([0.0, 0.0, 1.0])
        self.centroid = centroid if centroid is not None else np.array([0.0, 0.0, 1.5])


class _MockFspResult:
    def __init__(
        self,
        normal: np.ndarray | None = None,
        centroid: np.ndarray | None = None,
    ) -> None:
        self.plane = _MockPlane(normal=normal, centroid=centroid)
        self.slope_deg = 0.0
        self.direction_deg = 0.0
        self.clubhead_deviations = np.zeros(50)
        self.hand_deviations = np.zeros(50)


# ---------------------------------------------------------------------------
# render() — happy path & color coding
# ---------------------------------------------------------------------------


def test_render_adds_one_mesh() -> None:
    vp = MockViewport()
    handle = FspRenderer().render(vp, _MockFspResult(), mean_deviation=0.0)
    assert len(vp.added) == 1
    assert handle is not None


def test_render_uses_green_for_on_plane() -> None:
    vp = MockViewport()
    FspRenderer().render(vp, _MockFspResult(), mean_deviation=0.0)
    color = vp.added[0][2]
    # Green dominant: G > R and G > B.
    assert color[1] > color[0]
    assert color[1] > color[2]


def test_render_uses_orange_for_steep() -> None:
    vp = MockViewport()
    FspRenderer().render(vp, _MockFspResult(), mean_deviation=0.5)
    color = vp.added[0][2]
    # Orange ≈ (1.0, 0.5, 0.1): more red than blue.
    assert color[0] > color[2]


def test_render_uses_blue_for_shallow() -> None:
    vp = MockViewport()
    FspRenderer().render(vp, _MockFspResult(), mean_deviation=-0.5)
    color = vp.added[0][2]
    # Blue dominant: B > R.
    assert color[2] > color[0]


def test_render_threshold_treats_tiny_deviation_as_on_plane() -> None:
    """A deviation under the default threshold should still pick green."""
    vp = MockViewport()
    FspRenderer().render(vp, _MockFspResult(), mean_deviation=1e-6)
    color = vp.added[0][2]
    assert color[1] > color[0]
    assert color[1] > color[2]


# ---------------------------------------------------------------------------
# Mesh geometry
# ---------------------------------------------------------------------------


def test_mesh_has_four_vertices_and_two_triangles() -> None:
    vp = MockViewport()
    FspRenderer().render(vp, _MockFspResult())
    verts, faces, _color, _alpha = vp.added[0]
    assert verts.shape == (4, 3)
    assert faces.shape == (2, 3)


def test_mesh_vertices_lie_on_plane() -> None:
    """Every vertex should satisfy (v - centroid) . normal == 0."""
    vp = MockViewport()
    normal = np.array([0.3, -0.4, 0.866])
    normal = normal / np.linalg.norm(normal)
    centroid = np.array([1.0, 2.0, 1.2])
    result = _MockFspResult(normal=normal, centroid=centroid)
    FspRenderer().render(vp, result)
    verts = vp.added[0][0]
    for v in verts:
        residual = float(np.dot(v - centroid, normal))
        assert abs(residual) < 1e-9


def test_mesh_uses_configured_alpha() -> None:
    vp = MockViewport()
    cfg = FspRenderConfig(alpha=0.7)
    FspRenderer(cfg).render(vp, _MockFspResult())
    assert vp.added[0][3] == pytest.approx(0.7)


def test_mesh_uses_configured_plane_size() -> None:
    """Diagonal between opposite corners should be ~ 2 * sqrt(2) * plane_size."""
    vp = MockViewport()
    cfg = FspRenderConfig(plane_size=3.0)
    FspRenderer(cfg).render(vp, _MockFspResult())
    verts = vp.added[0][0]
    diag = float(np.linalg.norm(verts[0] - verts[2]))
    assert diag == pytest.approx(2.0 * np.sqrt(2.0) * 3.0, rel=1e-6)


# ---------------------------------------------------------------------------
# clear()
# ---------------------------------------------------------------------------


def test_clear_removes_mesh() -> None:
    vp = MockViewport()
    renderer = FspRenderer()
    renderer.render(vp, _MockFspResult())
    renderer.clear(vp)
    assert len(vp.removed) == 1


def test_clear_without_render_is_noop() -> None:
    vp = MockViewport()
    FspRenderer().clear(vp)
    assert vp.removed == []


def test_render_twice_replaces_previous_mesh() -> None:
    vp = MockViewport()
    renderer = FspRenderer()
    renderer.render(vp, _MockFspResult())
    renderer.render(vp, _MockFspResult())
    # First handle must have been removed before the second render.
    assert len(vp.removed) == 1
    assert len(vp.added) == 2


# ---------------------------------------------------------------------------
# DbC preconditions
# ---------------------------------------------------------------------------


def test_render_raises_when_viewport_missing_add_mesh() -> None:
    class Bad:
        def remove_mesh(self, h: object) -> None: ...

    with pytest.raises((TypeError, AttributeError)):
        FspRenderer().render(Bad(), _MockFspResult())


def test_render_raises_when_normal_has_wrong_shape() -> None:
    vp = MockViewport()
    bad = _MockFspResult(normal=np.array([0.0, 1.0]))  # 2-D, not 3-D
    with pytest.raises(ValueError):
        FspRenderer().render(vp, bad)
