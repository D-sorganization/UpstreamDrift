"""Unit tests for PyQtGLRenderer.

These tests are gated behind ``pytest.importorskip("pyqtgraph")`` so
they skip cleanly in environments without the optional
``body-part-viz-gl`` extra installed.

The tests run headless via ``QT_QPA_PLATFORM=offscreen`` — the
``GLViewWidget`` is instantiated but never shown.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

# Headless Qt platform — must be set before any Qt import.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("pyqtgraph")
pytest.importorskip("pyqtgraph.opengl")

# Skip the whole module if no Qt binding is importable.
_qt_app = pytest.importorskip("pyqtgraph.Qt")

import pyqtgraph as pg  # noqa: E402
import pyqtgraph.opengl as gl  # noqa: E402

from src.shared.python.body_part_viz import (  # noqa: E402
    BindingKind,
    FittedShape,
    MarkerBinding,
    ShapeRenderer,
    ShapeTheme,
)
from src.shared.python.body_part_viz.renderers.pyqtgl_renderer import (  # noqa: E402
    PyQtGLRenderer,
)

# --- Test fixtures / helpers ---------------------------------------


@pytest.fixture(scope="module")
def qapp():
    """Create or reuse a single ``QApplication`` for the test module."""
    app = pg.mkQApp("body_part_viz-pyqtgl-tests")
    yield app
    # Do not call app.quit() — pyqtgraph caches it across tests.


@pytest.fixture
def gl_widget(qapp):
    """Build a ``GLViewWidget`` without showing it."""
    w = gl.GLViewWidget()
    yield w
    w.deleteLater()


def _identity_fitted(shape_id: str, n_frames: int = 4) -> FittedShape:
    binding = MarkerBinding(kind=BindingKind.BETWEEN_TWO, marker_names=("a", "b"))
    rotation = np.broadcast_to(np.eye(3), (n_frames, 3, 3)).copy()
    return FittedShape(
        shape_id=shape_id,
        binding=binding,
        centroid=np.zeros((n_frames, 3)),
        rotation_matrix=rotation,
        scale=np.ones((n_frames, 3)),
        valid_mask=np.ones((n_frames,), dtype=bool),
    )


class _StubMeshShape:
    """Minimal ``BodyPartShape``-conformant mesh stub for tests.

    Builds a triangle fan around a unit circle so we get an actual
    mesh with ``n_verts`` vertices and ``n_verts - 2`` faces.
    """

    def __init__(self, shape_id: str, n_verts: int = 32) -> None:
        if n_verts < 3:
            raise ValueError("n_verts must be >= 3")
        self.shape_id = shape_id
        self.rest_dimensions = (1.0,)
        theta = np.linspace(0.0, 2.0 * np.pi, n_verts, endpoint=False)
        self._verts = np.stack(
            [np.cos(theta), np.sin(theta), np.zeros_like(theta)], axis=1
        ).astype(np.float64)
        # Triangle fan from vertex 0
        faces = np.stack(
            [
                np.zeros(n_verts - 2, dtype=np.int64),
                np.arange(1, n_verts - 1, dtype=np.int64),
                np.arange(2, n_verts, dtype=np.int64),
            ],
            axis=1,
        )
        self._faces = faces

    def vertices_at_rest(self) -> np.ndarray:
        return self._verts.copy()

    def faces(self) -> np.ndarray:
        return self._faces.copy()

    def transform(self, fitted: FittedShape) -> np.ndarray:
        from src.shared.python.body_part_viz.shapes._transform import (
            apply_fitted_to_rest_vertices,
        )

        return apply_fitted_to_rest_vertices(self._verts, fitted)


class _StubLineShape:
    """Minimal line-shape stub (faces is empty)."""

    def __init__(self, shape_id: str = "line") -> None:
        self.shape_id = shape_id
        self.rest_dimensions = (1.0,)

    def vertices_at_rest(self) -> np.ndarray:
        return np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)

    def faces(self) -> np.ndarray:
        return np.zeros((0, 3), dtype=np.int64)

    def transform(self, fitted: FittedShape) -> np.ndarray:
        from src.shared.python.body_part_viz.shapes._transform import (
            apply_fitted_to_rest_vertices,
        )

        return apply_fitted_to_rest_vertices(self.vertices_at_rest(), fitted)


# --- Construction --------------------------------------------------


def test_renderer_implements_shape_renderer_protocol(gl_widget):
    r = PyQtGLRenderer(gl_widget)
    assert isinstance(r, ShapeRenderer)


def test_renderer_rejects_none_widget():
    with pytest.raises(TypeError):
        PyQtGLRenderer(None)


# --- add_shape -----------------------------------------------------


def test_add_three_shapes_registers_items(gl_widget):
    r = PyQtGLRenderer(gl_widget)
    theme = ShapeTheme()
    handles = []
    n_existing = len(gl_widget.items)
    for i in range(3):
        shape = _StubMeshShape(f"m{i}", n_verts=16)
        handles.append(r.add_shape(shape, _identity_fitted(shape.shape_id), theme))
    assert len(set(handles)) == 3
    assert len(gl_widget.items) == n_existing + 3


def test_add_line_shape(gl_widget):
    r = PyQtGLRenderer(gl_widget)
    shape = _StubLineShape("line0")
    handle = r.add_shape(shape, _identity_fitted("line0"), ShapeTheme())
    assert handle.startswith("line0#")


def test_add_shape_rejects_mismatched_shape_id(gl_widget):
    r = PyQtGLRenderer(gl_widget)
    shape = _StubMeshShape("a", n_verts=8)
    fitted = _identity_fitted("b")
    with pytest.raises(ValueError):
        r.add_shape(shape, fitted, ShapeTheme())


def test_add_shape_rejects_non_theme(gl_widget):
    r = PyQtGLRenderer(gl_widget)
    shape = _StubMeshShape("a", n_verts=8)
    with pytest.raises(TypeError):
        r.add_shape(shape, _identity_fitted("a"), object())  # type: ignore[arg-type]


# --- update_frame --------------------------------------------------


def test_update_frame_mesh(gl_widget):
    r = PyQtGLRenderer(gl_widget)
    shape = _StubMeshShape("m", n_verts=12)
    handle = r.add_shape(shape, _identity_fitted("m", n_frames=3), ShapeTheme())
    r.update_frame(handle, 0)
    r.update_frame(handle, 2)


def test_update_frame_line(gl_widget):
    r = PyQtGLRenderer(gl_widget)
    shape = _StubLineShape("l")
    handle = r.add_shape(shape, _identity_fitted("l", n_frames=2), ShapeTheme())
    r.update_frame(handle, 1)


def test_update_frame_invalid_frame_hides_item(gl_widget):
    r = PyQtGLRenderer(gl_widget)
    shape = _StubMeshShape("m", n_verts=8)
    fitted = _identity_fitted("m", n_frames=3)
    # Mark frame 1 invalid
    mask = fitted.valid_mask.copy()
    mask[1] = False
    fitted = FittedShape(
        shape_id=fitted.shape_id,
        binding=fitted.binding,
        centroid=fitted.centroid,
        rotation_matrix=fitted.rotation_matrix,
        scale=fitted.scale,
        valid_mask=mask,
    )
    handle = r.add_shape(shape, fitted, ShapeTheme())
    r.update_frame(handle, 1)
    # Should be hidden after invalid update
    entry_item = r._entries[handle].item
    assert entry_item.visible() is False


def test_update_frame_unknown_handle(gl_widget):
    r = PyQtGLRenderer(gl_widget)
    with pytest.raises(KeyError):
        r.update_frame("nope", 0)


def test_update_frame_out_of_range(gl_widget):
    r = PyQtGLRenderer(gl_widget)
    shape = _StubMeshShape("m", n_verts=8)
    handle = r.add_shape(shape, _identity_fitted("m", n_frames=2), ShapeTheme())
    with pytest.raises(IndexError):
        r.update_frame(handle, 5)


def test_update_frame_rejects_non_int(gl_widget):
    r = PyQtGLRenderer(gl_widget)
    shape = _StubMeshShape("m", n_verts=8)
    handle = r.add_shape(shape, _identity_fitted("m"), ShapeTheme())
    with pytest.raises(TypeError):
        r.update_frame(handle, 1.5)  # type: ignore[arg-type]


# --- set_visible / remove ------------------------------------------


def test_set_visible_toggles(gl_widget):
    r = PyQtGLRenderer(gl_widget)
    shape = _StubMeshShape("m", n_verts=8)
    handle = r.add_shape(shape, _identity_fitted("m"), ShapeTheme())
    r.set_visible(handle, False)
    assert r._entries[handle].item.visible() is False
    r.set_visible(handle, True)
    assert r._entries[handle].item.visible() is True


def test_set_visible_unknown_handle(gl_widget):
    r = PyQtGLRenderer(gl_widget)
    with pytest.raises(KeyError):
        r.set_visible("nope", True)


def test_set_visible_rejects_non_bool(gl_widget):
    r = PyQtGLRenderer(gl_widget)
    shape = _StubMeshShape("m", n_verts=8)
    handle = r.add_shape(shape, _identity_fitted("m"), ShapeTheme())
    with pytest.raises(TypeError):
        r.set_visible(handle, "yes")  # type: ignore[arg-type]


def test_remove_drops_handle(gl_widget):
    r = PyQtGLRenderer(gl_widget)
    shape = _StubMeshShape("m", n_verts=8)
    handle = r.add_shape(shape, _identity_fitted("m"), ShapeTheme())
    n_before = len(gl_widget.items)
    r.remove(handle)
    assert handle not in r._entries
    assert len(gl_widget.items) == n_before - 1
    with pytest.raises(KeyError):
        r.remove(handle)


# --- Performance ----------------------------------------------------


@pytest.mark.benchmark
def test_update_frame_performance_5000_verts(gl_widget):
    """100 update_frame calls on a 5000-vertex mesh — assert mean <= 16 ms.

    This guards the EPIC's 60 fps target on the workload that
    matplotlib cannot comfortably hit.
    """
    n_verts = 5000
    shape = _StubMeshShape("perf", n_verts=n_verts)
    fitted = _identity_fitted("perf", n_frames=100)
    r = PyQtGLRenderer(gl_widget)
    handle = r.add_shape(shape, fitted, ShapeTheme())

    # Warm up
    for i in range(5):
        r.update_frame(handle, i)

    n_iters = 100
    t0 = time.perf_counter()
    for i in range(n_iters):
        r.update_frame(handle, i % fitted.centroid.shape[0])
    elapsed = time.perf_counter() - t0
    mean_ms = (elapsed / n_iters) * 1000.0
    # 16 ms == 60 fps. Use a generous CI-friendly margin (1.5x) since
    # headless GL on shared runners has variable wall-clock noise.
    assert mean_ms <= 24.0, f"mean per-frame update {mean_ms:.2f} ms exceeds 24 ms"
