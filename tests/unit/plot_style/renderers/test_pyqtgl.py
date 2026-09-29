"""Unit tests for ``PyQtGLMarkerRenderer``.

These tests are gated behind ``pytest.importorskip("pyqtgraph.opengl")``
so they skip cleanly in environments without the optional
``body-part-viz-gl`` extra installed.

The tests run headless via ``QT_QPA_PLATFORM=offscreen`` — the
``GLViewWidget`` is instantiated but never shown.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

# Headless Qt platform — must be set before any Qt import.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("pyqtgraph")
pytest.importorskip("pyqtgraph.opengl")
pytest.importorskip("pyqtgraph.Qt")

import pyqtgraph as pg  # noqa: E402
import pyqtgraph.opengl as gl  # noqa: E402

from src.shared.python.plot_style import (  # noqa: E402
    CustomMeshSpec,
    MarkerRenderer,
    MarkerShape,
    MarkerStyle,
    StaticColor,
)
from src.shared.python.plot_style.renderers.pyqtgl import (  # noqa: E402
    PyQtGLMarkerRenderer,
)

# --- Test fixtures / helpers ---------------------------------------


@pytest.fixture(scope="module")
def qapp():
    """Create or reuse a single ``QApplication`` for the test module."""
    app = pg.mkQApp("plot_style-pyqtgl-tests")
    yield app
    # Do not call app.quit() — pyqtgraph caches it across tests.


@pytest.fixture
def gl_widget(qapp):
    """Build a ``GLViewWidget`` without showing it."""
    w = gl.GLViewWidget()
    yield w
    w.deleteLater()


def _random_positions(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=(n, 3)).astype(np.float64)


def _custom_tetra_spec(name: str = "tetra") -> CustomMeshSpec:
    """Tetrahedron CustomMeshSpec — 4 verts, 4 faces."""
    verts = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [-0.5, 0.866, 0.0],
            [-0.5, -0.866, 0.0],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]],
        dtype=np.int64,
    )
    return CustomMeshSpec(name=name, vertices=verts, faces=faces)


# --- Construction --------------------------------------------------


def test_renderer_implements_marker_renderer_protocol(gl_widget):
    r = PyQtGLMarkerRenderer(gl_widget)
    assert isinstance(r, MarkerRenderer)


def test_renderer_rejects_none_widget():
    with pytest.raises(TypeError):
        PyQtGLMarkerRenderer(None)


def test_lazy_attribute_access_from_renderers_package():
    """``from plot_style.renderers import PyQtGLMarkerRenderer`` works lazily."""
    from src.shared.python.plot_style import renderers

    cls = renderers.PyQtGLMarkerRenderer
    assert cls is PyQtGLMarkerRenderer


def test_lazy_attribute_access_unknown_attr_raises():
    from src.shared.python.plot_style import renderers

    with pytest.raises(AttributeError):
        _ = renderers.NotAThing  # type: ignore[attr-defined]


def test_top_level_lazy_attribute_access():
    """``plot_style.PyQtGLMarkerRenderer`` works via top-level ``__getattr__``."""
    import src.shared.python.plot_style as plot_style

    cls = plot_style.PyQtGLMarkerRenderer
    assert cls is PyQtGLMarkerRenderer


def test_top_level_lazy_unknown_attr_raises():
    import src.shared.python.plot_style as plot_style

    with pytest.raises(AttributeError):
        _ = plot_style.NotAThing  # type: ignore[attr-defined]


# --- draw() smoke --------------------------------------------------


def test_draw_sphere_smoke_n100(gl_widget):
    """Render N=100 sphere markers via the convenience ``draw`` API."""
    r = PyQtGLMarkerRenderer(gl_widget)
    positions = _random_positions(100)
    style = MarkerStyle(
        shape=MarkerShape.SPHERE,
        size_px=8.0,
        fill_color=StaticColor("#ff0000"),
    )
    n_existing = len(gl_widget.items)
    item = r.draw(gl_widget, positions, style)
    # SPHERE goes through the scatter path — single item.
    assert not isinstance(item, list)
    assert len(gl_widget.items) == n_existing + 1
    # Size propagated.
    assert float(item.size) == 8.0


def test_draw_rejects_wrong_view(gl_widget, qapp):
    r = PyQtGLMarkerRenderer(gl_widget)
    other = gl.GLViewWidget()
    try:
        with pytest.raises(ValueError):
            r.draw(other, _random_positions(3), MarkerStyle())
    finally:
        other.deleteLater()


def test_draw_rejects_non_style(gl_widget):
    r = PyQtGLMarkerRenderer(gl_widget)
    with pytest.raises(TypeError):
        r.draw(gl_widget, _random_positions(3), object())  # type: ignore[arg-type]


def test_draw_rejects_bad_position_shape(gl_widget):
    r = PyQtGLMarkerRenderer(gl_widget)
    with pytest.raises(ValueError):
        r.draw(gl_widget, np.zeros((3, 2)), MarkerStyle())


def test_draw_with_explicit_colors(gl_widget):
    r = PyQtGLMarkerRenderer(gl_widget)
    positions = _random_positions(5)
    colors = np.tile(np.array([0.1, 0.2, 0.3, 1.0]), (5, 1))
    item = r.draw(gl_widget, positions, MarkerStyle(), colors)
    assert item is not None


def test_draw_rejects_bad_colors_shape(gl_widget):
    r = PyQtGLMarkerRenderer(gl_widget)
    with pytest.raises(ValueError):
        r.draw(
            gl_widget,
            _random_positions(3),
            MarkerStyle(),
            np.zeros((3, 3)),
        )


def test_draw_rejects_color_length_mismatch(gl_widget):
    r = PyQtGLMarkerRenderer(gl_widget)
    with pytest.raises(ValueError):
        r.draw(
            gl_widget,
            _random_positions(3),
            MarkerStyle(),
            np.zeros((4, 4)),
        )


# --- One test per built-in shape kind ------------------------------


@pytest.mark.parametrize(
    "shape",
    [
        MarkerShape.SPHERE,
        MarkerShape.POINT,
        MarkerShape.PLUS,
    ],
)
def test_scatter_path_shapes(gl_widget, shape):
    """Scatter-path shapes produce a single GLScatterPlotItem."""
    r = PyQtGLMarkerRenderer(gl_widget)
    style = MarkerStyle(shape=shape, size_px=5.0)
    item = r.draw(gl_widget, _random_positions(10), style)
    assert isinstance(item, gl.GLScatterPlotItem)


@pytest.mark.parametrize(
    "shape",
    [
        MarkerShape.CUBE,
        MarkerShape.CROSS,
        MarkerShape.STAR,
        MarkerShape.DIAMOND,
    ],
)
def test_mesh_path_shapes(gl_widget, shape):
    """Mesh-path shapes produce one GLMeshItem per marker."""
    r = PyQtGLMarkerRenderer(gl_widget)
    style = MarkerStyle(shape=shape, size_px=5.0)
    n = 4
    items = r.draw(gl_widget, _random_positions(n), style)
    assert isinstance(items, list)
    assert len(items) == n
    assert all(isinstance(it, gl.GLMeshItem) for it in items)


# --- Custom mesh path ----------------------------------------------


def test_custom_mesh_path_renders_mesh_items(gl_widget):
    spec = _custom_tetra_spec()
    style = MarkerStyle(shape=MarkerShape.CUSTOM_MESH, custom_mesh=spec)
    r = PyQtGLMarkerRenderer(gl_widget)
    items = r.draw(gl_widget, _random_positions(3), style)
    assert isinstance(items, list)
    assert len(items) == 3
    assert all(isinstance(it, gl.GLMeshItem) for it in items)


# --- add_markers / update_frame / update_style / set_visible / remove


def test_add_markers_returns_handle_and_registers_item(gl_widget):
    r = PyQtGLMarkerRenderer(gl_widget)
    # (T=2, M=4, 3) trajectory.
    positions = np.zeros((2, 4, 3))
    positions[1] = 1.0
    style = MarkerStyle(shape=MarkerShape.SPHERE)
    n_existing = len(gl_widget.items)
    handle = r.add_markers(positions, style, label="markers")
    assert handle.startswith("markers#")
    # Scatter path → 1 item.
    assert len(gl_widget.items) == n_existing + 1


def test_add_markers_default_label_uses_shape(gl_widget):
    r = PyQtGLMarkerRenderer(gl_widget)
    handle = r.add_markers(np.zeros((1, 2, 3)), MarkerStyle())
    assert handle.startswith("sphere#")


def test_add_markers_2d_positions_treated_as_single_frame(gl_widget):
    r = PyQtGLMarkerRenderer(gl_widget)
    handle = r.add_markers(np.zeros((5, 3)), MarkerStyle())
    entry = r._entries[handle]
    assert entry.positions.shape == (1, 5, 3)


def test_add_markers_rejects_palette_color(gl_widget):
    from src.shared.python.plot_style import PaletteColor

    r = PyQtGLMarkerRenderer(gl_widget)
    style = MarkerStyle(fill_color=PaletteColor("tab10", 0))
    with pytest.raises(NotImplementedError):
        r.add_markers(np.zeros((1, 3, 3)), style)


def test_add_markers_rejects_non_style(gl_widget):
    r = PyQtGLMarkerRenderer(gl_widget)
    with pytest.raises(TypeError):
        r.add_markers(np.zeros((1, 3, 3)), object())  # type: ignore[arg-type]


def test_add_markers_rejects_non_string_label(gl_widget):
    r = PyQtGLMarkerRenderer(gl_widget)
    with pytest.raises(TypeError):
        r.add_markers(
            np.zeros((1, 3, 3)),
            MarkerStyle(),
            label=42,  # type: ignore[arg-type]
        )


def test_add_markers_rejects_bad_position_shape(gl_widget):
    r = PyQtGLMarkerRenderer(gl_widget)
    with pytest.raises(ValueError):
        r.add_markers(np.zeros((3, 4)), MarkerStyle())  # last dim != 3


def test_add_markers_rejects_1d_positions(gl_widget):
    r = PyQtGLMarkerRenderer(gl_widget)
    with pytest.raises(ValueError):
        r.add_markers(np.zeros(9), MarkerStyle())


def test_add_markers_rejects_non_ndarray(gl_widget):
    r = PyQtGLMarkerRenderer(gl_widget)
    with pytest.raises(TypeError):
        r.add_markers([[0.0, 0.0, 0.0]], MarkerStyle())  # type: ignore[arg-type]


def test_update_frame_scatter(gl_widget):
    r = PyQtGLMarkerRenderer(gl_widget)
    pos = np.zeros((3, 4, 3))
    pos[1] = 1.0
    pos[2] = 2.0
    handle = r.add_markers(pos, MarkerStyle(shape=MarkerShape.SPHERE))
    r.update_frame(handle, 0)
    r.update_frame(handle, 2)


def test_update_frame_mesh(gl_widget):
    r = PyQtGLMarkerRenderer(gl_widget)
    pos = np.zeros((3, 2, 3))
    pos[2] = 5.0
    handle = r.add_markers(pos, MarkerStyle(shape=MarkerShape.CUBE))
    r.update_frame(handle, 2)


def test_update_frame_custom_mesh(gl_widget):
    r = PyQtGLMarkerRenderer(gl_widget)
    pos = np.zeros((2, 3, 3))
    pos[1] = 1.0
    style = MarkerStyle(shape=MarkerShape.CUSTOM_MESH, custom_mesh=_custom_tetra_spec())
    handle = r.add_markers(pos, style)
    r.update_frame(handle, 1)


def test_update_frame_unknown_handle(gl_widget):
    r = PyQtGLMarkerRenderer(gl_widget)
    with pytest.raises(KeyError):
        r.update_frame("nope", 0)


def test_update_frame_rejects_non_int(gl_widget):
    r = PyQtGLMarkerRenderer(gl_widget)
    handle = r.add_markers(np.zeros((1, 2, 3)), MarkerStyle())
    with pytest.raises(TypeError):
        r.update_frame(handle, 1.5)  # type: ignore[arg-type]


def test_update_frame_out_of_range(gl_widget):
    r = PyQtGLMarkerRenderer(gl_widget)
    handle = r.add_markers(np.zeros((2, 1, 3)), MarkerStyle())
    with pytest.raises(IndexError):
        r.update_frame(handle, 99)


def test_update_style_scatter(gl_widget):
    r = PyQtGLMarkerRenderer(gl_widget)
    handle = r.add_markers(np.zeros((1, 3, 3)), MarkerStyle(size_px=4.0))
    new = MarkerStyle(size_px=10.0, fill_color=StaticColor("#00ff00"))
    r.update_style(handle, new)
    assert r._entries[handle].style.size_px == 10.0


def test_update_style_mesh(gl_widget):
    r = PyQtGLMarkerRenderer(gl_widget)
    handle = r.add_markers(
        np.zeros((1, 2, 3)),
        MarkerStyle(shape=MarkerShape.CUBE, size_px=4.0),
    )
    new = MarkerStyle(
        shape=MarkerShape.CUBE,
        size_px=8.0,
        fill_color=StaticColor("#00ff00"),
    )
    r.update_style(handle, new)
    assert r._entries[handle].style.size_px == 8.0


def test_update_style_rejects_shape_change(gl_widget):
    r = PyQtGLMarkerRenderer(gl_widget)
    handle = r.add_markers(np.zeros((1, 2, 3)), MarkerStyle())
    with pytest.raises(ValueError):
        r.update_style(handle, MarkerStyle(shape=MarkerShape.CUBE))


def test_update_style_rejects_non_static_fill(gl_widget):
    from src.shared.python.plot_style import PaletteColor

    r = PyQtGLMarkerRenderer(gl_widget)
    handle = r.add_markers(np.zeros((1, 2, 3)), MarkerStyle())
    with pytest.raises(NotImplementedError):
        r.update_style(handle, MarkerStyle(fill_color=PaletteColor("tab10", 0)))


def test_update_style_unknown_handle(gl_widget):
    r = PyQtGLMarkerRenderer(gl_widget)
    with pytest.raises(KeyError):
        r.update_style("nope", MarkerStyle())


def test_update_style_rejects_non_style(gl_widget):
    r = PyQtGLMarkerRenderer(gl_widget)
    handle = r.add_markers(np.zeros((1, 2, 3)), MarkerStyle())
    with pytest.raises(TypeError):
        r.update_style(handle, object())  # type: ignore[arg-type]


def test_set_visible_toggles_scatter(gl_widget):
    r = PyQtGLMarkerRenderer(gl_widget)
    handle = r.add_markers(np.zeros((1, 2, 3)), MarkerStyle())
    r.set_visible(handle, False)
    assert r._entries[handle].items[0].visible() is False
    r.set_visible(handle, True)
    assert r._entries[handle].items[0].visible() is True


def test_set_visible_toggles_mesh(gl_widget):
    r = PyQtGLMarkerRenderer(gl_widget)
    handle = r.add_markers(np.zeros((1, 3, 3)), MarkerStyle(shape=MarkerShape.CUBE))
    r.set_visible(handle, False)
    for item in r._entries[handle].items:
        assert item.visible() is False


def test_set_visible_unknown_handle(gl_widget):
    r = PyQtGLMarkerRenderer(gl_widget)
    with pytest.raises(KeyError):
        r.set_visible("nope", True)


def test_set_visible_rejects_non_bool(gl_widget):
    r = PyQtGLMarkerRenderer(gl_widget)
    handle = r.add_markers(np.zeros((1, 2, 3)), MarkerStyle())
    with pytest.raises(TypeError):
        r.set_visible(handle, "yes")  # type: ignore[arg-type]


def test_remove_drops_handle_scatter(gl_widget):
    r = PyQtGLMarkerRenderer(gl_widget)
    handle = r.add_markers(np.zeros((1, 2, 3)), MarkerStyle())
    n_before = len(gl_widget.items)
    r.remove(handle)
    assert handle not in r._entries
    assert len(gl_widget.items) == n_before - 1
    with pytest.raises(KeyError):
        r.remove(handle)


def test_remove_drops_handle_mesh(gl_widget):
    r = PyQtGLMarkerRenderer(gl_widget)
    handle = r.add_markers(np.zeros((1, 4, 3)), MarkerStyle(shape=MarkerShape.CUBE))
    n_before = len(gl_widget.items)
    r.remove(handle)
    # 4 mesh items removed.
    assert len(gl_widget.items) == n_before - 4
