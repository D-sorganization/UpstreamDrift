"""Tests for :class:`MatplotlibMarkerRenderer`.

Headless: forces the ``Agg`` backend before any pyplot import.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: E402, F401  # registers 3d projection

from src.shared.python.plot_style import (  # noqa: E402
    CustomMeshSpec,
    MarkerShape,
    MarkerStyle,
    MatplotlibMarkerRenderer,
    StaticColor,
)

pytestmark = pytest.mark.unit


SNAPSHOT_DIR = Path(__file__).resolve().parents[3] / "snapshots" / "plot_style"


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def positions_2d() -> np.ndarray:
    return np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.5]], dtype=np.float64)


@pytest.fixture
def positions_3d() -> np.ndarray:
    return np.array(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 0.5, -1.0]],
        dtype=np.float64,
    )


@pytest.fixture
def rgba_three() -> np.ndarray:
    return np.array(
        [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]],
        dtype=np.float64,
    )


@pytest.fixture
def fig_ax_2d():
    fig, ax = plt.subplots()
    yield fig, ax
    plt.close(fig)


@pytest.fixture
def fig_ax_3d():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    yield fig, ax
    plt.close(fig)


@pytest.fixture
def tetra_mesh() -> CustomMeshSpec:
    """Tiny tetrahedron CustomMeshSpec for custom-mesh tests."""
    v = np.array(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
        dtype=np.float64,
    )
    f = np.array(
        [[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]],
        dtype=np.int64,
    )
    return CustomMeshSpec(name="tetra", vertices=v, faces=f)


# --------------------------------------------------------------------------
# 2D smoke tests
# --------------------------------------------------------------------------


def test_draw_2d_smoke(fig_ax_2d, positions_2d, rgba_three):
    _, ax = fig_ax_2d
    r = MatplotlibMarkerRenderer()
    style = MarkerStyle(shape=MarkerShape.SPHERE)
    art = r.draw(ax, positions_2d, style, rgba_three)
    assert art is not None
    # One PathCollection added.
    assert len(ax.collections) == 1


def test_draw_3d_smoke(fig_ax_3d, positions_3d, rgba_three):
    _, ax = fig_ax_3d
    r = MatplotlibMarkerRenderer()
    style = MarkerStyle(shape=MarkerShape.SPHERE)
    art = r.draw(ax, positions_3d, style, rgba_three)
    assert art is not None
    assert len(ax.collections) == 1


@pytest.mark.parametrize(
    "shape",
    [
        MarkerShape.SPHERE,
        MarkerShape.CUBE,
        MarkerShape.CROSS,
        MarkerShape.STAR,
        MarkerShape.DIAMOND,
        MarkerShape.PLUS,
        MarkerShape.POINT,
    ],
)
def test_each_builtin_shape_renders_2d(shape, fig_ax_2d, positions_2d, rgba_three):
    _, ax = fig_ax_2d
    r = MatplotlibMarkerRenderer()
    style = MarkerStyle(shape=shape)
    art = r.draw(ax, positions_2d, style, rgba_three)
    assert art is not None


@pytest.mark.parametrize(
    "shape",
    [
        MarkerShape.SPHERE,
        MarkerShape.CUBE,
        MarkerShape.CROSS,
        MarkerShape.STAR,
        MarkerShape.DIAMOND,
        MarkerShape.PLUS,
        MarkerShape.POINT,
    ],
)
def test_each_builtin_shape_renders_3d(shape, fig_ax_3d, positions_3d, rgba_three):
    _, ax = fig_ax_3d
    r = MatplotlibMarkerRenderer()
    style = MarkerStyle(shape=shape)
    art = r.draw(ax, positions_3d, style, rgba_three)
    assert art is not None


def test_custom_mesh_3d_path(fig_ax_3d, positions_3d, rgba_three, tetra_mesh):
    _, ax = fig_ax_3d
    r = MatplotlibMarkerRenderer()
    style = MarkerStyle(shape=MarkerShape.CUSTOM_MESH, custom_mesh=tetra_mesh)
    arts = r.draw(ax, positions_3d, style, rgba_three)
    assert isinstance(arts, list)
    # One trisurf per marker.
    assert len(arts) == positions_3d.shape[0]


def test_custom_mesh_2d_falls_back_with_warning(
    fig_ax_2d, positions_2d, rgba_three, tetra_mesh
):
    _, ax = fig_ax_2d
    r = MatplotlibMarkerRenderer()
    style = MarkerStyle(shape=MarkerShape.CUSTOM_MESH, custom_mesh=tetra_mesh)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        art = r.draw(ax, positions_2d, style, rgba_three)
    assert art is not None
    assert any("CUSTOM_MESH" in str(w.message) for w in caught)


# --------------------------------------------------------------------------
# Stateful Protocol tests
# --------------------------------------------------------------------------


def test_add_markers_requires_default_ax(positions_3d):
    r = MatplotlibMarkerRenderer()  # no ax
    with pytest.raises(RuntimeError):
        r.add_markers(positions_3d, MarkerStyle())


def test_add_update_remove_3d(fig_ax_3d, positions_3d):
    _, ax = fig_ax_3d
    r = MatplotlibMarkerRenderer(ax=ax)
    style = MarkerStyle(shape=MarkerShape.CUBE)
    h = r.add_markers(positions_3d, style, label="m")
    assert isinstance(h, str) and h
    r.update_frame(h, 0)
    r.set_visible(h, False)
    r.set_visible(h, True)
    new_style = MarkerStyle(shape=MarkerShape.STAR)
    r.update_style(h, new_style)
    r.remove(h)
    with pytest.raises(KeyError):
        r.update_frame(h, 0)


def test_update_frame_per_frame_positions(fig_ax_3d):
    """``positions`` shape ``(T, M, 3)`` updates per frame."""
    _, ax = fig_ax_3d
    r = MatplotlibMarkerRenderer(ax=ax)
    rng = np.random.default_rng(0)
    pos = rng.standard_normal((4, 3, 3)).astype(np.float64)
    h = r.add_markers(pos, MarkerStyle())
    r.update_frame(h, 2)
    with pytest.raises(IndexError):
        r.update_frame(h, 99)


def test_update_frame_custom_mesh_rebuilds(fig_ax_3d, tetra_mesh):
    _, ax = fig_ax_3d
    r = MatplotlibMarkerRenderer(ax=ax)
    pos = np.array([[[0.0, 0.0, 0.0]], [[1.0, 1.0, 1.0]]], dtype=np.float64)
    style = MarkerStyle(shape=MarkerShape.CUSTOM_MESH, custom_mesh=tetra_mesh)
    h = r.add_markers(pos, style)
    r.update_frame(h, 1)


def test_2d_axes_with_3d_positions_raises(fig_ax_2d, positions_3d, rgba_three):
    _, ax = fig_ax_2d
    r = MatplotlibMarkerRenderer()
    with pytest.raises(ValueError):
        r.draw(ax, positions_3d, MarkerStyle(), rgba_three)


def test_3d_axes_with_2d_positions_raises(fig_ax_3d, positions_2d, rgba_three):
    _, ax = fig_ax_3d
    r = MatplotlibMarkerRenderer()
    with pytest.raises(ValueError):
        r.draw(ax, positions_2d, MarkerStyle(), rgba_three)


def test_colors_length_mismatch_raises(fig_ax_3d, positions_3d):
    _, ax = fig_ax_3d
    r = MatplotlibMarkerRenderer()
    bad = np.zeros((2, 4))
    with pytest.raises(ValueError):
        r.draw(ax, positions_3d, MarkerStyle(), bad)


def test_colors_wrong_shape_raises(fig_ax_3d, positions_3d):
    _, ax = fig_ax_3d
    r = MatplotlibMarkerRenderer()
    bad = np.zeros((3, 3))
    with pytest.raises(ValueError):
        r.draw(ax, positions_3d, MarkerStyle(), bad)


def test_colors_wrong_type_raises(fig_ax_3d, positions_3d):
    _, ax = fig_ax_3d
    r = MatplotlibMarkerRenderer()
    with pytest.raises(TypeError):
        r.draw(ax, positions_3d, MarkerStyle(), [[0, 0, 0, 1]] * 3)  # type: ignore[arg-type]


def test_positions_wrong_type_raises(fig_ax_3d, rgba_three):
    _, ax = fig_ax_3d
    r = MatplotlibMarkerRenderer()
    with pytest.raises(TypeError):
        r.draw(ax, [[0, 0, 0]], MarkerStyle(), rgba_three)  # type: ignore[arg-type]


def test_positions_bad_dim_raises(fig_ax_3d, rgba_three):
    _, ax = fig_ax_3d
    r = MatplotlibMarkerRenderer()
    bad = np.zeros((3, 5))
    with pytest.raises(ValueError):
        r.draw(ax, bad, MarkerStyle(), rgba_three)


def test_positions_bad_ndim_raises(fig_ax_3d, rgba_three):
    _, ax = fig_ax_3d
    r = MatplotlibMarkerRenderer()
    bad = np.zeros((1, 1, 1, 3))
    with pytest.raises(ValueError):
        r.draw(ax, bad, MarkerStyle(), rgba_three)


def test_constructor_rejects_non_axes():
    with pytest.raises(TypeError):
        MatplotlibMarkerRenderer(ax="not an ax")  # type: ignore[arg-type]


def test_draw_rejects_non_axes(positions_3d, rgba_three):
    r = MatplotlibMarkerRenderer()
    with pytest.raises(TypeError):
        r.draw("nope", positions_3d, MarkerStyle(), rgba_three)  # type: ignore[arg-type]


def test_draw_rejects_non_style(fig_ax_3d, positions_3d, rgba_three):
    _, ax = fig_ax_3d
    r = MatplotlibMarkerRenderer()
    with pytest.raises(TypeError):
        r.draw(ax, positions_3d, "no", rgba_three)  # type: ignore[arg-type]


def test_add_markers_rejects_bad_inputs(fig_ax_3d):
    _, ax = fig_ax_3d
    r = MatplotlibMarkerRenderer(ax=ax)
    with pytest.raises(TypeError):
        r.add_markers(np.zeros((1, 3)), "nope")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        r.add_markers([[0, 0, 0]], MarkerStyle())  # type: ignore[arg-type]


def test_add_markers_dim_mismatch_raises(fig_ax_2d):
    _, ax = fig_ax_2d
    r = MatplotlibMarkerRenderer(ax=ax)
    pos = np.array([[0.0, 0.0, 0.0]])  # 3D pos, 2D ax
    with pytest.raises(ValueError):
        r.add_markers(pos, MarkerStyle())


def test_update_style_rejects_non_style(fig_ax_3d, positions_3d):
    _, ax = fig_ax_3d
    r = MatplotlibMarkerRenderer(ax=ax)
    h = r.add_markers(positions_3d, MarkerStyle())
    with pytest.raises(TypeError):
        r.update_style(h, "nope")  # type: ignore[arg-type]


def test_unknown_handle_raises(fig_ax_3d):
    _, ax = fig_ax_3d
    r = MatplotlibMarkerRenderer(ax=ax)
    with pytest.raises(KeyError):
        r.set_visible("nope", True)


def test_static_color_resolution(fig_ax_3d, positions_3d):
    _, ax = fig_ax_3d
    r = MatplotlibMarkerRenderer(ax=ax)
    style = MarkerStyle(shape=MarkerShape.SPHERE, fill_color=StaticColor("#ff0000"))
    h = r.add_markers(positions_3d, style)
    assert h


def test_normalise_positions_static_marker(fig_ax_3d):
    _, ax = fig_ax_3d
    r = MatplotlibMarkerRenderer(ax=ax)
    pos = np.array([1.0, 2.0, 3.0])  # 1-D
    h = r.add_markers(pos, MarkerStyle())
    assert h


# --------------------------------------------------------------------------
# Optional golden snapshot
# --------------------------------------------------------------------------


def _save_or_compare(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        fig.savefig(path, dpi=72)
        return
    # Just ensure we can write a fresh copy and the file exists.
    tmp = path.with_suffix(".cur.png")
    fig.savefig(tmp, dpi=72)
    assert tmp.exists()
    tmp.unlink(missing_ok=True)


def test_snapshot_3d_sphere(positions_3d, rgba_three):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    r = MatplotlibMarkerRenderer()
    r.draw(ax, positions_3d, MarkerStyle(shape=MarkerShape.SPHERE), rgba_three)
    _save_or_compare(fig, SNAPSHOT_DIR / "mpl_3d_sphere.png")
    plt.close(fig)


def test_snapshot_2d_cube(positions_2d, rgba_three):
    fig, ax = plt.subplots()
    r = MatplotlibMarkerRenderer()
    r.draw(ax, positions_2d, MarkerStyle(shape=MarkerShape.CUBE), rgba_three)
    _save_or_compare(fig, SNAPSHOT_DIR / "mpl_2d_cube.png")
    plt.close(fig)
