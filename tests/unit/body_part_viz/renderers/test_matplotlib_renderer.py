"""Unit tests for ``MatplotlibRenderer``.

Headless: forces matplotlib's ``Agg`` backend before importing pyplot.
The 60 fps performance contract is enforced via a loose 1 ms / call
mean budget over 1000 ``update_frame`` calls.
"""

from __future__ import annotations

import os
import time

import matplotlib

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from mpl_toolkits.mplot3d.art3d import (  # noqa: E402
    Line3DCollection,
    Poly3DCollection,
)

from src.shared.python.body_part_viz import (  # noqa: E402
    BindingKind,
    FittedShape,
    MarkerBinding,
    ShapeRenderer,
    ShapeTheme,
)
from src.shared.python.body_part_viz.renderers import (  # noqa: E402
    MatplotlibRenderer,
)
from src.shared.python.body_part_viz.shapes import (  # noqa: E402
    CylinderShape,
    LineShape,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _fitted(shape_id: str, n_frames: int = 4) -> FittedShape:
    binding = MarkerBinding(
        kind=BindingKind.BETWEEN_TWO,
        marker_names=("a", "b"),
    )
    centroid = np.zeros((n_frames, 3))
    centroid[:, 0] = np.arange(n_frames, dtype=float)
    rotation = np.broadcast_to(np.eye(3), (n_frames, 3, 3)).copy()
    scale = np.ones((n_frames, 3))
    mask = np.ones((n_frames,), dtype=bool)
    return FittedShape(
        shape_id=shape_id,
        binding=binding,
        centroid=centroid,
        rotation_matrix=rotation,
        scale=scale,
        valid_mask=mask,
    )


@pytest.fixture
def ax():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    yield ax
    plt.close(fig)


@pytest.fixture
def theme():
    return ShapeTheme(color="#1f77b4", opacity=0.5)


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------
def test_matplotlib_renderer_satisfies_shape_renderer_protocol(ax) -> None:
    renderer = MatplotlibRenderer(ax)
    assert isinstance(renderer, ShapeRenderer)


def test_constructor_rejects_none() -> None:
    with pytest.raises(TypeError):
        MatplotlibRenderer(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# add_shape: artist creation
# ---------------------------------------------------------------------------
def test_add_shape_line_creates_line3dcollection(ax, theme) -> None:
    renderer = MatplotlibRenderer(ax)
    shape = LineShape(length=1.0)
    handle = renderer.add_shape(shape, _fitted("line"), theme)
    entry = renderer._entries[handle]
    assert isinstance(entry.artist, Line3DCollection)


def test_add_shape_cylinder_creates_poly3dcollection(ax, theme) -> None:
    renderer = MatplotlibRenderer(ax)
    shape = CylinderShape(length=1.0, radius=0.5, n_facets=12)
    handle = renderer.add_shape(shape, _fitted("cyl"), theme)
    entry = renderer._entries[handle]
    assert isinstance(entry.artist, Poly3DCollection)


def test_add_shape_validates_arguments(ax, theme) -> None:
    renderer = MatplotlibRenderer(ax)
    shape = LineShape(length=1.0)
    fitted = _fitted("line")
    with pytest.raises(TypeError):
        renderer.add_shape("not a shape", fitted, theme)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        renderer.add_shape(shape, "not fitted", theme)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        renderer.add_shape(shape, fitted, "not theme")  # type: ignore[arg-type]


def test_add_three_cylinders_and_one_line_yields_four_artists(ax, theme) -> None:
    renderer = MatplotlibRenderer(ax)
    handles = []
    for i in range(3):
        cyl = CylinderShape(length=1.0, radius=0.4, n_facets=8)
        handles.append(renderer.add_shape(cyl, _fitted(f"cyl{i}"), theme))
    handles.append(renderer.add_shape(LineShape(length=1.0), _fitted("line"), theme))
    assert len(renderer._entries) == 4
    assert len(set(handles)) == 4


# ---------------------------------------------------------------------------
# update_frame
# ---------------------------------------------------------------------------
def test_update_frame_changes_geometry(ax, theme) -> None:
    renderer = MatplotlibRenderer(ax)
    shape = LineShape(length=1.0)
    handle = renderer.add_shape(shape, _fitted("line", n_frames=4), theme)
    artist = renderer._entries[handle].artist

    renderer.update_frame(handle, 0)
    segs0 = [np.asarray(s).copy() for s in artist._segments3d]

    renderer.update_frame(handle, 2)
    segs2 = [np.asarray(s) for s in artist._segments3d]

    assert len(segs0) == len(segs2) >= 1
    assert not np.allclose(segs0[0], segs2[0])


def test_update_frame_is_idempotent(ax, theme) -> None:
    renderer = MatplotlibRenderer(ax)
    shape = CylinderShape(length=1.0, radius=0.4, n_facets=8)
    handle = renderer.add_shape(shape, _fitted("cyl"), theme)
    # Calling update_frame twice with the same index must not raise and
    # must leave the renderer in a consistent state.
    renderer.update_frame(handle, 2)
    renderer.update_frame(handle, 2)
    assert handle in renderer._entries


def test_update_frame_unknown_handle_raises(ax) -> None:
    renderer = MatplotlibRenderer(ax)
    with pytest.raises(KeyError):
        renderer.update_frame("nope", 0)


def test_update_frame_validates_frame_idx(ax, theme) -> None:
    renderer = MatplotlibRenderer(ax)
    handle = renderer.add_shape(LineShape(length=1.0), _fitted("line"), theme)
    with pytest.raises(TypeError):
        renderer.update_frame(handle, "0")  # type: ignore[arg-type]
    with pytest.raises(IndexError):
        renderer.update_frame(handle, 999)
    with pytest.raises(IndexError):
        renderer.update_frame(handle, -1)


def test_update_frame_handles_invalid_frame_without_raising(ax, theme) -> None:
    """Invalid frames produce NaN vertices; renderer must substitute zeros."""
    renderer = MatplotlibRenderer(ax)
    binding = MarkerBinding(
        kind=BindingKind.BETWEEN_TWO,
        marker_names=("a", "b"),
    )
    mask = np.array([True, False, True], dtype=bool)
    fitted = FittedShape(
        shape_id="line",
        binding=binding,
        centroid=np.zeros((3, 3)),
        rotation_matrix=np.broadcast_to(np.eye(3), (3, 3, 3)).copy(),
        scale=np.ones((3, 3)),
        valid_mask=mask,
    )
    handle = renderer.add_shape(LineShape(length=1.0), fitted, theme)
    renderer.update_frame(handle, 1)  # invalid frame, must not raise


# ---------------------------------------------------------------------------
# set_visible
# ---------------------------------------------------------------------------
def test_set_visible_toggles_artist(ax, theme) -> None:
    renderer = MatplotlibRenderer(ax)
    handle = renderer.add_shape(LineShape(length=1.0), _fitted("line"), theme)
    renderer.set_visible(handle, False)
    assert renderer._entries[handle].artist.get_visible() is False
    renderer.set_visible(handle, True)
    assert renderer._entries[handle].artist.get_visible() is True


def test_set_visible_validates_args(ax, theme) -> None:
    renderer = MatplotlibRenderer(ax)
    handle = renderer.add_shape(LineShape(length=1.0), _fitted("line"), theme)
    with pytest.raises(TypeError):
        renderer.set_visible(handle, "yes")  # type: ignore[arg-type]
    with pytest.raises(KeyError):
        renderer.set_visible("missing", True)


# ---------------------------------------------------------------------------
# remove + clear
# ---------------------------------------------------------------------------
def test_remove_drops_artist(ax, theme) -> None:
    renderer = MatplotlibRenderer(ax)
    handle = renderer.add_shape(LineShape(length=1.0), _fitted("line"), theme)
    renderer.remove(handle)
    assert handle not in renderer._entries
    with pytest.raises(KeyError):
        renderer.update_frame(handle, 0)


def test_remove_unknown_raises(ax) -> None:
    renderer = MatplotlibRenderer(ax)
    with pytest.raises(KeyError):
        renderer.remove("missing")


def test_remove_tolerates_already_detached_artist(ax, theme) -> None:
    renderer = MatplotlibRenderer(ax)
    handle = renderer.add_shape(LineShape(length=1.0), _fitted("line"), theme)
    # Forcibly detach the artist from the axes to simulate a stale state.
    renderer._entries[handle].artist.remove()
    renderer.remove(handle)  # must not raise
    assert handle not in renderer._entries


def test_clear_drops_all(ax, theme) -> None:
    renderer = MatplotlibRenderer(ax)
    for i in range(3):
        renderer.add_shape(LineShape(length=1.0), _fitted(f"l{i}"), theme)
    renderer.clear()
    assert renderer._entries == {}


# ---------------------------------------------------------------------------
# add_segment_set
# ---------------------------------------------------------------------------
def test_add_segment_set_wraps_add_shape(ax) -> None:
    renderer = MatplotlibRenderer(ax)
    pairs = [
        (LineShape(length=1.0), _fitted("l0")),
        (CylinderShape(length=1.0, radius=0.3, n_facets=8), _fitted("c0")),
    ]
    handles = renderer.add_segment_set(
        pairs, lambda f: ShapeTheme(color="C1", opacity=0.6)
    )
    assert len(handles) == 2
    assert all(h in renderer._entries for h in handles)


def test_add_segment_set_validates(ax) -> None:
    renderer = MatplotlibRenderer(ax)
    with pytest.raises(TypeError):
        renderer.add_segment_set([], "not callable")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        renderer.add_segment_set(
            ["bad"],  # type: ignore[list-item]
            lambda f: ShapeTheme(),
        )


# ---------------------------------------------------------------------------
# No ax.clear() ever
# ---------------------------------------------------------------------------
def test_renderer_never_calls_axes_clear(ax, theme, monkeypatch) -> None:
    calls = {"n": 0}

    def fake_clear(*args, **kwargs):
        calls["n"] += 1

    monkeypatch.setattr(ax, "clear", fake_clear)
    monkeypatch.setattr(ax, "cla", fake_clear)
    renderer = MatplotlibRenderer(ax)
    h = renderer.add_shape(LineShape(length=1.0), _fitted("l"), theme)
    renderer.update_frame(h, 1)
    renderer.set_visible(h, False)
    renderer.remove(h)
    assert calls["n"] == 0


# ---------------------------------------------------------------------------
# Performance contract: 60 fps (1 ms / update_frame, mean)
# ---------------------------------------------------------------------------
@pytest.mark.benchmark
def test_update_frame_performance_contract(ax, theme) -> None:
    renderer = MatplotlibRenderer(ax)
    shape = CylinderShape(length=1.0, radius=0.5, n_facets=20)
    fitted = _fitted("cyl", n_frames=10)
    handle = renderer.add_shape(shape, fitted, theme)

    # Warm-up.
    for _ in range(10):
        renderer.update_frame(handle, 0)

    n = 1000
    start = time.perf_counter()
    for i in range(n):
        renderer.update_frame(handle, i % 10)
    elapsed = time.perf_counter() - start
    mean_ms = (elapsed / n) * 1000.0
    # 1 ms is the contract; print for visibility.
    assert mean_ms <= 1.0, f"mean update_frame time {mean_ms:.3f} ms exceeds budget"
