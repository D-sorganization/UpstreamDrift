"""Cross-tool consistency tests for the plot_style stack (#4814).

A single canonical :class:`PlotStyleSet` (the bundled ``"default"``
preset) is loaded from disk, persisted to a tmp file, reloaded, and then
fed to three downstream consumers that all draw on the same matplotlib
``Agg`` canvas:

1. The C3D Viewer's :class:`MatplotlibMarkerRenderer` (used directly by
   ``apps/ui/tabs/marker_plot_tab``, ``viewer_3d_tab``, etc.).
2. The matcher's :class:`StyledMarkerLayer` (the
   ``LiveViewController`` body-marker layer).
3. The cross-engine dashboard's renderer (constructed exactly as
   ``cross_engine_dashboard._traj_renderer``).

All three must produce pixel-identical PNGs (within the same RMS
tolerance the snapshot suite uses) when given the same five 2-D /
3-D positions and the same :class:`MarkerStyle`.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from src.shared.python.plot_style import (  # noqa: E402
    MatplotlibMarkerRenderer,
    PlotStyleSet,
    PresetLibrary,
)

pytestmark = pytest.mark.integration

RMS_TOLERANCE_FRAC = 0.005
SNAPSHOT_DPI = 100
SNAPSHOT_FIGSIZE = (3.0, 3.0)
N_MARKERS = 5

_FIXED_POSITIONS_3D = np.stack(
    [
        np.linspace(-0.4, 0.4, N_MARKERS),
        np.zeros(N_MARKERS),
        np.linspace(-0.2, 0.2, N_MARKERS),
    ],
    axis=1,
)


def _render_to_array(fig: Figure) -> np.ndarray:
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    buf = np.frombuffer(
        fig.canvas.buffer_rgba(),  # type: ignore[attr-defined]
        dtype=np.uint8,
    )
    return buf.reshape(int(height), int(width), 4).copy()


def _rms_normalised(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        h = min(a.shape[0], b.shape[0])
        w = min(a.shape[1], b.shape[1])
        a = a[:h, :w]
        b = b[:h, :w]
    diff = a.astype(np.float64) - b.astype(np.float64)
    return float(np.sqrt(np.mean(diff * diff)) / 255.0)


def _setup_3d_axes() -> tuple[Figure, Axes]:
    fig = plt.figure(figsize=SNAPSHOT_FIGSIZE, dpi=SNAPSHOT_DPI)
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=20.0, azim=45.0)
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(-0.5, 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    return fig, ax


def _viewer_render(style):
    """C3D Viewer code path: bare ``MatplotlibMarkerRenderer.draw``."""
    fig, ax = _setup_3d_axes()
    renderer = MatplotlibMarkerRenderer()
    rgba = np.broadcast_to(
        np.asarray(style.fill_color.resolve(0, 0), dtype=np.float64),
        (N_MARKERS, 4),
    ).copy()
    renderer.draw(ax, _FIXED_POSITIONS_3D, style, rgba)
    arr = _render_to_array(fig)
    plt.close(fig)
    return arr


def _matcher_render(style):
    """Matcher code path: stateful ``add_markers`` + ``update_frame(0)``.

    Mirrors :class:`StyledMarkerLayer.build` — that layer constructs
    its own :class:`MatplotlibMarkerRenderer`, calls ``add_markers``,
    then ``update_frame(0)`` to seed the scene. Doing the same here
    avoids importing the GUI-flavoured controller (PyQt6).
    """
    fig, ax = _setup_3d_axes()
    renderer = MatplotlibMarkerRenderer(ax)
    # (T, M, 3) per-frame layout used by StyledMarkerLayer for body
    # markers — single static frame here.
    positions = _FIXED_POSITIONS_3D.reshape(1, N_MARKERS, 3)
    handle = renderer.add_markers(positions, style, label="body")
    renderer.update_frame(handle, 0)
    arr = _render_to_array(fig)
    plt.close(fig)
    return arr


def _dashboard_render(style):
    """Dashboard code path: renderer bound at construction (#4810)."""
    fig, ax = _setup_3d_axes()
    renderer = MatplotlibMarkerRenderer(ax)
    rgba = np.broadcast_to(
        np.asarray(style.fill_color.resolve(0, 0), dtype=np.float64),
        (N_MARKERS, 4),
    ).copy()
    renderer.draw(ax, _FIXED_POSITIONS_3D, style, rgba)
    arr = _render_to_array(fig)
    plt.close(fig)
    return arr


def test_default_preset_round_trip_through_disk(tmp_path: Path) -> None:
    """Bundled ``default`` preset must round-trip cleanly through JSON."""
    library = PresetLibrary.default()
    canonical = library["default"]

    out = tmp_path / "default.json"
    canonical.save(out)
    assert out.is_file()

    reloaded = PlotStyleSet.load(out)
    assert reloaded.schema_version == canonical.schema_version
    assert len(reloaded.entries) == len(canonical.entries)
    for original, came_back in zip(canonical.entries, reloaded.entries, strict=True):
        assert came_back.name == original.name
        assert came_back.target == original.target
        assert came_back.style.shape == original.style.shape


def test_three_renderers_produce_identical_output(tmp_path: Path) -> None:
    """C3D Viewer, matcher, dashboard render the same MarkerStyle the same way.

    The three code paths instantiate :class:`MatplotlibMarkerRenderer`
    in slightly different ways (``draw`` vs ``add_markers`` vs
    pre-bound default ax). They must converge byte-for-byte (within
    the standard RMS tolerance) when handed the same style + positions.
    """
    library = PresetLibrary.default()
    canonical = library["default"]
    out = tmp_path / "default.json"
    canonical.save(out)
    reloaded = PlotStyleSet.load(out)
    assert reloaded.entries, "default preset has no entries"

    style = reloaded.entries[0].style

    viewer = _viewer_render(style)
    matcher = _matcher_render(style)
    dashboard = _dashboard_render(style)

    rms_vm = _rms_normalised(viewer, matcher)
    rms_vd = _rms_normalised(viewer, dashboard)
    rms_md = _rms_normalised(matcher, dashboard)

    assert (
        rms_vm < RMS_TOLERANCE_FRAC
    ), f"viewer vs matcher RMS {rms_vm:.4f} exceeds {RMS_TOLERANCE_FRAC:.4f}"
    assert (
        rms_vd < RMS_TOLERANCE_FRAC
    ), f"viewer vs dashboard RMS {rms_vd:.4f} exceeds {RMS_TOLERANCE_FRAC:.4f}"
    assert (
        rms_md < RMS_TOLERANCE_FRAC
    ), f"matcher vs dashboard RMS {rms_md:.4f} exceeds {RMS_TOLERANCE_FRAC:.4f}"


def test_every_builtin_preset_renders_via_matplotlib() -> None:
    """Smoke: every bundled preset entry renders without raising."""
    library = PresetLibrary.default()
    for preset_name in library.names():
        plot_style_set = library[preset_name]
        for spec in plot_style_set.entries:
            fig, ax = _setup_3d_axes()
            try:
                renderer = MatplotlibMarkerRenderer()
                rgba = np.broadcast_to(
                    np.asarray(spec.style.fill_color.resolve(0, 0), dtype=np.float64),
                    (N_MARKERS, 4),
                ).copy()
                renderer.draw(ax, _FIXED_POSITIONS_3D, spec.style, rgba)
            finally:
                plt.close(fig)
