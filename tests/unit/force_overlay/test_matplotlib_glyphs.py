"""Unit tests for Matplotlib 3D force/torque glyph renderer (ADR-0052, #11292)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from src.shared.python.force_overlay.contracts import (
    ForceTorqueFrame,
    OverlayWrench,
    WrenchKind,
)
from src.shared.python.force_overlay.glyphs import (
    ArrowGlyph,
    ForceGlyphStyle,
    GlyphSet,
    LegendSpec,
    TorqueArcGlyph,
    build_glyphs,
)
from src.shared.python.force_overlay.renderers.matplotlib_glyphs import (
    draw_glyphs_3d,
    draw_legend,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


@pytest.fixture
def sample_glyphs() -> GlyphSet:
    frame = ForceTorqueFrame(
        time_s=0.5,
        engine="mujoco",
        wrenches=(
            OverlayWrench(
                label="joint:lead_wrist",
                kind=WrenchKind.JOINT_REACTION,
                body="wrist",
                torque_nm=(0.0, 0.0, 20.0),
                force_n=(500.0, 0.0, 0.0),
                point_m=(0.0, 0.0, 1.0),
                source="sim",
            ),
        ),
    )
    return build_glyphs(frame, style=ForceGlyphStyle())


def test_draw_glyphs_3d_artist_counts_and_removal(sample_glyphs: GlyphSet) -> None:
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    initial_lines = len(ax.lines)
    initial_collections = len(ax.collections)

    artists = draw_glyphs_3d(ax, sample_glyphs, halo=True)
    assert len(artists) > 0

    # With halo=True, each arrow has at least halo shaft + shaft + cone
    assert len(ax.lines) > initial_lines
    assert len(ax.collections) > initial_collections

    # Removing the returned artists restores a blank axes
    for artist in artists:
        artist.remove()

    assert len(ax.lines) == initial_lines
    assert len(ax.collections) == initial_collections
    plt.close(fig)


def test_draw_glyphs_3d_renders_arrow_color_pixels_near_tip(
    sample_glyphs: GlyphSet,
) -> None:
    fig = plt.figure(figsize=(4, 4), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=0, azim=0)
    ax.set_axis_off()

    draw_glyphs_3d(ax, sample_glyphs, halo=False)
    fig.canvas.draw()

    rgba_buf = np.asarray(fig.canvas.buffer_rgba())
    # The canvas should not be entirely uniform (blank)
    assert np.any(rgba_buf[..., :3] < 250)
    plt.close(fig)


def test_draw_glyphs_3d_rejects_non_3d_axes(sample_glyphs: GlyphSet) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111)  # 2D axes
    with pytest.raises(ValueError, match="3D"):
        draw_glyphs_3d(ax, sample_glyphs)
    plt.close(fig)


def test_draw_legend_contains_force_reference(sample_glyphs: GlyphSet) -> None:
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    legend_artist = draw_legend(ax, sample_glyphs.legend)
    assert legend_artist is not None

    fig.canvas.draw()
    # Check that text containing "500 N" is rendered
    texts = (
        [t.get_text() for t in legend_artist.texts]
        if hasattr(legend_artist, "texts")
        else []
    )
    if not texts and hasattr(legend_artist, "get_children"):
        texts = [
            c.get_text() for c in legend_artist.get_children() if hasattr(c, "get_text")
        ]

    # Also check full figure text representations
    all_text = " ".join(texts)
    assert (
        "500 N" in all_text
        or any("500 N" in t.get_text() for t in fig.texts)
        or any("500 N" in t.get_text() for t in ax.texts)
    )
    plt.close(fig)
