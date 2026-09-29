# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Smoke tests for rendering.py: BodyRenderer, BarbellRenderer, style_axis, Palette."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from matplotlib.figure import Figure

from movement_optimizer.rendering import (
    BarbellRenderer,
    BodyRenderer,
    Palette,
    style_axis,
)


@pytest.fixture()
def ax():
    """Create a fresh matplotlib Axes for each test."""
    fig = Figure()
    ax = fig.add_subplot(111)
    yield ax
    import matplotlib.pyplot as plt

    plt.close("all")


@pytest.fixture()
def sample_joints() -> dict[str, np.ndarray]:
    """Sample joint positions for a rough standing pose."""
    return {
        "ankle": np.array([0.0, 0.0]),
        "knee": np.array([0.05, 0.45]),
        "hip": np.array([0.02, 0.90]),
        "shoulder": np.array([0.0, 1.40]),
    }


# ------------------------------------------------------------------
# Palette
# ------------------------------------------------------------------


class TestPalette:
    def test_palette_has_required_colours(self) -> None:
        assert isinstance(Palette.BG, str)
        assert isinstance(Palette.FG, str)
        assert isinstance(Palette.ACCENT, str)
        assert isinstance(Palette.SEG_COLORS, tuple)
        assert len(Palette.SEG_COLORS) == 3

    def test_seg_labels(self) -> None:
        assert len(Palette.SEG_LABELS) == 3
        assert len(Palette.BENCH_LABELS) == 3


# ------------------------------------------------------------------
# BarbellRenderer
# ------------------------------------------------------------------


class TestBarbellRenderer:
    def test_draw_adds_patches(self, ax) -> None:
        n_before = len(ax.patches)
        BarbellRenderer.draw(ax, (0.0, 1.0))
        assert len(ax.patches) == n_before + 2  # plate + bar circles

    def test_draw_at_different_positions(self, ax) -> None:
        BarbellRenderer.draw(ax, (0.5, 2.0))
        # No exception raised; patches added
        assert len(ax.patches) >= 2


# ------------------------------------------------------------------
# BodyRenderer
# ------------------------------------------------------------------


class TestBodyRenderer:
    def test_draw_segments_adds_lines(self, ax, sample_joints) -> None:
        n_lines_before = len(ax.lines)
        BodyRenderer.draw_segments(ax, sample_joints)
        assert len(ax.lines) > n_lines_before

    def test_draw_segments_adds_head_patch(self, ax, sample_joints) -> None:
        BodyRenderer.draw_segments(ax, sample_joints)
        assert len(ax.patches) >= 1  # head circle

    def test_draw_ground(self, ax) -> None:
        n_before = len(ax.lines)
        BodyRenderer.draw_ground(ax, -0.1, 0.25)
        assert len(ax.lines) == n_before + 2

    def test_draw_arms(self, ax, sample_joints) -> None:
        n_before = len(ax.lines)
        BodyRenderer.draw_arms(ax, sample_joints["shoulder"], 0.6)
        assert len(ax.lines) > n_before

    def test_draw_com_marker(self, ax) -> None:
        n_before = len(ax.lines)
        BodyRenderer.draw_com_marker(ax, np.array([0.05, 0.8]))
        assert len(ax.lines) > n_before

    def test_draw_ghost(self, ax, sample_joints) -> None:
        n_before = len(ax.lines)
        BodyRenderer.draw_ghost(ax, sample_joints, alpha=0.1)
        assert len(ax.lines) > n_before

    def test_draw_bar_trace(self, ax) -> None:
        bar_traj = np.column_stack([np.linspace(0, 0.1, 10), np.linspace(1.0, 1.5, 10)])
        BodyRenderer.draw_bar_trace(ax, bar_traj, current_idx=5)
        assert len(ax.lines) >= 2


# ------------------------------------------------------------------
# style_axis
# ------------------------------------------------------------------


class TestStyleAxis:
    def test_style_axis_applies_without_error(self, ax) -> None:
        style_axis(ax)
        # Verify grid is on
        assert ax.xaxis.get_gridlines()[0].get_visible()
