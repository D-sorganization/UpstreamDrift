# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Tests for fleet shared-theme integration.

Covers palette sourcing from the shared theme, matplotlib restyling, the
palette-derived stylesheet, the motion-canvas palette, and the main window's
Theme menu + live recolouring.
"""

from __future__ import annotations

import pytest

pytest.importorskip("shared.python.theme")  # fleet theme only; skip on bare envs

from matplotlib.figure import Figure
from shared.python.theme import BUILTIN_THEMES, ThemeManager, get_theme_manager

from movement_optimizer.rendering import (
    Palette,
    refresh_palette,
    restyle_figure,
)


@pytest.fixture()
def theme_manager(qapp):
    """Provide a clean theme-manager singleton, reset around each test."""
    ThemeManager.reset_instance()
    manager = get_theme_manager()
    yield manager
    ThemeManager.reset_instance()


@pytest.mark.parametrize("theme_name", ["Light", "Dark", "Slate Gray"])
def test_refresh_palette_sources_from_theme(theme_manager, theme_name: str) -> None:
    theme_manager.change_theme(theme_name)
    refresh_palette()

    colors = BUILTIN_THEMES[theme_name]
    assert colors["bg"] == Palette.BG
    assert colors["text"] == Palette.FG
    assert colors["accent"] == Palette.ACCENT
    assert colors["group_bg"] == Palette.BG_PANEL
    assert colors["input_bg"] == Palette.BG_INPUT


def test_palette_seeded_with_hex_strings_at_import() -> None:
    # Palette is importable and usable without a QApplication.
    for value in (Palette.BG, Palette.FG, Palette.ACCENT, Palette.GREEN, Palette.RED):
        assert isinstance(value, str)
        assert value.startswith("#")
    assert isinstance(Palette.SEG_COLORS, tuple)
    assert len(Palette.SEG_COLORS) == 3


def test_restyle_figure_recolors_between_themes(theme_manager) -> None:
    fig = Figure()

    theme_manager.change_theme("Light")
    refresh_palette()
    restyle_figure(fig)
    light_facecolor = fig.get_facecolor()

    theme_manager.change_theme("Dark")
    refresh_palette()
    restyle_figure(fig)
    dark_facecolor = fig.get_facecolor()

    assert light_facecolor != dark_facecolor


def test_restyle_figure_rejects_none() -> None:
    with pytest.raises(ValueError, match="Figure"):
        restyle_figure(None)


def test_build_qss_embeds_palette_colors(theme_manager) -> None:
    from movement_optimizer.gui.stylesheet import build_qss

    theme_manager.change_theme("Dark")
    refresh_palette()
    qss = build_qss()
    assert Palette.BG in qss
    assert Palette.ACCENT in qss
    assert Palette.FG in qss


def test_refresh_motion_palette_tracks_theme(theme_manager) -> None:
    from movement_optimizer.gui import motion_tabs

    theme_manager.change_theme("Light")
    refresh_palette()
    motion_tabs.refresh_motion_palette()
    light_surface = motion_tabs.SURFACE.name()

    theme_manager.change_theme("Dark")
    refresh_palette()
    motion_tabs.refresh_motion_palette()
    dark_surface = motion_tabs.SURFACE.name()

    assert light_surface != dark_surface


def test_main_window_has_theme_menu(theme_manager) -> None:
    from movement_optimizer.gui.main_window import MainWindow

    window = MainWindow()
    try:
        menu_bar = window.menuBar()
        titles = [
            action.menu().title().replace("&", "")
            for action in menu_bar.actions()
            if action.menu() is not None
        ]
        assert any("Theme" in title for title in titles)
    finally:
        window.close()


def test_changing_theme_updates_window_stylesheet(theme_manager) -> None:
    from movement_optimizer.gui.main_window import MainWindow

    window = MainWindow()
    try:
        window.change_theme("Light")
        light_qss = window.styleSheet()
        window.change_theme("Dark")
        dark_qss = window.styleSheet()
        assert light_qss
        assert dark_qss
        assert light_qss != dark_qss
    finally:
        window.close()
