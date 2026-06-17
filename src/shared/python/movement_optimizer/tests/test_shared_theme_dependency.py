# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Contract test guarding the hard dependency on the fleet shared theme.

Movement-Optimizer depends on the ``ud-tools`` ``shared.python.theme`` package
for its color themes. This test fails loudly if that dependency is missing or
its public surface drifts, so a broken CI install is caught immediately rather
than as an obscure import error deep in the GUI.
"""

from __future__ import annotations

import pytest

pytest.importorskip("shared.python.theme")  # fleet theme only; skip on bare envs


def test_shared_theme_public_surface_is_importable() -> None:
    from shared.python.theme import (
        BUILTIN_THEMES,
        THEME_COLOR_KEYS,
        ThemedWindowMixin,
        generate_stylesheet,
        get_theme_manager,
    )
    from shared.python.theme.matplotlib_style import apply_plot_theme, get_chart_color

    # The names must be the real callables/objects, not None placeholders.
    assert callable(get_theme_manager)
    assert isinstance(ThemedWindowMixin, type)
    assert callable(generate_stylesheet)
    assert callable(apply_plot_theme)
    assert callable(get_chart_color)

    # The themes we map onto must exist with the keys the Palette consumes.
    assert "Dark" in BUILTIN_THEMES
    assert "Light" in BUILTIN_THEMES
    required = {"bg", "group_bg", "input_bg", "text", "text_secondary", "accent", "button_hover"}
    assert required.issubset(set(THEME_COLOR_KEYS))
    assert required.issubset(set(BUILTIN_THEMES["Dark"]))


def test_get_chart_color_returns_hex_strings() -> None:
    from shared.python.theme.matplotlib_style import get_chart_color

    for index in range(3):
        color = get_chart_color(index)
        assert isinstance(color, str)
        assert color.startswith("#")
