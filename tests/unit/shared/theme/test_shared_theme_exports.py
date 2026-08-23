from __future__ import annotations

import pytest


@pytest.mark.unit
def test_theme_package_exports() -> None:
    from src.shared.python import theme

    assert hasattr(theme, "Colors")
    assert hasattr(theme, "Sizes")
    assert hasattr(theme, "Weights")
    assert hasattr(theme, "get_qfont")
    assert hasattr(theme, "get_display_font")
    assert hasattr(theme, "get_mono_font")
    assert hasattr(theme, "ThemePalette")
    assert hasattr(theme, "get_current_colors")

    assert "Colors" in theme.__all__
    assert "Sizes" in theme.__all__
    assert "Weights" in theme.__all__
    assert "get_qfont" in theme.__all__


@pytest.mark.unit
def test_colors_dynamic_tokens() -> None:
    from src.shared.python.theme import Colors

    assert isinstance(Colors.SUCCESS, str) and Colors.SUCCESS.startswith("#")
    assert isinstance(Colors.ERROR, str) and Colors.ERROR.startswith("#")
    assert isinstance(Colors.WARNING, str) and Colors.WARNING.startswith("#")
    assert isinstance(Colors.INFO, str) and Colors.INFO.startswith("#")
    assert isinstance(Colors.BG_ELEVATED, str) and Colors.BG_ELEVATED.startswith("#")
    assert isinstance(Colors.BG_BASE, str) and Colors.BG_BASE.startswith("#")
    assert isinstance(Colors.PRIMARY, str) and Colors.PRIMARY.startswith("#")
    assert isinstance(Colors.PRIMARY_HOVER, str) and Colors.PRIMARY_HOVER.startswith(
        "#"
    )
    assert isinstance(Colors.TEXT_PRIMARY, str) and Colors.TEXT_PRIMARY.startswith("#")
    assert isinstance(Colors.TEXT_SECONDARY, str) and Colors.TEXT_SECONDARY.startswith(
        "#"
    )
    assert isinstance(Colors.TEXT_TERTIARY, str) and Colors.TEXT_TERTIARY.startswith(
        "#"
    )


@pytest.mark.unit
def test_widgets_theme_available() -> None:
    from src.shared.python.ui import (
        loading_button,
        preferences_dialog,
        recent_models,
        shortcuts_overlay,
        toast,
    )

    assert toast.THEME_AVAILABLE, "toast.THEME_AVAILABLE must be True"
    assert shortcuts_overlay.THEME_AVAILABLE, (
        "shortcuts_overlay.THEME_AVAILABLE must be True"
    )
    assert recent_models.THEME_AVAILABLE, "recent_models.THEME_AVAILABLE must be True"
    assert preferences_dialog.THEME_AVAILABLE, (
        "preferences_dialog.THEME_AVAILABLE must be True"
    )
    assert loading_button.THEME_AVAILABLE, "loading_button.THEME_AVAILABLE must be True"
