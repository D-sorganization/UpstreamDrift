from __future__ import annotations

from upstream_drift_tools.ui.catppuccin_theme import COLORS, get_stylesheet


def test_colors_dict() -> None:
    """Test that the COLORS dict has the expected keys and format."""
    assert isinstance(COLORS, dict)
    assert len(COLORS) > 0
    # Check a few specific colors
    assert "base" in COLORS
    assert COLORS["base"].startswith("#")
    assert "blue" in COLORS
    assert "text" in COLORS


def test_get_stylesheet() -> None:
    """Test that getting the stylesheet returns a valid string with colors."""
    stylesheet = get_stylesheet()
    assert isinstance(stylesheet, str)
    assert len(stylesheet) > 0

    # Check that it contains some QSS rules
    assert "QMainWindow" in stylesheet
    assert "QPushButton:hover" in stylesheet

    # Check that it injected colors
    assert COLORS["base"] in stylesheet
    assert COLORS["blue"] in stylesheet
