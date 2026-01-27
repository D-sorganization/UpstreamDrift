import pytest
import sys
from unittest.mock import MagicMock, patch

# Mock PyQt6 if not available to ensure tests run in CI without GUI libs
try:
    from PyQt6.QtCore import QObject
except ImportError:
    # We rely on the fallback in theme_manager, but for tests we might want to ensure it works
    pass

from src.shared.python.theme.colors import Colors, LIGHT_PALETTE, DARK_PALETTE, get_active_palette
from src.shared.python.theme.theme_manager import ThemeManager
from src.shared.python.theme.matplotlib_style import apply_golf_suite_style
import matplotlib.pyplot as plt

def test_theme_switching_updates_colors():
    manager = ThemeManager.get_instance()

    # Start with default (Dark)
    manager.set_theme("Dark")
    assert get_active_palette() == DARK_PALETTE
    # Verify proxy works
    assert Colors.BG_BASE == DARK_PALETTE.BG_BASE

    # Switch to Light
    manager.set_theme("Light")
    assert get_active_palette() == LIGHT_PALETTE
    assert Colors.BG_BASE == LIGHT_PALETTE.BG_BASE

    # Switch back to Dark
    manager.set_theme("Dark")
    assert get_active_palette() == DARK_PALETTE
    assert Colors.BG_BASE == DARK_PALETTE.BG_BASE

def test_theme_switching_updates_matplotlib():
    manager = ThemeManager.get_instance()

    # Set to Light
    manager.set_theme("Light")
    apply_golf_suite_style()

    # Check if rcParams updated
    # Note: matplotlib normalizes colors (e.g., to lowercase or rgba), so exact string match might require normalization
    # But usually hex strings are preserved or convertible.

    # Colors.BG_BASE for Light is #FFFFFF
    assert plt.rcParams["figure.facecolor"].lower() == LIGHT_PALETTE.BG_BASE.lower()

    # Set to Dark
    manager.set_theme("Dark")
    apply_golf_suite_style()

    # Colors.BG_BASE for Dark is #1A1A1A
    assert plt.rcParams["figure.facecolor"].lower() == DARK_PALETTE.BG_BASE.lower()

def test_theme_signals():
    manager = ThemeManager.get_instance()

    # Mock slot
    mock_slot = MagicMock()

    # Connect signal
    # Note: If PyQt6 is real, this works. If stub, it also works (stub has connect method).
    manager.theme_changed.connect(mock_slot)

    manager.set_theme("Light")
    mock_slot.assert_called_with("Light")

    manager.set_theme("Dark")
    mock_slot.assert_called_with("Dark")

    # Test no emit if same theme
    mock_slot.reset_mock()
    manager.set_theme("Dark")
    mock_slot.assert_not_called()
