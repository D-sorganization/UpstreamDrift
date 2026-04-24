"""Regression tests ensuring launcher_theme.py uses correct ThemeManager API.

Verifies that launcher code references methods that actually exist on
ThemeManager, preventing silent failures in the theme menu.
"""

import inspect

from src.launchers import launcher_theme


class TestLauncherThemeApiCalls:
    """Parse launcher_theme.py and verify all ThemeManager method calls exist."""

    def test_no_set_theme_calls(self) -> None:
        """set_theme was renamed to change_theme."""
        source = inspect.getsource(launcher_theme)
        assert "manager.set_theme(" not in source, (
            "launcher_theme.py still calls manager.set_theme() — use change_theme()"
        )

    def test_no_theme_name_property(self) -> None:
        """theme_name was renamed to get_current_theme_name()."""
        source = inspect.getsource(launcher_theme)
        assert ".theme_name" not in source or "get_current_theme_name" in source, (
            "launcher_theme.py uses .theme_name — use get_current_theme_name()"
        )

    def test_no_load_saved_theme(self) -> None:
        """load_saved_theme() was removed."""
        source = inspect.getsource(launcher_theme)
        assert "load_saved_theme" not in source, (
            "launcher_theme.py calls load_saved_theme() which no longer exists"
        )

    def test_no_get_available_fleet_themes(self) -> None:
        """get_available_fleet_themes() was renamed to get_available_themes()."""
        source = inspect.getsource(launcher_theme)
        assert "get_available_fleet_themes" not in source, (
            "launcher_theme.py calls get_available_fleet_themes() — use get_available_themes()"
        )

    def test_no_set_fleet_theme(self) -> None:
        """set_fleet_theme() was removed."""
        source = inspect.getsource(launcher_theme)
        assert "set_fleet_theme" not in source, (
            "launcher_theme.py calls set_fleet_theme() — use change_theme()"
        )
