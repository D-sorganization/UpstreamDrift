"""Protocol interfaces for the theme system.

These protocols define the structural typing contracts for theme
management, allowing GUI applications to depend on the interface
rather than a concrete ThemeManager implementation. This enables
testing with lightweight stubs and avoids hard PyQt6 dependencies
in non-GUI code.

Usage:
    def apply_colors(provider: ThemeProvider) -> dict[str, str]:
        return provider.get_current_colors()
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ThemeProvider(Protocol):
    """Protocol for providing theme information.

    Any object that can report available themes, the current theme,
    and current colors satisfies this protocol. The full ThemeManager
    (PyQt6-dependent) implements it, but so can a lightweight test stub.
    """

    def get_available_themes(self) -> list[str]:
        """Return the names of all registered themes."""
        ...

    def get_current_colors(self) -> dict[str, str]:
        """Return the active theme's color map."""
        ...

    @property
    def current_theme_name(self) -> str:
        """The identifier of the currently active theme."""
        ...


@runtime_checkable
class ThemeSwitcher(Protocol):
    """Protocol for objects that can switch the active theme.

    Extends ThemeProvider with the ability to change the theme
    and persist the choice.
    """

    def change_theme(self, name: str) -> None:
        """Switch to the theme identified by *name*."""
        ...

    def get_available_themes(self) -> list[str]:
        """Return the names of all registered themes."""
        ...


@runtime_checkable
class StylesheetGenerator(Protocol):
    """Protocol for generating Qt stylesheets from theme colors.

    Implementations accept a color dictionary and produce a CSS-like
    stylesheet string suitable for ``QWidget.setStyleSheet()``.
    """

    def generate(self, colors: dict[str, str]) -> str:
        """Generate a Qt stylesheet string from *colors*."""
        ...
