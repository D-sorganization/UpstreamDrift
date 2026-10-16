"""Launcher-facing theme names that the pinned Tools theme package lacks (#12901).

``launcher_theme`` and the preferences dialog historically imported
``ThemePreset`` and ``apply_golf_suite_style`` from the package root. The
Tools-owned ``shared.python.theme`` no longer exports them, so the import failed
and the launcher logged "Theme system unavailable". This UD-owned module (kept
on the canonical ``__path__`` by the seam redirect) is their concrete home.
"""

from __future__ import annotations

import enum

from src.shared.python.theme.matplotlib_style import apply_global_style

__all__ = ["ThemePreset", "apply_golf_suite_style"]


class ThemePreset(enum.Enum):
    """Core launcher theme presets; values are the ThemeManager theme names."""

    DARK = "Dark"
    LIGHT = "Light"
    HIGH_CONTRAST = "High Contrast"


def apply_golf_suite_style() -> None:
    """Apply the suite-wide matplotlib defaults (delegates to Tools)."""
    apply_global_style()
