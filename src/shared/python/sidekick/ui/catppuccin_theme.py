"""Catppuccin Mocha theme - shared color palette and stylesheet.

This module centralises the Catppuccin Mocha color palette and the
standard application-wide QSS stylesheet.
"""

from __future__ import annotations

from src.shared.python.theme.catppuccin import CATPPUCCIN_MOCHA as COLORS
from src.shared.python.theme.catppuccin import get_stylesheet

__all__ = [
    "COLORS",
    "get_stylesheet",
]
