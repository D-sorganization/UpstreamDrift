"""Backward-compatible import shims for Frankenstein editor panel classes."""

from __future__ import annotations

from .frankenstein_editor.dialogs import StealComponentDialog
from .frankenstein_editor.panel import ModelPanel

__all__ = ["ModelPanel", "StealComponentDialog"]
