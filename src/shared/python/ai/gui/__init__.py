"""GUI components for AI Assistant integration.

This package provides PyQt6 widgets for the AI Assistant,
including the conversation panel, settings dialog, and
supporting utilities.

Example:
    >>> from shared.python.ai.gui import AIAssistantPanel, AISettingsDialog
    >>> panel = AIAssistantPanel()
    >>> settings_dialog = AISettingsDialog()
"""

from src.shared.python.ai.gui.assistant_panel import AIAssistantPanel
from src.shared.python.ai.gui.settings_dialog import (
    AIProvider,
    AISettings,
    AISettingsDialog,
    delete_api_key,
    get_api_key,
    set_api_key,
)

__all__ = [
    # Widgets
    "AIAssistantPanel",
    "AISettingsDialog",
    # Settings
    "AISettings",
    "AIProvider",
    # Key management
    "get_api_key",
    "set_api_key",
    "delete_api_key",
]
