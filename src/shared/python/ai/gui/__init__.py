"""GUI components for AI Assistant integration.

This package provides PyQt6 widgets for the AI Assistant,
including the conversation panel, settings dialog, and
supporting utilities.

Example:
    >>> from shared.python.ai.gui import AIAssistantPanel, AISettingsDialog
    >>> panel = AIAssistantPanel()
    >>> settings_dialog = AISettingsDialog()
"""

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


from typing import Any


def __getattr__(name: str) -> Any:
    if name == "AIAssistantPanel":
        from shared.python.ai.gui.assistant_panel import AIAssistantPanel

        return AIAssistantPanel
    if name == "AISettings":
        from shared.python.ai._settings_model import AISettings

        return AISettings
    if name == "AIProvider":
        from shared.python.ai.gui._provider_registry_data import AIProvider

        return AIProvider
    if name in ("AISettingsDialog", "get_api_key", "set_api_key", "delete_api_key"):
        import shared.python.ai.gui.settings_dialog as sd

        return getattr(sd, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
