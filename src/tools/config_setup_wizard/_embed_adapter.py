"""Embeddable-tool adapter for the canonical-core setup wizard."""

from __future__ import annotations

from typing import Any

from src.shared.python.launcher_embed import EmbedCapabilities
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

__all__ = ["ConfigSetupWizardAdapter"]


class ConfigSetupWizardAdapter:
    """Expose the deterministic setup wizard through the launcher contract."""

    tool_id = "config_setup_wizard"
    display_name = "Setup Wizard"

    def __init__(self) -> None:
        self._widgets: list[Any] = []

    def embed_capabilities(self) -> EmbedCapabilities:
        """Return how this tool wants to be embedded."""

        return EmbedCapabilities(
            supports_embedded=True,
            prefers_dock=False,
            min_size=(720, 520),
            requires_separate_qapplication=False,
        )

    def create_main_widget(self, parent: Any = None) -> Any:
        """Create and return the wizard's main widget."""

        from .gui import ConfigSetupWizardWidget

        widget = ConfigSetupWizardWidget(parent=parent)
        self._widgets.append(widget)
        return widget

    def cleanup(self) -> None:
        """Release widget references and call widget cleanup hooks."""

        widgets, self._widgets = self._widgets, []
        for widget in widgets:
            cleanup = getattr(widget, "cleanup", None)
            if callable(cleanup):
                try:
                    cleanup()
                except Exception:  # pragma: no cover - defensive
                    logger.exception("config setup wizard widget cleanup raised")

    def is_dirty(self) -> bool:
        """The wizard holds no unsaved state itself."""

        return False


def _register_adapter() -> None:
    """Register this adapter when the bootstrap imports the module directly."""

    from src.shared.python.launcher_embed import (
        get_embeddable_tool,
        register_embeddable_tool,
    )

    adapter = ConfigSetupWizardAdapter()
    if get_embeddable_tool(adapter.tool_id) is None:
        register_embeddable_tool(adapter)


_register_adapter()
