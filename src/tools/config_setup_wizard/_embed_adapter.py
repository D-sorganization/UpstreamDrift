"""Embeddable-tool adapter for the canonical-core setup wizard."""

from __future__ import annotations

from typing import Any

from src.shared.python.launcher_embed import EmbedCapabilities
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

__all__ = ["ConfigSetupWizardAdapter", "get_dockable_ui"]


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

        # Absolute, not relative: the launcher loads this file by path
        # (``spec_from_file_location``) and also runs it as a script, and in
        # both cases the module has no parent package, so ``from .gui`` died
        # with "attempted relative import with no known parent package"
        # (#8067). The module's other imports are already absolute.
        from src.tools.config_setup_wizard.gui import ConfigSetupWizardWidget

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


def get_dockable_ui(parent: Any = None) -> Any:
    """Return the wizard widget for the launcher's dockable-tile contract.

    ``models.yaml`` points the Setup Wizard tile at this module, and
    ``SpecialAppHandler.get_dockable_ui`` discovers tiles by loading their
    target file and looking for a module-level ``get_dockable_ui``. Without
    it the handler fell through to running this module as a *script*, which
    only re-registers the adapter and exits -- so the launcher reported
    "Setup Wizard Launched" while no window or tab ever appeared (#8067).
    """
    return ConfigSetupWizardAdapter().create_main_widget(parent)


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


def _run_standalone() -> int:
    """Show the wizard in its own window when this module is run as a script.

    The launcher falls back to spawning the tile's target file as a
    subprocess whenever embedding is unavailable. Without a ``__main__``
    body that path started a process which exited instantly, and the user
    saw a success toast with nothing behind it (#8067).
    """
    import sys

    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv)
    widget = get_dockable_ui()
    widget.setWindowTitle(ConfigSetupWizardAdapter.display_name)
    widget.resize(*ConfigSetupWizardAdapter().embed_capabilities().min_size)
    widget.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(_run_standalone())
