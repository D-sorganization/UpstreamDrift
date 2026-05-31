"""PyQt6-free embed adapter for the Simulation Backends tile.

The launcher bootstrap (:mod:`src.launchers.embedded_tool_bootstrap`) imports
this module to register the tile, and that bootstrap runs in environments where
PyQt6 may be absent. This module therefore imports **no** Qt binding and **no**
``gui`` module at import time: the only top-level import is the PyQt6-free
launcher-embed contract. The actual widget class is imported lazily inside
:meth:`_EmbedAdapter.create_main_widget`.
"""

# background: yes (defaults); cleanup idempotent (drops widget ref). Holds only
# transient in-memory traces; no scarce GPU context at the adapter level, so
# structural defaults apply (#6013).

from __future__ import annotations

from typing import Any

from src.shared.python.launcher_embed import (
    EmbedCapabilities,
    register_embeddable_tool,
)

#: Stable registry id; must match the launcher manifest entry.
TOOL_ID = "simulation_backends"

#: Minimum embedded size (width, height) in pixels. The tool packs a backend
#: selector, a parameter form, run controls, a matplotlib canvas, and a report
#: pane, so it needs a reasonably large minimum footprint.
_MIN_SIZE = (820, 560)


class _EmbedAdapter:
    """Embeddable-tool adapter for the Simulation Backends tile.

    Implements the :class:`~src.shared.python.launcher_embed.EmbeddableTool`
    protocol so the launcher can host the tool as a tab or dock widget. The
    PyQt6 widget is constructed lazily so importing this module never requires
    a Qt binding.
    """

    tool_id = TOOL_ID

    def __init__(self) -> None:
        # Typed as ``Any`` because the concrete widget type (``MainWidget``)
        # must not be imported at module top level (PyQt6-free contract).
        self._widget: Any | None = None

    def embed_capabilities(self) -> EmbedCapabilities:
        """Return how this tool wants to be embedded in the launcher."""
        return EmbedCapabilities(
            supports_embedded=True,
            prefers_dock=False,
            min_size=_MIN_SIZE,
            requires_separate_qapplication=False,
        )

    def create_main_widget(self, parent: Any) -> Any:
        """Construct and return the tool's top-level widget.

        The :class:`~src.tools.simulation_backends_launcher.gui.MainWidget`
        import is deliberately performed inside the method so that merely
        importing this adapter (as the bootstrap does) does not pull in PyQt6.

        Args:
            parent: The intended Qt parent (``QWidget | None``).

        Returns:
            A freshly constructed ``MainWidget`` parented to ``parent``.
        """
        from src.tools.simulation_backends_launcher.gui import MainWidget

        self._widget = MainWidget(parent=parent)
        return self._widget

    def cleanup(self) -> None:
        """Release the embedded widget reference.

        Idempotent: safe to call multiple times during shutdown.
        """
        self._widget = None

    def is_dirty(self) -> bool:
        """Return ``True`` if the tool holds unsaved state.

        The tool only holds transient in-memory traces, so it never reports a
        dirty state.
        """
        return False


# Register the adapter at import time so the launcher bootstrap picks it up.
register_embeddable_tool(_EmbedAdapter())
