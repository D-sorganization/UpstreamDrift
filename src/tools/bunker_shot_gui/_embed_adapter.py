"""Adapter exposing the BunkerShot3D workbench as an ``EmbeddableTool``.

This is the **only** adapter for ``bunker_shot_gui``. It used to compete with
a second one defined inside :mod:`src.tools.bunker_shot_gui.gui`; the registry
rejects a duplicate ``tool_id``, so one of the two always lost, silently, and
the launcher's view of the tool depended on import order (issue #8618).

The module deliberately imports no Qt at module scope -- the launcher_embed
contract spells widget types as :class:`typing.Any` for exactly this reason --
so importing this package on a machine where PyQt6 does not load still works.
"""

# background: yes (defaults); cleanup idempotent. CPU widget with no scarce
# resources to release on pause, so structural defaults apply (#6013).

from __future__ import annotations

from typing import Any

from src.shared.python.launcher_embed import EmbedCapabilities

__all__ = ["BunkerShotGuiAdapter"]

_MIN_SIZE_PX = (1000, 700)


class BunkerShotGuiAdapter:
    """Implements the ``EmbeddableTool`` protocol for the launcher."""

    tool_id = "bunker_shot_gui"
    display_name = "BunkerShot3D Designer Workbench"

    def __init__(self) -> None:
        """Start with no widget created."""
        self._widget: Any | None = None

    def embed_capabilities(self) -> EmbedCapabilities:
        """Return how this tool wants to be embedded.

        Returns:
            Embeddable as a child widget; the two side-by-side design
            columns and the two map panes need a wide pane to be readable.
        """
        return EmbedCapabilities(
            supports_embedded=True,
            prefers_dock=False,
            min_size=_MIN_SIZE_PX,
            requires_separate_qapplication=False,
        )

    def create_main_widget(self, parent: Any = None) -> Any:
        """Create and return the workbench widget.

        Args:
            parent: The intended Qt parent.

        Returns:
            A new :class:`~src.tools.bunker_shot_gui.gui.BunkerShotWidget`.
        """
        from .gui import BunkerShotWidget

        self._widget = BunkerShotWidget(parent=parent)
        return self._widget

    def cleanup(self) -> None:
        """Release the embedded widget. Idempotent."""
        if self._widget is not None:
            self._widget.cleanup()
        self._widget = None

    def is_dirty(self) -> bool:
        """Return ``False``: the workbench holds no unsaved state."""
        return False
