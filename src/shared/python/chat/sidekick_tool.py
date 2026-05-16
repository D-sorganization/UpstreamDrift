"""Embeddable-tool adapter for the Sidekick AI chat assistant.

Implements the :class:`~src.shared.python.launcher_embed.EmbeddableTool`
protocol so the launcher can host the Sidekick chat panel as a dockable
side panel instead of a standalone window.

Designed to be headless-safe: PyQt6 is imported lazily inside
:meth:`SidekickTool.create_main_widget` so that importing this module
during headless test collection never fails due to a missing Qt
installation.

Issue #5468 — ADR-0013 launcher composability.
"""

from __future__ import annotations

import logging
from typing import Any

from src.shared.python.launcher_embed import (
    EmbedCapabilities,
    get_embeddable_tool,
    register_embeddable_tool,
)

logger = logging.getLogger(__name__)

__all__ = ["SidekickTool"]

_TOOL_ID = "chat_assistant"


def _create_chat_dock_widget(parent: Any) -> Any:
    """Lazily construct a :class:`ChatDockWidget`.

    Isolated into its own function so tests can patch it without having
    to import PyQt6.

    Args:
        parent: The Qt parent widget (``QWidget | None``).

    Returns:
        A new :class:`~src.shared.python.chat.ChatDockWidget` instance.
    """
    from src.shared.python.chat.chat_dock_widget import ChatDockWidget  # noqa: PLC0415

    return ChatDockWidget(parent=parent)


class SidekickTool:
    """Embeddable-tool adapter for the Sidekick AI chat assistant.

    Exposes :class:`~src.shared.python.chat.ChatDockWidget` through the
    :class:`~src.shared.python.launcher_embed.EmbeddableTool` Protocol so
    the launcher can embed it as a dock panel.

    Attributes:
        tool_id: Stable registry identifier, matches the ``id`` in
            ``src/config/models.yaml``.
    """

    tool_id: str = _TOOL_ID

    def __init__(self) -> None:
        self._widgets: list[Any] = []

    def embed_capabilities(self) -> EmbedCapabilities:
        """Return embedding preferences for the Sidekick chat panel.

        The panel prefers to be docked on the side rather than shown in
        a central tab, but the host may override this hint when dock
        layouts are unavailable.

        Returns:
            :class:`EmbedCapabilities` with ``prefers_dock=True``.
        """
        return EmbedCapabilities(
            supports_embedded=True,
            prefers_dock=True,
            min_size=(320, 480),
            requires_separate_qapplication=False,
        )

    def create_main_widget(self, parent: Any) -> Any:
        """Construct and return a :class:`ChatDockWidget` for embedding.

        Preconditions:
            ``parent`` is ``None`` or a ``QWidget`` instance.

        Args:
            parent: The intended Qt parent (``QWidget | None``).

        Returns:
            A new :class:`~src.shared.python.chat.ChatDockWidget`.

        Raises:
            TypeError: If ``parent`` is not ``None`` and is not a
                ``QWidget``.
        """
        # DbC: validate parent type without importing PyQt6 at module level.
        if parent is not None:
            try:
                from PyQt6.QtWidgets import QWidget  # noqa: PLC0415

                if not isinstance(parent, QWidget):
                    raise TypeError(
                        "create_main_widget: parent must be a QWidget or None, "
                        f"got {type(parent).__name__}"
                    )
            except ImportError:
                # PyQt6 not available — accept any truthy parent and log a
                # warning so the issue surfaces in CI without crashing.
                logger.warning(
                    "create_main_widget: PyQt6 not available; parent type check skipped"
                )

        widget = _create_chat_dock_widget(parent)
        self._widgets.append(widget)
        logger.debug("SidekickTool: created ChatDockWidget (parent=%r)", parent)
        return widget

    def cleanup(self) -> None:
        """Release resources held by all widgets handed out so far.

        Idempotent: hosts may call this multiple times during shutdown.
        """
        widgets, self._widgets = self._widgets, []
        for widget in widgets:
            try:
                widget.close()
            except Exception:  # noqa: BLE001 - defensive, never raise in cleanup
                logger.exception("SidekickTool: widget close raised")

    def is_dirty(self) -> bool:
        """Return ``False`` — the chat panel does not track unsaved state."""
        return False


# ---------------------------------------------------------------------------
# Register on first import; guard against duplicate registration so
# importing from both the package __init__ and a test is idempotent.
# ---------------------------------------------------------------------------
_ADAPTER = SidekickTool()
if get_embeddable_tool(_TOOL_ID) is None:
    register_embeddable_tool(_ADAPTER)
