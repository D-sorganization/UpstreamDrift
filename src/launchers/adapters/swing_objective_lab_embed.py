"""Embeddable-tool adapter for the Tools-provided Swing Objective Lab.

Implements the :class:`~src.shared.python.launcher_embed.EmbeddableTool`
protocol (ADR-0013) so the launcher resolves this tile through the
process-wide registry rather than the deprecated ``embed_adapter`` string in
``models.yaml``.

The widget itself lives in the ``pendulum_simulator`` provider under
``../Tools``, which the launcher puts on ``sys.path`` during bootstrap. The
adapter lives here, on the consumer side, so the provider never has to import
UpstreamDrift — the same direction of dependency as
:mod:`src.launchers.adapters.simscape_embed`.

Tools epic #4766, child #4772.
"""

from __future__ import annotations

from typing import Any

from src.shared.python.launcher_embed import (
    EmbedCapabilities,
    get_embeddable_tool,
    register_embeddable_tool,
)
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

__all__ = ["TOOL_ID", "_SwingObjectiveLabEmbedAdapter"]

TOOL_ID = "swing_objective_lab"
"""Tool id the adapter registers under, matching the ``models.yaml`` tile."""

_MIN_SIZE = (1180, 760)


class _SwingObjectiveLabEmbedAdapter:
    """Adapter exposing the Swing Objective Lab through the embed contract."""

    tool_id: str = TOOL_ID

    def __init__(self) -> None:
        """Track handed-out widgets so cleanup can dispose of them."""
        self._widgets: list[Any] = []

    def embed_capabilities(self) -> EmbedCapabilities:
        """Declare how the launcher may host this surface."""
        return EmbedCapabilities(
            supports_embedded=True,
            prefers_dock=False,
            min_size=_MIN_SIZE,
            requires_separate_qapplication=False,
        )

    def create_main_widget(self, parent: Any) -> Any:
        """Build the comparison surface.

        The import is deferred because the provider pulls in PyQt6, SciPy and
        the pendulum physics stack; keeping it here lets this module — and its
        registry side effect — import cleanly in headless contexts.
        """
        from double_pendulum_golf.swing_objectives._embed_adapter import (
            get_dockable_ui,
        )

        widget = get_dockable_ui(parent)
        self._widgets.append(widget)
        return widget

    def cleanup(self) -> None:
        """Dispose of every widget handed out. Idempotent and never raises."""
        widgets, self._widgets = self._widgets, []
        for widget in widgets:
            try:
                widget.deleteLater()
            except Exception:  # pragma: no cover - defensive
                logger.exception("swing_objective_lab widget cleanup raised")

    def is_dirty(self) -> bool:
        """Report unsaved state.

        The lab recomputes every comparison from its controls and writes
        nothing implicitly, so there is never anything to prompt about on close.
        """
        return False


# Register on first import, guarding against re-registration so importing this
# module from both bootstrap and tests does not raise.
_ADAPTER = _SwingObjectiveLabEmbedAdapter()
if get_embeddable_tool(_ADAPTER.tool_id) is None:
    register_embeddable_tool(_ADAPTER)
