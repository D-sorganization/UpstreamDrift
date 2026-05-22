"""Embeddable-tool adapter for the C3D Motion Analysis viewer.

Implements the :class:`~src.shared.python.launcher_embed.EmbeddableTool`
protocol so the launcher can host the viewer as a tab or dock widget
instead of spawning a standalone process.

The viewer uses matplotlib canvases inside its plot tabs. :meth:`cleanup`
releases all matplotlib figures via ``plt.close('all')`` so closing the
embedded tab does not leak figure references in the host process.
"""

from __future__ import annotations

from typing import Any

from src.shared.python.launcher_embed import EmbedCapabilities

__all__ = ["_C3DViewerEmbedAdapter"]


class _C3DViewerEmbedAdapter:
    """Adapter exposing the viewer's ``MainWidget`` through the embed contract."""

    tool_id: str = "c3d_viewer"

    def embed_capabilities(self) -> EmbedCapabilities:
        return EmbedCapabilities(
            supports_embedded=True,
            prefers_dock=False,
            min_size=(900, 600),
            requires_separate_qapplication=False,
        )

    def create_main_widget(self, parent: Any) -> Any:
        # Lazy import: the viewer pulls in PyQt6 + matplotlib + the C3D
        # reader chain. Keeping this import inside ``create_main_widget``
        # lets the adapter (and therefore the registry side-effect at
        # import time) load cleanly in headless contexts.
        from .c3d_viewer import MainWidget

        self._widget = MainWidget(parent)
        return self._widget

    def cleanup(self) -> None:
        """Release matplotlib figures held by the plot tabs.

        Must be idempotent; explicitly closes only the figures owned by
        this tool's canvases. Wrapped in a defensive try/except so a
        misbehaving matplotlib never blocks host shutdown.
        """
        try:
            import matplotlib.pyplot as plt

            if not hasattr(self, "_widget") or self._widget is None:
                return

            w = self._widget
            canvases = [
                w.viewer3d_tab.canvas_3d,
                w.marker_plot_tab.canvas_marker,
                w.analog_plot_tab.canvas_analog,
                w.analysis_tab.canvas_analysis,
                w.force_plot_tab.time_series_canvas,
                w.force_plot_tab.cop_canvas,
            ]
            for canvas in canvases:
                if hasattr(canvas, "fig"):
                    plt.close(canvas.fig)
        except Exception:  # pragma: no cover - defensive  # noqa: BLE001
            # Host shutdown must not depend on us. Swallow rather than
            # raise; the registry contract requires ``cleanup`` to be
            # idempotent and non-fatal.
            pass

    def is_dirty(self) -> bool:
        # The viewer is read-only on the input C3D file; the export
        # dialog writes to a user-chosen path each time. Nothing to
        # prompt the user about on close.
        return False
