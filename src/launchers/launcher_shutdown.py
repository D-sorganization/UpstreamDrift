"""Explicit teardown of embedded tools when the launcher window closes (UD#12914).

``EmbeddableTool.cleanup`` was only connected to ``QWidget.destroyed``. That
signal fires when the C++ widget is already gone, i.e. after GL contexts the
tool's renderers need have been destroyed, so freeing them then crashed the
process with an access violation. Calling ``cleanup`` from ``closeEvent``,
while every widget and GL context is still alive, makes teardown ordered.
"""

from __future__ import annotations

import logging

from src.shared.python.launcher_embed import EMBEDDABLE_TOOL_REGISTRY

logger = logging.getLogger(__name__)

__all__ = ["cleanup_embedded_tools"]


def cleanup_embedded_tools() -> None:
    """Call ``cleanup()`` on every registered embeddable tool.

    Idempotent (the embed contract requires ``cleanup`` to be) and never
    raises: one failing tool must not stop the others or block window close.
    """
    for tool_id, tool in list(EMBEDDABLE_TOOL_REGISTRY.items()):
        cleanup = getattr(tool, "cleanup", None)
        if not callable(cleanup):
            continue
        try:
            cleanup()
        except Exception:  # noqa: BLE001 - shutdown must not raise
            logger.exception("Embedded tool %r cleanup raised", tool_id)
