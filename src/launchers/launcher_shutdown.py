"""Explicit teardown of embedded tools when the launcher window closes (UD#12914).

``EmbeddableTool.cleanup`` was only connected to ``QWidget.destroyed``. That
signal fires when the C++ widget is already gone, i.e. after GL contexts the
tool's renderers need have been destroyed, so freeing them then crashed the
process with an access violation. Calling ``cleanup`` from ``closeEvent``,
while every widget and GL context is still alive, makes teardown ordered.
"""

from __future__ import annotations

import logging

from src.shared.python.core.process_safety import narrow_catch
from src.shared.python.launcher_embed import EMBEDDABLE_TOOL_REGISTRY

logger = logging.getLogger(__name__)

__all__ = ["cleanup_embedded_tools"]

# What a tool's teardown can plausibly raise (deleted Qt wrappers, GL and
# file handles, half-initialised state); anything else still propagates.
_CLEANUP_FAILURES: tuple[type[Exception], ...] = (
    RuntimeError,
    OSError,
    AttributeError,
    TypeError,
    ValueError,
)


def cleanup_embedded_tools() -> None:
    """Call ``cleanup()`` on every registered embeddable tool.

    Idempotent (the embed contract requires ``cleanup`` to be) and never
    raises: one failing tool must not stop the others or block window close.
    """
    for tool_id, tool in list(EMBEDDABLE_TOOL_REGISTRY.items()):
        cleanup = getattr(tool, "cleanup", None)
        if not callable(cleanup):
            continue
        with narrow_catch(
            *_CLEANUP_FAILURES, log_message=f"Embedded tool {tool_id!r} cleanup raised"
        ):
            cleanup()
