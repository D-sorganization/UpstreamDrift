"""Public API for the UnifiedToolsSidebar.

Provides factory functions used by the launcher embed adapter and external
callers to create the sidebar widget.
"""

from __future__ import annotations

from typing import Any


def create_tools_sidebar(parent: Any = None) -> Any:
    """Create and return a new UnifiedToolsSidebar instance.

    Args:
        parent: Optional parent widget for the sidebar.

    Returns:
        A UnifiedToolsSidebar widget.
    """
    from src.shared.python.upstream_drift_tools.ui.tools_sidebar.sidebar import (
        UnifiedToolsSidebar,
    )

    return UnifiedToolsSidebar(parent=parent)


__all__ = ["create_tools_sidebar"]
