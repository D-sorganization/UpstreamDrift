"""UnifiedToolsSidebar package.

Provides the tab-based sidebar widget and its factory API.
"""

from __future__ import annotations

from src.shared.python.upstream_drift_tools.ui.tools_sidebar.api import (
    create_tools_sidebar,
)
from src.shared.python.upstream_drift_tools.ui.tools_sidebar.sidebar import (
    SidebarTabDefinition,
    UnifiedToolsSidebar,
)

__all__ = [
    "create_tools_sidebar",
    "SidebarTabDefinition",
    "UnifiedToolsSidebar",
]
