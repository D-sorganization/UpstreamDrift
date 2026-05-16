"""UnifiedToolsSidebar package.

Provides the tab-based sidebar widget and its factory API.
Issue #5616: adds WorkspaceRegistry, LayoutMode, WorkspaceTableModel,
PythonReplWidget, and build_workspace_tab.
"""

from __future__ import annotations

from .api import (
    create_tools_sidebar,
)
from .registry import (
    Subscription,
    WorkspaceRegistry,
    WorkspaceVariable,
)
from .sidebar import (
    LayoutMode,
    SidebarTabDefinition,
    UnifiedToolsSidebar,
)

__all__ = [
    "LayoutMode",
    "Subscription",
    "SidebarTabDefinition",
    "UnifiedToolsSidebar",
    "WorkspaceRegistry",
    "WorkspaceVariable",
    "create_tools_sidebar",
]
