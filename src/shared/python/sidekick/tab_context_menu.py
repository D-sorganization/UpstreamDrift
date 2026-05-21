"""Public re-export of the tab context-menu builder (issue #2929).

The actual implementation lives in
:mod:`sidekick.ui.tools_sidebar.tab_context_menu`.  This module provides a
stable top-level import path matching the acceptance-criteria file path from
issue #2929.

Design
------
- **DRY**: no logic is duplicated here.
- **Stable API**: top-level path insulates callers from internal
  restructuring of the ``ui.tools_sidebar`` subpackage.
"""

from __future__ import annotations

from sidekick.ui.tools_sidebar.tab_context_menu import (
    build_tab_context_menu,
    show_tab_context_menu,
)

__all__ = [
    "build_tab_context_menu",
    "show_tab_context_menu",
]
