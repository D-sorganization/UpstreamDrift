"""Public re-export of the selected-tab settings panel (issue #2929).

The actual implementation lives in
:mod:`sidekick.ui.tools_sidebar.tab_settings_panel`.  This module provides
a stable top-level import path that matches the acceptance-criteria path
specified in issue #2929.

Design
------
- **DRY**: no logic is duplicated here; all classes are re-exported.
- **Stable API**: callers importing from ``sidekick.selected_tab_panel``
  are insulated from internal restructuring of the ``ui.tools_sidebar``
  subpackage.
"""

from __future__ import annotations

from sidekick.ui.tools_sidebar.tab_settings_panel import (
    SIDEKICK_TAB_SETTINGS_BUTTON_OBJECT_NAME,
    TabSettingsMixin,
    build_tab_settings_dialog,
    build_tab_settings_toolbar,
)

__all__ = [
    "SIDEKICK_TAB_SETTINGS_BUTTON_OBJECT_NAME",
    "TabSettingsMixin",
    "build_tab_settings_dialog",
    "build_tab_settings_toolbar",
]
