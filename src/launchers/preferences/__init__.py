"""Launcher preferences subpages.

This package is the UpstreamDrift launcher's collection of preferences
panes. Most heavy lifting lives in shared Tools (`src/shared/python/...`)
— the modules here are thin embed wrappers so this launcher and the
Gasification_Model launcher render the same widget for shared concerns.

Subpages:

* :mod:`terminal_section` — default shell selection (Tools #2882)
* :mod:`workspace_section` — workspace layout-mode default (Tools #2883)
* :mod:`mcp_servers_section` — MCP server table editor; wraps
  :class:`McpServersPrefsWidget` from Tools (Tools #2884 / #2914)
* :mod:`jupyter_section` — notebook directory + kernel prefs (Tools #2889)

Each subpage is self-contained (orthogonal); removing or adding one
does not affect the others. They share a single helper
:func:`build_prefs_section` that wraps a list of widget rows in a
group-box header to keep DRY out of the per-section code.
"""

from __future__ import annotations

from src.launchers.preferences.mcp_servers_section import McpServersSection

__all__ = [
    "McpServersSection",
    "build_prefs_section",
]


def build_prefs_section(
    section_id: str,
    label: str,
    widgets: list,
) -> object:
    """Return a QWidget group-box containing *widgets* under *label*.

    Lazily imports PyQt6 so this module remains importable in headless
    contexts where the Qt binding is unavailable.

    Args:
        section_id: Stable identifier (used as objectName for tests).
        label: Human-readable section title.
        widgets: Ordered list of QWidget rows to add.

    Returns:
        A new QGroupBox containing the widgets in a vertical layout.

    Raises:
        ValueError: If *section_id* or *label* is empty.
    """
    if not section_id:
        raise ValueError("section_id must be non-empty")
    if not label:
        raise ValueError("label must be non-empty")
    if widgets is None:
        raise ValueError("widgets must be a list (possibly empty)")

    from PyQt6.QtWidgets import QGroupBox, QVBoxLayout

    box = QGroupBox(label)
    box.setObjectName(f"prefs_section_{section_id}")
    layout = QVBoxLayout(box)
    for widget in widgets:
        layout.addWidget(widget)
    return box
