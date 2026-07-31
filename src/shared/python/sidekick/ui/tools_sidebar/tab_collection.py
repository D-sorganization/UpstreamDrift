"""Tab bookkeeping collaborator for :class:`UnifiedToolsSidebar` (F4).

``TabCollection`` owns the id↔widget↔order mapping that was previously
scattered across :class:`~sidekick.ui.tools_sidebar.sidebar.UnifiedToolsSidebar`.
The sidebar delegates all tab-CRUD to this class; it never reaches into
``_tab_ids`` / ``_tab_widgets`` directly.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .qt_compat import QtWidgets

if TYPE_CHECKING:
    from .tab_definition import SidebarTabDefinition

_logger = logging.getLogger(__name__)

__all__ = ["TabCollection"]


class TabCollection:
    """Stable id↔widget↔order bookkeeping for a :class:`~PyQt6.QtWidgets.QTabWidget`.

    Collaborators must call :meth:`add` / :meth:`remove` / :meth:`replace`
    instead of mutating the Qt tab widget directly so that the parallel id
    list stays consistent.

    Args:
        qt_tabs: The :class:`~PyQt6.QtWidgets.QTabWidget` this collection
            wraps.  The widget must already be fully constructed.

    Raises:
        TypeError: If *qt_tabs* is ``None``.
    """

    def __init__(self, qt_tabs: QtWidgets.QTabWidget) -> None:
        if qt_tabs is None:
            raise TypeError("qt_tabs must not be None")
        self._qt_tabs = qt_tabs
        self._tab_ids: list[str] = []
        self._tab_widgets: dict[str, QtWidgets.QWidget] = {}
        self._tab_definitions: dict[str, SidebarTabDefinition] = {}

    # ── Read accessors ────────────────────────────────────────────────────────

    def visible_ids(self) -> list[str]:
        """Return a snapshot of ids in current visual order."""
        return list(self._tab_ids)

    def hidden_ids(self, popout_ids: set[str] | None = None) -> list[str]:
        """Return configured ids that are not visible *and* not popped out.

        Args:
            popout_ids: Optional set of ids currently in pop-out windows.
        """
        floating = popout_ids or set()
        return [
            tid
            for tid in self._tab_definitions
            if tid not in self._tab_ids and tid not in floating
        ]

    def widget_for(self, tab_id: str) -> QtWidgets.QWidget | None:
        """Return the live widget for *tab_id*, or *None*."""
        return self._tab_widgets.get(tab_id)

    def id_at(self, index: int) -> str | None:
        """Return the stable id at visual *index*, or *None*."""
        if 0 <= index < len(self._tab_ids):
            return self._tab_ids[index]
        return None

    def definition_for(self, tab_id: str) -> SidebarTabDefinition | None:
        """Return the :class:`~.tab_definition.SidebarTabDefinition` for *tab_id*."""
        return self._tab_definitions.get(tab_id)

    def all_ids(self) -> list[str]:
        """Return all configured ids (visible + hidden + popped-out)."""
        return list(self._tab_definitions)

    def contains(self, tab_id: str) -> bool:
        """Return ``True`` when *tab_id* is currently docked (visible)."""
        return tab_id in self._tab_ids

    def index_of(self, tab_id: str) -> int:
        """Return the visual index of *tab_id*, or ``-1`` if not docked."""
        try:
            return self._tab_ids.index(tab_id)
        except ValueError:
            return -1

    # ── Mutation helpers ──────────────────────────────────────────────────────

    def set_definitions(
        self,
        definitions: list[SidebarTabDefinition],
    ) -> None:
        """Replace the full set of registered definitions.

        This does **not** clear visible tabs — call :meth:`clear` first
        if you want a hard reset.

        The definition mapping is mutated **in place** (cleared and
        repopulated) rather than rebound to a new dict. The sidebar exposes
        ``self._tab_definitions`` as a live alias into this collaborator
        (see ``UnifiedToolsSidebar.__init__``); rebinding here would orphan
        that alias to a stale empty dict, silently emptying the Configure
        Tabs dialog and disabling the active-tab settings button. Mutating
        in place keeps the alias valid.
        """
        self._tab_definitions.clear()
        self._tab_definitions.update({d.tab_id: d for d in definitions})

    def clear(self) -> None:
        """Remove all tabs from the Qt widget and wipe bookkeeping."""
        self._qt_tabs.clear()
        self._tab_ids.clear()
        self._tab_widgets.clear()

    def add(
        self,
        tab_id: str,
        title: str,
        widget: QtWidgets.QWidget,
    ) -> None:
        """Add a new tab with a stable id.

        Args:
            tab_id: Stable string identifier.  Must be unique within this
                collection.
            title: User-visible tab label.
            widget: Content widget to display inside the tab.

        Raises:
            ValueError: If *tab_id* is already registered.
        """
        if tab_id in self._tab_ids:
            raise ValueError(f"Duplicate sidebar tab id: {tab_id!r}")
        self._tab_ids.append(tab_id)
        self._tab_widgets[tab_id] = widget
        self._qt_tabs.addTab(widget, title)

    def remove(self, tab_id: str) -> bool:
        """Remove the tab with *tab_id* from the Qt widget and bookkeeping.

        Returns ``True`` on success, ``False`` if the id was not visible.
        """
        if tab_id not in self._tab_ids:
            return False
        index = self._tab_ids.index(tab_id)
        widget = self._qt_tabs.widget(index)
        self._qt_tabs.removeTab(index)
        self._tab_ids.pop(index)
        widget_ref = self._tab_widgets.pop(tab_id, None)
        # Qt ownership transfer — neither instance owns the widget after removal.
        target = widget_ref if widget_ref is not None else widget
        if target is not None:
            target.setParent(None)
            target.deleteLater()
        return True

    def replace(
        self,
        old_widget: QtWidgets.QWidget,
        new_widget: QtWidgets.QWidget,
    ) -> bool:
        """Atomically swap *old_widget* for *new_widget* in the Qt tab bar.

        The stable id and order in ``_tab_ids`` are unchanged — only the
        widget reference is replaced.

        Returns:
            ``True`` when the swap succeeded; ``False`` when *old_widget* was
            not found in the current docked tabs.
        """
        tab_id: str | None = None
        for tid, w in self._tab_widgets.items():
            if w is old_widget:
                tab_id = tid
                break
        if tab_id is None:
            return False

        index = self._qt_tabs.indexOf(old_widget)
        if index < 0:
            return False

        title = self._qt_tabs.tabText(index)
        tooltip = self._qt_tabs.tabToolTip(index)

        self._qt_tabs.removeTab(index)
        self._qt_tabs.insertTab(index, new_widget, title)
        if tooltip:
            self._qt_tabs.setTabToolTip(index, tooltip)
        self._qt_tabs.setCurrentIndex(index)

        self._tab_widgets[tab_id] = new_widget

        old_widget.setParent(None)
        old_widget.deleteLater()
        return True

    def sync_order_from_widget(self) -> list[str]:
        """Re-build ``_tab_ids`` by walking the Qt tab bar's current order.

        The backing ``list`` is mutated **in place** (sliced reassignment)
        rather than rebound, so live aliases held by collaborators keep
        observing current state (issue #3138).

        Returns the new id list (may be shorter than expected if widgets
        have been orphaned).
        """
        ordered: list[str] = []
        for index in range(self._qt_tabs.count()):
            widget = self._qt_tabs.widget(index)
            for tab_id, tab_widget in self._tab_widgets.items():
                if tab_widget is widget:
                    ordered.append(tab_id)
                    break
        if len(ordered) == len(self._tab_ids):
            self._tab_ids[:] = ordered
        return list(self._tab_ids)
