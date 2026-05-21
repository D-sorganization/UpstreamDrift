"""Custom tab display-name helpers for the Sidekick sidebar."""

from __future__ import annotations

from typing import Any

from .qt_compat import QtWidgets


class TabDisplayNameMixin:
    """Mixin for stable tab ids with persisted user-facing display names."""

    _state: Any
    _tab_definitions: dict[str, Any]
    _tab_ids: list[str]
    _popout_windows: dict[str, QtWidgets.QMainWindow]
    tabs: QtWidgets.QTabWidget

    def tab_display_name(self, tab_id: str) -> str:
        """Return the user-facing name for ``tab_id``."""
        definition = self._tab_definitions.get(tab_id)
        if definition is None:
            raise KeyError(tab_id)
        return self._tab_display_name(tab_id, definition.title)

    def rename_tab(self, tab_id: str, title: str) -> None:
        """Persist a custom display name and refresh any visible tab label."""
        definition = self._tab_definitions.get(tab_id)
        if definition is None:
            raise KeyError(tab_id)
        normalized = title.strip()
        if not normalized:
            raise ValueError("Sidekick tab display name must be non-empty")
        if normalized == definition.title:
            self._state.tab_display_names.pop(tab_id, None)
        else:
            self._state.tab_display_names[tab_id] = normalized
        self._refresh_tab_display_name(tab_id)
        self._emit_context()

    def reset_tab_display_name(self, tab_id: str) -> None:
        """Restore the default display name for ``tab_id``."""
        if tab_id not in self._tab_definitions:
            raise KeyError(tab_id)
        self._state.tab_display_names.pop(tab_id, None)
        self._refresh_tab_display_name(tab_id)
        self._emit_context()

    def _tab_display_name(self, tab_id: str, default_title: str) -> str:
        return self._state.tab_display_names.get(tab_id, default_title)

    def _refresh_tab_display_name(self, tab_id: str) -> None:
        definition = self._tab_definitions.get(tab_id)
        if definition is None:
            return
        display_name = self._tab_display_name(tab_id, definition.title)
        if tab_id in self._tab_ids:
            self.tabs.setTabText(self._tab_ids.index(tab_id), display_name)
        popout = self._popout_windows.get(tab_id)
        if popout is not None:
            popout.setWindowTitle(f"Sidekick - {display_name}")

    def _prompt_rename_tab(self, tab_id: str) -> None:
        title, accepted = QtWidgets.QInputDialog.getText(
            self,
            "Rename Tab",
            "Tab name:",
            text=self.tab_display_name(tab_id),
        )
        if accepted:
            self.rename_tab(tab_id, title)

    def _emit_context(self) -> None:
        raise NotImplementedError
