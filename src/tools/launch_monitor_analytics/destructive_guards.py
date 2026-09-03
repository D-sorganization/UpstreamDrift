"""Confirm-or-cancel behaviour for the Launch Monitor's destructive actions.

Issue #8881. ``MainWidget.clear_project`` / ``load_project`` /
``remove_sessions`` stay unconditional primitives so tests and scripts can
drive them directly. Every *user-facing* path to them runs a guard from
:class:`DestructiveActionGuards` first, so the dirty-state contract lives
in exactly one place and ``QMessageBox`` is reached from exactly one module.

The unsaved-work answer is deliberately Save / Discard / Cancel rather than
Yes / No: "no" is ambiguous when the alternative is losing an hour of
imports, and Cancel is the default button so a stray Return keypress is
inert.

The mixin reads three things from the widget it is mixed into — ``_dirty``,
``project``, and ``session_tree`` — and calls three of its methods
(``is_dirty``, ``_on_save_project``, ``_refresh_all``). Those are declared
below so type checkers and readers see the contract instead of inferring it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import pandas as pd
from PyQt6 import QtCore, QtWidgets

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.tools.launch_monitor_model import LaunchMonitorProject

__all__ = ["DestructiveActionGuards"]


class _GuardedWorkspace(Protocol):
    """What :class:`DestructiveActionGuards` requires of its host widget."""

    project: LaunchMonitorProject
    analysis_frame: pd.DataFrame
    session_tree: QtWidgets.QTreeWidget
    _dirty: bool

    def is_dirty(self) -> bool: ...

    def clear_project(self) -> None: ...

    def _on_save_project(self) -> None: ...

    def _refresh_all(self) -> None: ...

    # Supplied by DestructiveActionGuards itself; declared so the guards can
    # call one another without mypy losing the mixin's own surface.
    def _confirm_discard_unsaved(self, action_label: str) -> bool: ...

    def _confirm_remove_sessions(self, sessions: int, shots: int) -> bool: ...

    def _selected_session_ids(self) -> list[str]: ...

    def remove_sessions(self, session_ids: list[str]) -> None: ...

    def _on_remove_selected_sessions(self) -> None: ...


class DestructiveActionGuards:
    """Guarded handlers for actions that can destroy unsaved project work."""

    # ---- confirmation seams (the only QMessageBox call sites) ----------

    def _confirm_discard_unsaved(self: _GuardedWorkspace, action_label: str) -> bool:
        """Ask before an action that would drop unsaved project changes.

        Returns ``True`` when the caller may proceed. A ``Save`` answer
        proceeds **only** when the save actually cleared the dirty flag:
        the user can still cancel the file dialog, or the write can fail,
        and a cancelled save must never fall through into the destructive
        action it was meant to protect against.
        """
        if not action_label:
            raise ValueError("action_label must be a non-empty phrase")
        if not self._dirty:
            return True
        buttons = QtWidgets.QMessageBox.StandardButton
        answer = QtWidgets.QMessageBox.question(
            self,  # type: ignore[arg-type]
            "Unsaved Changes",
            f"This project has unsaved changes. Save before {action_label}?",
            buttons.Save | buttons.Discard | buttons.Cancel,
            buttons.Cancel,
        )
        if answer == buttons.Discard:
            return True
        if answer != buttons.Save:
            return False
        self._on_save_project()
        return not self.is_dirty()

    def _confirm_remove_sessions(
        self: _GuardedWorkspace, sessions: int, shots: int
    ) -> bool:
        """Confirm a multi-select session removal, naming what is lost."""
        if sessions < 1:
            raise ValueError("sessions must be at least 1 to warrant a prompt")
        buttons = QtWidgets.QMessageBox.StandardButton
        answer = QtWidgets.QMessageBox.question(
            self,  # type: ignore[arg-type]
            "Remove Sessions",
            f"Remove {sessions} session(s) ({shots} shot(s))? This cannot be undone.",
            buttons.Yes | buttons.Cancel,
            buttons.Cancel,
        )
        return answer == buttons.Yes

    # ---- guarded handlers ---------------------------------------------

    def _on_new_project(self: _GuardedWorkspace) -> None:
        """Guarded "New Project..." handler."""
        if not self._confirm_discard_unsaved("starting a new project"):
            return
        self.clear_project()

    def _selected_session_ids(self: _GuardedWorkspace) -> list[str]:
        """Return the session IDs currently selected in the session tree."""
        ids: list[str] = []
        for item in self.session_tree.selectedItems():
            session_id = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
            if session_id is None:
                continue
            identifier = str(session_id)
            if identifier not in ids:
                ids.append(identifier)
        return ids

    def remove_sessions(self: _GuardedWorkspace, session_ids: list[str]) -> None:
        """Remove ``session_ids`` from the project (unconditional primitive)."""
        for session_id in session_ids:
            self.project.remove_session(session_id)
        self.analysis_frame = self.project.combined_shots()
        self._dirty = True
        self._refresh_all()

    def _on_remove_selected_sessions(self: _GuardedWorkspace) -> None:
        """Guarded "Remove Selected Sessions..." handler."""
        session_ids = self._selected_session_ids()
        if not session_ids:
            return
        targeted = set(session_ids)
        shots = sum(
            len(session.shots)
            for session in self.project.sessions
            if session.session_id in targeted
        )
        if not self._confirm_remove_sessions(len(session_ids), shots):
            return
        self.remove_sessions(session_ids)

    def _remove_selected_sessions(self: _GuardedWorkspace) -> None:
        """Back-compatible alias for the guarded removal handler."""
        self._on_remove_selected_sessions()
