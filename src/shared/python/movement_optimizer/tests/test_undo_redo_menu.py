# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Tests for MainWindow undo/redo menu state integration."""

from __future__ import annotations

from movement_optimizer.gui.commands import Command
from movement_optimizer.gui.main_window import MainWindow


class _NoopCommand(Command):
    def execute(self) -> None:
        pass

    def undo(self) -> None:
        pass


class _FakeAction:
    def __init__(self) -> None:
        self.enabled: bool | None = None

    def setEnabled(self, enabled: bool) -> None:
        self.enabled = enabled


class _FakeWindow:
    def __init__(self) -> None:
        from movement_optimizer.gui.commands import UndoStack

        self._undo_stack = UndoStack()
        self.action_undo = _FakeAction()
        self.action_redo = _FakeAction()

    def _update_undo_redo_actions(self) -> None:
        MainWindow._update_undo_redo_actions(self)


def test_edit_menu_actions_start_disabled() -> None:
    window = _FakeWindow()
    MainWindow._update_undo_redo_actions(window)

    assert window.action_undo.enabled is False
    assert window.action_redo.enabled is False


def test_edit_menu_actions_track_undo_redo_stack() -> None:
    window = _FakeWindow()
    window._undo_stack.record_executed(_NoopCommand())
    MainWindow._update_undo_redo_actions(window)

    assert window.action_undo.enabled is True
    assert window.action_redo.enabled is False

    MainWindow._undo(window)
    assert window.action_undo.enabled is False
    assert window.action_redo.enabled is True

    MainWindow._redo(window)
    assert window.action_undo.enabled is True
    assert window.action_redo.enabled is False
