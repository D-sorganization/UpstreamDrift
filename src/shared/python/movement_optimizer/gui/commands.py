# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Command pattern for undo/redo support in the GUI.

This module is intentionally free of PyQt6 imports so the core classes
(``Command``, ``UndoStack``) can be tested without a running QApplication.
``SliderChangeCommand`` imports ``QSlider`` only at the type-checking level;
the slider object is duck-typed at runtime to keep the module importable in
headless test environments.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PyQt6.QtWidgets import QSlider

logger = logging.getLogger(__name__)


class Command(ABC):
    """Abstract base for reversible GUI operations."""

    @abstractmethod
    def execute(self) -> None:
        """Apply (or re-apply) the operation."""
        ...

    @abstractmethod
    def undo(self) -> None:
        """Reverse the operation."""
        ...


class UndoStack:
    """Fixed-capacity stack of reversible :class:`Command` objects.

    Args:
        max_size: Maximum number of commands retained in each direction.
            Must be a positive integer.

    Raises:
        ValueError: If *max_size* is not a positive integer.
    """

    def __init__(self, max_size: int = 100) -> None:
        if max_size <= 0:
            raise ValueError(f"max_size must be a positive integer, got {max_size!r}")
        self._undo: deque[Command] = deque(maxlen=max_size)
        self._redo: deque[Command] = deque(maxlen=max_size)

    def push(self, cmd: Command) -> None:
        """Execute *cmd* and push it onto the undo stack.

        Clears the redo stack -- branching history is not supported.

        Args:
            cmd: The command to execute and record.
        """
        cmd.execute()
        self._undo.append(cmd)
        self._redo.clear()
        logger.debug("UndoStack: pushed %s (depth=%d)", type(cmd).__name__, len(self._undo))

    def record_executed(self, cmd: Command) -> None:
        """Record an already-applied command without calling ``execute``.

        This is useful for UI controls that have already changed by the time
        their completion signal fires, such as ``QSlider.sliderReleased``.
        """
        self._undo.append(cmd)
        self._redo.clear()
        logger.debug("UndoStack: recorded %s (depth=%d)", type(cmd).__name__, len(self._undo))

    def undo(self) -> bool:
        """Undo the most recently executed command.

        Returns:
            ``True`` if a command was undone, ``False`` if the stack was empty.
        """
        if not self._undo:
            logger.debug("UndoStack.undo: nothing to undo")
            return False
        cmd = self._undo.pop()
        cmd.undo()
        self._redo.append(cmd)
        logger.debug("UndoStack: undid %s (remaining=%d)", type(cmd).__name__, len(self._undo))
        return True

    def redo(self) -> bool:
        """Re-execute the most recently undone command.

        Returns:
            ``True`` if a command was re-applied, ``False`` if the stack was empty.
        """
        if not self._redo:
            logger.debug("UndoStack.redo: nothing to redo")
            return False
        cmd = self._redo.pop()
        cmd.execute()
        self._undo.append(cmd)
        logger.debug("UndoStack: redid %s (depth=%d)", type(cmd).__name__, len(self._undo))
        return True

    def clear(self) -> None:
        """Remove all undo and redo history."""
        self._undo.clear()
        self._redo.clear()
        logger.debug("UndoStack: cleared")

    @property
    def can_undo(self) -> bool:
        """``True`` when at least one command can be undone."""
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        """``True`` when at least one command can be re-applied."""
        return bool(self._redo)


class SliderChangeCommand(Command):
    """Record a single integer-tick change on a ``QSlider``.

    Signals on the slider are blocked during ``execute`` / ``undo`` to prevent
    re-entrant command creation.

    Args:
        slider: The ``QSlider`` (or duck-typed equivalent) to operate on.
        old_value: Tick value before the change.
        new_value: Tick value after the change.

    Raises:
        ValueError: If *old_value* equals *new_value* (no-op change).
    """

    def __init__(self, slider: QSlider, old_value: int, new_value: int) -> None:
        if old_value == new_value:
            raise ValueError(
                f"SliderChangeCommand: old_value and new_value are both {old_value}; "
                "no-op commands should not be recorded."
            )
        self._slider = slider
        self._old = old_value
        self._new = new_value

    def execute(self) -> None:
        self._slider.blockSignals(True)
        self._slider.setValue(self._new)
        self._slider.blockSignals(False)

    def undo(self) -> None:
        self._slider.blockSignals(True)
        self._slider.setValue(self._old)
        self._slider.blockSignals(False)
