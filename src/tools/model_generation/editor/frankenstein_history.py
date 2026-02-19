"""
Undo/redo history mixin for the Frankenstein Editor.

Extracted from FrankensteinEditor to respect SRP:
state management logic is independent of model editing operations.
"""

from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING

from .frankenstein_types import EditorState

if TYPE_CHECKING:
    from model_generation.converters.urdf_parser import ParsedModel


logger = logging.getLogger(__name__)


class HistoryMixin:
    """Undo/redo operations for the Frankenstein Editor.

    Requires host class to provide:
        _models: dict[str, ParsedModel]
        _clipboard: list[tuple[ComponentType, list[Link], list[Joint], dict[str, Material]]]
        _undo_stack: list[EditorState]
        _redo_stack: list[EditorState]
        _max_history: int
    """

    def undo(self) -> bool:
        """
        Undo the last operation.

        Returns:
            True if undone
        """
        if not self._undo_stack:
            logger.warning("Nothing to undo")
            return False

        # Save current state to redo stack
        current_state = self._create_state()
        self._redo_stack.append(current_state)

        # Restore previous state
        state = self._undo_stack.pop()
        self._restore_state(state)

        logger.info("Undone")
        return True

    def redo(self) -> bool:
        """
        Redo the last undone operation.

        Returns:
            True if redone
        """
        if not self._redo_stack:
            logger.warning("Nothing to redo")
            return False

        # Save current state to undo stack
        current_state = self._create_state()
        self._undo_stack.append(current_state)

        # Restore redo state
        state = self._redo_stack.pop()
        self._restore_state(state)

        logger.info("Redone")
        return True

    def _save_state(self) -> None:
        """Save current state to undo stack."""
        state = self._create_state()
        self._undo_stack.append(state)

        # Clear redo stack on new operation
        self._redo_stack = []

        # Limit history size
        while len(self._undo_stack) > self._max_history:
            self._undo_stack.pop(0)

    def _create_state(self) -> EditorState:
        """Create a state snapshot."""
        import time

        models_copy: dict[str, ParsedModel] = {}
        for model_id, model in self._models.items():
            models_copy[model_id] = model.copy()

        return EditorState(
            models=models_copy,
            clipboard=copy.deepcopy(self._clipboard),
            operation_history=[],
            timestamp=time.time(),
        )

    def _restore_state(self, state: EditorState) -> None:
        """Restore from a state snapshot."""
        self._models = state.models
        self._clipboard = state.clipboard
