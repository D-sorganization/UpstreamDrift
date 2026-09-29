# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Tests for the undo/redo command pattern (commands.py).

These tests are intentionally free of PyQt6 so they run in any environment,
including CI without a display.
"""

from __future__ import annotations

import pytest

from movement_optimizer.gui.commands import Command, SliderChangeCommand, UndoStack

# ---------------------------------------------------------------------------
# Minimal in-memory Command stub (no Qt required)
# ---------------------------------------------------------------------------


class _CounterCommand(Command):
    """Records how many times execute/undo have been called."""

    def __init__(self, log: list[str], label: str = "cmd") -> None:
        self._log = log
        self._label = label

    def execute(self) -> None:
        self._log.append(f"{self._label}:execute")

    def undo(self) -> None:
        self._log.append(f"{self._label}:undo")


class _FakeSlider:
    """Duck-typed QSlider substitute for testing SliderChangeCommand."""

    def __init__(self, initial: int = 0) -> None:
        self._value = initial
        self._signals_blocked = False

    def value(self) -> int:
        return self._value

    def setValue(self, v: int) -> None:
        if not self._signals_blocked:
            pass  # real slider would emit valueChanged; we don't need that here
        self._value = v

    def blockSignals(self, block: bool) -> None:
        self._signals_blocked = block


# ---------------------------------------------------------------------------
# UndoStack tests
# ---------------------------------------------------------------------------


class TestUndoStackBasics:
    def test_initial_state_empty(self) -> None:
        stack = UndoStack()
        assert not stack.can_undo
        assert not stack.can_redo

    def test_push_executes_command(self) -> None:
        log: list[str] = []
        stack = UndoStack()
        stack.push(_CounterCommand(log))
        assert log == ["cmd:execute"]

    def test_can_undo_after_push(self) -> None:
        stack = UndoStack()
        stack.push(_CounterCommand([]))
        assert stack.can_undo

    def test_undo_calls_undo_on_command(self) -> None:
        log: list[str] = []
        stack = UndoStack()
        stack.push(_CounterCommand(log))
        log.clear()
        result = stack.undo()
        assert result is True
        assert log == ["cmd:undo"]

    def test_undo_returns_false_when_empty(self) -> None:
        stack = UndoStack()
        assert stack.undo() is False

    def test_redo_returns_false_when_empty(self) -> None:
        stack = UndoStack()
        assert stack.redo() is False

    def test_undo_moves_to_redo_stack(self) -> None:
        stack = UndoStack()
        stack.push(_CounterCommand([]))
        stack.undo()
        assert not stack.can_undo
        assert stack.can_redo

    def test_redo_re_executes_command(self) -> None:
        log: list[str] = []
        stack = UndoStack()
        stack.push(_CounterCommand(log))
        stack.undo()
        log.clear()
        result = stack.redo()
        assert result is True
        assert log == ["cmd:execute"]

    def test_redo_moves_back_to_undo_stack(self) -> None:
        stack = UndoStack()
        stack.push(_CounterCommand([]))
        stack.undo()
        stack.redo()
        assert stack.can_undo
        assert not stack.can_redo

    def test_push_clears_redo_stack(self) -> None:
        stack = UndoStack()
        stack.push(_CounterCommand([]))
        stack.undo()
        assert stack.can_redo
        stack.push(_CounterCommand([]))
        assert not stack.can_redo

    def test_multiple_pushes_and_undos(self) -> None:
        log: list[str] = []
        stack = UndoStack()
        for i in range(3):
            stack.push(_CounterCommand(log, label=str(i)))
        log.clear()

        stack.undo()
        stack.undo()
        assert log == ["2:undo", "1:undo"]
        assert stack.can_undo
        assert stack.can_redo

    def test_full_undo_redo_sequence(self) -> None:
        log: list[str] = []
        stack = UndoStack()
        stack.push(_CounterCommand(log, "a"))
        stack.push(_CounterCommand(log, "b"))
        log.clear()

        stack.undo()  # undo b
        stack.undo()  # undo a
        stack.redo()  # redo a
        stack.redo()  # redo b

        assert log == ["b:undo", "a:undo", "a:execute", "b:execute"]

    def test_max_size_respected(self) -> None:
        stack = UndoStack(max_size=3)
        for i in range(5):
            stack.push(_CounterCommand([], label=str(i)))
        # Only the last 3 should remain
        assert len(stack._undo) == 3

    def test_default_max_size_is_one_hundred(self) -> None:
        stack = UndoStack()
        for i in range(105):
            stack.push(_CounterCommand([], label=str(i)))
        assert len(stack._undo) == 100

    def test_record_executed_does_not_execute_command(self) -> None:
        log: list[str] = []
        stack = UndoStack()
        stack.record_executed(_CounterCommand(log))
        assert log == []
        assert stack.can_undo

    def test_clear_removes_undo_and_redo_history(self) -> None:
        stack = UndoStack()
        stack.push(_CounterCommand([]))
        stack.undo()
        stack.clear()
        assert not stack.can_undo
        assert not stack.can_redo

    def test_invalid_max_size_raises(self) -> None:
        with pytest.raises(ValueError):
            UndoStack(max_size=0)
        with pytest.raises(ValueError):
            UndoStack(max_size=-1)


# ---------------------------------------------------------------------------
# SliderChangeCommand tests
# ---------------------------------------------------------------------------


class TestSliderChangeCommand:
    def test_execute_sets_new_value(self) -> None:
        slider = _FakeSlider(10)
        cmd = SliderChangeCommand(slider, old_value=10, new_value=20)  # type: ignore[arg-type]
        cmd.execute()
        assert slider.value() == 20

    def test_undo_restores_old_value(self) -> None:
        slider = _FakeSlider(20)
        cmd = SliderChangeCommand(slider, old_value=10, new_value=20)  # type: ignore[arg-type]
        cmd.undo()
        assert slider.value() == 10

    def test_execute_blocks_signals(self) -> None:
        """blockSignals(True) must be called before setValue and False after."""
        calls: list[str] = []

        class _TrackingSlider(_FakeSlider):
            def blockSignals(self, block: bool) -> None:
                calls.append(f"block:{block}")
                super().blockSignals(block)

            def setValue(self, v: int) -> None:
                calls.append(f"set:{v}")
                super().setValue(v)

        slider = _TrackingSlider(0)
        cmd = SliderChangeCommand(slider, old_value=0, new_value=5)  # type: ignore[arg-type]
        cmd.execute()
        assert calls == ["block:True", "set:5", "block:False"]

    def test_undo_blocks_signals(self) -> None:
        calls: list[str] = []

        class _TrackingSlider(_FakeSlider):
            def blockSignals(self, block: bool) -> None:
                calls.append(f"block:{block}")
                super().blockSignals(block)

            def setValue(self, v: int) -> None:
                calls.append(f"set:{v}")
                super().setValue(v)

        slider = _TrackingSlider(5)
        cmd = SliderChangeCommand(slider, old_value=0, new_value=5)  # type: ignore[arg-type]
        cmd.undo()
        assert calls == ["block:True", "set:0", "block:False"]

    def test_noop_command_raises(self) -> None:
        slider = _FakeSlider(7)
        with pytest.raises(ValueError):
            SliderChangeCommand(slider, old_value=7, new_value=7)  # type: ignore[arg-type]

    def test_round_trip_via_stack(self) -> None:
        slider = _FakeSlider(0)
        cmd = SliderChangeCommand(slider, old_value=0, new_value=42)  # type: ignore[arg-type]
        stack = UndoStack()
        stack.push(cmd)
        assert slider.value() == 42
        stack.undo()
        assert slider.value() == 0
        stack.redo()
        assert slider.value() == 42
