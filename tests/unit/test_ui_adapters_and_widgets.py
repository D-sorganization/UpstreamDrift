"""Importability tests for ui.adapters and ui.widgets (Issues #1949, #1744)."""

from __future__ import annotations

from src.shared.python.ui.adapters.canvas import CanvasAdapter, CanvasProtocol
from src.shared.python.ui.adapters.thread import BackgroundWorker, QtWorker
from src.shared.python.ui.widgets import LogPanel, SignalBlocker
from sidekick.ui.mixins.calculator_state_mixin import (
    CalculatorStateMixin,
)
from sidekick.ui.widgets.unit_aware_input import (
    UnitAwareDisplay,
    UnitAwareInput,
)


class TestCanvasAdapterImportable:
    def test_canvas_adapter_importable(self) -> None:
        assert CanvasAdapter is not None

    def test_canvas_protocol_importable(self) -> None:
        assert CanvasProtocol is not None


class TestThreadAdapterImportable:
    def test_background_worker_importable(self) -> None:
        assert BackgroundWorker is not None

    def test_qt_worker_importable(self) -> None:
        assert QtWorker is not None


class TestWidgetsImportable:
    def test_log_panel_importable(self) -> None:
        assert LogPanel is not None

    def test_signal_blocker_importable(self) -> None:
        assert SignalBlocker is not None


class TestUnitAwareInputImportable:
    def test_unit_aware_input_importable(self) -> None:
        assert UnitAwareInput is not None

    def test_unit_aware_display_importable(self) -> None:
        assert UnitAwareDisplay is not None


class TestCalculatorStateMixinImportable:
    def test_calculator_state_mixin_importable(self) -> None:
        assert CalculatorStateMixin is not None
