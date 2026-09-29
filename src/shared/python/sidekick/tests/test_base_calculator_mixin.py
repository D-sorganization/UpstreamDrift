# mypy: disable-error-code=no-untyped-def

from __future__ import annotations

import logging
from typing import Any

from PyQt6.QtWidgets import QWidget
from sidekick.ui.mixins import base_calculator_mixin
from sidekick.ui.mixins.base_calculator_mixin import BaseCalculatorMixin


class MockCalculator(QWidget, BaseCalculatorMixin):
    def __init__(self, name=None):
        QWidget.__init__(self)
        BaseCalculatorMixin.__init__(self, name)


def test_base_calculator_mixin_init_default(qapp) -> Any:
    calc = MockCalculator()
    assert calc.calculator_name == "MockCalculator"
    assert hasattr(calc, "_splitters")
    assert hasattr(calc, "_copyable_widgets")
    assert hasattr(calc, "_state")
    assert isinstance(calc._splitters, list)
    assert isinstance(calc._copyable_widgets, list)
    assert isinstance(calc._state, dict)
    assert calc._logger.name == f"{base_calculator_mixin.__name__}.MockCalculator"


def test_base_calculator_mixin_init_with_name(qapp) -> Any:
    calc = MockCalculator("CustomName")
    assert calc.calculator_name == "CustomName"
    assert calc._logger.name == f"{base_calculator_mixin.__name__}.MockCalculator"


def test_base_calculator_mixin_existing_attrs(qapp) -> Any:
    class ExtMockCalculator(QWidget, BaseCalculatorMixin):
        def __init__(self):
            QWidget.__init__(self)
            self._splitters = ["a"]
            self._copyable_widgets = ["b"]
            self._state = {"c": 1}
            BaseCalculatorMixin.__init__(self)

    calc = ExtMockCalculator()
    assert calc._splitters == ["a"]
    assert calc._copyable_widgets == ["b"]
    assert calc._state == {"c": 1}


def test_base_calculator_mixin_logging(qapp, caplog) -> Any:
    calc = MockCalculator("Test")
    with caplog.at_level(logging.INFO):
        calc.log_info("Info message")
        calc.log_warning("Warning message")
        calc.log_error("Error message")

    assert "Info message" in caplog.text
    assert "Warning message" in caplog.text
    assert "Error message" in caplog.text
