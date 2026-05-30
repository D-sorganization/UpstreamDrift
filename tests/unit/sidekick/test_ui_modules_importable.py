"""Importability tests for sidekick UI modules (Issues #1949, #1744)."""

from __future__ import annotations

from sidekick.process_calculators.scrubber.tests.test_scrubber_engine import (
    ScrubberEngine,
    ScrubberInputs,
    TestScrubberEngine,
)
from sidekick.ui.mixins.base_calculator_mixin import (
    BaseCalculatorMixin,
    CalculatorStateMixin,
)
from sidekick.ui.widgets.mixins.data_processor_ops import (
    DataProcessorOpsMixin,
)


class TestScubberModulesImportable:
    def test_scrubber_engine_importable(self) -> None:
        assert ScrubberEngine is not None

    def test_scrubber_inputs_importable(self) -> None:
        assert ScrubberInputs is not None

    def test_scrubber_test_class_importable(self) -> None:
        assert TestScrubberEngine is not None


class TestCalculatorMixinsImportable:
    def test_base_calculator_mixin(self) -> None:
        assert BaseCalculatorMixin is not None

    def test_calculator_state_mixin(self) -> None:
        assert CalculatorStateMixin is not None

    def test_data_processor_ops_mixin(self) -> None:
        assert DataProcessorOpsMixin is not None
