"""Importability tests for sidekick internal test modules (Issues #1949, #1744)."""

from __future__ import annotations

from sidekick.tests.calculators.conversion.test_conversion import (
    TestUnitConversion,
)
from sidekick.tests.calculators.electrical.test_electrical_model import (
    TestElectricalModel,
)
from sidekick.tests.calculators.mechanical.test_trc_geometry import (
    TestTRCGeometryEngine,
)


class TestInternalTestModulesImportable:
    def test_unit_conversion_test_class(self) -> None:
        assert TestUnitConversion is not None

    def test_electrical_model_test_class(self) -> None:
        assert TestElectricalModel is not None

    def test_trc_geometry_test_class(self) -> None:
        assert TestTRCGeometryEngine is not None
