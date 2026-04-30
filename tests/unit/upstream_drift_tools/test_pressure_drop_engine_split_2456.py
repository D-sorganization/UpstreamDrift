"""Contract tests for #2456: pressure_drop_calculation_engine.py split.

Tests run red before the split and green after.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).parents[3]
ENGINE_DIR = (
    REPO
    / "src/shared/python/upstream_drift_tools/process_calculators"
    / "pressure_drop_calculator/engine"
)
LOC_BUDGET = 600


def _count_lines(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


class TestPressureDropEngineSplitStructure:
    """Split modules must exist after refactor."""

    @pytest.mark.unit
    def test_friction_factors_module_exists(self) -> None:
        assert (ENGINE_DIR / "_friction_factors.py").exists()

    @pytest.mark.unit
    def test_flow_calculations_module_exists(self) -> None:
        assert (ENGINE_DIR / "_flow_calculations.py").exists()


class TestPressureDropEngineFileSizes:
    """Each file must be under 600 LOC after split."""

    @pytest.mark.unit
    def test_engine_coordinator_loc(self) -> None:
        loc = _count_lines(ENGINE_DIR / "pressure_drop_calculation_engine.py")
        assert (
            loc <= LOC_BUDGET
        ), f"pressure_drop_calculation_engine.py has {loc} LOC; budget {LOC_BUDGET}"

    @pytest.mark.unit
    def test_friction_factors_loc(self) -> None:
        loc = _count_lines(ENGINE_DIR / "_friction_factors.py")
        assert (
            loc <= LOC_BUDGET
        ), f"_friction_factors.py has {loc} LOC; budget {LOC_BUDGET}"

    @pytest.mark.unit
    def test_flow_calculations_loc(self) -> None:
        loc = _count_lines(ENGINE_DIR / "_flow_calculations.py")
        assert (
            loc <= LOC_BUDGET
        ), f"_flow_calculations.py has {loc} LOC; budget {LOC_BUDGET}"


class TestPressureDropEnginePublicAPI:
    """Public API must remain importable from engine module (backward compat)."""

    @pytest.mark.unit
    def test_import_friction_factor_laminar(self) -> None:
        from src.shared.python.upstream_drift_tools.process_calculators.pressure_drop_calculator.engine.pressure_drop_calculation_engine import (
            friction_factor_laminar,
        )

        assert callable(friction_factor_laminar)

    @pytest.mark.unit
    def test_import_calculate_flow_properties(self) -> None:
        from src.shared.python.upstream_drift_tools.process_calculators.pressure_drop_calculator.engine.pressure_drop_calculation_engine import (
            calculate_flow_properties,
        )

        assert callable(calculate_flow_properties)

    @pytest.mark.unit
    def test_import_calculation_engine_class(self) -> None:
        from src.shared.python.upstream_drift_tools.process_calculators.pressure_drop_calculator.engine.pressure_drop_calculation_engine import (
            PressureDropCalculationEngine,
        )

        assert PressureDropCalculationEngine is not None
