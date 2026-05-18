"""Contract tests for #2486: pressure_drop_interface.py split.

Tests run red before the split and green after.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_upstream_tools_available = importlib.util.find_spec("sympy") is not None

REPO = Path(__file__).parents[2]
CALC_DIR = (
    REPO / "src/shared/python/sidekick/process_calculators/pressure_drop_calculator"
)
LOC_BUDGET_HELPERS = 450
LOC_BUDGET_VALIDATION = 280
LOC_BUDGET_OUTPUT = 450
LOC_BUDGET_COORDINATOR = 600


def _count_lines(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


class TestPressureDropSplitStructure:
    """Split modules must exist after refactor."""

    @pytest.mark.unit
    def test_helpers_module_exists(self) -> None:
        assert (CALC_DIR / "_pressure_drop_helpers.py").exists()

    @pytest.mark.unit
    def test_validation_module_exists(self) -> None:
        assert (CALC_DIR / "_pressure_drop_validation.py").exists()

    @pytest.mark.unit
    def test_output_module_exists(self) -> None:
        assert (CALC_DIR / "_pressure_drop_output.py").exists()


class TestPressureDropFileSizes:
    """Each file must be within LOC budget after split."""

    @pytest.mark.unit
    def test_pressure_drop_split_2486_coordinator_loc(self) -> None:
        loc = _count_lines(CALC_DIR / "pressure_drop_interface.py")
        assert (
            loc <= LOC_BUDGET_COORDINATOR
        ), f"pressure_drop_interface.py has {loc} LOC; budget {LOC_BUDGET_COORDINATOR}"

    @pytest.mark.unit
    def test_helpers_loc(self) -> None:
        loc = _count_lines(CALC_DIR / "_pressure_drop_helpers.py")
        assert (
            loc <= LOC_BUDGET_HELPERS
        ), f"_pressure_drop_helpers.py has {loc} LOC; budget {LOC_BUDGET_HELPERS}"

    @pytest.mark.unit
    def test_validation_loc(self) -> None:
        loc = _count_lines(CALC_DIR / "_pressure_drop_validation.py")
        assert (
            loc <= LOC_BUDGET_VALIDATION
        ), f"_pressure_drop_validation.py has {loc} LOC; budget {LOC_BUDGET_VALIDATION}"

    @pytest.mark.unit
    def test_output_loc(self) -> None:
        loc = _count_lines(CALC_DIR / "_pressure_drop_output.py")
        assert (
            loc <= LOC_BUDGET_OUTPUT
        ), f"_pressure_drop_output.py has {loc} LOC; budget {LOC_BUDGET_OUTPUT}"


@pytest.mark.skipif(not _upstream_tools_available, reason="sympy not installed")
class TestPressureDropPublicAPI:
    """Public API must remain importable from pressure_drop_interface."""

    @pytest.mark.unit
    def test_import_calculate_pressure_drop(self) -> None:
        from sidekick.process_calculators.pressure_drop_calculator.pressure_drop_interface import (
            calculate_pressure_drop,
        )

        assert calculate_pressure_drop is not None

    @pytest.mark.unit
    def test_import_show_help(self) -> None:
        from sidekick.process_calculators.pressure_drop_calculator.pressure_drop_interface import (
            show_help,
        )

        assert show_help is not None

    @pytest.mark.unit
    def test_import_validate_inputs(self) -> None:
        from sidekick.process_calculators.pressure_drop_calculator.pressure_drop_interface import (
            validate_inputs,
        )

        assert validate_inputs is not None

    @pytest.mark.unit
    def test_import_print_results(self) -> None:
        from sidekick.process_calculators.pressure_drop_calculator.pressure_drop_interface import (
            print_results,
        )

        assert print_results is not None

    @pytest.mark.unit
    def test_import_list_gas_components(self) -> None:
        from sidekick.process_calculators.pressure_drop_calculator.pressure_drop_interface import (
            list_gas_components,
        )

        assert list_gas_components is not None
