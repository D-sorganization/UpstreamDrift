"""Tests for issue #2515: split monolithic scripts to <= 300 LOC.

These tests define the acceptance criteria - they must pass after the split.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).parents[2]

SYNGAS_CALC = (
    REPO
    / "src/shared/python/sidekick/process_calculators"
    / "syngas_compression_calculator.py"
)
CONTROLS_TAB = (
    REPO
    / "src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/gui/tabs"
    / "controls_tab.py"
)

LOC_BUDGET = 300


def _count_lines(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


class TestSyngasCompressionCalculatorSize:
    """syngas_compression_calculator.py must be <= 300 LOC after the split."""

    @pytest.mark.unit
    def test_file_exists(self) -> None:
        assert SYNGAS_CALC.exists(), f"File not found: {SYNGAS_CALC}"

    @pytest.mark.unit
    def test_loc_within_budget(self) -> None:
        loc = _count_lines(SYNGAS_CALC)
        assert loc <= LOC_BUDGET, (
            f"syngas_compression_calculator.py has {loc} LOC; "
            f"budget is <= {LOC_BUDGET}. "
            "Extract SyngasCompressionEngine, worker, tab builders, and "
            "display helpers to separate modules."
        )


class TestControlsTabSize:
    """controls_tab.py must be <= 300 LOC after the split."""

    @pytest.mark.unit
    def test_file_exists(self) -> None:
        assert CONTROLS_TAB.exists(), f"File not found: {CONTROLS_TAB}"

    @pytest.mark.unit
    def test_loc_within_budget(self) -> None:
        loc = _count_lines(CONTROLS_TAB)
        assert loc <= LOC_BUDGET, (
            f"controls_tab.py has {loc} LOC; "
            f"budget is <= {LOC_BUDGET}. "
            "Extract ActuatorDetailDialog, actuator controls mixin, "
            "kinematic controls mixin, and simulation controls mixin "
            "to separate modules."
        )


class TestNewModulesExist:
    """Verify the extracted modules are in place after the split."""

    _CALC_DIR = SYNGAS_CALC.parent
    _TAB_DIR = CONTROLS_TAB.parent

    @pytest.mark.unit
    def test_syngas_engine_module_exists(self) -> None:
        assert (self._CALC_DIR / "syngas_compression_engine.py").exists()

    @pytest.mark.unit
    def test_syngas_worker_module_exists(self) -> None:
        assert (self._CALC_DIR / "syngas_compression_worker.py").exists()

    @pytest.mark.unit
    def test_syngas_tabs_mixin_module_exists(self) -> None:
        assert (self._CALC_DIR / "syngas_compression_tabs_mixin.py").exists()

    @pytest.mark.unit
    def test_syngas_display_module_exists(self) -> None:
        assert (self._CALC_DIR / "syngas_compression_display.py").exists()

    @pytest.mark.unit
    def test_actuator_detail_dialog_module_exists(self) -> None:
        assert (self._TAB_DIR / "actuator_detail_dialog.py").exists()

    @pytest.mark.unit
    def test_actuator_controls_mixin_module_exists(self) -> None:
        assert (self._TAB_DIR / "actuator_controls_mixin.py").exists()

    @pytest.mark.unit
    def test_kinematic_controls_mixin_module_exists(self) -> None:
        assert (self._TAB_DIR / "kinematic_controls_mixin.py").exists()

    @pytest.mark.unit
    def test_simulation_controls_mixin_module_exists(self) -> None:
        assert (self._TAB_DIR / "simulation_controls_mixin.py").exists()
