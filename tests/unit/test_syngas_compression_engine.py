"""Unit tests for the extracted SyngasCompressionEngine module."""

from __future__ import annotations

import pytest

pytest.importorskip("sympy")  # sidekick package chain requires sympy


class TestCompressionStageImport:
    @pytest.mark.unit
    def test_can_import(self) -> None:
        from sidekick.process_calculators.syngas_compression_engine import (
            CompressionStage,
        )

        assert CompressionStage is not None

    @pytest.mark.unit
    def test_dataclass_fields(self) -> None:
        from sidekick.process_calculators.syngas_compression_engine import (
            CompressionStage,
        )

        stage = CompressionStage(
            inlet_pressure=1.0,
            outlet_pressure=3.0,
            inlet_temperature=313.15,
            efficiency=0.85,
            compression_type="isentropic",
        )
        assert stage.inlet_pressure == 1.0
        assert stage.outlet_pressure == 3.0
        assert stage.compression_type == "isentropic"


class TestSyngasCompressionEngineImport:
    @pytest.mark.unit
    def test_can_import(self) -> None:
        from sidekick.process_calculators.syngas_compression_engine import (
            SyngasCompressionEngine,
        )

        assert SyngasCompressionEngine is not None

    @pytest.mark.unit
    def test_calculate_water_dropout_no_condensation(self) -> None:
        from sidekick.process_calculators.syngas_compression_engine import (
            SyngasCompressionEngine,
        )

        engine = SyngasCompressionEngine()
        result = engine.calculate_water_dropout(
            temperature=313.15,  # 40 deg C
            pressure=1.0,  # 1 bar
            water_content=1.0,  # 1 mol%
        )
        assert "water_dropout" in result
        assert "relative_humidity" in result
        assert result["water_dropout"] >= 0.0

    @pytest.mark.unit
    def test_calculate_water_dropout_invalid_pressure(self) -> None:
        from sidekick.process_calculators.syngas_compression_engine import (
            SyngasCompressionEngine,
        )

        engine = SyngasCompressionEngine()
        with pytest.raises(ValueError, match="pressure must be > 0"):
            engine.calculate_water_dropout(
                temperature=313.15,
                pressure=0.0,
                water_content=1.0,
            )

    @pytest.mark.unit
    def test_calculate_compression_work_isentropic(self) -> None:
        from sidekick.process_calculators.syngas_compression_engine import (
            CompressionStage,
            SyngasCompressionEngine,
        )

        engine = SyngasCompressionEngine()
        stage = CompressionStage(
            inlet_pressure=1.0,
            outlet_pressure=3.0,
            inlet_temperature=313.15,
            efficiency=0.85,
            compression_type="isentropic",
        )
        mixture_props = {"heat_capacity_ratio": 1.4, "molecular_weight": 28.0}
        result = engine.calculate_compression_work(stage, 100.0, mixture_props)
        assert result["power_hp"] > 0
        assert result["heat_rise"] > 0
        assert result["pressure_ratio"] == pytest.approx(3.0)

    @pytest.mark.unit
    def test_calculate_compression_work_isothermal(self) -> None:
        from sidekick.process_calculators.syngas_compression_engine import (
            CompressionStage,
            SyngasCompressionEngine,
        )

        engine = SyngasCompressionEngine()
        stage = CompressionStage(
            inlet_pressure=1.0,
            outlet_pressure=3.0,
            inlet_temperature=313.15,
            efficiency=0.85,
            compression_type="isothermal",
        )
        mixture_props = {"heat_capacity_ratio": 1.4}
        result = engine.calculate_compression_work(stage, 100.0, mixture_props)
        assert result["power_hp"] > 0
        assert result["heat_rise"] == 0.0

    @pytest.mark.unit
    def test_calculate_compression_work_unknown_type(self) -> None:
        from sidekick.process_calculators.syngas_compression_engine import (
            CompressionStage,
            SyngasCompressionEngine,
        )

        engine = SyngasCompressionEngine()
        stage = CompressionStage(
            inlet_pressure=1.0,
            outlet_pressure=3.0,
            inlet_temperature=313.15,
            efficiency=0.85,
            compression_type="unknown_type",
        )
        with pytest.raises(ValueError, match="Unknown compression type"):
            engine.calculate_compression_work(
                stage, 100.0, {"heat_capacity_ratio": 1.4}
            )

    @pytest.mark.unit
    def test_analyze_process_conditions_no_concerns(self) -> None:
        from sidekick.process_calculators.syngas_compression_engine import (
            SyngasCompressionEngine,
        )

        engine = SyngasCompressionEngine()
        # Low pressure, low temperature, low power - no concerns
        mock_result = {
            "final_temperature": 350.0,  # below warning threshold
            "final_pressure": 10.0,  # below high pressure threshold
            "total_power_hp": 50.0,  # below high power threshold
            "stages": [
                {
                    "work_isentropic": None,  # no efficiency check
                    "water_dropout": {"water_dropout": 0.0},
                }
            ],
        }
        analysis = engine.analyze_process_conditions(mock_result)
        assert "concerns" in analysis
        assert "warnings" in analysis
        assert "recommendations" in analysis
        assert len(analysis["concerns"]) == 0
        assert len(analysis["warnings"]) == 0


class TestSyngasDisplayFunctions:
    @pytest.mark.unit
    def test_format_results_text(self) -> None:
        from sidekick.process_calculators.syngas_compression_display import (
            format_results_text,
        )

        mock_result = {
            "mixture_properties": {
                "molecular_weight": 20.0,
                "critical_temperature": 300.0,
                "critical_pressure": 40.0,
                "heat_capacity_ratio": 1.35,
            },
            "stages": [
                {
                    "stage_number": 1,
                    "inlet_temp": 313.15,
                    "outlet_temp": 400.0,
                    "heat_rise": 86.85,
                    "pressure_ratio": 3.0,
                    "power_hp": 100.0,
                    "water_dropout": {"water_dropout": 0.0},
                }
            ],
            "total_power_hp": 100.0,
            "final_temperature": 400.0,
            "final_pressure": 3.0,
        }
        mock_analysis = {
            "total_water_dropout": 0.0,
            "average_efficiency": None,
        }
        text = format_results_text(mock_result, mock_analysis)
        assert "SYNGAS COMPRESSION" in text
        assert "Stage 1" in text
        assert "100.0 HP" in text

    @pytest.mark.unit
    def test_format_analysis_text_no_issues(self) -> None:
        from sidekick.process_calculators.syngas_compression_display import (
            format_analysis_text,
        )

        analysis = {
            "warnings": [],
            "concerns": [],
            "recommendations": [],
        }
        text = format_analysis_text(analysis)
        assert "No significant concerns" in text

    @pytest.mark.unit
    def test_format_analysis_text_with_concerns(self) -> None:
        from sidekick.process_calculators.syngas_compression_display import (
            format_analysis_text,
        )

        analysis = {
            "warnings": ["High temp warning"],
            "concerns": ["High pressure concern"],
            "recommendations": ["Add intercooler"],
        }
        text = format_analysis_text(analysis)
        assert "High temp warning" in text
        assert "High pressure concern" in text
        assert "Add intercooler" in text
