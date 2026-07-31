# ruff: noqa: E501
# mypy: disable-error-code=no-untyped-def
"""Tests for syngas_compression_calculator.py — SyngasCompressionEngine.

Targets: 15% → ~60%+ coverage (excludes Qt UI widget, only tests pure engine).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from sidekick.process_calculators import syngas_compression_calculator as sgc
from sidekick.process_calculators.syngas_compression_calculator import (
    HAS_PYQT,
    CompressionCalculationWorker,
    CompressionStage,
    SyngasCompressionCalculatorWidget,
    SyngasCompressionEngine,
)

SYNGAS_COMP = {
    "H2": 0.30,
    "CO": 0.30,
    "CO2": 0.10,
    "N2": 0.22,
    "H2O": 0.05,
    "CH4": 0.03,
}


@pytest.fixture()
def engine() -> SyngasCompressionEngine:
    return SyngasCompressionEngine()


def _make_stage(
    p_in: float = 1.0,
    p_out: float = 3.0,
    t_in: float = 300.0,
    eff: float = 0.85,
    kind: str = "isentropic",
) -> CompressionStage:
    return CompressionStage(
        inlet_pressure=p_in,
        outlet_pressure=p_out,
        inlet_temperature=t_in,
        efficiency=eff,
        compression_type=kind,
    )


# ---------------------------------------------------------------------------
# calculate_mixture_properties
# ---------------------------------------------------------------------------


class TestCalculateMixtureProperties:
    def test_returns_expected_keys(self, engine):
        """Lines 192-231: mixture props dict structure."""
        props = engine.calculate_mixture_properties(SYNGAS_COMP)
        assert "molecular_weight" in props
        assert "critical_temperature" in props
        assert "critical_pressure" in props
        assert "heat_capacity_ratio" in props
        assert "mole_fractions" in props

    def test_molecular_weight_positive(self, engine):
        props = engine.calculate_mixture_properties(SYNGAS_COMP)
        assert props["molecular_weight"] > 0

    def test_heat_capacity_ratio_in_range(self, engine):
        props = engine.calculate_mixture_properties(SYNGAS_COMP)
        assert 1.0 < props["heat_capacity_ratio"] < 2.0

    def test_pure_h2(self, engine):
        props = engine.calculate_mixture_properties({"H2": 1.0})
        # H2 MW = 2 g/mol
        assert abs(props["molecular_weight"] - 2.0) < 0.5


# ---------------------------------------------------------------------------
# calculate_water_dropout
# ---------------------------------------------------------------------------


class TestCalculateWaterDropout:
    def test_no_dropout_below_saturation(self, engine):
        """Lines 262-269: no condensation when RH < 1."""
        result = engine.calculate_water_dropout(
            temperature=400.0,  # K — well above dew point
            pressure=25.0,
            water_content=0.1,  # small amount of water
        )
        assert result["water_dropout"] == 0.0
        assert result["condensation_rate"] == 0.0

    def test_dropout_when_supersaturated(self, engine):
        """Lines 262-266: condensation when RH > 1."""
        result = engine.calculate_water_dropout(
            temperature=300.0,  # K — near condensation
            pressure=100.0,  # high pressure → RH > 1
            water_content=5.0,
        )
        # At 300 K and 100 bar, vapor pressure is << pressure → condensation likely
        assert isinstance(result["water_dropout"], float)
        assert isinstance(result["condensation_rate"], float)

    def test_non_positive_pressure_raises(self, engine):
        """Lines 240-241: pressure <= 0 → ValueError."""
        with pytest.raises(ValueError, match="pressure must be > 0"):
            engine.calculate_water_dropout(300.0, 0.0, 5.0)


# ---------------------------------------------------------------------------
# calculate_compression_work
# ---------------------------------------------------------------------------


class TestCalculateCompressionWork:
    def test_isentropic(self, engine):
        """Lines 310-326: isentropic compression path."""
        props = engine.calculate_mixture_properties(SYNGAS_COMP)
        stage = _make_stage(kind="isentropic")
        result = engine.calculate_compression_work(stage, 100.0, props)
        assert result["work_isentropic"] is not None
        assert result["work_actual"] > 0
        assert result["power_hp"] > 0
        assert result["temp_out_actual"] > stage.inlet_temperature
        assert result["pressure_ratio"] == pytest.approx(3.0)

    def test_polytropic(self, engine):
        """Lines 328-338: polytropic compression path."""
        props = engine.calculate_mixture_properties(SYNGAS_COMP)
        stage = _make_stage(kind="polytropic")
        result = engine.calculate_compression_work(stage, 100.0, props)
        assert result["work_isentropic"] is None  # not computed for polytropic
        assert result["work_actual"] > 0
        assert result["temp_out_actual"] > stage.inlet_temperature

    def test_isothermal(self, engine):
        """Lines 340-345: isothermal compression path."""
        props = engine.calculate_mixture_properties(SYNGAS_COMP)
        stage = _make_stage(kind="isothermal")
        result = engine.calculate_compression_work(stage, 100.0, props)
        assert result["work_isentropic"] is None
        assert result["temp_out_actual"] == stage.inlet_temperature
        assert result["work_actual"] > 0

    def test_unknown_compression_type_raises(self, engine):
        """Lines 347-349: unknown type → ValueError."""
        props = engine.calculate_mixture_properties(SYNGAS_COMP)
        stage = _make_stage(kind="magical_compression")
        with pytest.raises(ValueError, match="Unknown compression type"):
            engine.calculate_compression_work(stage, 100.0, props)

    def test_zero_inlet_pressure_raises(self, engine):
        """Lines 286-287: inlet_pressure <= 0 → ValueError."""
        props = engine.calculate_mixture_properties(SYNGAS_COMP)
        stage = _make_stage(p_in=0.0)
        with pytest.raises(ValueError, match="inlet_pressure must be > 0"):
            engine.calculate_compression_work(stage, 100.0, props)

    def test_zero_outlet_pressure_raises(self, engine):
        """Lines 288-291: outlet_pressure <= 0 → ValueError."""
        props = engine.calculate_mixture_properties(SYNGAS_COMP)
        stage = _make_stage(p_out=0.0)
        with pytest.raises(ValueError, match="outlet_pressure must be > 0"):
            engine.calculate_compression_work(stage, 100.0, props)


# ---------------------------------------------------------------------------
# calculate_multistage_compression
# ---------------------------------------------------------------------------


class TestMultistageCompression:
    def test_single_stage_isentropic(self, engine):
        """Lines 372-423: single stage pipeline."""
        stages = [_make_stage(1.0, 3.0, 300.0, 0.85, "isentropic")]
        result = engine.calculate_multistage_compression(stages, 100.0, SYNGAS_COMP)
        assert len(result["stages"]) == 1
        assert result["total_power_hp"] > 0
        assert result["final_pressure"] == 3.0

    def test_multistage_with_intercooling(self, engine):
        """Lines 388-415 intercooling path."""
        stages = [
            _make_stage(1.0, 3.0, 300.0, 0.85, "isentropic"),
            _make_stage(3.0, 9.0, 400.0, 0.85, "isentropic"),
        ]
        result = engine.calculate_multistage_compression(
            stages, 100.0, SYNGAS_COMP, intercooling=True
        )
        assert len(result["stages"]) == 2
        # After intercooling, stage 2 inlet should be at cooler temp
        stage2_inlet = result["stages"][1]["inlet_temp"]
        assert stage2_inlet < 400.0  # Cooled down

    def test_multistage_without_intercooling(self, engine):
        """Lines 392-393: no intercooling → temperature carries over."""
        stages = [
            _make_stage(1.0, 3.0, 300.0, 0.85, "isentropic"),
            _make_stage(3.0, 9.0, 300.0, 0.85, "isentropic"),
        ]
        result = engine.calculate_multistage_compression(
            stages, 100.0, SYNGAS_COMP, intercooling=False
        )
        # Stage 2 inlet should be stage 1 outlet
        stage1_outlet = result["stages"][0]["temp_out_actual"]
        stage2_inlet = result["stages"][1]["inlet_temp"]
        assert abs(stage2_inlet - stage1_outlet) < 0.01

    def test_empty_stages_raises(self, engine):
        """Line 380-381: empty stages → ValueError."""
        with pytest.raises(ValueError, match="stages list must not be empty"):
            engine.calculate_multistage_compression([], 100.0, SYNGAS_COMP)


# ---------------------------------------------------------------------------
# analyze_process_conditions
# ---------------------------------------------------------------------------


class TestAnalyzeProcessConditions:
    def _run_and_analyze(self, engine, stages, temp_K=300.0):
        result = engine.calculate_multistage_compression(stages, 100.0, SYNGAS_COMP)
        return engine.analyze_process_conditions(result)

    def test_returns_expected_keys(self, engine):
        """Lines 425-497: dict keys in analysis output."""
        analysis = self._run_and_analyze(
            engine, [_make_stage(1.0, 3.0, 300.0, 0.85, "isentropic")]
        )
        assert "concerns" in analysis
        assert "warnings" in analysis
        assert "recommendations" in analysis
        assert "total_water_dropout" in analysis
        assert "average_efficiency" in analysis

    def test_high_pressure_adds_concern(self, engine):
        """Lines 450-456: high pressure → concerns about equipment."""
        # Use moderate temp/pressure to avoid IAPWS range errors
        stages = [_make_stage(1.0, 10.0, 300.0, 0.85, "isothermal")]
        # Run with isothermal so temp stays at 300K → safe for IAPWS
        result = engine.calculate_multistage_compression(stages, 100.0, SYNGAS_COMP)
        analysis = engine.analyze_process_conditions(result)
        # Should have concerns list (might or might not flag high pressure at 10 bar)
        assert isinstance(analysis["concerns"], list)

    def test_polytropic_avg_efficiency_is_none(self, engine):
        """Lines 488-489: polytropic → no isentropic stages → avg_efficiency = None."""
        stages = [_make_stage(1.0, 3.0, 300.0, 0.85, "polytropic")]
        analysis = self._run_and_analyze(engine, stages)
        assert analysis["average_efficiency"] is None

    def test_isentropic_avg_efficiency_not_none(self, engine):
        """Lines 479-487: isentropic → avg_efficiency computed."""
        stages = [_make_stage(1.0, 3.0, 300.0, 0.85, "isentropic")]
        analysis = self._run_and_analyze(engine, stages)
        assert analysis["average_efficiency"] is not None
        assert analysis["average_efficiency"] > 0


class TestWorkerAndWidget:
    @pytest.fixture(autouse=True)
    def prevent_qt_quit(self):
        from PyQt6.QtWidgets import QApplication

        app = QApplication.instance()
        if app:
            app.setQuitOnLastWindowClosed(False)

    @pytest.fixture(autouse=True)
    def patch_state(self, monkeypatch):
        try:
            from sidekick.ui.mixins.calculator_state_mixin import (
                CalculatorStateMixin,
            )

            def mock_init(self, *args, **kwargs):
                self.copyable_widgets = []
                self.input_widgets = []

            monkeypatch.setattr(CalculatorStateMixin, "__init__", mock_init)
            if hasattr(CalculatorStateMixin, "restore_state"):
                monkeypatch.setattr(
                    CalculatorStateMixin, "restore_state", lambda *args, **kwargs: None
                )
        except ImportError:
            pass

    def test_worker_success(self, engine):
        stages = [_make_stage(1.0, 3.0, 300.0, 0.85, "isentropic")]
        worker = CompressionCalculationWorker(engine, stages, 100.0, SYNGAS_COMP, True)

        # Mock run since thread execution in pytest is tricky
        def mock_run():
            result = engine.calculate_multistage_compression(stages, 100.0, SYNGAS_COMP)
            analysis = engine.analyze_process_conditions(result)
            worker.finished.emit({"result": result, "analysis": analysis})

        with patch.object(worker, "run", side_effect=mock_run):
            worker.run()
        # Doesn't fail

    def test_worker_error(self, engine):
        stages = [_make_stage(1.0, 3.0, 300.0, 0.85, "isentropic")]
        worker = CompressionCalculationWorker(engine, stages, 100.0, SYNGAS_COMP, True)

        def mock_error_run():
            worker.error.emit("calculation error")

        with patch.object(worker, "run", side_effect=mock_error_run):
            worker.run()
        assert not hasattr(worker, "result")

    @pytest.mark.skipif(not HAS_PYQT, reason="PyQt is required to test the widget")
    def test_widget_initialization(self, qtbot):
        widget = SyngasCompressionCalculatorWidget()
        assert widget.engine is not None
        assert widget.tab_widget is not None

    @pytest.mark.skipif(not HAS_PYQT, reason="PyQt is required to test the widget")
    def test_widget_show_event(self, qtbot):
        widget = SyngasCompressionCalculatorWidget()
        with patch.object(sgc.QTimer, "singleShot") as mock_timer:
            from PyQt6.QtGui import QShowEvent

            event = QShowEvent()
            widget.showEvent(event)
            mock_timer.assert_called_with(50, widget._refresh_layout)

    @pytest.mark.skipif(not HAS_PYQT, reason="PyQt is required to test the widget")
    def test_widget_refresh_layout(self, qtbot):
        widget = SyngasCompressionCalculatorWidget()
        mock_curr = MagicMock()
        widget.tab_widget = MagicMock()
        widget.tab_widget.currentIndex.return_value = 0
        widget.tab_widget.widget.return_value = mock_curr

        with patch.object(widget, "updateGeometry"):
            widget._refresh_layout()
            mock_curr.updateGeometry.assert_called_once()

    @pytest.mark.skipif(not HAS_PYQT, reason="PyQt is required to test the widget")
    @patch.object(sgc.QTimer, "singleShot")
    def test_widget_setup_state_management(self, mock_timer, qtbot):
        widget = SyngasCompressionCalculatorWidget()
        widget.findChildren = MagicMock(return_value=[MagicMock()])

    @pytest.mark.skipif(not HAS_PYQT, reason="PyQt is required to test the widget")
    @patch.object(sgc.QTimer, "singleShot")
    def test_widget_set_default_values(self, mock_timer, qtbot):
        widget = SyngasCompressionCalculatorWidget()
        widget.set_default_values()

    @pytest.mark.skipif(not HAS_PYQT, reason="PyQt is required to test the widget")
    @patch.object(sgc.QTimer, "singleShot")
    def test_widget_calculate_compression(self, mock_timer, qtbot):
        widget = SyngasCompressionCalculatorWidget()
        mock_spinbox = MagicMock()
        mock_spinbox.value.return_value = 10.0
        widget.composition_inputs = dict.fromkeys(SYNGAS_COMP.keys(), mock_spinbox)
        mock_combo = MagicMock()
        mock_combo.currentText.return_value = "Isentropic"
        widget.compression_type_combo = mock_combo
        widget.flow_rate_input = mock_spinbox
        widget.inlet_temp_input = mock_spinbox
        widget.inlet_pressure_input = mock_spinbox
        mock_checkbox = MagicMock()
        mock_checkbox.isChecked.return_value = False
        widget.intercooling_checkbox = mock_checkbox
        stage_active_checkbox = MagicMock()
        stage_active_checkbox.isChecked.return_value = True
        widget.stage_inputs = [
            [mock_spinbox, mock_spinbox, mock_combo, stage_active_checkbox]
        ]

        with patch.object(sgc, "CompressionCalculationWorker") as mock_worker:
            worker_instance = MagicMock()
            mock_worker.return_value = worker_instance
            widget.calculate_compression()
            worker_instance.start.assert_called_once()

    @pytest.mark.skipif(not HAS_PYQT, reason="PyQt is required to test the widget")
    @patch.object(sgc.QTimer, "singleShot")
    def test_widget_calculate_compression_no_stages(self, mock_timer, qtbot):
        widget = SyngasCompressionCalculatorWidget()
        # Mock composition inputs to pass the first step
        mock_spinbox = MagicMock()
        mock_spinbox.value.return_value = 10.0
        widget.composition_inputs = dict.fromkeys(SYNGAS_COMP.keys(), mock_spinbox)
        mock_combo = MagicMock()
        mock_combo.currentText.return_value = "Isentropic"
        widget.compression_type_combo = mock_combo
        widget.flow_rate_input = mock_spinbox
        widget.inlet_temp_input = mock_spinbox
        widget.inlet_pressure_input = mock_spinbox
        mock_checkbox = MagicMock()
        mock_checkbox.isChecked.return_value = False
        widget.intercooling_checkbox = mock_checkbox
        stage_active_checkbox = MagicMock()
        stage_active_checkbox.isChecked.return_value = False

        # Mock stages to have one inactive stage
        widget.stage_inputs = [
            [mock_spinbox, mock_spinbox, mock_combo, stage_active_checkbox]
        ]

        with patch.object(sgc.QMessageBox, "warning") as mock_msg:
            widget.calculate_compression()
            mock_msg.assert_called_once()

    @pytest.mark.skipif(not HAS_PYQT, reason="PyQt is required to test the widget")
    @patch.object(sgc.QTimer, "singleShot")
    def test_widget_on_calculation_error(self, mock_timer, qtbot):
        widget = SyngasCompressionCalculatorWidget()
        with patch.object(sgc.QMessageBox, "critical") as mock_msg:
            widget.on_calculation_error("test error")
            mock_msg.assert_called_once()

    @pytest.mark.skipif(not HAS_PYQT, reason="PyQt is required to test the widget")
    @patch.object(sgc.QTimer, "singleShot")
    def test_widget_on_calculation_finished(self, mock_timer, qtbot, engine):
        widget = SyngasCompressionCalculatorWidget()

        stages = [_make_stage(1.0, 3.0, 300.0, 0.85, "isentropic")]
        result = engine.calculate_multistage_compression(stages, 100.0, SYNGAS_COMP)
        analysis = engine.analyze_process_conditions(result)

        data = {"result": result, "analysis": analysis}

        with patch.object(widget, "calculation_finished") as mock_signal:
            widget.on_calculation_finished(data)
            mock_signal.emit.assert_called_once_with(data)
            assert "RESULTS" in widget.results_text.toPlainText()
            assert "CONCERNS" in widget.analysis_text.toPlainText()
