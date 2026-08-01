# ruff: noqa: E501
# mypy: disable-error-code=no-untyped-def
from typing import Any

import pytest
from sidekick.process_calculators.acid_gas_dewpoint_calculator import (
    AcidGasComposition,
    AcidGasDewpointCalculator,
    DewpointResult,
    estimate_condensation_risk,
    quick_dewpoint_calculation,
)


@pytest.fixture
def dewpoint_calc() -> AcidGasDewpointCalculator:
    return AcidGasDewpointCalculator()


def test_calculate_vapor_pressure(dewpoint_calc: AcidGasDewpointCalculator) -> None:
    # Test valid components
    h2o_vp = dewpoint_calc.calculate_vapor_pressure(100.0, "H2O")
    assert h2o_vp > 0.0

    hf_vp = dewpoint_calc.calculate_vapor_pressure(25.0, "HF")
    assert hf_vp > 0.0

    # Extended antoine
    h2o_vp_high = dewpoint_calc.calculate_vapor_pressure(
        150.0, "H2O", "extended_antoine"
    )
    assert h2o_vp_high > 0.0

    # Invalid component
    with pytest.raises(ValueError, match="Unknown component"):
        dewpoint_calc.calculate_vapor_pressure(100.0, "InvalidGas")


def test_calculate_dewpoint(dewpoint_calc: AcidGasDewpointCalculator) -> None:
    # Test reverse calculation
    vp = dewpoint_calc.calculate_vapor_pressure(80.0, "H2O")
    dp = dewpoint_calc.calculate_dewpoint(vp, "H2O")
    assert dp == pytest.approx(80.0, abs=1e-3)

    with pytest.raises(ValueError):
        dewpoint_calc.calculate_dewpoint(0, "H2O")

    with pytest.raises(ValueError, match="unknown component"):
        dewpoint_calc.calculate_dewpoint(1000, "Invalid")


def test_calculate_dewpoint_mixture(dewpoint_calc: AcidGasDewpointCalculator) -> None:
    comp = AcidGasComposition(h2o=0.1, hf=0.01, hcl=0.02, h2s=0.05)
    result = dewpoint_calc.calculate_dewpoint_mixture(
        temperature_c=150.0,
        pressure_bar=10.0,
        composition=comp,
    )

    assert isinstance(result, DewpointResult)
    assert result.overall_dewpoint_c > 0.0
    assert result.limiting_component in ["H2O", "HF", "HCl", "H2S"]
    assert result.dewpoint_margin_c > 0.0
    assert result.condensation_risk != "Unknown"


def test_calculate_dewpoint_mixture_invalid(
    dewpoint_calc: AcidGasDewpointCalculator,
) -> None:
    comp = AcidGasComposition(h2o=0.1)
    with pytest.raises(ValueError, match="pressure_bar must be positive"):
        dewpoint_calc.calculate_dewpoint_mixture(150.0, -1.0, comp)

    with pytest.raises(ValueError, match="temperature must yield a positive Kelvin"):
        dewpoint_calc.calculate_dewpoint_mixture(-300.0, 10.0, comp)


def test_quick_dewpoint_calculation() -> None:
    res = quick_dewpoint_calculation(
        temperature_c=150.0,
        pressure_bar=10.0,
        h2o_fraction=0.1,
    )
    assert res["overall_dewpoint_c"] > 0
    assert isinstance(res["condensation_risk"], str)


def test_estimate_condensation_risk() -> None:
    comp = AcidGasComposition(h2o=0.1)
    res = estimate_condensation_risk(150.0, 10.0, comp, safety_margin_c=10.0)
    assert "risk_level" in res
    assert "recommendation" in res
    assert res["limiting_component"] == "H2O"


def test_acid_gas_composition_normalize() -> None:
    comp = AcidGasComposition(h2o=0.1, hf=0.1, hcl=0, h2s=0, other=0.2)
    norm = comp.normalize()
    assert norm.h2o == 0.25
    assert norm.total == 1.0
    assert norm.hf == 0.25
    assert norm.other == 0.5
    comp2 = AcidGasComposition()
    assert comp2.normalize() == comp2
    assert comp.to_dict()["H2O"] == 0.1


def test_dewpoint_result_to_dict(dewpoint_calc: AcidGasDewpointCalculator) -> None:
    comp = AcidGasComposition(h2o=0.1)
    res = dewpoint_calc.calculate_dewpoint_mixture(150.0, 10.0, comp)
    d = res.to_dict()
    assert "timestamp" in d
    assert d["input"]["temperature_c"] == 150.0
    assert "H2O" in d["dewpoints"]
    assert "dewpoint_margin_c" in d["safety"]


def test_calculate_vapor_pressure_libraries(
    dewpoint_calc: AcidGasDewpointCalculator, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sidekick.process_calculators import (
        acid_gas_dewpoint_calculator as agdc,
    )

    monkeypatch.setattr(agdc, "THERMO_AVAILABLE", False)
    with pytest.raises(RuntimeError):
        dewpoint_calc.calculate_vapor_pressure(100.0, "H2O", "thermo")

    monkeypatch.setattr(agdc, "COOLPROP_AVAILABLE", False)
    with pytest.raises(RuntimeError):
        dewpoint_calc.calculate_vapor_pressure(100.0, "H2O", "coolprop")

    with pytest.raises(ValueError, match="Unknown method"):
        dewpoint_calc.calculate_vapor_pressure(100.0, "H2O", "invalid_method")


def test_calculate_dewpoint_zero_denominator(
    dewpoint_calc: AcidGasDewpointCalculator,
) -> None:
    A = dewpoint_calc.antoine_constants["H2O"]["A"]
    P_mmhg = 10**A
    from sidekick.process_calculators.acid_gas_dewpoint_calculator import (
        MMHG_TO_PA_CONV,
    )

    P_pa = P_mmhg * MMHG_TO_PA_CONV
    with pytest.raises(ValueError, match="zero denominator"):
        dewpoint_calc.calculate_dewpoint(P_pa, "H2O")

    with pytest.raises(ValueError, match="must be > 0"):
        dewpoint_calc.calculate_dewpoint(-100, "H2O")

    with pytest.raises(ValueError, match="must be > 0"):
        dewpoint_calc.calculate_dewpoint(0, "H2O")


from unittest.mock import MagicMock, patch

import numpy as np


def test_calculate_dewpoint_mixture_warnings(
    dewpoint_calc: AcidGasDewpointCalculator,
) -> None:
    comp = AcidGasComposition(h2o=0.1)
    res = dewpoint_calc.calculate_dewpoint_mixture(500.0, 500.0, comp)
    assert len(res.warnings) >= 2
    assert any("temperature" in w.lower() for w in res.warnings)
    assert any("pressure" in w.lower() for w in res.warnings)

    with patch.object(
        dewpoint_calc,
        "_calculate_all_individual_dewpoints",
        return_value={"H2O": np.nan, "HF": np.nan, "HCl": np.nan, "H2S": np.nan},
    ):
        res_nan = dewpoint_calc.calculate_dewpoint_mixture(150.0, 10.0, comp)
        assert np.isnan(res_nan.overall_dewpoint_c)
        assert res_nan.condensation_risk == "Unknown"


def test_assess_condensation_risk(dewpoint_calc: AcidGasDewpointCalculator) -> None:
    assert dewpoint_calc._assess_condensation_risk(np.nan) == "Unknown"
    assert "HIGH" in dewpoint_calc._assess_condensation_risk(-5.0)
    assert "MEDIUM" in dewpoint_calc._assess_condensation_risk(5.0)
    assert "LOW - Safe" in dewpoint_calc._assess_condensation_risk(20.0)
    assert "VERY LOW" in dewpoint_calc._assess_condensation_risk(50.0)


def test_generate_dewpoint_curves(dewpoint_calc: AcidGasDewpointCalculator) -> None:
    comp = AcidGasComposition(h2o=0.1)
    df = dewpoint_calc.generate_dewpoint_curves(10.0, comp, num_points=3)
    assert len(df) == 3
    assert "Temperature_C" in df.columns
    assert "Overall_Dewpoint_C" in df.columns


def test_estimate_condensation_risk_branches() -> None:
    from sidekick.process_calculators.acid_gas_dewpoint_calculator import (
        DewpointResult,
        estimate_condensation_risk,
    )

    comp = AcidGasComposition(h2o=0.1)

    mock_res = MagicMock(spec=DewpointResult)
    mock_res.limiting_component = "H2O"

    with patch(
        "upstream_drift_tools.process_calculators.acid_gas_dewpoint_calculator.AcidGasDewpointCalculator.calculate_dewpoint_mixture",
        return_value=mock_res,
    ):
        mock_res.dewpoint_margin_c = np.nan
        res = estimate_condensation_risk(10.0, 10.0, comp)
        assert res["risk_level"] == "Unknown"

        mock_res.dewpoint_margin_c = -5.0
        res = estimate_condensation_risk(10.0, 10.0, comp)
        assert res["risk_level"] == "Critical"

        mock_res.dewpoint_margin_c = 5.0
        res = estimate_condensation_risk(10.0, 10.0, comp, safety_margin_c=10.0)
        assert res["risk_level"] == "High"

        mock_res.dewpoint_margin_c = 15.0
        res = estimate_condensation_risk(10.0, 10.0, comp, safety_margin_c=10.0)
        assert res["risk_level"] == "Medium"

        mock_res.dewpoint_margin_c = 30.0
        res = estimate_condensation_risk(10.0, 10.0, comp, safety_margin_c=10.0)
        assert res["risk_level"] == "Low"


try:
    import importlib.util

    HAS_PYQT = importlib.util.find_spec("PyQt6") is not None
except ImportError:
    HAS_PYQT = False


class TestAcidGasDewpointWidget:
    @pytest.fixture(autouse=True)
    def prevent_qt_quit(self) -> Any:
        if HAS_PYQT:
            from PyQt6.QtWidgets import QApplication

            app = QApplication.instance()
            if app:
                app.setQuitOnLastWindowClosed(False)

    @pytest.fixture(autouse=True)
    def patch_state(self, monkeypatch) -> Any:
        if not HAS_PYQT:
            return
        try:
            from sidekick.ui.mixins.calculator_state_mixin import (
                CalculatorStateMixin,
            )

            def mock_init(self, *args, **kwargs) -> Any:
                self.copyable_widgets = []
                self.input_widgets = []

            monkeypatch.setattr(CalculatorStateMixin, "__init__", mock_init)
            if hasattr(CalculatorStateMixin, "restore_state"):
                monkeypatch.setattr(
                    CalculatorStateMixin, "restore_state", lambda *args, **kwargs: None
                )
        except ImportError:
            pass

    @pytest.mark.skipif(not HAS_PYQT, reason="PyQt is required to test the widget")
    def test_widget_initialization(self, qtbot) -> Any:
        from sidekick.process_calculators.acid_gas_dewpoint_calculator import (
            AcidGasDewpointCalculatorWidget,
        )

        widget = AcidGasDewpointCalculatorWidget()
        widget.show()
        qtbot.waitExposed(widget)
        assert widget.calculator is not None
        assert widget.layout() is not None

    @pytest.mark.skipif(not HAS_PYQT, reason="PyQt is required to test the widget")
    def test_widget_calculate(self, qtbot) -> Any:
        from sidekick.process_calculators.acid_gas_dewpoint_calculator import (
            AcidGasComposition,
            AcidGasDewpointCalculatorWidget,
            DewpointResult,
        )

        widget = AcidGasDewpointCalculatorWidget()
        widget.show()
        qtbot.waitExposed(widget)

        mock_result = DewpointResult(
            temperature_c=150.0,
            temperature_k=423.15,
            pressure_bar=30.0,
            pressure_pa=3000000.0,
            composition=AcidGasComposition(),
            h2o_dewpoint_c=130.0,
            hf_dewpoint_c=25.0,
            hcl_dewpoint_c=10.0,
            h2s_dewpoint_c=0.0,
            overall_dewpoint_c=130.0,
            limiting_component="H2O",
            h2o_vapor_pressure_pa=1000.0,
            hf_vapor_pressure_pa=10.0,
            hcl_vapor_pressure_pa=5.0,
            h2s_vapor_pressure_pa=1.0,
            h2o_partial_pressure_pa=500.0,
            hf_partial_pressure_pa=5.0,
            hcl_partial_pressure_pa=1.0,
            h2s_partial_pressure_pa=0.5,
            dewpoint_margin_c=20.0,
            condensation_risk="LOW - Safe margin",
            calculation_method="antoine",
            warnings=[],
            sources=[],
        )

        with patch.object(
            widget.calculator, "calculate_dewpoint_mixture", return_value=mock_result
        ):
            widget.calculate()
            assert "130.00" in widget.result_area.toPlainText()

    @pytest.mark.skipif(not HAS_PYQT, reason="PyQt is required to test the widget")
    def test_widget_calculate_error(self, qtbot) -> Any:
        from sidekick.process_calculators.acid_gas_dewpoint_calculator import (
            AcidGasDewpointCalculatorWidget,
        )

        widget = AcidGasDewpointCalculatorWidget()
        widget.show()
        qtbot.waitExposed(widget)

        with (
            patch.object(
                widget.calculator,
                "calculate_dewpoint_mixture",
                side_effect=ValueError("Test Error DP"),
            ),
            pytest.raises(ValueError, match="Test Error DP"),
        ):
            widget.calculate()

    @pytest.mark.skipif(not HAS_PYQT, reason="PyQt is required to test the widget")
    def test_widget_close_event(self, qtbot) -> Any:
        from PyQt6.QtGui import QCloseEvent
        from sidekick.process_calculators.acid_gas_dewpoint_calculator import (
            AcidGasDewpointCalculatorWidget,
        )

        widget = AcidGasDewpointCalculatorWidget()
        widget.show()
        qtbot.waitExposed(widget)
        widget.save_state = MagicMock()
        event = QCloseEvent()
        widget.closeEvent(event)
        widget.save_state.assert_called_once()
