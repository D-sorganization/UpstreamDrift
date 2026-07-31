# ruff: noqa: E501
from typing import Any

from sidekick.process_calculators.wgs_reactor_calculator import (
    WGSReactorEngine,
)


def test_wgs_reactor_engine_equilibrium_constant() -> None:
    engine = WGSReactorEngine()
    # At standard temp 298.15 the constant should be very large (shift to right)
    k_eq_298 = engine.calculate_equilibrium_constant(298.15)
    assert k_eq_298 > 1.0


def test_wgs_reactor_engine_equilibrium_composition() -> None:
    engine = WGSReactorEngine()

    inlet_comp = {
        "CO": 50.0,
        "H2O": 0.0,  # Steam will be added
        "CO2": 0.0,
        "H2": 50.0,
    }

    # 500 C = 773.15 K
    result = engine.calculate_equilibrium_composition(
        inlet_comp, 773.15, 25.0, steam_ratio=2.0
    )

    # Check that reaction shifted
    assert result["conversion"] > 0
    assert result["equilibrium_constant"] > 0
    assert result["composition"]["CO2"] > 0.0


def test_wgs_reactor_size() -> None:
    engine = WGSReactorEngine()
    sizing = engine.size_wgs_reactor(
        feed_rate=1000.0,
        conversion=80.0,
        temperature=773.15,
        catalyst_type="HTS",
    )

    assert sizing["reactor_volume"] > 0
    assert sizing["catalyst_volume"] > 0
    assert sizing["diameter"] > 0
    assert sizing["length"] > 0
    assert sizing["heat_duty"] > 0


def test_engine_load_data_none() -> None:
    # Coverage for passing None to load_data
    engine = WGSReactorEngine(data_file="dummy_nonexistent.json")
    from unittest.mock import patch

    with patch(
        "upstream_drift_tools.process_calculators.wgs_reactor_calculator.safe_read_json",
        return_value={},
    ):
        engine._load_data(None)
        assert engine.catalysts == {}


def test_equilibrium_composition_missing_species() -> None:
    engine = WGSReactorEngine()
    # Force the species DB to return None for a species to test the failure branch
    from unittest.mock import patch

    with patch.object(engine.species_db, "get_species", return_value=None):
        res = engine.calculate_equilibrium_composition({"CO": 50.0}, 773.15, 25.0)
        # Should return mostly zero or fail gracefully, but g_f defaults to 0
        assert "conversion" in res


def test_equilibrium_composition_zero_total() -> None:
    engine = WGSReactorEngine()
    res = engine.calculate_equilibrium_composition({"CO": 0.0}, 773.15, 25.0)
    assert res["conversion"] == 0.0
    assert res["composition"]["CO"] == 0.0


def test_minimal_species_db_fallback() -> None:
    # Need to manipulate the module attributes to test the fallback DB if it wasn't defined
    import importlib
    import sys

    from sidekick.process_calculators import wgs_reactor_calculator

    # Temporarily mock the db import out
    original_modules = dict(sys.modules)
    sys.modules[
        "integrated_process_simulator.calculators.thermodynamic_properties.species_database"
    ] = None

    try:
        importlib.reload(wgs_reactor_calculator)
        db = wgs_reactor_calculator._MinimalSpeciesDB()
        sp = db.get_species("CO_g")
        assert sp is not None
        assert sp.formation_enthalpy < 0
        assert sp.formation_entropy > 0
        assert sp.molecular_weight > 0
        assert db.get_species("Unknown") is None
    finally:
        sys.modules.clear()
        sys.modules.update(original_modules)
        importlib.reload(wgs_reactor_calculator)


import importlib.util
from unittest.mock import MagicMock, patch

import pytest

HAS_PYQT = importlib.util.find_spec("PyQt6") is not None


class TestWGSReactorWidget:
    @pytest.fixture(autouse=True)
    def prevent_qt_quit(self) -> Any:
        if HAS_PYQT:
            from PyQt6.QtWidgets import QApplication

            app = QApplication.instance()
            if app:
                app.setQuitOnLastWindowClosed(False)

    @pytest.fixture(autouse=True)
    def patch_state(self, monkeypatch) -> Any:
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
        from sidekick.process_calculators.wgs_reactor_calculator import (
            WGSReactorCalculatorWidget,
        )

        widget = WGSReactorCalculatorWidget()
        assert widget.engine is not None
        assert widget.layout() is not None

    @pytest.mark.skipif(not HAS_PYQT, reason="PyQt is required to test the widget")
    def test_widget_setup_state_management(self, qtbot) -> Any:
        from sidekick.process_calculators.wgs_reactor_calculator import (
            WGSReactorCalculatorWidget,
        )

        widget = WGSReactorCalculatorWidget()
        widget.findChildren = MagicMock(return_value=[MagicMock()])
        widget.register_splitter = MagicMock()
        widget.register_copyable_widget = MagicMock()
        widget.setup_state_management()

    @pytest.mark.skipif(not HAS_PYQT, reason="PyQt is required to test the widget")
    def test_widget_close_event(self, qtbot) -> Any:
        from PyQt6.QtGui import QCloseEvent
        from sidekick.process_calculators.wgs_reactor_calculator import (
            WGSReactorCalculatorWidget,
        )

        widget = WGSReactorCalculatorWidget()
        widget.show()
        qtbot.waitExposed(widget)
        widget.save_state = MagicMock()
        event = QCloseEvent()
        widget.closeEvent(event)
        widget.save_state.assert_called_once()

    @pytest.mark.skipif(not HAS_PYQT, reason="PyQt is required to test the widget")
    def test_widget_calculate(self, qtbot) -> Any:
        from sidekick.process_calculators.wgs_reactor_calculator import (
            WGSReactorCalculatorWidget,
        )

        widget = WGSReactorCalculatorWidget()
        widget.show()
        qtbot.waitExposed(widget)
        mock_combo = MagicMock()
        mock_combo.currentText.return_value = "HTS"
        widget.catalyst_combo = mock_combo

        # Mock values directly since spinning up full Qt event cycle isn't necessary
        with (
            patch.object(
                widget.engine,
                "calculate_equilibrium_composition",
                return_value={
                    "conversion": 50.0,
                    "equilibrium_constant": 10.0,
                    "h2_co_ratio": 2.0,
                    "heat_released": 100.0,
                    "composition": {"H2": 50.0},
                },
            ),
            patch.object(
                widget.engine,
                "size_wgs_reactor",
                return_value={
                    "reactor_volume": 1.0,
                    "catalyst_volume": 0.8,
                    "diameter": 1.0,
                    "length": 3.0,
                    "heat_duty": 500.0,
                    "ghsv": 3000,
                },
            ),
            patch.object(widget, "create_plots"),
        ):
            # Execute
            widget.calculate()
            # Verify text edit has output
            assert len(widget.results_text.toPlainText()) > 10

    @pytest.mark.skipif(not HAS_PYQT, reason="PyQt is required to test the widget")
    def test_widget_calculate_error(self, qtbot) -> Any:
        from sidekick.process_calculators.wgs_reactor_calculator import (
            WGSReactorCalculatorWidget,
        )

        widget = WGSReactorCalculatorWidget()
        widget.show()
        qtbot.waitExposed(widget)
        with (
            patch.object(
                widget.engine,
                "calculate_equilibrium_composition",
                side_effect=ValueError("Test Error"),
            ),
            patch("PyQt6.QtWidgets.QMessageBox.critical") as mock_msg,
        ):
            widget.calculate()
            mock_msg.assert_called_once()

    @pytest.mark.skipif(not HAS_PYQT, reason="PyQt is required to test the widget")
    def test_widget_create_plots(self, qtbot) -> Any:
        from sidekick.process_calculators.wgs_reactor_calculator import (
            WGSReactorCalculatorWidget,
        )

        widget = WGSReactorCalculatorWidget()
        inlet = {"H2": 10.0, "CO": 10.0}
        outlet = {"H2": 15.0, "CO": 5.0}
        with patch.object(widget.figure, "add_subplot") as mock_subplot:
            widget.create_plots(inlet, outlet)
            mock_subplot.assert_called_once()

    @pytest.mark.skipif(not HAS_PYQT, reason="PyQt is required to test the widget")
    def test_create_wgs_reactor_calculator(self, qtbot) -> Any:
        from sidekick.process_calculators.wgs_reactor_calculator import (
            create_wgs_reactor_calculator,
        )

        widget = create_wgs_reactor_calculator()
        widget.show()
        qtbot.waitExposed(widget)
        assert widget is not None
