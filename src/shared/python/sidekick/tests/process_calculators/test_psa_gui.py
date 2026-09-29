# ruff: noqa: E501
"""Tests for psa_gui.py."""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from PyQt6.QtWidgets import QApplication
from sidekick.process_calculators.psa_package.psa_gui import (
    InputPanel,
    PFDWidget,
    PSAMainWindow,
    ResultsPanel,
    SensitivityPlotWidget,
)
from sidekick.process_calculators.psa_package.ui import (
    main_window as refactored_main_window,
)
from sidekick.process_calculators.psa_package.ui.input_panel import (
    InputPanel as RefactoredInputPanel,
)
from sidekick.process_calculators.psa_package.ui.main_window import (
    PSAMainWindow as RefactoredPSAMainWindow,
)

PSA_PACKAGE = (
    Path(__file__).resolve().parents[2] / "process_calculators" / "psa_package"
)


@pytest.fixture
def dummy_qapp() -> QApplication:
    """Create a dummy QApplication for UI testing."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def test_input_panel_initialization(dummy_qapp: QApplication) -> None:
    """Test that the input panel initializes correctly."""
    panel = InputPanel()
    assert panel.feed_input.text() == "1100"

    # Check default values
    feed, s2, prod, components = panel.get_parameters()
    assert feed == 1100.0
    assert s2 == 1.0
    assert prod == 0.0
    assert len(components) == 7

    # Test reset defaults
    panel.feed_input.setText("500")
    panel._reset_defaults()
    assert panel.feed_input.text() == "1100"


@pytest.mark.parametrize("panel_cls", [InputPanel, RefactoredInputPanel])
def test_input_panel_emits_change_signal(
    panel_cls: type[Any], dummy_qapp: QApplication, qtbot: Any
) -> None:
    """InputPanel owns child-widget wiring and exposes a panel-level signal."""
    panel = panel_cls()
    qtbot.addWidget(panel)

    with qtbot.waitSignal(panel.input_changed, timeout=1000):
        panel.s2_recycle_slider.setValue(99)

    with qtbot.waitSignal(panel.input_changed, timeout=1000):
        panel.prod_recycle_slider.setValue(1)

    with qtbot.waitSignal(panel.input_changed, timeout=1000):
        panel.feed_input.setText("1200")

    with qtbot.waitSignal(panel.input_changed, timeout=1000):
        item = panel.component_table.item(0, 1)
        assert item is not None
        item.setText("31")


@pytest.mark.parametrize(
    "path",
    [
        PSA_PACKAGE / "psa_gui.py",
        PSA_PACKAGE / "ui" / "main_window.py",
    ],
)
def test_psa_main_window_uses_panel_change_contract(path: Path) -> None:
    """Main windows must subscribe to InputPanel's public change signal."""
    tree = ast.parse(path.read_text(encoding="utf-8"))

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_connect_signals":
            source = ast.get_source_segment(path.read_text(encoding="utf-8"), node)
            assert source is not None
            assert "self.input_panel.input_changed.connect" in source
            assert "self.input_panel.s2_recycle_slider" not in source
            assert "self.input_panel.prod_recycle_slider" not in source
            assert "self.input_panel.feed_input" not in source
            assert "self.input_panel.component_table" not in source
            return

    raise AssertionError(f"{path} has no _connect_signals method")


def _mock_psa_results() -> MagicMock:
    results = MagicMock()
    results.h2_recovery_pct = 85.5
    results.h2_purity_pct = 99.99
    results.total_net_product_scfm = 500.0
    results.total_exhaust_scfm = 200.0
    results.mass_balance_error = 0.0001
    results.s2_tail_h2_pct = 5.0
    results.s2_tail_o2_pct = 1.0
    results.component_names = ["H2", "CH4"]

    flows = MagicMock()
    flows.fresh_feed = [100.0, 50.0]
    flows.mixed_feed = [110.0, 55.0]
    flows.exhaust = [10.0, 5.0]
    flows.interstage = [90.0, 45.0]
    flows.s2_tail = [5.0, 2.5]
    flows.s2_tail_recycle = [5.0, 2.5]
    flows.gross_product = [85.0, 42.5]
    flows.net_product = [80.0, 40.0]
    results.flows = flows
    results.total_feed_scfm = 150.0

    compositions = MagicMock()
    compositions.fresh_feed = [66.6, 33.3]
    compositions.mixed_feed = [66.6, 33.3]
    compositions.exhaust = [66.6, 33.3]
    compositions.interstage = [66.6, 33.3]
    compositions.s2_tail = [66.6, 33.3]
    compositions.net_product = [66.6, 33.3]
    results.compositions = compositions
    return results


def test_refactored_psa_main_window_calculation_and_signal_paths(
    dummy_qapp: QApplication, qtbot: Any
) -> None:
    """Exercise extracted PSA main-window behavior, not only the facade module."""
    with (
        patch.object(refactored_main_window, "PSAModel") as mock_model,
        patch.object(refactored_main_window.QMessageBox, "warning") as mock_warning,
        patch.object(refactored_main_window.QMessageBox, "critical") as mock_critical,
    ):
        model_instance = MagicMock()
        model_instance.calculate.return_value = _mock_psa_results()
        mock_model.return_value = model_instance

        window = RefactoredPSAMainWindow()
        qtbot.addWidget(window)

        assert window.windowTitle() == "Two-Stage PSA System Analysis"
        assert model_instance.calculate.called

        model_instance.calculate.reset_mock()
        window.input_panel.input_changed.emit()
        assert model_instance.calculate.called

        update_plot = MagicMock()
        window.sensitivity_widget._update_plot = update_plot
        window._on_tab_change(window.tab_widget.indexOf(window.sensitivity_widget))
        update_plot.assert_called_once_with()

        model_instance.calculate.side_effect = ValueError("bad input")
        window._calculate()
        assert mock_warning.called

        model_instance.calculate.side_effect = RuntimeError("boom")
        window._calculate()
        assert mock_critical.called


def test_refactored_psa_main_window_launch_and_help_actions(
    dummy_qapp: QApplication, qtbot: Any
) -> None:
    """Cover extracted launch/help branches with dialogs and processes patched."""
    with (
        patch.object(refactored_main_window.PSAMainWindow, "_calculate"),
        patch.object(refactored_main_window.QMessageBox, "warning") as mock_warning,
        patch.object(refactored_main_window.QMessageBox, "information") as mock_info,
        patch.object(refactored_main_window.QMessageBox, "about") as mock_about,
        patch.object(refactored_main_window.subprocess, "Popen") as mock_popen,
        patch.object(refactored_main_window.webbrowser, "open") as mock_open,
    ):
        window = RefactoredPSAMainWindow()
        qtbot.addWidget(window)

        with patch.object(refactored_main_window.os.path, "exists", return_value=False):
            window._launch_jupyter()
            window._launch_webapp()
        assert mock_warning.call_count >= 2

        with patch.object(refactored_main_window.os.path, "exists", return_value=True):
            window._launch_jupyter()
            window._launch_webapp()
        assert mock_popen.call_count == 2
        assert mock_info.call_count == 2

        standard_buttons = refactored_main_window.QMessageBox.StandardButton

        class FakeMessageBox:
            StandardButton = standard_buttons

            def __init__(self, *_args: Any, **_kwargs: Any) -> None:
                pass

            def setWindowTitle(self, _title: str) -> None:
                pass

            def setText(self, _text: str) -> None:
                pass

            def setInformativeText(self, _text: str) -> None:
                pass

            def setStandardButtons(self, _buttons: Any) -> None:
                pass

            def setDefaultButton(self, _button: Any) -> None:
                pass

            def exec(self) -> Any:
                return standard_buttons.Open

        with patch.object(refactored_main_window, "QMessageBox", FakeMessageBox):
            window._launch_colab()
        mock_open.assert_called_once_with("https://colab.research.google.com/")

        window._show_about()
        assert mock_about.called


def test_results_panel_initialization(dummy_qapp: QApplication) -> None:
    """Test that the results panel initializes correctly."""
    panel = ResultsPanel()
    assert panel.h2_recovery_label.text() == "--"

    # Mock PSAResults
    mock_results = MagicMock()
    mock_results.h2_recovery_pct = 85.5
    mock_results.h2_purity_pct = 99.99
    mock_results.total_net_product_scfm = 500.0
    mock_results.total_exhaust_scfm = 200.0
    mock_results.mass_balance_error = 0.0001
    mock_results.s2_tail_h2_pct = 5.0
    mock_results.s2_tail_o2_pct = 1.0
    mock_results.component_names = ["H2", "CH4"]

    # Mock flows
    mock_flows = MagicMock()
    mock_flows.fresh_feed = [100.0, 50.0]
    mock_flows.mixed_feed = [110.0, 55.0]
    mock_flows.exhaust = [10.0, 5.0]
    mock_flows.interstage = [90.0, 45.0]
    mock_flows.s2_tail = [5.0, 2.5]
    mock_flows.s2_tail_recycle = [5.0, 2.5]
    mock_flows.gross_product = [85.0, 42.5]
    mock_flows.net_product = [80.0, 40.0]
    mock_results.flows = mock_flows
    mock_results.total_feed_scfm = 150.0

    # Mock compositions
    mock_comps = MagicMock()
    mock_comps.fresh_feed = [66.6, 33.3]
    mock_comps.mixed_feed = [66.6, 33.3]
    mock_comps.exhaust = [66.6, 33.3]
    mock_comps.interstage = [66.6, 33.3]
    mock_comps.s2_tail = [66.6, 33.3]
    mock_comps.net_product = [66.6, 33.3]
    mock_results.compositions = mock_comps

    panel.update_results(mock_results)
    assert panel.h2_recovery_label.text() == "85.50%"


@patch(
    "upstream_drift_tools.process_calculators.psa_package.psa_gui.calculate_o2_safety_analysis"
)
@patch(
    "upstream_drift_tools.process_calculators.psa_package.psa_gui.calculate_sensitivity"
)
def test_sensitivity_plot_widget(
    mock_calc_sens: Any, mock_calc_o2: Any, dummy_qapp: QApplication
) -> None:
    """Test the sensitivity plot widget."""
    import numpy as np

    mock_sensitivity = {
        "h2_recovery": np.ones((51, 25)) * np.linspace(70.0, 80.0, 25),
        "net_product": np.ones((51, 25)) * np.linspace(400.0, 500.0, 25),
    }
    mock_calc_sens.return_value = mock_sensitivity

    mock_analysis = {
        "stage1_o2_removal": np.linspace(50, 95, 51),
        "s2_tail_o2": np.zeros((51, 4)),
    }
    mock_calc_o2.return_value = mock_analysis

    widget = SensitivityPlotWidget()

    # Update plot for each type
    plot_types = [
        "H2 Recovery vs Recycle",
        "Net Product vs Recycle",
        "O2 Safety Analysis",
        "3D Recovery Surface",
        "Contour Map",
    ]
    for p_type in plot_types:
        widget.plot_type_combo.setCurrentText(p_type)
        widget._update_plot()


@patch("upstream_drift_tools.process_calculators.psa_package.psa_gui.QMessageBox")
@patch("PyQt6.QtWidgets.QMainWindow.show")
def test_psa_main_window_initialization(
    mock_show: Any, mock_msg_box: Any, dummy_qapp: QApplication
) -> None:
    """Test PSA main window initialization."""
    with patch(
        "upstream_drift_tools.process_calculators.psa_package.psa_gui.PSAModel"
    ) as mock_model:
        mock_results = MagicMock()
        mock_results.h2_recovery_pct = 85.5
        mock_results.h2_purity_pct = 99.99
        mock_results.total_net_product_scfm = 500.0
        mock_results.total_exhaust_scfm = 200.0
        mock_results.mass_balance_error = 0.0001
        mock_results.s2_tail_h2_pct = 5.0
        mock_results.s2_tail_o2_pct = 1.0
        mock_results.component_names = ["H2", "CH4"]

        mock_flows = MagicMock()
        mock_flows.fresh_feed = [100.0, 50.0]
        mock_flows.mixed_feed = [110.0, 55.0]
        mock_flows.exhaust = [10.0, 5.0]
        mock_flows.interstage = [90.0, 45.0]
        mock_flows.s2_tail = [5.0, 2.5]
        mock_flows.s2_tail_recycle = [5.0, 2.5]
        mock_flows.gross_product = [85.0, 42.5]
        mock_flows.net_product = [80.0, 40.0]
        mock_results.flows = mock_flows
        mock_results.total_feed_scfm = 150.0

        # Mock compositions
        mock_comps = MagicMock()
        mock_comps.fresh_feed = [66.6, 33.3]
        mock_comps.mixed_feed = [66.6, 33.3]
        mock_comps.exhaust = [66.6, 33.3]
        mock_comps.interstage = [66.6, 33.3]
        mock_comps.s2_tail = [66.6, 33.3]
        mock_comps.net_product = [66.6, 33.3]
        mock_results.compositions = mock_comps

        mock_instance = MagicMock()
        mock_instance.calculate.return_value = mock_results
        mock_model.return_value = mock_instance
        window = PSAMainWindow()
        assert window.windowTitle() == "Two-Stage PSA System Analysis"

        # Test calculation
        window._calculate()
        assert mock_instance.calculate.called

        # Test calculation with error ValueError
        mock_instance.calculate.side_effect = ValueError("Test PSAModel Error")
        window._calculate()
        assert mock_msg_box.warning.called

        # Test calculation with error RuntimeError
        mock_instance.calculate.side_effect = RuntimeError("Test general error")
        window._calculate()
        assert mock_msg_box.critical.called

        # Test notebook launches
        with patch(
            "upstream_drift_tools.process_calculators.psa_package.psa_gui.subprocess.Popen"
        ) as mock_popen:
            with patch(
                "upstream_drift_tools.process_calculators.psa_package.psa_gui.os.path.exists",
                return_value=True,
            ):
                window._launch_jupyter()
                assert mock_popen.called

        with patch(
            "upstream_drift_tools.process_calculators.psa_package.psa_gui.subprocess.Popen"
        ) as mock_popen2:
            with patch(
                "upstream_drift_tools.process_calculators.psa_package.psa_gui.os.path.exists",
                return_value=True,
            ):
                window._launch_webapp()
                assert mock_popen2.called

        # Test Colab launch
        with patch(
            "upstream_drift_tools.process_calculators.psa_package.psa_gui.webbrowser.open"
        ) as mock_web:
            window._launch_colab()
            assert not mock_web.called

        # Test about
        window._show_about()
        assert mock_msg_box.about.called


def test_results_panel_update_results_rejects_none(dummy_qapp: QApplication) -> None:
    """update_results validates at the public boundary with ValueError.

    Under ``-O`` the previous assert-based guard would be stripped and a None
    would reach ``results.h2_recovery_pct`` producing an opaque AttributeError.
    The boundary check must hold regardless of optimization level.
    """
    panel = ResultsPanel()
    with pytest.raises(ValueError, match="results must be provided"):
        panel.update_results(None)  # type: ignore[arg-type]


def test_pfd_widget(dummy_qapp: QApplication) -> None:
    """Test the Process Flow Diagram widget."""
    widget = PFDWidget()
    assert widget.image_label.text() is not None
