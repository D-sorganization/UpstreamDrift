"""Tests for drake_gui_analysis.py."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.engines.physics_engines.drake.python.src.drake_gui_analysis import (
    AnalysisMixin,
)


class DummyGUI(AnalysisMixin):
    """Dummy GUI class to test AnalysisMixin."""

    def __init__(self):
        """Initialize dummy attributes."""
        self.recorder = MagicMock()
        self.plant = MagicMock()
        self.eval_context = MagicMock()
        self.chk_live_analysis = MagicMock()
        self.chk_induced_vec = MagicMock()
        self.combo_induced_source = MagicMock()


@pytest.fixture
def dummy_gui() -> DummyGUI:
    """Fixture providing a DummyGUI instance."""
    return DummyGUI()


class TestAnalysisMixin:
    """Tests for AnalysisMixin."""

    def test_is_analysis_enabled_checkbox(self, dummy_gui: DummyGUI) -> None:
        """Test _is_analysis_enabled checks the UI checkbox."""
        dummy_gui.chk_live_analysis.isChecked.return_value = True
        
        # Should be true because checkbox is checked
        assert dummy_gui._is_analysis_enabled() is True
        
        dummy_gui.chk_live_analysis.isChecked.return_value = False
        assert dummy_gui._is_analysis_enabled() is False

    def test_is_analysis_enabled_config(self, dummy_gui: DummyGUI) -> None:
        """Test _is_analysis_enabled checks the recorder config."""
        dummy_gui.chk_live_analysis.isChecked.return_value = False
        
        # Should be false if config is empty
        dummy_gui.recorder.analysis_config = {}
        assert dummy_gui._is_analysis_enabled() is False
        
        # Should be true if any of the specific keys are true
        dummy_gui.recorder.analysis_config = {"ztcf": True}
        assert dummy_gui._is_analysis_enabled() is True

    @patch("src.engines.physics_engines.drake.python.src.drake_gui_analysis.DrakeInducedAccelerationAnalyzer")
    def test_compute_live_analysis(self, mock_analyzer_cls: MagicMock, dummy_gui: DummyGUI) -> None:
        """Test _compute_live_analysis updates contexts and calls analyzer."""
        mock_analyzer = MagicMock()
        mock_analyzer.compute_components.return_value = {"gravity": np.array([1.0])}
        mock_analyzer.compute_counterfactuals.return_value = {"ztcf_accel": np.array([2.0])}
        mock_analyzer_cls.return_value = mock_analyzer
        
        dummy_gui.recorder.induced_accelerations = {}
        dummy_gui.recorder.counterfactuals = {}
        
        q = np.array([0.0])
        v = np.array([0.0])
        
        # Stub the specific sources method to do nothing for this test
        dummy_gui._compute_specific_sources = MagicMock()  # type: ignore
        
        dummy_gui._compute_live_analysis(q, v)
        
        dummy_gui.plant.SetPositions.assert_called_once_with(dummy_gui.eval_context, q)
        dummy_gui.plant.SetVelocities.assert_called_once_with(dummy_gui.eval_context, v)
        
        mock_analyzer.compute_components.assert_called_once_with(dummy_gui.eval_context)
        mock_analyzer.compute_counterfactuals.assert_called_once_with(dummy_gui.eval_context)
        
        # Check that it appended to recorder
        assert "gravity" in dummy_gui.recorder.induced_accelerations
        assert len(dummy_gui.recorder.induced_accelerations["gravity"]) == 1
        
        assert "ztcf_accel" in dummy_gui.recorder.counterfactuals
        assert len(dummy_gui.recorder.counterfactuals["ztcf_accel"]) == 1

    def test_compute_live_analysis_missing_preconditions(self, dummy_gui: DummyGUI) -> None:
        """Test _compute_live_analysis raises error if preconditions missing."""
        dummy_gui.plant = None
        with pytest.raises(ValueError, match="DbC Blocked: Precondition failed."):
            dummy_gui._compute_live_analysis(np.array([]), np.array([]))

    def test_compute_specific_sources(self, dummy_gui: DummyGUI) -> None:
        """Test _compute_specific_sources parses config and UI correctly."""
        dummy_gui.chk_induced_vec.isChecked.return_value = True
        dummy_gui.combo_induced_source.currentText.return_value = "0"
        dummy_gui.recorder.analysis_config = {"induced_accel_sources": ["1", "gravity"]}
        
        dummy_gui.plant.num_velocities.return_value = 2
        
        mock_analyzer = MagicMock()
        mock_analyzer.compute_specific_control.side_effect = lambda ctx, tau: tau * 2.0
        
        res = {}
        dummy_gui._compute_specific_sources(mock_analyzer, res)
        
        # Should have computed for '0' and '1', ignoring 'gravity'
        assert "0" in res
        assert "1" in res
        assert "gravity" not in res
        
        # Check that tau was formulated correctly
        np.testing.assert_array_equal(res["0"], np.array([2.0, 0.0]))
        np.testing.assert_array_equal(res["1"], np.array([0.0, 2.0]))

    @patch("src.engines.physics_engines.drake.python.src.drake_gui_analysis.HAS_MATPLOTLIB", False)
    def test_show_induced_acceleration_plot_no_matplotlib(self, dummy_gui: DummyGUI) -> None:
        """Test plotting aborts if matplotlib is not available."""
        with patch("src.engines.physics_engines.drake.python.src.drake_gui_analysis.QtWidgets.QMessageBox.warning") as mock_warn:
            dummy_gui._show_induced_acceleration_plot()
            mock_warn.assert_called_once_with(dummy_gui, "Error", "Matplotlib not found.")

    @patch("src.engines.physics_engines.drake.python.src.drake_gui_analysis.HAS_MATPLOTLIB", False)
    def test_show_counterfactuals_plot_no_matplotlib(self, dummy_gui: DummyGUI) -> None:
        """Test plotting aborts if matplotlib is not available."""
        with patch("src.engines.physics_engines.drake.python.src.drake_gui_analysis.QtWidgets.QMessageBox.warning") as mock_warn:
            dummy_gui._show_counterfactuals_plot()
            mock_warn.assert_called_once_with(dummy_gui, "Error", "Matplotlib not found.")
