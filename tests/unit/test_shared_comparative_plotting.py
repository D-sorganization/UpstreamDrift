from unittest.mock import MagicMock

import matplotlib.figure
import numpy as np
import pytest

from src.shared.python.comparative_analysis import (
    AlignedSignals,
    ComparativeSwingAnalyzer,
    ComparisonMetric,
)
from src.shared.python.comparative_plotting import ComparativePlotter


class TestComparativePlotter:
    """Tests for ComparativePlotter class."""

    @pytest.fixture
    def mock_analyzer(self):
        """Mock ComparativeSwingAnalyzer."""
        analyzer = MagicMock(spec=ComparativeSwingAnalyzer)
        analyzer.name_a = "Swing A"
        analyzer.name_b = "Swing B"
        return analyzer

    @pytest.fixture
    def plotter(self, mock_analyzer):
        """Create ComparativePlotter instance."""
        return ComparativePlotter(mock_analyzer)

    @pytest.fixture
    def mock_figure(self):
        """Mock matplotlib Figure."""
        fig = MagicMock(spec=matplotlib.figure.Figure)
        # Setup subplot mocks
        ax = MagicMock()
        fig.add_subplot.return_value = ax
        fig.add_subfigure.return_value = fig  # Simplify for testing

        # Mock GridSpec
        mock_gs = MagicMock()
        mock_gs.__getitem__.return_value = (
            MagicMock()
        )  # Support gs[0,0] returning a subspec
        fig.add_gridspec.return_value = mock_gs

        return fig

    def test_init(self, plotter, mock_analyzer):
        """Test initialization."""
        assert plotter.analyzer == mock_analyzer
        assert "a" in plotter.colors

    def test_plot_comparison_success(self, plotter, mock_analyzer, mock_figure):
        """Test plot_comparison with valid data."""
        # Setup mock aligned signals
        aligned = AlignedSignals(
            times=np.linspace(0, 1, 100),
            signal_a=np.zeros(100),
            signal_b=np.zeros(100),
            error_curve=np.zeros(100),
            rms_error=0.0,
            correlation=1.0,
        )
        mock_analyzer.align_signals.return_value = aligned

        # Call method
        plotter.plot_comparison(mock_figure, "test_field")

        # Verify calls
        mock_analyzer.align_signals.assert_called_with("test_field", joint_idx=None)
        assert mock_figure.add_gridspec.called
        assert mock_figure.add_subplot.called

    def test_plot_comparison_no_data(self, plotter, mock_analyzer, mock_figure):
        """Test plot_comparison when no data is available."""
        mock_analyzer.align_signals.return_value = None

        plotter.plot_comparison(mock_figure, "test_field")

        # Should just add one subplot with text
        mock_figure.add_subplot.assert_called()
        ax = mock_figure.add_subplot.return_value
        ax.text.assert_called()

    def test_plot_phase_comparison(self, plotter, mock_analyzer, mock_figure):
        """Test plot_phase_comparison."""
        aligned = AlignedSignals(
            times=np.linspace(0, 1, 10),
            signal_a=np.zeros(10),
            signal_b=np.zeros(10),
            error_curve=np.zeros(10),
            rms_error=0.0,
            correlation=1.0,
        )
        mock_analyzer.align_signals.return_value = aligned

        plotter.plot_phase_comparison(fig=mock_figure, joint_idx=0)

        # Should align positions and velocities
        mock_analyzer.align_signals.assert_any_call("joint_positions", joint_idx=0)
        mock_analyzer.align_signals.assert_any_call("joint_velocities", joint_idx=0)

    def test_plot_coordination_comparison(self, plotter, mock_analyzer, mock_figure):
        """Test plot_coordination_comparison."""
        aligned = AlignedSignals(
            times=np.linspace(0, 1, 10),
            signal_a=np.zeros(10),
            signal_b=np.zeros(10),
            error_curve=np.zeros(10),
            rms_error=0.0,
            correlation=1.0,
        )
        mock_analyzer.align_signals.return_value = aligned

        plotter.plot_coordination_comparison(mock_figure, 0, 1)

        # Should align both joints
        mock_analyzer.align_signals.assert_any_call("joint_positions", joint_idx=0)
        mock_analyzer.align_signals.assert_any_call("joint_positions", joint_idx=1)

    def test_plot_3d_trajectory_comparison(self, plotter, mock_analyzer, mock_figure):
        """Test plot_3d_trajectory_comparison."""
        # Setup recorder mocks within analyzer
        rec_a = MagicMock()
        rec_b = MagicMock()
        mock_analyzer.recorder_a = rec_a
        mock_analyzer.recorder_b = rec_b

        rec_a.get_time_series.return_value = (np.arange(10), np.zeros((10, 3)))
        rec_b.get_time_series.return_value = (np.arange(10), np.zeros((10, 3)))

        plotter.plot_3d_trajectory_comparison(mock_figure)

        rec_a.get_time_series.assert_called_with("club_head_position")
        rec_b.get_time_series.assert_called_with("club_head_position")

    def test_plot_dashboard(self, plotter, mock_analyzer, mock_figure):
        """Test plot_dashboard."""
        # Mock metric report
        metric = ComparisonMetric("Test", 1.0, 1.0, 0.0, 0.0)
        mock_analyzer.generate_comparison_report.return_value = {"metrics": [metric]}

        # Mock align_signals for subplots
        aligned = AlignedSignals(
            times=np.linspace(0, 1, 10),
            signal_a=np.zeros(10),
            signal_b=np.zeros(10),
            error_curve=np.zeros(10),
            rms_error=0.0,
            correlation=1.0,
        )
        mock_analyzer.align_signals.return_value = aligned

        plotter.plot_dashboard(mock_figure)

        mock_figure.add_gridspec.assert_called()
        assert mock_analyzer.generate_comparison_report.called
