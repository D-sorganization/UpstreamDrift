from unittest.mock import MagicMock

import numpy as np
import pytest
from src.shared.python.plotting import GolfSwingPlotter
from src.shared.python.validation_pkg.statistical_analysis import StatisticalAnalyzer


class TestAdvancedAnalysis:
    def test_plotter_advanced_methods(self) -> None:
        # Mock recorder and canvas
        recorder = MagicMock()
        # Mock time series
        t = np.linspace(0, 1, 100)
        data = np.random.randn(100, 3)
        recorder.get_time_series.return_value = (t, data)
        recorder.get_induced_acceleration_series.return_value = (t, data)
        recorder.get_counterfactual_series.return_value = (t, data)
        # Mock club induced accel
        recorder.get_club_induced_acceleration_series.return_value = (t, data)

        plotter = GolfSwingPlotter(recorder, enable_cache=False)
        fig = MagicMock()
        ax = MagicMock()
        fig.add_subplot.return_value = ax

        # Test Poincaré Plot
        # Requires dimensions list of strings/ints
        plotter.plot_poincare_map_3d(
            fig,
            dimensions=[("position", 0), ("velocity", 0), ("acceleration", 0)],
            section_condition=("velocity", 0, 0.0),
        )
        assert fig.add_subplot.called

        # Test Lyapunov Plot
        plotter.plot_lyapunov_exponent(fig, joint_idx=0)
        assert fig.add_subplot.called

        # Test Recurrence Plot
        rm = np.random.randint(0, 2, (50, 50))
        plotter.plot_recurrence_plot(fig, recurrence_matrix=rm)
        assert fig.add_subplot.called
