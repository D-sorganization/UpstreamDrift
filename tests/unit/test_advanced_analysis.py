
from unittest.mock import MagicMock

import numpy as np
import pytest

from shared.python.plotting import GolfSwingPlotter
from shared.python.statistical_analysis import StatisticalAnalyzer


class TestAdvancedAnalysis:
    def test_compute_poincare_map(self) -> None:
        # Create a simple periodic signal: sin(t), cos(t)
        # Using enough points to get good interpolation
        t = np.linspace(0, 4*np.pi, 200)
        pos = np.sin(t).reshape(-1, 1)
        vel = np.cos(t).reshape(-1, 1)
        acc = -np.sin(t).reshape(-1, 1)

        analyzer = StatisticalAnalyzer(
            times=t,
            joint_positions=pos,
            joint_velocities=vel,
            joint_torques=np.zeros_like(pos),
            joint_accelerations=acc
        )

        # Section: vel = 0 (peaks/troughs of position)
        dimensions = [('position', 0), ('velocity', 0), ('acceleration', 0)]
        section = ('velocity', 0, 0.0)

        # Test "both" directions
        result = analyzer.compute_poincare_map(dimensions, section, direction="both")
        if result is None:
            pytest.fail("Returned None")
        points, times = result

        # vel crosses 0 at pi/2, 3pi/2, 5pi/2, 7pi/2 (approx 1.57, 4.71, 7.85, 10.99)
        # Should have 4 crossings
        assert len(points) == 4
        assert len(times) == 4

        # Check values: At vel=0, pos should be +/- 1
        # Allow some tolerance due to linear interpolation
        assert np.allclose(np.abs(points[:, 0]), 1.0, atol=0.01)
        assert np.allclose(points[:, 1], 0.0, atol=0.01)

        # Test "positive" direction (slope of vel > 0 => acc > 0 => pos < 0 troughs)
        result_pos = analyzer.compute_poincare_map(dimensions, section, direction="positive")
        assert result_pos is not None
        points_pos, _ = result_pos

        # Troughs are at 3pi/2 and 7pi/2
        assert len(points_pos) == 2
        assert np.allclose(points_pos[:, 0], -1.0, atol=0.01)

    def test_compute_lyapunov_divergence(self) -> None:
        # Generate some data
        t = np.linspace(0, 10, 500)
        # Sine wave shouldn't diverge (LLE ~ 0)
        data = np.sin(t)

        analyzer = StatisticalAnalyzer(
            times=t,
            joint_positions=data.reshape(-1, 1),
            joint_velocities=data.reshape(-1, 1),
            joint_torques=np.zeros((500, 1))
        )

        time_axis, log_div, slope = analyzer.compute_lyapunov_divergence(
            data, tau=10, dim=2, window=50
        )

        assert len(time_axis) > 0
        assert len(time_axis) == len(log_div)
        assert isinstance(slope, float)

        # For a sine wave, slope should be small/near zero
        assert abs(slope) < 0.5

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
            dimensions=[('position', 0), ('velocity', 0), ('acceleration', 0)],
            section_condition=('velocity', 0, 0.0)
        )
        assert fig.add_subplot.called

        # Test Lyapunov Plot
        plotter.plot_lyapunov_exponent(fig, joint_idx=0)
        assert fig.add_subplot.called

        # Test Recurrence Plot
        rm = np.random.randint(0, 2, (50, 50))
        plotter.plot_recurrence_plot(fig, recurrence_matrix=rm)
        assert fig.add_subplot.called
