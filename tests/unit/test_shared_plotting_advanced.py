"""Tests for advanced features in shared plotting module."""

import unittest
from unittest.mock import MagicMock

import matplotlib
import matplotlib.backend_bases
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from shared.python.plotting import GolfSwingPlotter


class MockRecorder:
    """Mock recorder for testing."""

    def __init__(self):
        self.times = np.linspace(0, 1, 100)
        self.cop_position = np.random.rand(100, 3)
        self.com_position = np.random.rand(100, 3)
        self.ground_forces = np.random.rand(100, 3)
        self.angular_momentum = np.random.rand(100, 3)
        self.joint_positions = np.zeros((100, 2))
        self.joint_velocities = np.zeros((100, 2))

    def get_time_series(self, field_name):
        if field_name == "cop_position":
            return self.times, self.cop_position
        if field_name == "com_position":
            return self.times, self.com_position
        if field_name == "ground_forces":
            return self.times, self.ground_forces
        if field_name == "angular_momentum":
            return self.times, self.angular_momentum
        if field_name == "joint_positions":
            return self.times, self.joint_positions
        if field_name == "joint_velocities":
            return self.times, self.joint_velocities
        raise KeyError(f"Unknown field: {field_name}")

    def get_induced_acceleration_series(self, source_name):
        return self.times, np.zeros((100, 3))

    def get_counterfactual_series(self, cf_name):
        return self.times, np.zeros((100, 3))


class TestSharedPlottingAdvanced(unittest.TestCase):
    """Test advanced plotting features."""

    def setUp(self):
        self.recorder = MockRecorder()
        self.plotter = GolfSwingPlotter(self.recorder)
        self.fig = plt.figure()

    def tearDown(self):
        plt.close(self.fig)

    def test_plot_grf_butterfly_diagram(self):
        """Test plotting GRF butterfly diagram."""
        self.plotter.plot_grf_butterfly_diagram(self.fig)
        # Check if axes were created (projection='3d' creates an Axes3D object)
        self.assertTrue(len(self.fig.axes) > 0)
        ax = self.fig.axes[0]
        self.assertEqual(ax.name, "3d")
        self.assertEqual(ax.get_title(), "GRF Butterfly Diagram")

    def test_plot_grf_butterfly_diagram_no_data(self):
        """Test graceful handling of missing data."""
        empty_recorder = MagicMock()
        empty_recorder.get_time_series.side_effect = KeyError("Data missing")
        plotter = GolfSwingPlotter(empty_recorder)
        plotter.plot_grf_butterfly_diagram(self.fig)
        # Should still create a subplot with error message
        self.assertTrue(len(self.fig.axes) > 0)
        # Should catch exception and not crash

    def test_plot_angular_momentum_3d(self):
        """Test plotting 3D angular momentum."""
        self.plotter.plot_angular_momentum_3d(self.fig)
        self.assertTrue(len(self.fig.axes) > 0)
        ax = self.fig.axes[0]
        self.assertEqual(ax.name, "3d")
        self.assertEqual(ax.get_title(), "3D Angular Momentum Trajectory")

    def test_plot_stability_diagram(self):
        """Test plotting stability diagram."""
        self.plotter.plot_stability_diagram(self.fig)
        self.assertTrue(len(self.fig.axes) > 0)
        ax = self.fig.axes[0]
        self.assertNotEqual(ax.name, "3d")  # Should be 2D
        self.assertEqual(ax.get_title(), "Stability Diagram (CoM vs CoP)")

    def test_plot_stability_diagram_missing_data(self):
        """Test stability diagram with missing data."""
        mock_rec = MagicMock()
        # Return time but empty arrays
        mock_rec.get_time_series.return_value = (np.array([]), np.array([]))
        plotter = GolfSwingPlotter(mock_rec)
        plotter.plot_stability_diagram(self.fig)
        # Should execute without error
        self.assertTrue(len(self.fig.axes) > 0)
