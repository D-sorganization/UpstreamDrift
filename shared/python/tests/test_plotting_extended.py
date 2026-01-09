
import pytest
import numpy as np
from unittest.mock import MagicMock
from matplotlib.figure import Figure
from shared.python.plotting import GolfSwingPlotter
from shared.python.tests.test_plotting import MockRecorder

@pytest.fixture
def extended_plotter():
    recorder = MockRecorder()
    # Add dummy com data for vector field
    recorder.data["com_position"] = (np.linspace(0, 1, 10), np.random.rand(10, 3))
    return GolfSwingPlotter(recorder, joint_names=["Joint1", "Joint2", "Joint3"])

@pytest.fixture
def fig():
    return Figure()

def test_plot_dynamic_correlation(extended_plotter, fig):
    extended_plotter.plot_dynamic_correlation(fig, joint_idx_1=0, joint_idx_2=1, window_size=5)
    assert len(fig.axes) > 0
    # Check if plot was drawn
    ax = fig.axes[0]
    assert ax.get_title().startswith("Dynamic Correlation")

def test_plot_synergy_trajectory(extended_plotter, fig):
    # Mock synergy result
    synergy_result = MagicMock()
    synergy_result.activations = np.random.rand(2, 10)

    extended_plotter.plot_synergy_trajectory(fig, synergy_result)
    assert len(fig.axes) > 0
    ax = fig.axes[0]
    assert ax.get_title() == "Synergy Space Trajectory"

def test_plot_3d_vector_field(extended_plotter, fig):
    # Setup recorder with necessary data
    # Angular momentum and CoM position
    extended_plotter.recorder.data["angular_momentum"] = (np.linspace(0, 1, 10), np.random.rand(10, 3))
    extended_plotter.recorder.data["com_position"] = (np.linspace(0, 1, 10), np.random.rand(10, 3))

    extended_plotter.plot_3d_vector_field(fig, "angular_momentum", "com_position")
    assert len(fig.axes) > 0
    assert fig.axes[0].name == "3d"
    ax = fig.axes[0]
    assert ax.get_title().startswith("3D Vector Field")

def test_plot_local_stability(extended_plotter, fig):
    # Setup data
    extended_plotter.recorder.data["joint_velocities"] = (np.linspace(0, 1, 50), np.random.rand(50, 3))
    extended_plotter.recorder.data["joint_positions"] = (np.linspace(0, 1, 50), np.random.rand(50, 3))

    extended_plotter.plot_local_stability(fig, joint_idx=0, tau=1, embedding_dim=2)
    assert len(fig.axes) > 0
    ax = fig.axes[0]
    assert ax.get_title().startswith("Local Stability")

def test_plot_dynamic_correlation_insufficient_data(extended_plotter, fig):
    extended_plotter.recorder.data["joint_velocities"] = (np.array([]), np.array([]))
    extended_plotter.plot_dynamic_correlation(fig, 0, 1)
    assert len(fig.axes) > 0
    # Should display text
    texts = [t.get_text() for t in fig.axes[0].texts]
    assert any("No data" in t or "Insufficient" in t for t in texts)
