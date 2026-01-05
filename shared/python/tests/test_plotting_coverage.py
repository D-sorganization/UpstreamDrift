"""Tests for shared.python.plotting coverage."""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Mock matplotlib before importing plotting
mock_matplotlib = MagicMock()
mock_pyplot = MagicMock()
mock_figure = MagicMock()
mock_axes = MagicMock()
mock_backend = MagicMock()
mock_patches = MagicMock()
mock_collections = MagicMock()

# Setup module structure for mocks
mock_matplotlib.pyplot = mock_pyplot
mock_matplotlib.figure = mock_figure
mock_matplotlib.axes = mock_axes
mock_matplotlib.patches = mock_patches
mock_matplotlib.collections = mock_collections

with patch.dict(
    sys.modules,
    {
        "matplotlib": mock_matplotlib,
        "matplotlib.pyplot": mock_pyplot,
        "matplotlib.figure": mock_figure,
        "matplotlib.axes": mock_axes,
        "matplotlib.patches": mock_patches,
        "matplotlib.collections": mock_collections,
        "matplotlib.backends.backend_qt5agg": mock_backend,
        "matplotlib.backends.backend_qtagg": mock_backend,
    },
):
    from shared.python.plotting import GolfSwingPlotter


@pytest.fixture
def mock_plotter() -> GolfSwingPlotter:
    """Fixture to create a GolfSwingPlotter with mocked dependencies."""
    mock_recorder = MagicMock()
    # Configure mock recorder to return valid data for get_time_series
    mock_recorder.get_time_series.return_value = (
        np.array([0, 1]),
        np.array([]),
    )  # type: ignore
    return GolfSwingPlotter(mock_recorder)


def test_plotter_initialization(mock_plotter: GolfSwingPlotter) -> None:
    """Test initialization of GolfSwingPlotter."""
    assert mock_plotter is not None
    # Verify color scheme
    assert mock_plotter.colors["primary"] == "#1f77b4"


def test_plot_kinematic_sequence_bars(mock_plotter: GolfSwingPlotter) -> None:
    """Test plotting kinematic sequence bars."""
    # Create dummy data
    # Note: The method signature expects segment_indices as the second argument,
    # and optional impact_time as third.
    # It does NOT take KinematicSequenceResult as second argument based on my reading of the code.
    # But wait, there is another method plot_kinematic_sequence that takes analyzer_result.
    # Let's test both if possible, or correct this one.

    # plot_kinematic_sequence_bars(self, fig: Figure, segment_indices: dict[str, int], impact_time: float | None = None)

    segment_indices = {"pelvis": 0, "torso": 1}
    fig = MagicMock()

    # Configure recorder to return some data
    mock_plotter.recorder.get_time_series.return_value = (  # type: ignore
        np.array([0, 1]),
        np.array([[0, 0], [1, 1]]),
    )

    mock_plotter.plot_kinematic_sequence_bars(fig, segment_indices)
    fig.add_subplot.assert_called()


def test_plot_angle_angle_diagram(mock_plotter: GolfSwingPlotter) -> None:
    """Test plotting angle-angle diagram."""
    fig = MagicMock()
    # Configure recorder
    mock_plotter.recorder.get_time_series.return_value = (  # type: ignore
        np.array([0, 1]),
        np.array([[0, 0], [1, 1]]),
    )

    mock_plotter.plot_angle_angle_diagram(fig, 0, 1, "Test Diagram")
    fig.add_subplot.assert_called()


def test_plot_continuous_relative_phase(mock_plotter: GolfSwingPlotter) -> None:
    """Test plotting CRP."""
    fig = MagicMock()
    crp_data = np.linspace(0, 1, 100)
    # Configure recorder (needed for time base)
    times = np.linspace(0, 1, 100)
    mock_plotter.recorder.get_time_series.return_value = (times, np.zeros((100, 2)))  # type: ignore

    mock_plotter.plot_continuous_relative_phase(fig, crp_data, "Test CRP")
    fig.add_subplot.assert_called()


def test_plot_coupling_angle(mock_plotter: GolfSwingPlotter) -> None:
    """Test plotting coupling angle."""
    fig = MagicMock()
    coupling_angle = np.linspace(0, 1, 100)
    times = np.linspace(0, 1, 100)
    # Configure recorder
    mock_plotter.recorder.get_time_series.return_value = (times, np.zeros((100, 2)))  # type: ignore

    mock_plotter.plot_coupling_angle(fig, coupling_angle, "Test Coupling Angle")
    fig.add_subplot.assert_called()
