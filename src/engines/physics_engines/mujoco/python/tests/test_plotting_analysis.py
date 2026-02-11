"""Tests for advanced plotting analysis."""

import importlib.util
import os
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

# Mock mujoco before importing plotting
sys.modules["mujoco"] = MagicMock()

# Dynamic import for Plotting
file_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../mujoco_humanoid_golf/plotting.py")
)
spec = importlib.util.spec_from_file_location("plotting", file_path)
assert spec is not None
assert spec.loader is not None
plotting = importlib.util.module_from_spec(spec)
sys.modules["plotting"] = plotting
spec.loader.exec_module(plotting)

GolfSwingPlotter = plotting.GolfSwingPlotter


@pytest.fixture
def mock_recorder():
    recorder = MagicMock()

    # Mock data
    times = np.linspace(0, 1, 1000)
    data = np.zeros((1000, 2))  # 2 joints
    data[:, 0] = np.sin(2 * np.pi * 5 * times)  # 5 Hz
    data[:, 1] = np.cos(2 * np.pi * 10 * times)  # 10 Hz

    recorder.get_time_series.return_value = (times, data)
    return recorder


def test_plot_frequency_analysis(mock_recorder):
    plotter = GolfSwingPlotter(mock_recorder)
    fig = MagicMock()

    plotter.plot_frequency_analysis(fig, joint_idx=0, signal_type="velocity")

    # Check if ax.semilogy was called
    assert fig.add_subplot.called
    ax = fig.add_subplot.return_value
    assert ax.semilogy.called


def test_plot_spectrogram(mock_recorder):
    plotter = GolfSwingPlotter(mock_recorder)
    fig = MagicMock()

    plotter.plot_spectrogram(fig, joint_idx=0, signal_type="velocity")

    assert fig.add_subplot.called
    ax = fig.add_subplot.return_value
    assert ax.pcolormesh.called
