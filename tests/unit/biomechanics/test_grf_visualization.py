"""Tests for GRF visualization tools."""

import numpy as np
import pandas as pd
import pytest

from src.shared.python.biomechanics.grf_visualization import plot_grf_and_com_3d


def test_plot_grf_and_com_3d() -> None:
    """Test that the 3D plot function executes without errors."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        pytest.skip("matplotlib not installed")

    # Generate synthetic data
    n_frames = 50
    com_trajectory = np.zeros((n_frames, 3))
    com_trajectory[:, 2] = 1.0  # COM at z=1.0
    com_trajectory[:, 0] = np.linspace(0, 1, n_frames)  # Moving along X

    force_df = pd.DataFrame(
        {
            "fx": np.zeros(n_frames),
            "fy": np.zeros(n_frames),
            "fz": np.ones(n_frames) * 800.0,
            "cop_x": np.linspace(0, 1, n_frames),
            "cop_y": np.zeros(n_frames),
            "cop_z": np.zeros(n_frames),
        }
    )

    fig, ax = plot_grf_and_com_3d(force_df, com_trajectory, downsample_factor=5)

    assert fig is not None
    assert ax is not None
    assert ax.name == "3d"

    # Clean up
    plt.close(fig)


def test_plot_grf_length_mismatch() -> None:
    """Test that length mismatches raise an error."""
    n_frames = 10
    com_trajectory = np.zeros((n_frames, 3))
    force_df = pd.DataFrame(
        {
            "fx": [0.0],
            "fy": [0.0],
            "fz": [0.0],
            "cop_x": [0.0],
            "cop_y": [0.0],
            "cop_z": [0.0],
        }
    )

    with pytest.raises(Exception, match="matching lengths"):
        plot_grf_and_com_3d(force_df, com_trajectory)
