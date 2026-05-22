"""Visualization tools for Ground Reaction Forces and Center of Mass trajectories."""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any

from src.shared.python.contracts import require

try:
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes  # noqa: F401
    from matplotlib.figure import Figure  # noqa: F401
except ImportError:
    plt = None  # type: ignore


def plot_grf_and_com_3d(
    force_df: pd.DataFrame,
    com_trajectory: np.ndarray,
    downsample_factor: int = 10,
    vector_scale: float = 0.001,
) -> tuple[Any, Any]:
    """Plot 3D GRF vectors and dynamic COM trajectory.

    Args:
        force_df: DataFrame containing combined force plate data with
            columns fx, fy, fz, cop_x, cop_y, cop_z.
        com_trajectory: (N, 3) array of Center of Mass positions over time.
        downsample_factor: Factor by which to downsample vectors for visualization
            clarity (e.g., plot every 10th vector).
        vector_scale: Scaling factor for force vectors to fit the spatial plot.

    Returns:
        Tuple of (matplotlib Figure, matplotlib Axes3D).

    Raises:
        ImportError: If matplotlib is not installed.
    """
    if plt is None:
        raise ImportError("matplotlib is required for visualization.")

    require(len(com_trajectory) > 0, "COM trajectory must not be empty")
    require(
        len(force_df) == len(com_trajectory),
        "force_df and com_trajectory must have matching lengths",
    )

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot COM trajectory
    ax.plot(
        com_trajectory[:, 0],
        com_trajectory[:, 1],
        com_trajectory[:, 2],
        label="Dynamic COM",
        color="blue",
        linewidth=2,
        alpha=0.8,
    )

    # Plot COP trajectory
    ax.plot(
        force_df["cop_x"],
        force_df["cop_y"],
        force_df["cop_z"],
        label="Center of Pressure",
        color="red",
        linewidth=2,
        alpha=0.6,
    )

    # Downsample for vector plotting
    idx = np.arange(0, len(force_df), downsample_factor)
    sampled_df = force_df.iloc[idx]
    sampled_com = com_trajectory[idx]

    # Plot GRF Vectors originating from COP
    ax.quiver(
        sampled_df["cop_x"],
        sampled_df["cop_y"],
        sampled_df["cop_z"],
        sampled_df["fx"],
        sampled_df["fy"],
        sampled_df["fz"],
        length=vector_scale,
        normalize=False,
        color="green",
        alpha=0.5,
        arrow_length_ratio=0.1,
        label="GRF Vectors",
    )

    # Plot lines from COP to COM (moment arms)
    for i in range(len(idx)):
        ax.plot(
            [sampled_df["cop_x"].iloc[i], sampled_com[i, 0]],
            [sampled_df["cop_y"].iloc[i], sampled_com[i, 1]],
            [sampled_df["cop_z"].iloc[i], sampled_com[i, 2]],
            color="gray",
            linestyle="--",
            alpha=0.2,
        )

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")  # type: ignore
    ax.set_title("Dynamic Center of Mass and Ground Reaction Forces")
    ax.legend()

    # Auto-scale axes to be equal
    x_limits = ax.get_xlim3d()  # type: ignore
    y_limits = ax.get_ylim3d()  # type: ignore
    z_limits = ax.get_zlim3d()  # type: ignore

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])  # type: ignore
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])  # type: ignore
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])  # type: ignore

    return fig, ax
