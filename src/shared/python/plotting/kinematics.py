"""Kinematic Plotting Module.

Provides specialized plots for kinematic analysis:
- Joint positions over time
- Joint velocities over time
- Joint accelerations over time
- Angular kinematics
- Linear kinematics (COM, club head)

All functions follow consistent interface and styling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from src.shared.python.plotting.base import RecorderInterface
from src.shared.python.plotting.config import PlotConfig, DEFAULT_CONFIG

if TYPE_CHECKING:
    from matplotlib.lines import Line2D


def plot_joint_positions(
    recorder: RecorderInterface,
    ax: Axes | None = None,
    joint_indices: list[int] | None = None,
    joint_names: list[str] | None = None,
    config: PlotConfig | None = None,
) -> tuple[Figure, Axes]:
    """Plot joint positions over time.

    Args:
        recorder: Data source implementing RecorderInterface
        ax: Optional axes to plot on (creates new figure if None)
        joint_indices: Indices of joints to plot (None = all)
        joint_names: Names for legend (uses "Joint N" if None)
        config: Plot configuration

    Returns:
        Tuple of (figure, axes)
    """
    config = config or DEFAULT_CONFIG

    if ax is None:
        fig, ax = config.create_figure()
    else:
        fig = ax.figure

    times, positions = recorder.get_time_series("joint_positions")

    if len(times) == 0:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        return fig, ax

    positions = np.asarray(positions)
    if positions.ndim == 1:
        positions = positions.reshape(-1, 1)

    n_joints = positions.shape[1]
    indices = joint_indices or list(range(n_joints))
    names = joint_names or [f"Joint {i}" for i in indices]

    for i, (idx, name) in enumerate(zip(indices, names)):
        if idx < n_joints:
            color = config.colors.get_color(i)
            ax.plot(times, np.rad2deg(positions[:, idx]), label=name, color=color)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Position [deg]")
    ax.set_title("Joint Positions")
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(config.show_grid, alpha=config.grid_alpha)

    return fig, ax


def plot_joint_velocities(
    recorder: RecorderInterface,
    ax: Axes | None = None,
    joint_indices: list[int] | None = None,
    joint_names: list[str] | None = None,
    config: PlotConfig | None = None,
) -> tuple[Figure, Axes]:
    """Plot joint velocities over time.

    Args:
        recorder: Data source implementing RecorderInterface
        ax: Optional axes to plot on (creates new figure if None)
        joint_indices: Indices of joints to plot (None = all)
        joint_names: Names for legend (uses "Joint N" if None)
        config: Plot configuration

    Returns:
        Tuple of (figure, axes)
    """
    config = config or DEFAULT_CONFIG

    if ax is None:
        fig, ax = config.create_figure()
    else:
        fig = ax.figure

    times, velocities = recorder.get_time_series("joint_velocities")

    if len(times) == 0:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        return fig, ax

    velocities = np.asarray(velocities)
    if velocities.ndim == 1:
        velocities = velocities.reshape(-1, 1)

    n_joints = velocities.shape[1]
    indices = joint_indices or list(range(n_joints))
    names = joint_names or [f"Joint {i}" for i in indices]

    for i, (idx, name) in enumerate(zip(indices, names)):
        if idx < n_joints:
            color = config.colors.get_color(i)
            ax.plot(times, np.rad2deg(velocities[:, idx]), label=name, color=color)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Velocity [deg/s]")
    ax.set_title("Joint Velocities")
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(config.show_grid, alpha=config.grid_alpha)

    return fig, ax


def plot_joint_accelerations(
    recorder: RecorderInterface,
    ax: Axes | None = None,
    joint_indices: list[int] | None = None,
    joint_names: list[str] | None = None,
    config: PlotConfig | None = None,
) -> tuple[Figure, Axes]:
    """Plot joint accelerations over time.

    Accelerations are computed from velocities if not directly available.

    Args:
        recorder: Data source implementing RecorderInterface
        ax: Optional axes to plot on (creates new figure if None)
        joint_indices: Indices of joints to plot (None = all)
        joint_names: Names for legend (uses "Joint N" if None)
        config: Plot configuration

    Returns:
        Tuple of (figure, axes)
    """
    config = config or DEFAULT_CONFIG

    if ax is None:
        fig, ax = config.create_figure()
    else:
        fig = ax.figure

    # Try to get accelerations directly, compute from velocities if not available
    try:
        times, accelerations = recorder.get_time_series("joint_accelerations")
    except (KeyError, AttributeError):
        # Compute from velocities
        times, velocities = recorder.get_time_series("joint_velocities")
        velocities = np.asarray(velocities)
        if len(times) > 1:
            dt = np.diff(times)
            accelerations = np.diff(velocities, axis=0) / dt[:, np.newaxis]
            times = times[:-1]
        else:
            accelerations = np.array([])

    if len(times) == 0:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        return fig, ax

    accelerations = np.asarray(accelerations)
    if accelerations.ndim == 1:
        accelerations = accelerations.reshape(-1, 1)

    n_joints = accelerations.shape[1]
    indices = joint_indices or list(range(n_joints))
    names = joint_names or [f"Joint {i}" for i in indices]

    for i, (idx, name) in enumerate(zip(indices, names)):
        if idx < n_joints:
            color = config.colors.get_color(i)
            ax.plot(
                times, np.rad2deg(accelerations[:, idx]), label=name, color=color
            )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Acceleration [deg/s²]")
    ax.set_title("Joint Accelerations")
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(config.show_grid, alpha=config.grid_alpha)

    return fig, ax


def plot_club_head_speed(
    recorder: RecorderInterface,
    ax: Axes | None = None,
    config: PlotConfig | None = None,
    show_peak: bool = True,
) -> tuple[Figure, Axes]:
    """Plot club head speed over time.

    Args:
        recorder: Data source implementing RecorderInterface
        ax: Optional axes to plot on
        config: Plot configuration
        show_peak: Whether to annotate peak speed

    Returns:
        Tuple of (figure, axes)
    """
    config = config or DEFAULT_CONFIG

    if ax is None:
        fig, ax = config.create_figure()
    else:
        fig = ax.figure

    times, speeds = recorder.get_time_series("club_head_speed")

    if len(times) == 0:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        return fig, ax

    speeds = np.asarray(speeds)

    # Convert to mph if in m/s (assume m/s if max > 100)
    if np.max(speeds) < 100:
        speeds_mph = speeds * 2.237  # m/s to mph
        ylabel = "Speed [mph]"
    else:
        speeds_mph = speeds
        ylabel = "Speed [mph]"

    ax.plot(times, speeds_mph, color=config.colors.primary, linewidth=config.line_width)

    if show_peak and len(speeds_mph) > 0:
        peak_idx = np.argmax(speeds_mph)
        peak_time = times[peak_idx]
        peak_speed = speeds_mph[peak_idx]
        ax.axhline(
            peak_speed, color=config.colors.secondary, linestyle="--", alpha=0.5
        )
        ax.annotate(
            f"Peak: {peak_speed:.1f} mph",
            xy=(peak_time, peak_speed),
            xytext=(peak_time + 0.05, peak_speed + 5),
            fontsize=config.legend_size,
            color=config.colors.secondary,
            arrowprops=dict(arrowstyle="->", color=config.colors.secondary),
        )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel(ylabel)
    ax.set_title("Club Head Speed")
    ax.grid(config.show_grid, alpha=config.grid_alpha)

    return fig, ax


def plot_com_trajectory(
    recorder: RecorderInterface,
    ax: Axes | None = None,
    config: PlotConfig | None = None,
    dimensions: str = "xy",
) -> tuple[Figure, Axes]:
    """Plot center of mass trajectory.

    Args:
        recorder: Data source implementing RecorderInterface
        ax: Optional axes to plot on
        config: Plot configuration
        dimensions: Which dimensions to plot ("xy", "xz", "yz", or "3d")

    Returns:
        Tuple of (figure, axes)
    """
    config = config or DEFAULT_CONFIG

    times, positions = recorder.get_time_series("com_position")

    if len(times) == 0:
        if ax is None:
            fig, ax = config.create_figure()
        else:
            fig = ax.figure
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        return fig, ax

    positions = np.asarray(positions)
    if positions.ndim == 1:
        positions = positions.reshape(-1, 3)

    dim_map = {"x": 0, "y": 1, "z": 2}

    if dimensions == "3d":
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        if ax is None:
            fig = Figure(figsize=(config.width, config.height), dpi=config.dpi)
            ax = fig.add_subplot(111, projection="3d")
        else:
            fig = ax.figure

        ax.plot(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            color=config.colors.primary,
        )
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")
    else:
        if ax is None:
            fig, ax = config.create_figure()
        else:
            fig = ax.figure

        d1, d2 = dimensions[0], dimensions[1]
        i1, i2 = dim_map[d1], dim_map[d2]

        # Color by time
        scatter = ax.scatter(
            positions[:, i1],
            positions[:, i2],
            c=times,
            cmap="viridis",
            s=config.marker_size,
        )
        ax.plot(
            positions[:, i1],
            positions[:, i2],
            color=config.colors.primary,
            alpha=0.3,
        )

        fig.colorbar(scatter, ax=ax, label="Time [s]")
        ax.set_xlabel(f"{d1.upper()} [m]")
        ax.set_ylabel(f"{d2.upper()} [m]")

    ax.set_title("Center of Mass Trajectory")
    ax.set_aspect("equal", adjustable="box")

    return fig, ax


def plot_phase_diagram(
    recorder: RecorderInterface,
    joint_index: int = 0,
    ax: Axes | None = None,
    joint_name: str | None = None,
    config: PlotConfig | None = None,
) -> tuple[Figure, Axes]:
    """Plot phase diagram (position vs velocity) for a joint.

    Args:
        recorder: Data source implementing RecorderInterface
        joint_index: Index of joint to analyze
        ax: Optional axes to plot on
        joint_name: Name of joint for title
        config: Plot configuration

    Returns:
        Tuple of (figure, axes)
    """
    config = config or DEFAULT_CONFIG

    if ax is None:
        fig, ax = config.create_figure()
    else:
        fig = ax.figure

    times, positions = recorder.get_time_series("joint_positions")
    _, velocities = recorder.get_time_series("joint_velocities")

    if len(times) == 0:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        return fig, ax

    positions = np.asarray(positions)
    velocities = np.asarray(velocities)

    if positions.ndim == 1:
        positions = positions.reshape(-1, 1)
        velocities = velocities.reshape(-1, 1)

    if joint_index >= positions.shape[1]:
        ax.text(0.5, 0.5, "Joint index out of range", ha="center", va="center")
        return fig, ax

    pos = np.rad2deg(positions[:, joint_index])
    vel = np.rad2deg(velocities[:, joint_index])

    # Plot trajectory colored by time
    scatter = ax.scatter(pos, vel, c=times, cmap="viridis", s=config.marker_size)
    ax.plot(pos, vel, color=config.colors.primary, alpha=0.3, linewidth=0.5)

    # Mark start and end
    ax.scatter(pos[0], vel[0], color=config.colors.tertiary, s=100, marker="o", label="Start", zorder=5)
    ax.scatter(pos[-1], vel[-1], color=config.colors.quaternary, s=100, marker="s", label="End", zorder=5)

    fig.colorbar(scatter, ax=ax, label="Time [s]")

    name = joint_name or f"Joint {joint_index}"
    ax.set_xlabel("Position [deg]")
    ax.set_ylabel("Velocity [deg/s]")
    ax.set_title(f"Phase Diagram: {name}")
    ax.legend(loc="best")
    ax.grid(config.show_grid, alpha=config.grid_alpha)

    return fig, ax


__all__ = [
    "plot_joint_positions",
    "plot_joint_velocities",
    "plot_joint_accelerations",
    "plot_club_head_speed",
    "plot_com_trajectory",
    "plot_phase_diagram",
]
