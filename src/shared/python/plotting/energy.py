"""Energy Analysis Plotting Module.

Provides specialized plots for energy analysis:
- Kinetic, potential, total energy over time
- Energy flow diagrams
- Power analysis
- Work done by actuators

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
    pass


def plot_energy_overview(
    recorder: RecorderInterface,
    ax: Axes | None = None,
    config: PlotConfig | None = None,
    show_components: bool = True,
) -> tuple[Figure, Axes]:
    """Plot total, kinetic, and potential energy over time.

    Args:
        recorder: Data source implementing RecorderInterface
        ax: Optional axes to plot on
        config: Plot configuration
        show_components: Whether to show KE/PE breakdown

    Returns:
        Tuple of (figure, axes)
    """
    config = config or DEFAULT_CONFIG

    if ax is None:
        fig, ax = config.create_figure()
    else:
        fig = ax.figure

    # Get energy data
    try:
        t_total, total = recorder.get_time_series("total_energy")
    except (KeyError, AttributeError):
        total = np.array([])
        t_total = np.array([])

    try:
        t_kin, kinetic = recorder.get_time_series("kinetic_energy")
    except (KeyError, AttributeError):
        kinetic = np.array([])
        t_kin = np.array([])

    try:
        t_pot, potential = recorder.get_time_series("potential_energy")
    except (KeyError, AttributeError):
        potential = np.array([])
        t_pot = np.array([])

    if len(t_total) == 0 and len(t_kin) == 0 and len(t_pot) == 0:
        ax.text(0.5, 0.5, "No energy data available", ha="center", va="center")
        return fig, ax

    # Plot total energy
    if len(t_total) > 0:
        ax.plot(
            t_total,
            np.asarray(total),
            label="Total",
            color=config.colors.primary,
            linewidth=config.line_width * 1.5,
        )

    # Plot components
    if show_components:
        if len(t_kin) > 0:
            ax.plot(
                t_kin,
                np.asarray(kinetic),
                label="Kinetic",
                color=config.colors.secondary,
                linewidth=config.line_width,
            )
        if len(t_pot) > 0:
            ax.plot(
                t_pot,
                np.asarray(potential),
                label="Potential",
                color=config.colors.tertiary,
                linewidth=config.line_width,
            )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Energy [J]")
    ax.set_title("System Energy")
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(config.show_grid, alpha=config.grid_alpha)

    return fig, ax


def plot_energy_breakdown(
    recorder: RecorderInterface,
    ax: Axes | None = None,
    config: PlotConfig | None = None,
) -> tuple[Figure, Axes]:
    """Plot stacked area chart of energy components.

    Args:
        recorder: Data source implementing RecorderInterface
        ax: Optional axes to plot on
        config: Plot configuration

    Returns:
        Tuple of (figure, axes)
    """
    config = config or DEFAULT_CONFIG

    if ax is None:
        fig, ax = config.create_figure()
    else:
        fig = ax.figure

    try:
        times, kinetic = recorder.get_time_series("kinetic_energy")
        _, potential = recorder.get_time_series("potential_energy")
    except (KeyError, AttributeError):
        ax.text(0.5, 0.5, "No energy data available", ha="center", va="center")
        return fig, ax

    if len(times) == 0:
        ax.text(0.5, 0.5, "No energy data available", ha="center", va="center")
        return fig, ax

    kinetic = np.asarray(kinetic)
    potential = np.asarray(potential)

    # Stacked area plot
    ax.fill_between(
        times,
        0,
        kinetic,
        alpha=0.7,
        label="Kinetic Energy",
        color=config.colors.secondary,
    )
    ax.fill_between(
        times,
        kinetic,
        kinetic + potential,
        alpha=0.7,
        label="Potential Energy",
        color=config.colors.tertiary,
    )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Energy [J]")
    ax.set_title("Energy Breakdown")
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(config.show_grid, alpha=config.grid_alpha)

    return fig, ax


def plot_power_analysis(
    recorder: RecorderInterface,
    ax: Axes | None = None,
    joint_indices: list[int] | None = None,
    joint_names: list[str] | None = None,
    config: PlotConfig | None = None,
) -> tuple[Figure, Axes]:
    """Plot instantaneous power for each actuator.

    Power = torque × angular_velocity

    Args:
        recorder: Data source implementing RecorderInterface
        ax: Optional axes to plot on
        joint_indices: Indices of joints to plot
        joint_names: Names for legend
        config: Plot configuration

    Returns:
        Tuple of (figure, axes)
    """
    config = config or DEFAULT_CONFIG

    if ax is None:
        fig, ax = config.create_figure()
    else:
        fig = ax.figure

    try:
        times, powers = recorder.get_time_series("actuator_powers")
    except (KeyError, AttributeError):
        # Compute from torques and velocities
        try:
            times, torques = recorder.get_time_series("joint_torques")
            _, velocities = recorder.get_time_series("joint_velocities")
            torques = np.asarray(torques)
            velocities = np.asarray(velocities)
            powers = torques * velocities
        except (KeyError, AttributeError):
            ax.text(0.5, 0.5, "No power data available", ha="center", va="center")
            return fig, ax

    if len(times) == 0:
        ax.text(0.5, 0.5, "No power data available", ha="center", va="center")
        return fig, ax

    powers = np.asarray(powers)
    if powers.ndim == 1:
        powers = powers.reshape(-1, 1)

    n_joints = powers.shape[1]
    indices = joint_indices or list(range(min(n_joints, 6)))  # Limit to 6 for clarity
    names = joint_names or [f"Joint {i}" for i in indices]

    for i, (idx, name) in enumerate(zip(indices, names)):
        if idx < n_joints:
            color = config.colors.get_color(i)
            ax.plot(times, powers[:, idx], label=name, color=color)

    ax.axhline(0, color=config.colors.foreground, linewidth=0.5, alpha=0.5)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Power [W]")
    ax.set_title("Actuator Power")
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(config.show_grid, alpha=config.grid_alpha)

    return fig, ax


def plot_cumulative_work(
    recorder: RecorderInterface,
    ax: Axes | None = None,
    joint_indices: list[int] | None = None,
    joint_names: list[str] | None = None,
    config: PlotConfig | None = None,
) -> tuple[Figure, Axes]:
    """Plot cumulative work done by each actuator.

    Work = ∫ power dt

    Args:
        recorder: Data source implementing RecorderInterface
        ax: Optional axes to plot on
        joint_indices: Indices of joints to plot
        joint_names: Names for legend
        config: Plot configuration

    Returns:
        Tuple of (figure, axes)
    """
    config = config or DEFAULT_CONFIG

    if ax is None:
        fig, ax = config.create_figure()
    else:
        fig = ax.figure

    try:
        times, powers = recorder.get_time_series("actuator_powers")
    except (KeyError, AttributeError):
        try:
            times, torques = recorder.get_time_series("joint_torques")
            _, velocities = recorder.get_time_series("joint_velocities")
            torques = np.asarray(torques)
            velocities = np.asarray(velocities)
            powers = torques * velocities
        except (KeyError, AttributeError):
            ax.text(0.5, 0.5, "No power data available", ha="center", va="center")
            return fig, ax

    if len(times) == 0:
        ax.text(0.5, 0.5, "No power data available", ha="center", va="center")
        return fig, ax

    powers = np.asarray(powers)
    if powers.ndim == 1:
        powers = powers.reshape(-1, 1)

    # Compute cumulative work via trapezoidal integration
    n_joints = powers.shape[1]
    work = np.zeros_like(powers)
    dt = np.diff(times, prepend=times[0])

    for j in range(n_joints):
        work[:, j] = np.cumsum(powers[:, j] * dt)

    indices = joint_indices or list(range(min(n_joints, 6)))
    names = joint_names or [f"Joint {i}" for i in indices]

    for i, (idx, name) in enumerate(zip(indices, names)):
        if idx < n_joints:
            color = config.colors.get_color(i)
            ax.plot(times, work[:, idx], label=name, color=color)

    ax.axhline(0, color=config.colors.foreground, linewidth=0.5, alpha=0.5)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Work [J]")
    ax.set_title("Cumulative Work by Actuator")
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(config.show_grid, alpha=config.grid_alpha)

    return fig, ax


def plot_energy_flow(
    recorder: RecorderInterface,
    ax: Axes | None = None,
    config: PlotConfig | None = None,
) -> tuple[Figure, Axes]:
    """Plot energy flow: work in, dissipation, stored energy.

    Shows the balance between work done by actuators and
    energy stored in the system.

    Args:
        recorder: Data source implementing RecorderInterface
        ax: Optional axes to plot on
        config: Plot configuration

    Returns:
        Tuple of (figure, axes)
    """
    config = config or DEFAULT_CONFIG

    if ax is None:
        fig, ax = config.create_figure()
    else:
        fig = ax.figure

    try:
        times, kinetic = recorder.get_time_series("kinetic_energy")
        _, potential = recorder.get_time_series("potential_energy")
    except (KeyError, AttributeError):
        ax.text(0.5, 0.5, "No energy data available", ha="center", va="center")
        return fig, ax

    if len(times) == 0:
        ax.text(0.5, 0.5, "No energy data available", ha="center", va="center")
        return fig, ax

    kinetic = np.asarray(kinetic)
    potential = np.asarray(potential)
    total = kinetic + potential

    # Rate of energy change
    dt = np.diff(times, prepend=times[0])
    energy_rate = np.diff(total, prepend=total[0]) / (dt + 1e-10)

    ax.fill_between(
        times,
        0,
        energy_rate,
        where=energy_rate >= 0,
        alpha=0.7,
        label="Energy Input",
        color=config.colors.tertiary,
    )
    ax.fill_between(
        times,
        0,
        energy_rate,
        where=energy_rate < 0,
        alpha=0.7,
        label="Energy Output",
        color=config.colors.quaternary,
    )

    ax.axhline(0, color=config.colors.foreground, linewidth=0.5)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Energy Rate [W]")
    ax.set_title("Energy Flow")
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(config.show_grid, alpha=config.grid_alpha)

    return fig, ax


__all__ = [
    "plot_energy_overview",
    "plot_energy_breakdown",
    "plot_power_analysis",
    "plot_cumulative_work",
    "plot_energy_flow",
]
