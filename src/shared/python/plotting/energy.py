"""Energy Analysis Plotting Module.

Provides specialized plots for energy analysis:
- Kinetic, potential, total energy over time
- Energy flow diagrams
- Power analysis
- Work done by actuators

All functions follow consistent interface and styling.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from src.shared.python.plotting.base import RecorderInterface
from src.shared.python.plotting.config import PlotConfig, resolve_figure
from src.shared.python.plotting.identity import (
    PlotIdentity,
    resolve_and_apply_identity_footer,
)

if TYPE_CHECKING:
    pass

_T = TypeVar("_T")


def _retrieve_power_data(
    recorder: RecorderInterface,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Retrieve power data from recorder, computing from torques if needed.

    DRY helper: eliminates duplicated power retrieval logic in
    plot_power_analysis and plot_cumulative_work.

    Returns:
        Tuple of (times, powers) or None if no data available.
    """
    try:
        times, powers = recorder.get_time_series("actuator_powers")
        return np.asarray(times), np.asarray(powers)
    except (KeyError, AttributeError):
        pass
    try:
        times, torques = recorder.get_time_series("joint_torques")
        _, velocities = recorder.get_time_series("joint_velocities")
        return np.asarray(times), np.asarray(torques) * np.asarray(velocities)
    except (KeyError, AttributeError):
        return None


def _prepare_power_series(
    recorder: RecorderInterface,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Retrieve and normalize power data to a 2D (times, powers[N, D]) pair.

    DRY helper: eliminates duplicated fetch/validate/reshape logic shared
    by plot_power_analysis and plot_cumulative_work.

    Returns:
        Tuple of (times, powers) with powers guaranteed 2D, or None if no
        data is available.
    """
    result = _retrieve_power_data(recorder)
    if result is None:
        return None
    times, powers = result
    if len(times) == 0:
        return None
    if powers.ndim == 1:
        powers = powers.reshape(-1, 1)
    return times, powers


def _retrieve_kinetic_potential(
    recorder: RecorderInterface,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Retrieve kinetic + potential energy series from recorder.

    DRY helper: eliminates duplicated retrieval/validation logic shared by
    plot_energy_breakdown and plot_energy_flow.

    Returns:
        Tuple of (times, kinetic, potential) or None if unavailable.
    """
    try:
        times, kinetic = recorder.get_time_series("kinetic_energy")
        _, potential = recorder.get_time_series("potential_energy")
    except (KeyError, AttributeError):
        return None

    if len(times) == 0:
        return None

    return np.asarray(times), np.asarray(kinetic), np.asarray(potential)


def _init_energy_figure(
    recorder: RecorderInterface,
    ax: Axes | None,
    config: PlotConfig | None,
    identity: PlotIdentity | None,
) -> tuple[Figure, Axes, PlotConfig]:
    """Validate recorder, resolve fig/ax/config, and stamp the identity footer.

    DRY helper: every function in this module shares this exact preamble
    (recorder validation, figure resolution, identity footer) before
    branching into its own data retrieval.
    """
    if recorder is None:
        raise ValueError("recorder must be provided")
    fig, ax, config = resolve_figure(ax, config)
    resolve_and_apply_identity_footer(fig, recorder, identity)
    return fig, ax, config


def _init_energy_plot(
    recorder: RecorderInterface,
    ax: Axes | None,
    config: PlotConfig | None,
    identity: PlotIdentity | None,
    retrieve: Callable[[RecorderInterface], _T | None],
    no_data_message: str,
) -> tuple[Figure, Axes, PlotConfig, _T | None]:
    """Init the figure/identity, then retrieve data or annotate "no data".

    DRY helper shared by every ``plot_*`` function that can bail out early
    when its data series is unavailable (breakdown, power analysis,
    cumulative work, energy flow). Callers still need to check whether the
    returned data is ``None`` and, if so, return ``(fig, ax)`` immediately.
    """
    fig, ax, config = _init_energy_figure(recorder, ax, config, identity)
    data = retrieve(recorder)
    if data is None:
        ax.text(0.5, 0.5, no_data_message, ha="center", va="center")
    return fig, ax, config, data


def _kp_plot_init(
    recorder: RecorderInterface,
    ax: Axes | None,
    config: PlotConfig | None,
    identity: PlotIdentity | None,
) -> tuple[Figure, Axes, PlotConfig, tuple[np.ndarray, np.ndarray, np.ndarray] | None]:
    """Bind ``_init_energy_plot`` to the kinetic/potential retriever.

    DRY helper shared by plot_energy_breakdown and plot_energy_flow, the
    two functions that both start from a kinetic+potential energy series.
    """
    return _init_energy_plot(
        recorder,
        ax,
        config,
        identity,
        _retrieve_kinetic_potential,
        "No energy data available",
    )


def _init_power_plot(
    recorder: RecorderInterface,
    ax: Axes | None,
    config: PlotConfig | None,
    identity: PlotIdentity | None,
) -> tuple[Figure, Axes, PlotConfig, tuple[np.ndarray, np.ndarray] | None]:
    """Bind ``_init_energy_plot`` to the per-actuator power retriever.

    DRY helper shared by plot_power_analysis and plot_cumulative_work, the
    two functions that both start from a per-actuator power series.
    """
    return _init_energy_plot(
        recorder, ax, config, identity, _prepare_power_series, "No power data available"
    )


def plot_energy_overview(  # noqa: C901
    recorder: RecorderInterface,
    ax: Axes | None = None,
    config: PlotConfig | None = None,
    show_components: bool = True,
    identity: PlotIdentity | None = None,
) -> tuple[Figure, Axes]:
    """Plot total, kinetic, and potential energy over time.

    Args:
        recorder: Data source implementing RecorderInterface
        ax: Optional axes to plot on
        config: Plot configuration
        show_components: Whether to show KE/PE breakdown
        identity: Optional engine/model/run identity rendered as a figure
            footer. Derived from ``recorder.engine`` when not provided.

    Returns:
        Tuple of (figure, axes)
    """
    fig, ax, config = _init_energy_figure(recorder, ax, config, identity)

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
    identity: PlotIdentity | None = None,
) -> tuple[Figure, Axes]:
    """Plot stacked area chart of energy components.

    Args:
        recorder: Data source implementing RecorderInterface
        ax: Optional axes to plot on
        config: Plot configuration
        identity: Optional engine/model/run identity rendered as a figure
            footer. Derived from ``recorder.engine`` when not provided.

    Returns:
        Tuple of (figure, axes)
    """
    fig, ax, config, result = _kp_plot_init(recorder, ax, config, identity)
    if result is None:
        return fig, ax
    times, kinetic, potential = result

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
    identity: PlotIdentity | None = None,
) -> tuple[Figure, Axes]:
    """Plot instantaneous power for each actuator.

    Power = torque * angular_velocity

    Args:
        recorder: Data source implementing RecorderInterface
        ax: Optional axes to plot on
        joint_indices: Indices of joints to plot
        joint_names: Names for legend
        config: Plot configuration
        identity: Optional engine/model/run identity rendered as a figure
            footer. Derived from ``recorder.engine`` when not provided.

    Returns:
        Tuple of (figure, axes)
    """
    fig, ax, config, result = _init_power_plot(recorder, ax, config, identity)
    if result is None:
        return fig, ax
    times, powers = result

    n_joints = powers.shape[1]
    indices = joint_indices or list(range(min(n_joints, 6)))  # Limit to 6 for clarity
    names = joint_names or [f"Joint {i}" for i in indices]

    for i, (idx, name) in enumerate(zip(indices, names, strict=False)):
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
    identity: PlotIdentity | None = None,
) -> tuple[Figure, Axes]:
    """Plot cumulative work done by each actuator.

    Work = integral of power dt

    Args:
        recorder: Data source implementing RecorderInterface
        ax: Optional axes to plot on
        joint_indices: Indices of joints to plot
        joint_names: Names for legend
        config: Plot configuration
        identity: Optional engine/model/run identity rendered as a figure
            footer. Derived from ``recorder.engine`` when not provided.

    Returns:
        Tuple of (figure, axes)
    """
    fig, ax, config, result = _init_power_plot(recorder, ax, config, identity)
    if result is None:
        return fig, ax
    times, powers = result

    # Compute cumulative work via trapezoidal integration
    n_joints = powers.shape[1]
    work = np.zeros_like(powers)
    dt = np.diff(times, prepend=times[0])

    for j in range(n_joints):
        work[:, j] = np.cumsum(powers[:, j] * dt)

    indices = joint_indices or list(range(min(n_joints, 6)))
    names = joint_names or [f"Joint {i}" for i in indices]

    for i, (idx, name) in enumerate(zip(indices, names, strict=False)):
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
    identity: PlotIdentity | None = None,
) -> tuple[Figure, Axes]:
    """Plot energy flow: work in, dissipation, stored energy.

    Shows the balance between work done by actuators and
    energy stored in the system.

    Args:
        recorder: Data source implementing RecorderInterface
        ax: Optional axes to plot on
        config: Plot configuration
        identity: Optional engine/model/run identity rendered as a figure
            footer. Derived from ``recorder.engine`` when not provided.

    Returns:
        Tuple of (figure, axes)
    """
    fig, ax, config, result = _kp_plot_init(recorder, ax, config, identity)
    if result is None:
        return fig, ax
    times, kinetic, potential = result
    total = kinetic + potential

    # Rate of energy change
    dt = np.diff(times, prepend=times[0])
    energy_rate = np.diff(total, prepend=total[0]) / (dt + 1e-10)

    ax.fill_between(
        times,
        0,
        energy_rate,
        where=energy_rate >= 0,  # type: ignore[arg-type]
        alpha=0.7,
        label="Energy Input",
        color=config.colors.tertiary,
    )
    ax.fill_between(
        times,
        0,
        energy_rate,
        where=energy_rate < 0,  # type: ignore[arg-type]
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
