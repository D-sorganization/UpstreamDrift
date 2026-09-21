"""Visualization generation for OpenSim tour matching trajectories and errors (OS-6).

Provides headless matplotlib-based plotting for:
- Synchronized target vs. model 3D trajectory overlays.
- Error-versus-time residual timecourses.
- Actuator effort and effort-rate profiles.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def plot_marker_error_timecourse(
    times: Sequence[float] | NDArray[Any],
    errors_by_marker: Mapping[str, Sequence[float] | NDArray[Any]],
    output_path: Path | str,
    title: str = "OpenSim Tour Matching: Marker Error vs Time",
) -> Path:
    """Plot marker residual errors over time."""
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    t_arr = np.asarray(times, dtype=np.float64)

    for name, err_series in errors_by_marker.items():
        e_arr = np.asarray(err_series, dtype=np.float64)
        ax.plot(t_arr, e_arr, label=name, lw=1.5)

    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Marker Error (m)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)
    if len(errors_by_marker) <= 15:
        ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(target)
    plt.close(fig)
    logger.info("Saved marker error plot to %s", target)
    return target


def plot_effort_and_rates(
    times: Sequence[float] | NDArray[Any],
    efforts: Mapping[str, Sequence[float] | NDArray[Any]],
    rates: Mapping[str, Sequence[float] | NDArray[Any]],
    output_path: Path | str,
) -> Path:
    """Plot actuator efforts and effort rates across two subplots."""
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, dpi=150)
    t_arr = np.asarray(times, dtype=np.float64)

    for name, u_vals in efforts.items():
        ax1.plot(t_arr, np.asarray(u_vals, dtype=np.float64), label=name, lw=1.2)

    for name, r_vals in rates.items():
        ax2.plot(t_arr, np.asarray(r_vals, dtype=np.float64), label=name, lw=1.2)

    ax1.set_ylabel("Effort (N*m / N)", fontsize=11)
    ax1.set_title("Actuator Continuous Torque Profiles", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2.set_xlabel("Time (s)", fontsize=11)
    ax2.set_ylabel("Effort Rate (N*m/s)", fontsize=11)
    ax2.set_title(
        "Actuator Torque Derivatives (dtau/dt)", fontsize=12, fontweight="bold"
    )
    ax2.grid(True, linestyle="--", alpha=0.6)

    fig.tight_layout()
    fig.savefig(target)
    plt.close(fig)
    logger.info("Saved effort/rate plot to %s", target)
    return target


def plot_3d_trajectory_overlay(
    target_trajectories: Mapping[str, NDArray[Any]],
    model_trajectories: Mapping[str, NDArray[Any]],
    output_path: Path | str,
    title: str = "Synchronized 3D Marker Trajectories: Target vs Model",
    camera_preset: Any = None,
) -> Path:
    """Plot synchronized 3D marker trajectory overlays with optional golf camera preset."""
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(10, 8), dpi=150)
    ax = fig.add_subplot(111, projection="3d")

    for name, traj in target_trajectories.items():
        t_data = np.asarray(traj, dtype=np.float64)
        if t_data.ndim == 2 and t_data.shape[1] == 3:
            ax.plot(
                t_data[:, 0],
                t_data[:, 1],
                t_data[:, 2],
                label=f"{name} (target)",
                linestyle="--",
                lw=1.5,
            )

    for name, traj in model_trajectories.items():
        m_data = np.asarray(traj, dtype=np.float64)
        if m_data.ndim == 2 and m_data.shape[1] == 3:
            ax.plot(
                m_data[:, 0],
                m_data[:, 1],
                m_data[:, 2],
                label=f"{name} (model)",
                lw=2.0,
            )

    ax.set_xlabel("X (forward/target) [m]")
    ax.set_ylabel("Y (up) [m]")
    ax.set_zlabel("Z (lateral) [m]")  # type: ignore[attr-defined]
    ax.set_title(title, fontsize=12, fontweight="bold")

    # Set camera elevation and azimuth if preset is supplied
    if camera_preset is not None and hasattr(ax, "view_init"):
        preset_val = getattr(camera_preset, "value", str(camera_preset)).lower()
        if "front" in preset_val:
            ax.view_init(elev=5, azim=-90)  # type: ignore[attr-defined]
        elif "side" in preset_val:
            ax.view_init(elev=5, azim=0)  # type: ignore[attr-defined]
        elif "down_the_line" in preset_val or "dtl" in preset_val:
            ax.view_init(elev=10, azim=180)  # type: ignore[attr-defined]
        elif "overhead" in preset_val:
            ax.view_init(elev=90, azim=-90)  # type: ignore[attr-defined]

    ax.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    fig.savefig(target)
    plt.close(fig)
    logger.info("Saved 3D trajectory plot to %s", target)
    return target
