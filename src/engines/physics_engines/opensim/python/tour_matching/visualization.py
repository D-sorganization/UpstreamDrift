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
) -> Path:
    """Plot synchronized 3D marker trajectory overlays."""
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(10, 8), dpi=150)
    ax = fig.add_subplot(111, projection="3d")

    for name, traj in target_trajectories.items():
        t_data = np.asarray(traj, dtype=np.float64)
        if t_data.shape[1] == 3:
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
        if m_data.shape[1] == 3:
            ax.plot(
                m_data[:, 0],
                m_data[:, 1],
                m_data[:, 2],
                label=f"{name} (model)",
                lw=2.0,
            )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")  # type: ignore[attr-defined]
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    fig.savefig(target)
    plt.close(fig)
    logger.info("Saved 3D trajectory plot to %s", target)
    return target


def animate_marker_overlay(
    observed_m: NDArray[Any],
    predicted_m: NDArray[Any],
    valid_mask: NDArray[Any],
    time_s: NDArray[Any],
    output_path: Path | str,
    stride: int = 6,
    title: str = "Tour Driver Matching",
) -> Path:
    """Animated GIF of observed (blue) against model (red) markers, Y up.

    Shared by the OS-3b IK overlay and the OS-7 Moco replay playback so both
    lanes draw the same picture. Invalid observed samples are skipped.
    """
    from matplotlib.animation import FuncAnimation

    if stride < 1:
        raise ValueError("stride must be >= 1")
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frames_to_plot = np.arange(0, len(time_s), stride)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    def update(frame_idx: int) -> list[Any]:
        ax.clear()
        f = frames_to_plot[frame_idx]
        v = valid_mask[f]
        obs = observed_m[f, v]
        pred = predicted_m[f, v]
        ax.scatter(obs[:, 0], obs[:, 2], obs[:, 1], c="blue", label="Observed", s=20)
        ax.scatter(pred[:, 0], pred[:, 2], pred[:, 1], c="red", label="Model", s=20)
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)
        ax.set_zlim(0.0, 2.0)  # type: ignore[attr-defined]
        ax.set_xlabel("X (forward/target) [m]")
        ax.set_ylabel("Z (lateral) [m]")
        ax.set_zlabel("Y (up) [m]")  # type: ignore[attr-defined]
        ax.set_title(f"{title} - frame {f}/{len(time_s)} (t = {time_s[f]:.3f} s)")
        ax.legend(loc="upper right")
        return []

    anim = FuncAnimation(fig, update, frames=len(frames_to_plot), blit=False)
    anim.save(str(target), writer="pillow", fps=15)
    plt.close(fig)
    logger.info("Saved marker overlay animation to %s", target)
    return target
