"""Render the reviewer-facing trajectory control-authority figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
ARTICLE = ROOT / "docs/research/proximal_distal_energy_transfer"
REPORT_PATH = ARTICLE / "data/trajectory_control_authority.json"
ARRAY_PATH = ARTICLE / "data/trajectory_control_authority.npz"
FIGURE_PATH = ARTICLE / "figures/fig_trajectory_control_authority.pdf"

COLORS = {
    "full": "#3a0ca3",
    "shoulder": "#e76f51",
    "wrist": "#2a9d8f",
    "varying": "#4361ee",
    "frozen": "#adb5bd",
}


def _plot_event(axis, phase: np.ndarray, guard: np.ndarray) -> None:
    axis.plot(phase, guard, color="#264653", linewidth=1.5)
    axis.axhline(0.0, color="#6c757d", linewidth=0.9, linestyle="--")
    axis.scatter([1.0], [guard[-1]], color="#d00000", zorder=3, label="Event")
    axis.set_xlim(0.0, 1.0)
    axis.set_xlabel("Fraction of Delivery-Event Time")
    axis.set_ylabel(r"Guard $\theta_s+\theta_w$ [rad]")
    axis.set_title("Exact-Step Transverse Event")
    axis.legend(frameon=False)


def _plot_authority_histories(
    axis, phase: np.ndarray, histories: tuple[np.ndarray, ...]
) -> None:
    styles = (
        (COLORS["full"], "Both Torque Channels"),
        (COLORS["shoulder"], "Shoulder Only"),
        (COLORS["wrist"], "Wrist Only"),
    )
    for history, (color, label) in zip(histories, styles, strict=True):
        trace = np.trace(history, axis1=1, axis2=2)
        axis.plot(phase[1:], trace[1:], color=color, label=label)
    axis.set_yscale("log")
    axis.set_xlim(0.0, 1.0)
    axis.set_xlabel("Fraction of Delivery-Event Time")
    axis.set_ylabel("Scaled Energy-Gramian Trace")
    axis.set_title("Trajectory-Varying Finite-Window Authority")
    axis.legend(frameon=False)


def _plot_tangent_spectrum(axis, tangent_eigenvalues: np.ndarray) -> None:
    channel_names = ("Full", "Shoulder Only", "Wrist Only")
    x = np.arange(3)
    width = 0.24
    for mode in range(3):
        axis.bar(
            x + (mode - 1) * width,
            tangent_eigenvalues[:, mode],
            width,
            label=f"Tangent Mode {mode + 1}",
        )
    axis.set_yscale("log")
    axis.set_xticks(x, channel_names)
    axis.set_ylabel("Event-Tangent Gramian Eigenvalue")
    axis.set_title("Three-Dimensional Event-Tangent Spectrum")
    axis.legend(frameon=False, ncol=3, loc="upper center")
    axis.text(
        0.02,
        0.03,
        "Scale-dependent diagnostic; not a controller ranking",
        transform=axis.transAxes,
        fontsize=7.5,
        color="#495057",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
    )


def _plot_frozen_countermodel(
    axis, varying_windows: np.ndarray, frozen_windows: np.ndarray
) -> None:
    centers = np.arange(4)
    varying_trace = np.trace(varying_windows, axis1=1, axis2=2)
    frozen_trace = np.trace(frozen_windows, axis1=1, axis2=2)
    axis.bar(
        centers - 0.18,
        varying_trace,
        0.36,
        color=COLORS["varying"],
        label="Trajectory-Varying",
    )
    axis.bar(
        centers + 0.18,
        frozen_trace,
        0.36,
        color=COLORS["frozen"],
        label="Frozen Local",
    )
    axis.set_ylim(bottom=0.0)
    axis.set_xticks(centers, ("0–25%", "25–50%", "50–75%", "75–100%"))
    axis.set_xlabel("Matched Phase Window")
    axis.set_ylabel("Scaled Energy-Gramian Trace")
    axis.set_title("Frozen-Local Countermodel Divergence")
    axis.legend(frameon=False)


def main() -> None:
    """Render trajectory, channel, event, and countermodel diagnostics."""

    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    with np.load(ARRAY_PATH, allow_pickle=False) as arrays:
        time_s = arrays["time_s"]
        state = arrays["state"]
        full = arrays["full_gramian_history"]
        shoulder = arrays["shoulder_gramian_history"]
        wrist = arrays["wrist_gramian_history"]
        varying_windows = arrays["trajectory_varying_window_gramians"]
        frozen_windows = arrays["frozen_local_window_gramians"]

    event_time_s = float(report["event_conditioned_authority"]["event_time_s"])
    phase = time_s / event_time_s
    guard = state[:, 0] + state[:, 1]
    channel_cases = report["channel_cases"]
    channel_keys = ("full", "shoulder_only", "wrist_only")
    tangent_eigenvalues = np.asarray(
        [channel_cases[key]["event_tangent"]["eigenvalues"] for key in channel_keys]
    )
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 160,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(10.6, 6.5), constrained_layout=True)

    _plot_event(axes[0, 0], phase, guard)
    _plot_authority_histories(axes[0, 1], phase, (full, shoulder, wrist))
    _plot_tangent_spectrum(axes[1, 0], tangent_eigenvalues)
    _plot_frozen_countermodel(axes[1, 1], varying_windows, frozen_windows)

    figure.suptitle(
        "Trajectory Variation and Event Geometry Condition Local Control Authority",
        fontsize=11,
    )
    figure.savefig(FIGURE_PATH, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
