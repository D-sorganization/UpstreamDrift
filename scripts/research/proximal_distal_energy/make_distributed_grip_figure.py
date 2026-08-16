"""Render the distributed-grip horizon and discretization figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
ARTICLE = ROOT / "docs/research/proximal_distal_energy_transfer"
DATA = ARTICLE / "data/articulated_distributed_grip_atlas.npz"
OUTPUT = ARTICLE / "figures/fig_articulated_distributed_grip_atlas"


def _fine_engine_mean(array: np.ndarray) -> np.ndarray:
    """Average engines after selecting the fine time step."""

    return np.mean(array[:, :, :, -1, :, :], axis=3)


def _envelope(
    axis: plt.Axes,
    horizons_ms: np.ndarray,
    values: np.ndarray,
    station_counts: np.ndarray,
    ylabel: str,
) -> None:
    for station, count in enumerate(station_counts):
        samples = values[:, station].reshape(-1, values.shape[-1])
        median = np.median(samples, axis=0)
        low, high = np.min(samples, axis=0), np.max(samples, axis=0)
        line = axis.plot(horizons_ms, median, marker="o", label=f"{count} / hand")[0]
        axis.fill_between(horizons_ms, low, high, alpha=0.16, color=line.get_color())
    axis.set_xlabel("Nested Horizon (ms)")
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.25)


def _speed_difference(
    axis: plt.Axes,
    horizons_ms: np.ndarray,
    speed: np.ndarray,
    station_counts: np.ndarray,
) -> None:
    reference = speed[:, 0]
    for station in range(1, station_counts.size):
        difference = (speed[:, station] - reference).reshape(-1, speed.shape[-1])
        median = np.median(difference, axis=0)
        low, high = np.min(difference, axis=0), np.max(difference, axis=0)
        line = axis.plot(
            horizons_ms,
            median,
            marker="o",
            label=f"{station_counts[station]} minus 1",
        )[0]
        axis.fill_between(horizons_ms, low, high, alpha=0.16, color=line.get_color())
    axis.axhline(0.0, color="black", linewidth=0.9)
    axis.set_xlabel("Nested Horizon (ms)")
    axis.set_ylabel("Final Translation-Speed Difference (m/s)")
    axis.grid(alpha=0.25)


def _numerical_panel(axis: plt.Axes, arrays: dict[str, np.ndarray]) -> None:
    steps_ms = arrays["time_steps_s"] * 1000.0
    residual = arrays["time_refinement_worst_normalized_residual"]
    axis.semilogy(steps_ms, residual, marker="o", label="Energy Residual")
    axis.semilogy(
        steps_ms[-1],
        np.max(arrays["trajectory_relative_error"]),
        marker="s",
        label="Trajectory Parity",
    )
    station_error = np.max(
        arrays["station_refinement_relative_error"], axis=(0, 2, 3, 4)
    )
    for index, value in enumerate(station_error):
        axis.semilogy(
            steps_ms[-1],
            value,
            marker="D",
            label=f"{arrays['station_counts'][index]}→{arrays['station_counts'][index + 1]} Stations",
        )
    axis.invert_xaxis()
    axis.set_xlabel("Time Step (ms; Finer to the Right)")
    axis.set_ylabel("Registered Relative Error")
    axis.grid(alpha=0.25)


def main() -> None:
    with np.load(DATA) as source:
        arrays = {key: np.asarray(source[key]) for key in source.files}
    horizons_ms = arrays["horizons_s"] * 1000.0
    counts = arrays["station_counts"]
    force = _fine_engine_mean(arrays["peak_station_force_n"])
    opening = _fine_engine_mean(arrays["open_fraction"])
    speed = _fine_engine_mean(arrays["final_club_translation_speed_m_s"])
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.5), constrained_layout=True)
    _envelope(axes[0, 0], horizons_ms, force, counts, "Peak Station Force (N)")
    axes[0, 0].set_title("A. Per-Station Force Envelope")
    _envelope(axes[0, 1], horizons_ms, opening, counts, "Fraction With Any Open Fiber")
    axes[0, 1].set_title("B. Active-Set Classification")
    _speed_difference(axes[1, 0], horizons_ms, speed, counts)
    axes[1, 0].set_title("C. Discretization Difference From One Fiber")
    _numerical_panel(axes[1, 1], arrays)
    axes[1, 1].set_title("D. Refinement and Native-Engine Parity")
    for axis in axes.flat:
        axis.legend(fontsize=7, loc="best")
    fig.suptitle(
        "Distributed Grip Discretization Across Nested Articulated Horizons\n"
        "Equal Total Stiffness; No Shaft, Ground, or Human Strategy Inference",
        fontsize=13,
    )
    fig.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight")
    svg_path = OUTPUT.with_suffix(".svg")
    fig.savefig(svg_path, bbox_inches="tight")
    svg_path.write_text(
        "\n".join(
            line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()
        )
        + "\n",
        encoding="utf-8",
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
