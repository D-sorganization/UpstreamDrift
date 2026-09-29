"""Render matched-state counterfactual ensemble figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["svg.hashsalt"] = "upstreamdrift-counterfactual-ensemble-v2"

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = REPO_ROOT / "docs" / "research" / "proximal_distal_energy_transfer"
DATA_DIR = OUTPUT_ROOT / "data"
FIG_DIR = OUTPUT_ROOT / "figures"


def _load() -> tuple[dict, dict[str, np.ndarray]]:
    with (DATA_DIR / "counterfactual_ensemble.json").open(encoding="utf-8") as stream:
        record = json.load(stream)
    arrays = dict(np.load(DATA_DIR / "counterfactual_selected_traces.npz"))
    return record, arrays


def _save(fig: plt.Figure, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(
        FIG_DIR / f"{stem}.pdf",
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    svg_path = FIG_DIR / f"{stem}.svg"
    fig.savefig(
        svg_path,
        bbox_inches="tight",
        metadata={"Date": None},
    )
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n",
        encoding="utf-8",
    )
    plt.close(fig)


def fig_divergence_maps(record: dict) -> None:
    """Map work and speed divergence over cut time and horizon."""
    rows = [row for row in record["rows"] if row["dt_s"] == 0.001]
    cuts = record["provenance"]["cut_times_s"]
    horizons = record["provenance"]["horizons_s"]
    metrics = (
        ("force_work_difference_j", "Commanded Minus Zero-Torque Force Work [J]"),
        (
            "terminal_clubhead_speed_difference_m_s",
            "Terminal Clubhead-Speed Difference [m/s]",
        ),
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    for ax, (metric, title) in zip(axes, metrics, strict=True):
        grid = np.array(
            [
                [
                    next(
                        row[metric]
                        for row in rows
                        if row["cut_time_s"] == cut and row["horizon_s"] == horizon
                    )
                    for cut in cuts
                ]
                for horizon in horizons
            ]
        )
        limit = float(np.max(np.abs(grid)))
        image = ax.imshow(
            grid,
            origin="lower",
            aspect="auto",
            cmap="coolwarm",
            vmin=-limit,
            vmax=limit,
        )
        ax.set_xticks(range(len(cuts)), [f"{cut:.3f}" for cut in cuts], rotation=45)
        ax.set_yticks(range(len(horizons)), [f"{value:.2f}" for value in horizons])
        ax.set_xlabel("Killswitch Cut Time [s]")
        ax.set_ylabel("Forward Horizon [s]")
        ax.set_title(title)
        fig.colorbar(image, ax=ax, shrink=0.82)
    fig.suptitle("Matched-State Counterfactual Divergence Depends on Cut and Horizon")
    _save(fig, "fig_counterfactual_divergence_maps")


def fig_selected_futures(arrays: dict[str, np.ndarray]) -> None:
    """Compare state, force, and power futures at three cut times."""
    cuts = (0.12, 0.22, 0.30)
    fig, axes = plt.subplots(3, 3, figsize=(11.0, 8.8), sharex="col")
    for col, cut in enumerate(cuts):
        prefix = f"cut_{cut:.2f}"
        t = arrays[f"{prefix}__time"]
        q_cmd = arrays[f"{prefix}__commanded_q"]
        q_zero = arrays[f"{prefix}__zero_q"]
        f_cmd = np.linalg.norm(arrays[f"{prefix}__commanded_force"], axis=1)
        f_zero = np.linalg.norm(arrays[f"{prefix}__zero_force"], axis=1)
        p_cmd = arrays[f"{prefix}__commanded_power"]
        p_zero = arrays[f"{prefix}__zero_power"]
        axes[0, col].plot(t, q_cmd.sum(axis=1), label="Commanded")
        axes[0, col].plot(t, q_zero.sum(axis=1), ls="--", label="Zero Torque")
        axes[1, col].plot(t, f_cmd)
        axes[1, col].plot(t, f_zero, ls="--")
        axes[2, col].plot(t, p_cmd)
        axes[2, col].plot(t, p_zero, ls="--")
        axes[0, col].set_title(f"Cut at {cut:.2f} s")
        axes[2, col].set_xlabel("Source-Trace Time [s]")
        for row in range(3):
            axes[row, col].axvline(cut, color="k", lw=0.8, ls=":")
            axes[row, col].grid(alpha=0.25)
    axes[0, 0].set_ylabel("Club Angle [rad]")
    axes[1, 0].set_ylabel("Wrist-Force Magnitude [N]")
    axes[2, 0].set_ylabel("Wrist-Force Power [W]")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Commanded and Zero-Torque Futures Separate After the Matched State")
    _save(fig, "fig_counterfactual_selected_futures")


def fig_timestep_sensitivity(record: dict) -> None:
    """Show maximum ensemble difference from the 0.5 ms reference."""
    rows = record["rows"]
    timesteps = record["provenance"]["timesteps_s"]
    metrics = (
        ("terminal_q_distance_rad", "Terminal Angle Distance [rad]"),
        ("terminal_v_distance_rad_s", "Terminal Angular-Velocity Distance [rad/s]"),
        ("terminal_force_distance_n", "Terminal Force Distance [N]"),
        ("force_work_difference_j", "Force-Work Difference [J]"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0))
    for ax, (metric, title) in zip(axes.flat, metrics, strict=True):
        errors: list[float] = []
        for timestep in timesteps:
            differences: list[float] = []
            for row in rows:
                if row["dt_s"] != timestep:
                    continue
                reference = next(
                    candidate
                    for candidate in rows
                    if candidate["cut_time_s"] == row["cut_time_s"]
                    and candidate["horizon_s"] == row["horizon_s"]
                    and candidate["dt_s"] == 0.0005
                )
                differences.append(abs(row[metric] - reference[metric]))
            errors.append(max(differences))
        ax.plot(np.asarray(timesteps) * 1000.0, errors, marker="o")
        ax.set_title(title)
        ax.set_xlabel("Integrator Step [ms]")
        ax.set_ylabel("Maximum Absolute Difference")
        ax.grid(alpha=0.25)
    fig.suptitle("Timestep Sensitivity Relative to the 0.5 ms Reference")
    _save(fig, "fig_counterfactual_timestep_sensitivity")


def fig_physics_variants(record: dict) -> None:
    """Compare baseline, gravity-disabled, and damping-disabled futures."""
    rows = record["variant_rows"]
    variants = ("baseline", "gravity_disabled", "damping_disabled")
    labels = ("Baseline", "Gravity Disabled", "Damping Disabled")
    cuts = sorted({row["cut_time_s"] for row in rows})
    x = np.arange(len(cuts))
    width = 0.24
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    metrics = (
        ("force_work_difference_j", "Commanded Minus Zero-Torque Force Work [J]"),
        (
            "terminal_clubhead_speed_difference_m_s",
            "Terminal Clubhead-Speed Difference [m/s]",
        ),
    )
    for ax, (metric, title) in zip(axes, metrics, strict=True):
        for index, (variant, label) in enumerate(zip(variants, labels, strict=True)):
            values = [
                next(
                    row[metric]
                    for row in rows
                    if row["variant"] == variant and row["cut_time_s"] == cut
                )
                for cut in cuts
            ]
            ax.bar(x + (index - 1) * width, values, width, label=label)
        ax.axhline(0.0, color="k", lw=0.8)
        ax.set_xticks(x, [f"{cut:.2f}" for cut in cuts])
        ax.set_xlabel("Killswitch Cut Time [s]")
        ax.set_title(title)
        ax.grid(alpha=0.2, axis="y")
    axes[0].legend(fontsize=8)
    fig.suptitle("Whole-Model Gravity and Damping Variants Remain Phase Dependent")
    _save(fig, "fig_counterfactual_physics_variants")


def main() -> None:
    """Render all matched-state ensemble figures as PDF and SVG."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    record, arrays = _load()
    fig_divergence_maps(record)
    fig_selected_futures(arrays)
    fig_timestep_sensitivity(record)
    fig_physics_variants(record)


if __name__ == "__main__":
    main()
