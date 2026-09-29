"""Render publication figures for coupled uncertainty and control evidence."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

REPO_ROOT = Path(__file__).resolve().parents[3]
ARTICLE_DIR = REPO_ROOT / "docs" / "research" / "proximal_distal_energy_transfer"
DATA_DIR = ARTICLE_DIR / "data"
FIGURE_DIR = ARTICLE_DIR / "figures"

COLORS = {
    "navy": "#17324D",
    "blue": "#2C7FB8",
    "orange": "#D95F0E",
    "green": "#238B45",
    "red": "#B2182B",
    "gray": "#657786",
}

METRIC_LABELS = (
    "Delivery Speed",
    "Face--Path Error",
    "Peak Hand Force",
    "Torque-Squared Proxy",
    "Minimum Force Couple",
    "Peak Shaft Flex",
)


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
            "savefig.bbox": "tight",
        }
    )


def _save(fig: Figure, stem: str) -> tuple[Path, Path]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FIGURE_DIR / f"{stem}.pdf"
    svg = FIGURE_DIR / f"{stem}.svg"
    fig.savefig(pdf)
    fig.savefig(svg)
    text = svg.read_text(encoding="utf-8")
    svg.write_text(
        "\n".join(line.rstrip() for line in text.splitlines()) + "\n",
        encoding="utf-8",
    )
    plt.close(fig)
    return pdf, svg


def _load() -> tuple[dict, dict[str, np.ndarray]]:
    record = json.loads((DATA_DIR / "uncertainty_control_study.json").read_text())
    with np.load(DATA_DIR / record["array_artifact"]) as bundle:
        arrays = {name: bundle[name].copy() for name in bundle.files}
    return record, arrays


def make_uncertainty_figure(
    record: dict, arrays: dict[str, np.ndarray]
) -> tuple[Path, Path]:
    """Show ensemble intervals and global partial-rank correlations."""

    fig, axes = plt.subplots(
        1, 2, figsize=(11.3, 5.4), gridspec_kw={"width_ratios": [0.8, 1.45]}
    )
    intervals = record["uncertainty_intervals"]
    metric_names = record["global_sensitivity"]["metric_names"]
    centers = np.zeros(len(metric_names))
    lower = np.empty(len(metric_names))
    upper = np.empty(len(metric_names))
    for index, name in enumerate(metric_names):
        row = intervals[name]
        scale = max(
            abs(float(row["q05"])),
            abs(float(row["median"])),
            abs(float(row["q95"])),
            1e-8,
        )
        lower[index] = 100.0 * (float(row["median"]) - float(row["q05"])) / scale
        upper[index] = 100.0 * (float(row["q95"]) - float(row["median"])) / scale
    y = np.arange(len(metric_names))
    axes[0].errorbar(
        centers,
        y,
        xerr=np.vstack([lower, upper]),
        fmt="o",
        color=COLORS["navy"],
        ecolor=COLORS["blue"],
        capsize=4,
        lw=2,
    )
    axes[0].axvline(0.0, color="#94A3B8", lw=1)
    axes[0].set_yticks(y, METRIC_LABELS)
    axes[0].invert_yaxis()
    axes[0].set_xlabel(
        "5th--95th Percentile Deviation (% of Maximum Absolute Interval Value)"
    )
    axes[0].set_title("Declared Coupled-Parameter Ensemble")

    prcc = arrays["prcc"]
    image = axes[1].imshow(prcc, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")
    parameter_labels = []
    for name in record["global_sensitivity"]["parameter_names"]:
        label = name.removesuffix("_scale").removesuffix("_s")
        parameter_labels.append(label.replace("_", " ").title())
    axes[1].set_yticks(np.arange(len(parameter_labels)), parameter_labels)
    axes[1].set_xticks(
        np.arange(len(METRIC_LABELS)),
        METRIC_LABELS,
        rotation=35,
        ha="right",
    )
    axes[1].set_title("Global Partial-Rank Correlation Screening")
    colorbar = fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)
    colorbar.set_label("Partial Rank Correlation")
    fig.suptitle(
        "Coupled Uncertainty Changes Magnitude and Strategy-Relevant Outcomes",
        fontweight="bold",
    )
    fig.tight_layout()
    return _save(fig, "fig_uncertainty_intervals_and_prcc")


def make_identifiability_figure(
    record: dict, arrays: dict[str, np.ndarray]
) -> tuple[Path, Path]:
    """Expose structural and practical nonidentifiability."""

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8))
    singular = np.asarray(
        record["identifiability"]["coupled_parameter_screen"][
            "standardized_sensitivity_singular_values"
        ]
    )
    threshold = 0.05 * singular[0]
    axes[0].bar(np.arange(1, singular.size + 1), singular, color=COLORS["blue"])
    axes[0].axhline(
        threshold,
        color=COLORS["red"],
        ls="--",
        label="5% Effective-Rank Threshold",
    )
    axes[0].set_xlabel("Observable-Sensitivity Mode")
    axes[0].set_ylabel("Singular Value")
    axes[0].set_title("Twelve Parameters Cannot Be Identified From Six Summaries")
    axes[0].legend(frameon=False, fontsize=8)

    mapping = arrays["individual_hand_wrench_map"]
    image = axes[1].imshow(mapping, cmap="PuOr", vmin=-0.07, vmax=1.0, aspect="auto")
    axes[1].set_xticks(
        np.arange(4),
        ("Lead $F_x$", "Lead $F_y$", "Trail $F_x$", "Trail $F_y$"),
        rotation=25,
        ha="right",
    )
    axes[1].set_yticks(np.arange(3), ("Net $F_x$", "Net $F_y$", "Net $M_z$"))
    for row in range(mapping.shape[0]):
        for column in range(mapping.shape[1]):
            axes[1].text(
                column,
                row,
                f"{mapping[row, column]:.3g}",
                ha="center",
                va="center",
                color="white" if abs(mapping[row, column]) > 0.4 else "black",
            )
    axes[1].set_title("Net Planar Wrench Leaves One Force Nullspace Mode")
    colorbar = fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)
    colorbar.set_label("Mapping Coefficient")
    fig.suptitle(
        "Identifiability Is Tested Before Individual-Hand or Parameter Fitting",
        fontweight="bold",
    )
    fig.tight_layout()
    return _save(fig, "fig_identifiability_audit")


def make_pareto_figure(record: dict) -> tuple[Path, Path]:
    """Compare training and independently held-out Pareto surfaces."""

    candidates = record["control_comparison"]["candidates"]
    pareto = {
        "training": set(record["control_comparison"]["training_pareto_programs"]),
        "held_out": set(record["control_comparison"]["held_out_pareto_programs"]),
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.2), sharex=True, sharey=True)
    markers = ("o", "s", "^", "D", "P", "X", "v", "*")
    short_names = (
        "Passive Wrist",
        "Early Drive",
        "Late Drive",
        "Restrain + Drive",
        "Higher Impedance",
        "Later Release",
        "Lower Drive",
        "Early Restrain",
    )
    for axis, split, title in zip(
        axes,
        ("training", "held_out"),
        ("Registered Training Ensemble", "Independent Held-Out Ensemble"),
        strict=True,
    ):
        speed = np.asarray([row[split]["delivery_speed_q10_m_s"] for row in candidates])
        load = np.asarray([row[split]["peak_hand_force_q90_n"] for row in candidates])
        face = np.asarray(
            [row[split]["face_path_error_mean_deg"] for row in candidates]
        )
        effort = np.asarray([row[split]["effort_proxy_mean_nms"] for row in candidates])
        sizes = 55.0 + 8.0 * (effort - np.min(effort))
        for index, row in enumerate(candidates):
            scatter = axis.scatter(
                load[index],
                speed[index],
                c=[face[index]],
                s=sizes[index],
                cmap="viridis",
                vmin=0.0,
                vmax=30.0,
                marker=markers[index],
                edgecolors=(COLORS["red"] if row["name"] in pareto[split] else "white"),
                linewidths=1.8,
                label=short_names[index],
            )
        axis.set_title(title)
        axis.set_xlabel("90th-Percentile Peak Individual-Hand Force (N)")
        axis.set_xlim(130.0, 242.0)
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("10th-Percentile Delivery Speed (m/s)")
    colorbar = fig.colorbar(scatter, ax=axes, fraction=0.035, pad=0.03)
    colorbar.set_label("Mean Face--Path Proxy Error (deg)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=8,
    )
    fig.suptitle(
        "Pareto Membership Changes on Held-Out Parameter Cases",
        fontweight="bold",
    )
    fig.subplots_adjust(top=0.84, bottom=0.24, wspace=0.12, right=0.9)
    return _save(fig, "fig_control_pareto_train_holdout")


def make_strategy_figure(record: dict) -> tuple[Path, Path]:
    """Show why no single strategy is best for every declared objective."""

    candidates = record["control_comparison"]["candidates"]
    names = [row["name"].replace("_", " ").title() for row in candidates]
    held = [row["held_out"] for row in candidates]
    panels = (
        ("delivery_speed_q10_m_s", "Lower-Tail Delivery Speed (m/s)", True),
        ("face_path_error_mean_deg", "Mean Face--Path Proxy Error (deg)", False),
        ("peak_hand_force_q90_n", "Upper-Tail Peak Hand Force (N)", False),
        ("effort_proxy_mean_nms", "Mean Torque-Squared Proxy (N²·m²·s)", False),
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.6))
    for axis, (key, title, higher_is_better) in zip(axes.flat, panels, strict=True):
        values = np.asarray([row[key] for row in held])
        colors = [
            (
                COLORS["green"]
                if value == (np.max(values) if higher_is_better else np.min(values))
                else COLORS["blue"]
            )
            for value in values
        ]
        axis.bar(np.arange(len(names)), values, color=colors)
        axis.set_xticks(np.arange(len(names)), names, rotation=35, ha="right")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle(
        "Held-Out Strategy Comparisons Retain Explicit Objective Tradeoffs",
        fontweight="bold",
    )
    fig.tight_layout()
    return _save(fig, "fig_control_strategy_tradeoffs")


def main() -> None:
    _style()
    record, arrays = _load()
    for paths in (
        make_uncertainty_figure(record, arrays),
        make_identifiability_figure(record, arrays),
        make_pareto_figure(record),
        make_strategy_figure(record),
    ):
        for path in paths:
            print(path)


if __name__ == "__main__":
    main()
