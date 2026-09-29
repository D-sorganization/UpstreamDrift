"""Render figures for the transmission-pathway and robustness study."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

REPO_ROOT = Path(__file__).resolve().parents[3]
ARTICLE_DIR = REPO_ROOT / "docs/research/proximal_distal_energy_transfer"
DATA_DIR = ARTICLE_DIR / "data"
FIGURE_DIR = ARTICLE_DIR / "figures"

COLORS = {
    "navy": "#17324D",
    "blue": "#2C7FB8",
    "orange": "#D95F0E",
    "green": "#238B45",
    "red": "#B2182B",
    "violet": "#756BB1",
    "gray": "#657786",
}
PROGRAM_LABELS = {
    "clock_restrain_then_drive": "Clock Trigger",
    "state_triggered_handoff": "State Trigger",
    "state_triggered_higher_impedance": "State Trigger + Impedance",
    "early_drive": "Early Drive",
}


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
    svg.write_text(
        "\n".join(
            line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()
        )
        + "\n",
        encoding="utf-8",
    )
    plt.close(fig)
    return pdf, svg


def _load() -> tuple[dict, dict[str, np.ndarray]]:
    record = json.loads((DATA_DIR / "transmission_robustness_study.json").read_text())
    with np.load(DATA_DIR / record["array_artifact"]) as bundle:
        arrays = {name: bundle[name].copy() for name in bundle.files}
    return record, arrays


def make_pathway_framework() -> tuple[Path, Path]:
    """Draw the declared transmission ledger and non-equivalence boundaries."""

    fig, axis = plt.subplots(figsize=(11.4, 5.8))
    axis.set_xlim(0.0, 12.0)
    axis.set_ylim(0.0, 6.2)
    axis.axis("off")
    nodes = (
        (0.8, 4.3, 2.0, 1.1, "Ground / Base\nWrench", COLORS["gray"]),
        (3.3, 4.3, 2.1, 1.1, "Proximal\nActuation", COLORS["blue"]),
        (6.0, 4.3, 2.1, 1.1, "Constraint-Force\nTransport", COLORS["green"]),
        (8.7, 4.3, 2.1, 1.1, "Club + Shaft\nEnergy", COLORS["orange"]),
        (5.0, 1.3, 2.1, 1.1, "Direct Distal\nMoment", COLORS["violet"]),
        (8.7, 1.3, 2.1, 1.1, "Impact State\nSpeed + Face + Path", COLORS["navy"]),
    )
    for x, y, width, height, label, color in nodes:
        patch = plt.Rectangle(
            (x, y), width, height, facecolor=color, alpha=0.12, edgecolor=color, lw=2
        )
        axis.add_patch(patch)
        axis.text(
            x + width / 2,
            y + height / 2,
            label,
            ha="center",
            va="center",
            fontweight="bold",
            color=color,
        )
    arrows = (
        ((2.8, 4.85), (3.3, 4.85), "base power"),
        ((5.4, 4.85), (6.0, 4.85), "joint power"),
        ((8.1, 4.85), (8.7, 4.85), "wrench--twist power"),
        ((7.1, 4.3), (6.1, 2.4), "allocation / couple"),
        ((7.1, 1.85), (8.7, 1.85), "moment power"),
        ((9.75, 4.3), (9.75, 2.4), "storage / release"),
    )
    for start, stop, label in arrows:
        axis.annotate(
            "",
            xy=stop,
            xytext=start,
            arrowprops={"arrowstyle": "->", "lw": 1.8, "color": COLORS["navy"]},
        )
        midpoint = ((start[0] + stop[0]) / 2, (start[1] + stop[1]) / 2)
        axis.text(midpoint[0], midpoint[1] + 0.18, label, ha="center", fontsize=8)
    axis.text(
        0.8,
        0.35,
        "Not equivalent: peak sequence ≠ pathway identity; torque sign ≠ power sign;\n"
        "pointwise drift ≠ forward future; nominal speed ≠ robustness; model robustness ≠ human stability.",
        color=COLORS["red"],
        fontsize=10,
        fontweight="bold",
    )
    axis.set_title(
        "A Closed Transmission Ledger With Explicit Inferential Boundaries",
        fontsize=13,
        fontweight="bold",
    )
    return _save(fig, "fig_transmission_pathway_framework")


def make_speed_variability(record: dict) -> tuple[Path, Path]:
    """Show held-out lower-tail speed, dispersion, and peak-load tradeoffs."""

    summaries = record["program_summaries"]
    names = list(record["programs"])
    q10 = np.asarray(
        [summaries[name]["held_out"]["delivery_speed_m_s"]["q10"] for name in names]
    )
    dispersion = np.asarray(
        [summaries[name]["held_out"]["delivery_speed_m_s"]["std"] for name in names]
    )
    face = np.asarray(
        [summaries[name]["held_out"]["face_path_error_deg"]["mean"] for name in names]
    )
    load = np.asarray(
        [summaries[name]["held_out"]["peak_hand_force_n"]["q90"] for name in names]
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.2))
    colors = (COLORS["blue"], COLORS["green"], COLORS["violet"], COLORS["orange"])
    for index, name in enumerate(names):
        axes[0].scatter(
            dispersion[index],
            q10[index],
            s=70 + 0.6 * load[index],
            color=colors[index],
            edgecolor="white",
            linewidth=1.2,
        )
        axes[0].annotate(
            PROGRAM_LABELS[name],
            (dispersion[index], q10[index]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
        axes[1].scatter(
            load[index],
            face[index],
            s=100,
            color=colors[index],
            label=PROGRAM_LABELS[name],
        )
    axes[0].set_xlabel("Held-Out Delivery-Speed Standard Deviation (m/s)")
    axes[0].set_ylabel("Held-Out 10th-Percentile Delivery Speed (m/s)")
    axes[0].set_title("Speed and Repeatability Are Separate Objectives")
    axes[0].grid(alpha=0.25)
    axes[1].set_xlabel("Held-Out 90th-Percentile Peak Hand Force (N)")
    axes[1].set_ylabel("Mean Face--Path Proxy Error (deg)")
    axes[1].set_title("Accuracy Proxy Trades Against Loading")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Every Registered Strategy Remains Pareto-Nondominated", fontweight="bold"
    )
    fig.tight_layout()
    return _save(fig, "fig_robust_speed_variability_pareto")


def make_paired_amplification(record: dict) -> tuple[Path, Path]:
    """Compare clock and state triggering on paired held-out perturbations."""

    paired = record["clock_vs_state_paired_held_out"]
    metrics = (
        ("delivery_speed_m_s", "Delivery Speed"),
        ("face_path_error_deg", "Face--Path Error"),
        ("peak_hand_force_n", "Peak Hand Force"),
        ("event_time_s", "Handoff Time"),
    )
    baseline = np.asarray([paired[key]["baseline_amplification"] for key, _ in metrics])
    candidate = np.asarray(
        [paired[key]["candidate_amplification"] for key, _ in metrics]
    )
    ratios = np.divide(
        candidate, baseline, out=np.ones_like(candidate), where=baseline > 1e-12
    )
    x = np.arange(len(metrics))
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.9))
    axes[0].bar(
        x,
        ratios,
        color=[COLORS["green"] if value < 1.0 else COLORS["red"] for value in ratios],
    )
    axes[0].axhline(1.0, color=COLORS["gray"], ls="--", label="Equal Amplification")
    axes[0].set_xticks(x, [label for _, label in metrics], rotation=20, ha="right")
    axes[0].set_ylabel("State / Clock Amplification Ratio")
    axes[0].set_title("Paired Ensemble Amplification Ratio")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.25)

    changes = np.asarray(
        [
            100.0
            * paired[key]["paired_mean_delta"]
            / max(abs(paired[key]["baseline_mean"]), 1e-12)
            for key, _ in metrics[:3]
        ]
    )
    change_labels = ("Speed", "Face--Path Error", "Peak Force")
    change_colors = (COLORS["green"], COLORS["blue"], COLORS["red"])
    axes[1].bar(np.arange(3), changes, color=change_colors)
    axes[1].axhline(0.0, color="black", lw=0.8)
    axes[1].set_xticks(np.arange(3), change_labels, rotation=20, ha="right")
    axes[1].set_ylabel("Mean Change From Clock Trigger (%)")
    axes[1].set_title("Mean Benefit Comes With a Force Cost")
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle(
        "State Triggering Attenuates Some Errors, Not Every Cost", fontweight="bold"
    )
    fig.tight_layout()
    return _save(fig, "fig_clock_vs_state_perturbation_response")


def make_task_null_map(
    record: dict, arrays: dict[str, np.ndarray]
) -> tuple[Path, Path]:
    """Visualize the local input--outcome map and task-null variance result."""

    jacobian = arrays["local_outcome_jacobian"]
    scale = np.maximum(np.max(np.abs(jacobian), axis=1, keepdims=True), 1e-12)
    normalized = jacobian / scale
    input_labels = [
        name.replace("_", " ").replace(" rad s", " rad/s").title()
        for name in record["perturbation_names"]
    ]
    output_labels = ("Delivery Speed", "Face--Path Error", "Peak Hand Force")
    fig, axes = plt.subplots(
        1, 2, figsize=(11.4, 4.9), gridspec_kw={"width_ratios": [1.55, 0.65]}
    )
    image = axes[0].imshow(
        normalized, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto"
    )
    axes[0].set_xticks(
        np.arange(len(input_labels)), input_labels, rotation=35, ha="right"
    )
    axes[0].set_yticks(np.arange(3), output_labels)
    axes[0].set_title("Row-Normalized Local Outcome Jacobian")
    colorbar = fig.colorbar(image, ax=axes[0], fraction=0.04, pad=0.03)
    colorbar.set_label("Signed Local Sensitivity")
    local = record["local_task_map"]
    values = [local["null_variance"], local["task_relevant_variance"]]
    axes[1].bar(
        ("Task-Null", "Task-Relevant"), values, color=(COLORS["green"], COLORS["red"])
    )
    axes[1].set_ylabel("Projected Perturbation Variance")
    axes[1].set_title(
        f"Local Nullity = {local['nullity']}\nIndex = {local['synergy_index']:.3f}"
    )
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle(
        "Large Elemental Variability Can Be Compatible With Stable Task Outcomes",
        fontweight="bold",
    )
    fig.tight_layout()
    return _save(fig, "fig_task_null_variability_map")


def main() -> None:
    """Render all transmission-robustness figures."""

    _style()
    record, arrays = _load()
    outputs = (
        make_pathway_framework(),
        make_speed_variability(record),
        make_paired_amplification(record),
        make_task_null_map(record, arrays),
    )
    for pair in outputs:
        print(*pair)


if __name__ == "__main__":
    main()
