"""Render the typed articulated slack/contact qualification figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
ARTICLE = ROOT / "docs/research/proximal_distal_energy_transfer"
DATA = ARTICLE / "data/articulated_slack_atlas.npz"
OUTPUT = ARTICLE / "figures/fig_articulated_slack_atlas"


def _law_curves(axis: plt.Axes) -> None:
    distance_mm = np.linspace(0.0, 3.0, 301)
    stiffness = 1800.0
    axis.plot(distance_mm, stiffness * distance_mm / 1000.0, label="Tension-Only")
    for slack_mm in (0.5, 1.5):
        force = stiffness * np.maximum(0.0, distance_mm - slack_mm) / 1000.0
        axis.plot(distance_mm, force, label=f"Dead Zone {slack_mm:g} mm")
    axis.axvline(1.0, color="black", linestyle="--", linewidth=1, label="Common 1 mm")
    axis.set_title("A. Declared Radial Force Laws")
    axis.set_xlabel("Hand–Grip Separation (mm)")
    axis.set_ylabel("Elastic Force Magnitude (N)")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.25)


def _natural_open_fraction(axis: plt.Axes, arrays: dict[str, np.ndarray]) -> None:
    names = arrays["condition_names"].astype(str)
    natural = np.asarray(["event_probe" not in name for name in names])
    law = arrays["law_kinds"].astype(str)
    slack_mm = 1000.0 * arrays["slack_distance_m"]
    preload = arrays["preload_modes"].astype(str)
    keys = []
    for index in np.flatnonzero(natural):
        key = (law[index], float(slack_mm[index]), preload[index])
        if key not in keys:
            keys.append(key)
    values, labels = [], []
    for law_name, slack, preload_name in keys:
        selected = (
            natural
            & (law == law_name)
            & (slack_mm == slack)
            & (preload == preload_name)
        )
        values.append(float(np.mean(arrays["open_fraction"][:, selected, -1, :])))
        short_law = law_name.replace("_", " ").title()
        if slack > 0.0:
            short_law = f"Dead Zone {slack:g} mm"
        short_preload = preload_name.replace("_", " ").title()
        labels.append(f"{short_law}\n{short_preload}")
    axis.bar(np.arange(len(values)), values, color="#4472c4")
    axis.set_title("B. Natural-State Open Fraction at Finest Step")
    axis.set_ylabel("Fraction With Fewer Than Two Active Interfaces")
    axis.set_xticks(np.arange(len(values)), labels, rotation=45, ha="right", fontsize=7)
    axis.set_ylim(0.0, 1.05)
    axis.grid(axis="y", alpha=0.25)


def _event_counts(axis: plt.Axes, arrays: dict[str, np.ndarray]) -> None:
    names = arrays["condition_names"].astype(str)
    probes = np.asarray(["event_probe" in name for name in names])
    openings = np.count_nonzero(arrays["opening_observed"][:, probes], axis=(0, 2, 3))
    reattachments = np.count_nonzero(
        arrays["reattachment_observed"][:, probes], axis=(0, 2, 3)
    )
    x = np.arange(np.count_nonzero(probes))
    axis.bar(x - 0.18, openings, width=0.36, label="Opening")
    axis.bar(x + 0.18, reattachments, width=0.36, label="Reattachment")
    axis.set_title("C. Boundary-Crossing Event Probes")
    axis.set_ylabel("Observed Engine–Step–State Cells")
    axis.set_xticks(x, ["Open to Taut", "Taut to Open"])
    axis.legend()
    axis.grid(axis="y", alpha=0.25)


def _numerical_controls(axis: plt.Axes, arrays: dict[str, np.ndarray]) -> None:
    step_ms = 1000.0 * arrays["time_steps_s"]
    energy = arrays["refinement_worst_normalized_residual"]
    trajectory = np.max(arrays["trajectory_relative_error"], axis=(0, 1))
    axis.semilogy(step_ms, energy, "o-", label="Normalized Energy Residual")
    axis.semilogy(step_ms, trajectory, "s-", label="Trajectory Parity Error")
    axis.invert_xaxis()
    axis.set_title("D. Refinement and Native-Engine Parity")
    axis.set_xlabel("Time Step (ms; Finer to the Right)")
    axis.set_ylabel("Worst Registered Residual")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.25, which="both")


def main() -> int:
    """Write stable PDF and SVG views of the registered slack atlas."""

    with np.load(DATA) as source:
        arrays = {key: np.asarray(source[key]) for key in source.files}
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 9.5), constrained_layout=True)
    figure.suptitle(
        "Typed Unilateral Slack in Articulated Contact\n"
        "Five-Millisecond Synthetic Falsification Screen; No Human Strategy Inference",
        fontsize=14,
    )
    _law_curves(axes[0, 0])
    _natural_open_fraction(axes[0, 1], arrays)
    _event_counts(axes[1, 0], arrays)
    _numerical_controls(axes[1, 1], arrays)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(OUTPUT.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)
    svg = OUTPUT.with_suffix(".svg")
    svg.write_text(
        "\n".join(
            line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
