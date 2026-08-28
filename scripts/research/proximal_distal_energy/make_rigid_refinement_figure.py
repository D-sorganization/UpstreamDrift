"""Render the preregistered rigid-refinement extension evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
ARTICLE = ROOT / "docs/research/proximal_distal_energy_transfer"
DATA = ARTICLE / "data/articulated_rigid_refinement/summary.json"
OUTPUT = ARTICLE / "figures/fig_rigid_refinement_screening_states"
CASE_INDICES = (0, 4, 8, 9, 13, 17)
SAMPLE_INDICES = (0, 6, 12)
VARIANT_COLORS = {"nominal": "#1f4e79", "damping_high": "#d97706"}


def load_summary(path: Path = DATA) -> dict[str, Any]:
    """Load and minimally validate the published refinement summary."""

    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("schema_version") != "articulated-forward-attribution-summary/v1":
        raise ValueError("unsupported rigid-refinement summary schema")
    if summary.get("counts", {}).get("registered") != 216:
        raise ValueError("rigid-refinement summary must retain 216 registered cases")
    return summary


def completed_groups(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the 36 completed MuJoCo state--variant groups."""

    return [
        group
        for group in summary["groups"]
        if group.get("engine") == "mujoco" and group.get("status") == "completed"
    ]


def _ratio_matrix(groups: list[dict[str, Any]], variant: str) -> np.ndarray:
    matrix = np.full((len(CASE_INDICES), len(SAMPLE_INDICES)), np.nan)
    for group in groups:
        if group["variant"] != variant:
            continue
        row = CASE_INDICES.index(group["source_case_index"])
        column = SAMPLE_INDICES.index(group["source_sample_index"])
        matrix[row, column] = max(group["work_refinement_ratios"])
    if np.any(~np.isfinite(matrix)):
        raise ValueError(f"incomplete screening matrix for {variant}")
    return matrix


def _heatmap(
    axis: plt.Axes,
    matrix: np.ndarray,
    title: str,
) -> None:
    image = axis.imshow(matrix, vmin=0.0, vmax=2.0, cmap="RdYlGn_r", aspect="auto")
    axis.set_xticks(range(3), ["0.00", "0.12", "0.24"])
    axis.set_yticks(range(6), [str(index) for index in CASE_INDICES])
    axis.set_xlabel("Source Time (s)")
    axis.set_ylabel("Source Case Index")
    axis.set_title(title)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            axis.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value > 1.25 else "black",
                fontweight="bold" if value > 0.8 else "normal",
            )
            if value > 0.8:
                axis.add_patch(
                    plt.Rectangle(
                        (column - 0.49, row - 0.49),
                        0.98,
                        0.98,
                        fill=False,
                        edgecolor="#8b0000",
                        linewidth=2.0,
                    )
                )
    colorbar = axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Maximum Successive Work-Residual Ratio")


def _failed_curves(axis: plt.Axes, groups: list[dict[str, Any]]) -> None:
    failed = [group for group in groups if not group["passes"]]
    for group in failed:
        label = f"Case {group['source_case_index']}, t={group['source_time_s']:.2f} s"
        axis.semilogy(
            np.asarray(group["time_steps_s"]) * 1_000.0,
            group["work_relative_residuals"],
            marker="o",
            linewidth=1.8,
            label=label,
        )
    axis.invert_xaxis()
    axis.set_xlabel("Time Step (ms; Finer to the Right)")
    axis.set_ylabel("Work Relative Residual")
    axis.set_title("C. Retained Nonmonotonic Nominal Cases")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)


def _ratio_scatter(axis: plt.Axes, groups: list[dict[str, Any]]) -> None:
    states = [(case, sample) for case in CASE_INDICES for sample in SAMPLE_INDICES]
    x = np.arange(len(states), dtype=float)
    for offset, variant in ((-0.12, "nominal"), (0.12, "damping_high")):
        lookup = {
            (group["source_case_index"], group["source_sample_index"]): max(
                group["work_refinement_ratios"]
            )
            for group in groups
            if group["variant"] == variant
        }
        values = np.asarray([lookup[state] for state in states])
        axis.scatter(
            x + offset,
            values,
            s=28,
            color=VARIANT_COLORS[variant],
            label=variant.replace("_", " ").title(),
        )
    axis.axhline(0.8, color="#8b0000", linestyle="--", linewidth=1.3)
    axis.set_xticks(x, [f"{case}/{sample}" for case, sample in states], rotation=60)
    axis.set_ylabel("Maximum Successive Work-Residual Ratio")
    axis.set_xlabel("Source Case / Sample Index")
    axis.set_title("D. Frozen 0.80 Gate Across All Screening States")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=8)


def render_figure(summary: dict[str, Any], output: Path = OUTPUT) -> None:
    """Render deterministic PDF and SVG evidence views."""

    groups = completed_groups(summary)
    if len(groups) != 36:
        raise ValueError("expected 36 completed MuJoCo refinement groups")
    figure, axes = plt.subplots(2, 2, figsize=(12.4, 9.0), constrained_layout=True)
    _heatmap(axes[0, 0], _ratio_matrix(groups, "nominal"), "A. Nominal Variant")
    _heatmap(
        axes[0, 1],
        _ratio_matrix(groups, "damping_high"),
        "B. High-Damping Variant",
    )
    _failed_curves(axes[1, 0], groups)
    _ratio_scatter(axes[1, 1], groups)
    figure.suptitle(
        "Rigid Refinement Across Screening States\n"
        "33 of 36 MuJoCo Groups Pass; Native Cross-Engine Parity Remains Unavailable",
        fontsize=13,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    svg_path = output.with_suffix(".svg")
    figure.savefig(svg_path, bbox_inches="tight")
    svg_path.write_text(
        "\n".join(
            line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()
        )
        + "\n",
        encoding="utf-8",
    )
    plt.close(figure)


def main() -> None:
    """Render the committed rigid-refinement summary."""

    render_figure(load_summary())


if __name__ == "__main__":
    main()
