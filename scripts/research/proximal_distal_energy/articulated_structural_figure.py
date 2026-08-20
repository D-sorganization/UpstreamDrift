"""Render the preregistered articulated structural sensitivity figure."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from scripts.research.proximal_distal_energy.articulated_structural_figure_data import (  # noqa: E402
    validate_structural_figure_data_record,
)


def _label(corner_id: str, pathway: str) -> str:
    corner = corner_id.replace("_scale", "").replace("_", " ").replace("-", " ").title()
    return f"{corner} - {pathway.title()}"


def _support_panel(axis: Any, rows: list[dict[str, Any]]) -> None:
    positions = np.arange(len(rows))
    labels = [_label(row["corner_id"], row["pathway"]) for row in rows]
    styles = (
        ("planned_cell_count", "|", "Planned"),
        ("feasible_cell_count", "x", "Feasible"),
        ("executed_cell_count", "s", "Executed"),
        ("matched_cell_count", "o", "Matched"),
    )
    for name, marker, label in styles:
        marker_colors = (
            {"color": "#1f2937"}
            if marker in ("|", "x")
            else {"facecolors": "none", "edgecolors": "#1f2937"}
        )
        axis.scatter(
            [row[name] for row in rows],
            positions,
            marker=marker,
            s=55,
            linewidths=1.2,
            label=label,
            **marker_colors,
        )
    axis.set_yticks(positions, labels, fontsize=7)
    axis.invert_yaxis()
    axis.set_xlabel("Cell Count")
    axis.set_title("A. Support Denominators")
    axis.grid(axis="x", alpha=0.25)
    axis.legend(ncols=2, fontsize=7, loc="lower right")


def _transition_panel(axis: Any, rows: list[dict[str, Any]]) -> None:
    rows = [row for row in rows if row["corner_id"] != "nominal"]
    positions = np.arange(len(rows))
    left = np.zeros(len(rows))
    styles = (
        ("persistent_cell_count", "Persistent", "#4c78a8", ""),
        ("entered_cell_count", "Entered", "#f2cf5b", "//"),
        ("exited_cell_count", "Exited", "#e45756", "xx"),
    )
    for name, label, color, hatch in styles:
        values = np.asarray([row[name] for row in rows])
        axis.barh(
            positions,
            values,
            left=left,
            label=label,
            color=color,
            edgecolor="#111827",
            linewidth=0.5,
            hatch=hatch,
        )
        left += values
    axis.set_yticks(
        positions,
        [_label(row["corner_id"], row["pathway"]) for row in rows],
        fontsize=7,
    )
    axis.invert_yaxis()
    axis.set_xlabel("Common Matching-Support Cell Count")
    axis.set_title("B. Persistent, Entered, And Exited Support")
    axis.grid(axis="x", alpha=0.25)
    axis.legend(ncols=3, fontsize=7, loc="lower right")


def _outcome_panel(axis: Any, rows: list[dict[str, Any]]) -> None:
    keys = list(dict.fromkeys((row["corner_id"], row["pathway"]) for row in rows))
    positions = {key: index for index, key in enumerate(keys)}
    if not rows:
        axis.text(
            0.5,
            0.5,
            "No Persistent Paired Outcomes",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
    for resolved, marker, label in (
        (False, "o", "Unresolved At Registered Threshold"),
        (True, "^", "Resolved"),
    ):
        selected = [row for row in rows if row["resolved"] is resolved]
        axis.scatter(
            [positions[(row["corner_id"], row["pathway"])] for row in selected],
            [row["change_m_s"] for row in selected],
            marker=marker,
            s=24,
            facecolors="#2f855a" if resolved else "none",
            edgecolors="#1f2937",
            linewidths=0.8,
            alpha=0.8,
            label=label,
        )
    axis.axhline(0.0, color="#111827", linewidth=0.8)
    axis.set_xticks(
        range(len(keys)),
        [_label(*key) for key in keys],
        rotation=55,
        ha="right",
        fontsize=7,
    )
    axis.set_ylabel("Corner - Nominal Speed Difference (m/s)")
    axis.set_title("C. Persistent Outcome Change And Resolution Status")
    axis.grid(axis="y", alpha=0.25)
    if rows:
        axis.legend(fontsize=7, loc="best")


def _range_error(value: float, value_range: list[float]) -> np.ndarray:
    return np.asarray([[value - value_range[0]], [value_range[1] - value]])


def _secant_panel(axis: Any, rows: list[dict[str, Any]]) -> None:
    for index, row in enumerate(rows):
        if row["shared_persistent_cell_count"] == 0:
            axis.scatter(
                index,
                0.0,
                marker="x",
                color="#6b7280",
                s=45,
                label="Insufficient Shared Support" if index == 0 else None,
            )
            continue
        for offset, value_name, range_name, marker, label in (
            (
                -0.12,
                "low_to_nominal_secant_m_s_per_unit_scale",
                "low_to_nominal_secant_range_m_s_per_unit_scale",
                "o",
                "Low To Nominal",
            ),
            (
                0.12,
                "nominal_to_high_secant_m_s_per_unit_scale",
                "nominal_to_high_secant_range_m_s_per_unit_scale",
                "s",
                "Nominal To High",
            ),
        ):
            value = float(row[value_name])
            axis.errorbar(
                index + offset,
                value,
                yerr=_range_error(value, row[range_name]),
                marker=marker,
                markerfacecolor="none",
                markeredgecolor="#1f2937",
                color="#1f2937",
                capsize=2,
                linestyle="none",
                label=label if index == 0 else None,
            )
    axis.axhline(0.0, color="#111827", linewidth=0.8)
    axis.set_xticks(
        range(len(rows)),
        [
            f"{row['axis_name'].replace('_scale', '').replace('_', ' ').title()} - {row['pathway'].title()}"
            for row in rows
        ],
        rotation=35,
        ha="right",
        fontsize=7,
    )
    axis.set_ylabel("One-Sided Secant (m/s Per Unit Scale)")
    axis.set_title("D. Engineering Secants - Not Parameter Importance")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=7, loc="best")


def _failure_panel(axis: Any, rows: list[dict[str, Any]]) -> None:
    axis.axis("off")
    axis.set_title("E. Retained Failures", loc="left")
    if not rows:
        text = "No Retained State Or Gate Failures In This Result Bundle"
    else:
        counts = Counter(
            (row["corner_id"], row["pathway"], row["failure_class"]) for row in rows
        )
        text = "\n".join(
            f"{_label(corner, pathway)}: {failure} (n={count})"
            for (corner, pathway, failure), count in sorted(counts.items())
        )
    axis.text(0.01, 0.78, text, va="top", fontsize=9, family="monospace")
    axis.text(
        0.01,
        0.18,
        "Retained failures remain evidence; absence of a failure does not establish human validity.",
        va="top",
        fontsize=8,
    )


def render_structural_figure(record: dict[str, Any], output: Path) -> None:
    """Render all registered panels after fail-closed data validation."""

    validate_structural_figure_data_record(record)
    if output.suffix.lower() not in (".svg", ".pdf"):
        raise ValueError("structural figure output must be SVG or PDF")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.stem}.tmp{output.suffix}")
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    ):
        figure = plt.figure(figsize=(14, 16), constrained_layout=True)
        grid = figure.add_gridspec(3, 2, height_ratios=(1.25, 1.0, 0.35))
        _support_panel(figure.add_subplot(grid[0, 0]), record["support"])
        _transition_panel(figure.add_subplot(grid[0, 1]), record["support"])
        _outcome_panel(figure.add_subplot(grid[1, 0]), record["persistent_outcomes"])
        _secant_panel(figure.add_subplot(grid[1, 1]), record["axis_secants"])
        _failure_panel(figure.add_subplot(grid[2, :]), record["retained_failures"])
        figure.suptitle(
            "Articulated Structural Propagation - Synthetic Engineering Sensitivity",
            fontsize=15,
            weight="bold",
            y=0.99,
        )
        figure.text(
            0.5,
            0.962,
            "Nominal Ground Matching: 0/384 | No Human Or Coaching Inference",
            ha="center",
            fontsize=10,
            weight="bold",
        )
        figure.get_layout_engine().set(rect=(0.0, 0.0, 1.0, 0.93))
        description = (
            "Support, persistent outcomes, secants, and retained failures from "
            "synthetic engineering models. "
            f"Figure data SHA-256: {record['figure_data_sha256']}. "
            f"Result SHA-256: {record['result_sha256']}."
        )
        metadata = {"Title": "Articulated Structural Propagation Sensitivity"}
        metadata["Description" if output.suffix.lower() == ".svg" else "Subject"] = (
            description
        )
        figure.savefig(temporary, metadata=metadata)
        plt.close(figure)
    temporary.replace(output)


__all__ = ["render_structural_figure"]
