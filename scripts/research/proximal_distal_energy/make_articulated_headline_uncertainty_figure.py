"""Render registered articulated shaft and ground headline uncertainty."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
ARTICLE = ROOT / "docs/research/proximal_distal_energy_transfer"
RECORD = ARTICLE / "data/articulated_headline_uncertainty.json"
OUTPUT = ARTICLE / "figures/fig_articulated_headline_uncertainty"
LEVEL_STYLE = {
    "low": ("#1f77b4", "o"),
    "high": ("#d62728", "s"),
}


def _label(name: str) -> str:
    return name.replace("_", " ").title()


def _pathway_panel(
    axis: plt.Axes,
    record: dict[str, Any],
    pathway: str,
) -> None:
    axes = record["design"]["axes"]
    names = [item["name"] for item in axes if pathway in item["pathways"]]
    positions = {name: index for index, name in enumerate(reversed(names))}
    rows = [
        row
        for row in record["corners"]
        if row["corner_id"] != "nominal" and row["axis_name"] in positions
    ]
    for row in rows:
        result = row[pathway]
        y = positions[row["axis_name"]]
        color, marker = LEVEL_STYLE[row["level"]]
        change = result.get("matched_cell_count_change_from_nominal")
        if result["status"] == "completed" and change is not None:
            axis.scatter(
                float(change),
                y,
                color=color,
                marker=marker,
                s=48,
                zorder=3,
            )
        elif result["status"] == "failed_retained":
            axis.scatter(0.0, y, color="#7f3c8d", marker="X", s=68, zorder=4)
            axis.annotate(
                "Failed Corner Retained",
                (0.0, y),
                xytext=(6, 5),
                textcoords="offset points",
                fontsize=7,
                color="#7f3c8d",
            )
    axis.axvline(0.0, color="0.2", linewidth=0.9)
    axis.set_yticks(range(len(names)), [_label(name) for name in reversed(names)])
    axis.set_xlabel("Matched-Cell Count Change From Nominal")
    axis.grid(axis="x", alpha=0.22)
    nominal = next(row for row in record["corners"] if row["corner_id"] == "nominal")
    nominal_count = nominal[pathway]["matched_cell_count"]
    axis.set_title(
        f"{pathway.title()} Screen: Nominal {nominal_count}/384 Matched Cells"
    )


def _legend(axis: plt.Axes) -> None:
    for level, (color, marker) in LEVEL_STYLE.items():
        axis.scatter([], [], color=color, marker=marker, s=48, label=level.title())
    axis.scatter(
        [],
        [],
        color="#7f3c8d",
        marker="X",
        s=68,
        label="Failed Corner Retained",
    )
    axis.legend(loc="lower right", fontsize=8, frameon=False)


def render_headline_uncertainty_figure(
    record_path: Path = RECORD,
    output_base: Path = OUTPUT,
) -> None:
    """Render the completed registered campaign to PDF and SVG."""

    record = json.loads(record_path.read_text(encoding="utf-8"))
    if (
        record.get("schema_version") != "articulated-headline-uncertainty/v1"
        or record.get("status") != "complete"
    ):
        raise ValueError("figure rendering requires a complete campaign record")
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.8), constrained_layout=True)
    _pathway_panel(axes[0], record, "shaft")
    _pathway_panel(axes[1], record, "ground")
    _legend(axes[1])
    fig.suptitle(
        "Articulated Headline Sensitivity Across Registered Engineering Bounds\n"
        "One-at-a-Time Screening; Not Participant Calibration or Human Inference",
        fontsize=13,
    )
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    svg_path = output_base.with_suffix(".svg")
    fig.savefig(svg_path, bbox_inches="tight")
    svg_path.write_text(
        "\n".join(
            line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()
        )
        + "\n",
        encoding="utf-8",
    )
    plt.close(fig)


def main() -> None:
    render_headline_uncertainty_figure()
    print(f"Saved: {OUTPUT.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
