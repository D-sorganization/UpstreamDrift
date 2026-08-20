"""Render the registered articulated structural-authority corner evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
ARTICLE = ROOT / "docs/research/proximal_distal_energy_transfer"
RECORD = ARTICLE / "data/articulated_structural_authority_campaign.json"
OUTPUT = ARTICLE / "figures/fig_articulated_structural_authority"


def _corner_label(corner_id: str) -> str:
    labels = {
        "nominal": "Nominal",
        "height_scale-low": "Height Scale Low",
        "height_scale-high": "Height Scale High",
        "body_mass_scale-low": "Body-Mass Scale Low",
        "body_mass_scale-high": "Body-Mass Scale High",
        "joint_limit_scale-low": "Joint-Limit Scale Low",
        "joint_limit_scale-high": "Joint-Limit Scale High",
    }
    return labels.get(corner_id, corner_id.replace("_", " ").replace("-", " ").title())


def _load_evidence(campaign_path: Path) -> list[dict[str, Any]]:
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    if campaign.get("schema_version") != (
        "articulated-structural-authority-campaign/v1"
    ):
        raise ValueError("figure rendering requires the governed campaign record")
    if campaign.get("status") != "complete":
        raise ValueError("figure rendering requires a complete campaign")

    rows: list[dict[str, Any]] = []
    for corner in campaign.get("corners", []):
        artifact = corner.get("record_artifact")
        if not artifact:
            raise ValueError("figure rendering requires every retained corner artifact")
        record = json.loads(
            (campaign_path.parent / artifact).read_text(encoding="utf-8")
        )
        results = record["results"]
        feasible = int(results["selected_feasible_sample_count"])
        total = int(results["selected_total_sample_count"])
        failures = int(corner["failure_count"])
        if total <= 0 or feasible < 0 or feasible > total:
            raise ValueError("corner feasibility counts are invalid")
        if total - feasible != failures:
            raise ValueError(
                "corner failure count does not match its retained artifact"
            )
        metrics = {
            "maximum_closure_error_m": float(results["maximum_closure_error_m"]),
            "minimum_joint_limit_margin_rad": float(
                results["minimum_joint_limit_margin_rad"]
            ),
            "minimum_collision_clearance_m": float(
                results["minimum_collision_clearance_m"]
            ),
        }
        if not all(np.isfinite(value) and value >= 0.0 for value in metrics.values()):
            raise ValueError("corner structural metrics must be finite and nonnegative")
        rows.append(
            {
                "corner_id": str(corner["corner_id"]),
                "status": str(corner["status"]),
                "feasible": feasible,
                "total": total,
                **metrics,
            }
        )
    if not rows:
        raise ValueError("figure rendering requires at least one registered corner")
    return rows


def render_articulated_structural_authority_figure(
    campaign_path: Path = RECORD,
    output_base: Path = OUTPUT,
) -> None:
    """Render retained feasibility, joint margin, and collision clearance."""

    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    rows = _load_evidence(campaign_path)
    labels = [_corner_label(row["corner_id"]) for row in rows]
    positions = np.arange(len(rows))
    feasible_fraction = np.asarray(
        [100.0 * row["feasible"] / row["total"] for row in rows]
    )
    joint_margin = np.asarray([row["minimum_joint_limit_margin_rad"] for row in rows])
    collision_clearance = np.asarray(
        [row["minimum_collision_clearance_m"] for row in rows]
    )
    colors = ["#c44e52" if row["status"] != "feasible" else "#4c78a8" for row in rows]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15.2, 6.6),
        sharey=True,
        constrained_layout=True,
    )
    axes[0].barh(positions, feasible_fraction, color=colors)
    axes[0].set_yticks(positions, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0.0, 106.0)
    axes[0].set_xlabel("Feasible Phase States (%)")
    axes[0].set_title("Retained Feasibility")
    for position, row, value in zip(positions, rows, feasible_fraction, strict=True):
        axes[0].text(
            min(value + 0.8, 101.0),
            position,
            f"{row['feasible']}/{row['total']}",
            va="center",
            fontsize=9,
        )

    axes[1].barh(positions, joint_margin, color=colors)
    boundary_rows = np.flatnonzero(joint_margin <= np.finfo(float).eps)
    if boundary_rows.size:
        axes[1].scatter(
            np.zeros(boundary_rows.size),
            boundary_rows,
            marker="D",
            s=42,
            color=[colors[index] for index in boundary_rows],
            zorder=3,
            clip_on=False,
        )
    axes[1].set_xlabel("Minimum Margin (rad)")
    axes[1].set_title("Joint-Limit Margin")
    axes[2].barh(positions, collision_clearance, color=colors)
    axes[2].set_xlabel("Minimum Clearance (m)")
    axes[2].set_title("Coarse Collision Clearance")
    for axis in axes:
        axis.grid(axis="x", alpha=0.22)
        axis.set_axisbelow(True)

    maximum_closure = max(row["maximum_closure_error_m"] for row in rows)
    fig.suptitle(
        "Structural Authority Across Registered Engineering Corners\n"
        f"Maximum Closure Error {maximum_closure:.2e} m; "
        "Red Retains a Registered Failure",
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
    render_articulated_structural_authority_figure()
    print(f"Saved: {OUTPUT.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
