"""Render a visibly pilot-only layout preview from immutable 32-shot matrices.

Usage: python -m scripts.research.shot_pattern_analysis.build_pilot_preview \
    PRIMARY_PILOT CONTROL_PILOT PREVIEW.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from . import build_final_report as final


def validate_pilot_matrices(primary: dict, control: dict) -> None:
    """Reject final, partial, or non-32-shot inputs before layout publication."""
    for label, report in (("primary", primary), ("control", control)):
        if (
            report.get("schema") != "matched-study-report/1"
            or report.get("stage") != "pilot"
        ):
            raise ValueError(f"{label} must be a validated pilot matrix")
        counts = report.get("counts", {})
        if (
            counts.get("not_run", 0)
            or sum(
                counts.get(status, 0)
                for status in ("completed", "unavailable", "not_run")
            )
            != report.get("planned_cells")
            or len(report.get("cells", [])) != report.get("planned_cells")
        ):
            raise ValueError(f"{label} pilot matrix is incomplete")
        for cell in report["cells"]:
            if cell["config"]["n_shots"] != 32:
                raise ValueError("pilot preview requires exactly 32 shots per pattern")
            if cell["status"] == "completed" and any(
                cell["summary"][name]["total_n"] != 32 for name in final.PATTERNS
            ):
                raise ValueError("completed pilot pattern must contain 32 attempts")


def render_pilot_preview(primary: dict, control: dict, lean: dict | None = None) -> str:
    """Reuse the final report's compact tables with unmistakable pilot scope."""
    validate_pilot_matrices(primary, control)
    primary_counts, control_counts = primary["counts"], control["counts"]
    lines = [
        "# PILOT-ONLY: Matched Shot Report Layout Preview",
        "",
        "## PILOT-ONLY Scope and Status",
        "",
        "**PILOT ONLY — 32 indexed errors attempted per physical pattern per completed cell. Layout review only. These results are not the final study and must not be used to infer a shot-shape advantage.**",
        "",
        f"The primary pilot completed {primary_counts['completed']} / {primary['planned_cells']} planned cells; {primary_counts['unavailable']} were unavailable under the stated model/search bounds. The fixed-target control completed {control_counts['completed']} / {control['planned_cells']} planned cells; {control_counts['unavailable']} were unavailable under the stated model/search bounds. Unsupported shots remain in attempted counts; paired comparisons use common supported indices.",
        "",
        "The layout and table formatters below are shared with the final report, but all statistics are from the saved 32-shot pilot. Monte Carlo intervals at this size are unstable. The model is an illustrative geometric simulation rather than an empirical player study.",
        "",
        "## PILOT-ONLY Controlled Inputs and Selected Results",
        "",
    ]
    unavailable = next(
        (cell for cell in primary["cells"] if cell["status"] == "unavailable"),
        None,
    )
    if unavailable is not None:
        lines.extend(
            [
                f"Illustrative unavailable pilot cell: `{unavailable['cell']}` (`{unavailable.get('reason', 'unspecified')}`). No shot outcomes are assigned to it.",
                "",
            ]
        )
    lines.extend(final._profile_control_lines(primary))
    lines.extend(final._anchor_lines(primary, n_shots=32))
    lines.extend(final._selected_comparison_lines(primary))
    lines.extend(final._effect_summary(primary, "Primary Pilot Matrix"))
    lines.extend(
        [
            "## PILOT-ONLY Graphic Layout",
            "",
            "The following figures were rendered from the same saved 32-shot pilot and are included to review page composition and labels only.",
            "",
            "![PILOT ONLY: Driver Overhead Flights](../presentation_preview_v8/details/driver_lean0_curve1_rotation1_shaft_only/overhead.png)",
            "",
            "[Full-Resolution Overhead PNG](../presentation_preview_v8/details/driver_lean0_curve1_rotation1_shaft_only/overhead.png)",
            "",
            "![PILOT ONLY: Driver Landing Density](../presentation_preview_v8/details/driver_lean0_curve1_rotation1_shaft_only/density.png)",
            "",
            "[Full-Resolution Density PNG](../presentation_preview_v8/details/driver_lean0_curve1_rotation1_shaft_only/density.png)",
            "",
            "![PILOT ONLY: Draw Shaft-Twist Comparison](../presentation_preview_v8/comparisons/driver/shaft_only/Draw/shaft_rotation_sd.png)",
            "",
            "[Full-Resolution Comparison PNG](../presentation_preview_v8/comparisons/driver/shaft_only/Draw/shaft_rotation_sd.png)",
            "",
            "## PILOT-ONLY Evidence Boundary",
            "",
            f"Primary pilot plan SHA-256: `{primary['plan_sha256']}`. Control pilot plan SHA-256: `{control['plan_sha256']}`. Original per-cell manifests and execution source snapshots remain in the saved pilot matrices. The [Cell Inventory](REPORT_CELLS.csv) and [Lean Comparison Inventory](REPORT_LEAN.csv) preserve unavailable cases. This document is a report-layout preview, not a final scientific result.",
            "",
            "## PILOT-ONLY Across-Lean Control",
            "",
        ]
    )
    lines.extend(final._effect_summary(control, "Fixed-Target Pilot Matrix"))
    if lean is not None:
        lines.extend(final._lean_lines(lean))
    return "\n".join(lines)


def build_pilot_preview(
    primary_root: Path, control_root: Path, destination: Path
) -> Path:
    """Validate immutable pilot inputs and publish a separate editable preview."""
    from src.tools.shot_pattern_analysis.matched_lean_control import analyze_across_lean
    from src.tools.shot_pattern_analysis.matched_report import load_matrix_report

    primary_root, control_root, destination = map(
        Path, (primary_root, control_root, destination)
    )
    primary = load_matrix_report(primary_root)
    control = load_matrix_report(control_root)
    validate_pilot_matrices(primary, control)
    lean = analyze_across_lean(control_root)
    rendered = render_pilot_preview(primary, control, lean)
    for root, report in ((primary_root, primary), (control_root, control)):
        final.assert_matrix_inputs_unchanged(root, report)
    final._write_atomic(
        destination.with_name("REPORT_CELLS.csv"),
        final.render_cell_inventory(primary, control),
    )
    final._write_atomic(
        destination.with_name("REPORT_LEAN.csv"), final.render_lean_inventory(lean)
    )
    final._write_atomic(destination, rendered)
    receipt = {
        "schema": "matched-pilot-layout-preview/1",
        "pilot_only": True,
        "n_shots_per_pattern": 32,
        "primary_plan_sha256": primary["plan_sha256"],
        "control_plan_sha256": control["plan_sha256"],
        "markdown_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "cell_inventory_sha256": hashlib.sha256(
            destination.with_name("REPORT_CELLS.csv").read_bytes()
        ).hexdigest(),
        "lean_inventory_sha256": hashlib.sha256(
            destination.with_name("REPORT_LEAN.csv").read_bytes()
        ).hexdigest(),
    }
    final._write_atomic(
        destination.with_name("preview_receipt.json"),
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
    )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("primary_root", type=Path)
    parser.add_argument("control_root", type=Path)
    parser.add_argument("destination", type=Path)
    arguments = parser.parse_args()
    print(
        build_pilot_preview(
            arguments.primary_root, arguments.control_root, arguments.destination
        )
    )


if __name__ == "__main__":
    main()
