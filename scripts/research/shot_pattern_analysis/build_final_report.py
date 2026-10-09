"""Build a source-bound narrative from immutable matched-shot matrix results.

Usage: python -m scripts.research.shot_pattern_analysis.build_final_report \
    PRIMARY_ROOT CONTROL_ROOT REPORT.md
The input cells are validated by the public matched-report and scoring APIs.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
from collections import Counter
from pathlib import Path
from tempfile import NamedTemporaryFile


PATTERNS = ("Straight", "Draw", "Fade")


def _count_matrix(report: dict, label: str) -> tuple[int, int, int]:
    if report.get("schema") != "matched-study-report/1":
        raise ValueError(f"{label} is not a validated matched study report")
    counts = report["counts"]
    total = report["planned_cells"]
    if (
        sum(counts.get(key, 0) for key in ("completed", "unavailable", "not_run"))
        != total
        or len(report["cells"]) != total
    ):
        raise ValueError(f"{label} cell accounting is inconsistent")
    if counts.get("not_run", 0):
        raise ValueError(f"{label} matrix is incomplete")
    for cell in report["cells"]:
        config = cell["config"]
        if config["n_shots"] != 10_000:
            raise ValueError("final study must use 10,000 shots per pattern")
        if cell["status"] != "completed":
            continue
        if any(cell["summary"][name]["total_n"] != 10_000 for name in PATTERNS):
            raise ValueError("completed pattern does not contain 10,000 shots")
    return total, counts.get("completed", 0), counts.get("unavailable", 0)


def _interval(data: dict | None, digits: int = 4) -> str:
    if not isinstance(data, dict):
        return "Unavailable"
    try:
        values = (float(data[key]) for key in ("estimate", "lower_95", "upper_95"))
        estimate, low, high = values
    except (KeyError, TypeError, ValueError):
        return "Unavailable"
    return f"{estimate:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def _profile(name: str) -> str:
    # The validated plan is the authority for the cell name; parse its suffix
    # without importing the simulation runner into the pure report formatter.
    match = re.search(r"_rotation\d+_(.+)$", name)
    if match is None:
        raise ValueError(f"cell name lacks a noise profile: {name}")
    return match.group(1)


def _row(cell: dict, protocol: str) -> str:
    config = cell["config"]
    club = config["club"]
    descriptor = [
        protocol,
        club["club_id"],
        f"{club['shaft_lean_deg']:g}",
        f"{config['curve_scale']:g}",
        f"{config['noise']['standard_deviations'][0]:g}",
        _profile(cell["cell"]),
    ]
    if cell["status"] != "completed":
        reason = cell.get("reason", "unspecified")
        return (
            "| "
            + " | ".join([*descriptor, "Unavailable", "—", "—", "—", reason])
            + " |"
        )
    summary = cell["summary"]
    minimum_supported = min(summary[name]["supported_n"] for name in PATTERNS)
    paired = cell.get("paired_comparisons", {})
    course = cell.get("course_scoring", {})
    sg = course.get("paired_benefit_vs_straight", {})
    radial = paired.get("mean_radial_error_difference_vs_straight", {})
    fields = [
        f"{minimum_supported:,}/10,000",
        _interval(radial.get("Draw")),
        _interval(radial.get("Fade")),
        _interval(sg.get("Draw")),
        _interval(sg.get("Fade")),
    ]
    return "| " + " | ".join([*descriptor, *fields]) + " |"


def _direction_counts(cells: list[dict], key: str) -> dict[str, Counter]:
    result: dict[str, Counter[str]] = {name: Counter() for name in ("Draw", "Fade")}
    for cell in cells:
        if cell["status"] != "completed":
            continue
        comparisons = (
            cell.get("course_scoring", {}).get("paired_benefit_vs_straight", {})
            if key == "course_sg"
            else cell.get("paired_comparisons", {}).get(
                "mean_radial_error_difference_vs_straight", {}
            )
        )
        for pattern in result:
            interval = comparisons.get(pattern)
            if not isinstance(interval, dict):
                result[pattern]["unavailable"] += 1
            elif interval["lower_95"] > 0:
                result[pattern]["positive"] += 1
            elif interval["upper_95"] < 0:
                result[pattern]["negative"] += 1
            else:
                result[pattern]["crosses_zero"] += 1
    return result


def _effect_summary(report: dict, label: str) -> list[str]:
    lines = [
        f"### {label} Direction Counts",
        "",
        "| Metric | Pattern | Positive Interval | Negative Interval | Crosses Zero | Unavailable |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for key, metric in (
        ("radial", "Mean Radial Error Difference"),
        ("course_sg", "Paired Strokes Gained Difference"),
    ):
        for pattern, counts in _direction_counts(report["cells"], key).items():
            lines.append(
                f"| {metric} | {pattern} | {counts['positive']} | {counts['negative']} | "
                f"{counts['crosses_zero']} | {counts['unavailable']} |"
            )
    lines.extend(
        (
            "",
            "A direction count describes this finite grid of assumed cells. It is not a pooled effect, an independent replication count, or a multiple-comparison-adjusted discovery.",
            "",
        )
    )
    return lines


def _sensitivity_lines(sensitivity: dict[str, dict]) -> list[str]:
    lines = ["## Course-Dimension Sensitivity", ""]
    if not sensitivity:
        return lines + [
            "Course-dimension sensitivity was not computed from the saved final cells.",
            "",
        ]
    lines.extend(
        (
            "Each case re-scores the same validated, commonly supported endpoints; no new flights are simulated. Driver dimensions are fairway half-width and hole distance. Iron and wedge dimensions are green radius.",
            "",
            "| Cell | Scenario | Paired n | Draw SG − Straight [95%] | Fade SG − Straight [95%] |",
            "| --- | --- | ---: | --- | --- |",
        )
    )
    for cell_name, result in sorted(sensitivity.items()):
        population = result["population"]
        for scenario, scored in sorted(result["scenarios"].items()):
            paired = scored.get("paired_benefit_vs_straight", {})
            lines.append(
                f"| {cell_name} | {scenario} | {population['common_supported_n']:,} | "
                f"{_interval(paired.get('Draw'))} | {_interval(paired.get('Fade'))} |"
            )
    return lines + [""]


def _lean_lines(lean: dict) -> list[str]:
    if lean.get("schema") != "matched-across-lean/1":
        raise ValueError("across-lean result has an unexpected schema")
    pairs = lean["pairs"]
    counts = Counter(pair["status"] for pair in pairs)
    lines = [
        "## Across-Lean Fixed-Target Control",
        "",
        f"The fixed-target control planned {lean['population']['planned_pairs']} lean pairs: {counts['available']} fully supported, {counts['conditional_only']} conditional, and {counts['unavailable']} unavailable. All comparisons use the intersection of supported shot indices across six physical populations: three patterns at each of two lean settings. Lean differences include rematched nominal speed, face, and path; they are not an isolated causal effect of shaft lean.",
        "",
        "| Club | Profile | Twist SD (°) | Curve Scale | Lean Pair (°) | Common n | Pattern | Radial Error Difference (m) [95%] | SG Difference [95%] |",
        "| --- | --- | ---: | ---: | --- | ---: | --- | --- | --- |",
    ]
    for pair in pairs:
        descriptor = [
            pair["club_id"],
            pair["profile"],
            f"{pair['rotation_sd_deg']:g}",
            f"{pair['curve_scale']:g}",
            f"{pair['leans_deg'][0]:g}→{pair['leans_deg'][1]:g}",
        ]
        if pair["status"] == "unavailable":
            lines.append(
                "| "
                + " | ".join(
                    [
                        *descriptor,
                        "—",
                        "Unavailable",
                        pair.get("reason", "unavailable"),
                        "—",
                    ]
                )
                + " |"
            )
            continue
        for pattern in PATTERNS:
            metrics = pair["patterns"][pattern]
            lines.append(
                "| "
                + " | ".join(
                    [
                        *descriptor,
                        f"{pair['common_supported_n']:,}",
                        pattern,
                        _interval(metrics["radial_error_difference_m"]),
                        _interval(metrics.get("course_sg_difference")),
                    ]
                )
                + " |"
            )
    return lines + [""]


def render_final_report(
    primary: dict, control: dict, lean: dict, sensitivity: dict[str, dict]
) -> str:
    """Describe only integrity-checked final cells; never infer unavailable outcomes."""
    primary_total, primary_done, primary_unavailable = _count_matrix(primary, "primary")
    control_total, control_done, control_unavailable = _count_matrix(control, "control")
    lines = [
        "# Matched Shot Patterns: Final Conditional Simulation Report",
        "",
        "## Scope and Status",
        "",
        f"The primary within-lean matrix completed {primary_done:,} / {primary_total:,} planned cells; {primary_unavailable:,} were unavailable. The across-lean fixed-target control completed {control_done:,} / {control_total:,} planned cells; {control_unavailable:,} were unavailable. Each completed cell attempted 10,000 paired Gaussian delivery errors for each of Straight, Draw, and Fade. Unsupported shots remain in the population accounting and are never retried.",
        "",
        "These are model-conditional sensitivities, not measured player outcomes or a universal advantage for a shot shape. Positive paired strokes gained favors the named curve; negative paired radial-error difference favors the named curve. The 95% intervals quantify Monte Carlo sampling only.",
        "",
        "## Design and Physical Interpretation",
        "",
        "The [Scientific Design](DESIGN.md) states the assumptions, equations, sourced static geometry, and numerical gates. Straight, draw, and fade nominals are solved to one physical zero-error landing point within each lean. The control attempts the same fixed target across leans. Neither endpoints nor paths are rotated after simulation. Matching landing X/Y does not equate ball speed, launch, peak height, descent, or spin.",
        "",
        "The model uses illustrative club masses, inertias, CG, face curvature, friction, and energetic restitution, plus a rigid shaft/face orientation. It omits roll, hazards, turf interaction, shaft deflection, compliant contact, and measured human delivery distributions. Landing is treated as final lie for hypothetical course scoring.",
        "",
        "### Relation to Phase One",
        "",
        "[Phase One](../README.md) varied face angle around constant delivered loft and earlier controls. Its historical outputs remain separate. This phase pitches the complete club, matches raw nominal landings, samples shaft rotation and additional delivery errors, and uses the independently reviewed two-body incremental contact law. Numeric phase-one and phase-two effects must not be pooled as if they were the same experiment.",
        "",
        "## Primary Within-Lean Results",
        "",
        "The source-bound [Primary Matrix Plan](final_v1/matrix_plan.json) and per-cell manifests identify every input and output. Supported-shot statistics and course scores are conditional whenever a cell has unsupported deliveries. The table reports curve-minus-straight paired means and sampling-only 95% intervals, one comparison within each cell.",
        "",
    ]
    lines.extend(_effect_summary(primary, "Primary Matrix"))
    lines.extend(
        (
            "### Complete Cell Inventory",
            "",
            "| Protocol | Club | Lean (°) | Curve Scale | Twist SD (°) | Noise Profile | Minimum Supported | Draw Radial Δ (m) [95%] | Fade Radial Δ (m) [95%] | Draw SG Δ [95%] | Fade SG Δ [95%] |",
            "| --- | --- | ---: | ---: | ---: | --- | ---: | --- | --- | --- | --- |",
        )
    )
    lines.extend(_row(cell, "Within Lean") for cell in primary["cells"])
    lines.extend(
        (
            "",
            "## Across-Lean Control Cell Accounting",
            "",
            "The [Control Matrix Plan](control_final_v1/matrix_plan.json) fixes target distance to the zero-lean reference. Infeasible cells are retained and named. This does not by itself identify a pure lean effect because the optimizer rematches delivery inputs.",
            "",
        )
    )
    lines.extend(_effect_summary(control, "Control Matrix"))
    lines.extend(
        (
            "| Protocol | Club | Lean (°) | Curve Scale | Twist SD (°) | Noise Profile | Minimum Supported | Draw Radial Δ (m) [95%] | Fade Radial Δ (m) [95%] | Draw SG Δ [95%] | Fade SG Δ [95%] |",
            "| --- | --- | ---: | ---: | ---: | --- | ---: | --- | --- | --- | --- |",
        )
    )
    lines.extend(_row(cell, "Across Lean") for cell in control["cells"])
    lines.extend(("",))
    lines.extend(_lean_lines(lean))
    lines.extend(_sensitivity_lines(sensitivity))
    lines.extend(
        (
            "## Reproducibility and Limitations",
            "",
            f"Primary plan SHA-256: `{primary['plan_sha256']}`. Control plan SHA-256: `{control['plan_sha256']}`. Source and native-binary identities, numerical settings, seeds, and artifact hashes are frozen in each `run_start.json`, `receipt.json`, and `manifest.json`; the cell loader validates them before aggregation. Review the [Independent Validation Record](validation/README.md) and [Final Run Acceptance](FINAL_RUN_ACCEPTANCE.md).",
            "",
            "An interval excluding zero is not empirical proof of a golfer's best shot shape. Course SG relies on a historical, source-backed but approximate baseline and assumed green/fairway geometry. Unsupported terminal lies prevent full-population SG; conditional estimates use common supported indices. Absolute carry and across-lean target feasibility constrain interpretation, especially at 10° driver lean.",
            "",
            "## Rebuild Commands",
            "",
            "```bash",
            "python -m scripts.research.shot_pattern_analysis.build_final_report docs/research/shot_pattern_analysis/phase2/final_v1 docs/research/shot_pattern_analysis/phase2/control_final_v1 docs/research/shot_pattern_analysis/phase2/REPORT.md",
            "python -m scripts.research.shot_pattern_analysis.build_pdf docs/research/shot_pattern_analysis/phase2/REPORT.md docs/research/shot_pattern_analysis/phase2/REPORT.pdf",
            "```",
            "",
        )
    )
    return "\n".join(lines)


def assert_matrix_inputs_unchanged(root: Path, report: dict) -> None:
    """Reject a report if source-bound input files changed during aggregation."""
    root = Path(root)
    paths = {"matrix_plan.json": report["plan_sha256"]}
    paths.update(
        {
            f"{name}/manifest.json": digest
            for name, digest in report["cell_manifest_sha256"].items()
        }
    )
    for name, digest in paths.items():
        path = root / name
        if (
            not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != digest
        ):
            raise RuntimeError(f"matrix input changed during report assembly: {name}")


def validate_final_plans(primary: dict, control: dict) -> None:
    """Keep a pilot or partial profile subset out of the final-report pathway."""
    if primary.get("stage") != "final_v1" or primary.get("planned_cells") != 324:
        raise ValueError("primary final protocol requires 324 final_v1 cells")
    if control.get("stage") != "control_final_v1" or control.get("planned_cells") != 72:
        raise ValueError("control final protocol requires 72 control_final_v1 cells")


def build_final_report(
    primary_root: Path, control_root: Path, destination: Path
) -> Path:
    """Validate saved matrices, then atomically publish editable Markdown."""
    from src.tools.shot_pattern_analysis.matched_lean_control import analyze_across_lean
    from src.tools.shot_pattern_analysis.matched_report import load_matrix_report
    from src.tools.shot_pattern_analysis.matched_sensitivity import (
        analyze_saved_sensitivity,
    )

    primary_root, control_root, destination = map(
        Path, (primary_root, control_root, destination)
    )
    primary = load_matrix_report(primary_root)
    control = load_matrix_report(control_root)
    validate_final_plans(primary, control)
    lean = analyze_across_lean(control_root)
    # A compact, predeclared set isolates course geometry from the main grid.
    sensitivity: dict[str, dict] = {}
    for club in ("driver", "seven_iron", "pitching_wedge"):
        candidates = [
            cell
            for cell in primary["cells"]
            if cell["status"] == "completed"
            and cell["config"]["club"]["club_id"] == club
            and cell["config"]["club"]["shaft_lean_deg"] == 0
            and cell["config"]["curve_scale"] == 1
            and cell["config"]["noise"]["standard_deviations"][0] == 1
            and _profile(cell["cell"]) == "combined2"
        ]
        if len(candidates) == 1:
            name = candidates[0]["cell"]
            sensitivity[name] = analyze_saved_sensitivity(
                primary_root / name, primary_root / "matrix_plan.json"
            )
    rendered = render_final_report(primary, control, lean, sensitivity)
    # Recheck plan identity after potentially long read-only sensitivity scoring.
    for root, report in ((primary_root, primary), (control_root, control)):
        assert_matrix_inputs_unchanged(root, report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w", encoding="utf-8", dir=destination.parent, delete=False
    ) as handle:
        staged = Path(handle.name)
        handle.write(rendered)
    try:
        os.replace(staged, destination)
    finally:
        staged.unlink(missing_ok=True)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("primary_root", type=Path)
    parser.add_argument("control_root", type=Path)
    parser.add_argument("destination", type=Path)
    arguments = parser.parse_args()
    print(
        build_final_report(
            arguments.primary_root, arguments.control_root, arguments.destination
        )
    )


if __name__ == "__main__":
    main()
