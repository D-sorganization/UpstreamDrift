"""Build a source-bound narrative from immutable matched-shot matrix results.

Usage: python -m scripts.research.shot_pattern_analysis.build_final_report \
    PRIMARY_ROOT CONTROL_ROOT REPORT.md
The input cells are validated by the public matched-report and scoring APIs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
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


def render_cell_inventory(primary: dict, control: dict) -> str:
    """Export every physical pattern, including explicit unavailable cells."""
    fields = (
        "protocol",
        "cell",
        "status",
        "reason",
        "club",
        "lean_deg",
        "profile",
        "shaft_rotation_sd_deg",
        "curve_scale",
        "seed",
        "target_x_m",
        "pattern",
        "planned_n_per_pattern",
        "attempted_n_per_pattern",
        "supported_n",
        "paired_n",
        "nominal_face_deg",
        "nominal_path_deg",
        "nominal_speed_mps",
        "mean_carry_m",
        "lateral_sd_m",
        "downrange_sd_m",
        "target_rmse_m",
        "target_containment_fraction",
        "realized_face_path_correlation",
        "radial_difference_vs_straight_m",
        "course_sg_difference_vs_straight",
    )
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for protocol, report in (("within_lean", primary), ("across_lean", control)):
        for cell in report["cells"]:
            config = cell["config"]
            for pattern in PATTERNS:
                summary = cell.get("summary", {}).get(pattern, {})
                nominal = cell.get("nominals", {}).get(pattern, {})
                paired = cell.get("paired_comparisons", {})
                radial = paired.get("mean_radial_error_difference_vs_straight", {}).get(
                    pattern, {}
                )
                course = cell.get("course_scoring", {})
                sg = course.get("paired_benefit_vs_straight", {}).get(pattern, {})
                writer.writerow(
                    {
                        "protocol": protocol,
                        "cell": cell["cell"],
                        "status": cell["status"],
                        "reason": cell.get("reason", ""),
                        "club": config["club"]["club_id"],
                        "lean_deg": config["club"]["shaft_lean_deg"],
                        "profile": _profile(cell["cell"]),
                        "shaft_rotation_sd_deg": config["noise"]["standard_deviations"][
                            0
                        ],
                        "curve_scale": config["curve_scale"],
                        "seed": config.get("seed", ""),
                        "target_x_m": cell.get("target_x_m", ""),
                        "pattern": pattern,
                        "planned_n_per_pattern": config["n_shots"],
                        "attempted_n_per_pattern": summary.get("total_n", 0),
                        "supported_n": summary.get("supported_n", ""),
                        "paired_n": paired.get("paired_n", ""),
                        "nominal_face_deg": nominal.get("face_deg", ""),
                        "nominal_path_deg": nominal.get("path_deg", ""),
                        "nominal_speed_mps": nominal.get("speed_mps", ""),
                        "mean_carry_m": summary.get("mean_carry_m", ""),
                        "lateral_sd_m": summary.get("lateral_sd_m", ""),
                        "downrange_sd_m": summary.get("downrange_sd_m", ""),
                        "target_rmse_m": summary.get("target_rmse_m", ""),
                        "target_containment_fraction": summary.get(
                            "target_containment_fraction", ""
                        ),
                        "realized_face_path_correlation": summary.get(
                            "realized_face_path_correlation", ""
                        ),
                        "radial_difference_vs_straight_m": radial.get("estimate", ""),
                        "course_sg_difference_vs_straight": sg.get("estimate", ""),
                    }
                )
    return stream.getvalue()


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
    unavailable_cells = report["counts"]["unavailable"]
    suffix = "cell" if unavailable_cells == 1 else "cells"
    lines = [
        f"### {label} Direction Counts",
        "",
        f"Counts below cover only {report['counts']['completed']} completed cells; {unavailable_cells} unavailable planned {suffix} have no interval and remain in the complete inventory.",
        "",
        "| Metric | Pattern | Positive Interval | Negative Interval | Crosses Zero | Missing Metric in Completed Cells |",
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


def _profile_control_lines(primary: dict) -> list[str]:
    sample = {}
    for cell in primary["cells"]:
        config = cell["config"]
        if (
            config["club"]["club_id"] == "driver"
            and config["club"]["shaft_lean_deg"] == 0
            and config["curve_scale"] == 1
            and config["noise"]["standard_deviations"][0] == 1
        ):
            sample[_profile(cell["cell"])] = config["noise"]
    lines = [
        "### Nine-Profile Input Controls",
        "",
        "All rows also use shaft-rotation SD 1° or 2° according to the cell. Speed SD is shown in mph; contact SD is per axis, not radial. Correlations below describe primitive shaft rotation and path errors, not the derived face/path correlation. The full seven-dimensional covariance and realized output covariance remain in the source-bound cell receipts.",
        "",
        "| Profile | Path SD (°) | Speed SD (mph) | Strike SD (mm Each Axis) | Primitive Shaft/Path ρ |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, noise in sample.items():
        sd = noise["standard_deviations"]
        correlation = noise["correlation"]
        rho = 0.0 if correlation is None else correlation[0][1]
        lines.append(
            f"| {name} | {sd[1]:g} | {sd[2] / 0.44704:.1f} | {sd[3] * 1000:.2f} / {sd[4] * 1000:.2f} | {rho:+.2f} |"
        )
    return lines + [""]


def _experimental_control_lines(primary: dict) -> list[str]:
    reference = primary["cells"][0]["config"]
    clubs = {}
    for cell in primary["cells"]:
        config = cell["config"]
        if config["club"]["shaft_lean_deg"] == 0:
            clubs[config["club"]["club_id"]] = config["club"]
    lines = [
        "## Controlled Inputs",
        "",
        f"Each completed cell uses {reference['n_shots']:,} indexed errors per physical pattern with seed `{reference.get('seed', 'see cell plan')}`. Nominal matching accepts raw X/Y residual at most {reference.get('match_tolerance_m', 'see cell plan')} m. Nominal speed is searched within 75–125% of each listed reference speed; no post-simulation aiming rotation is applied. Flight timestep, contact resolution, runtime backend and exact source hashes are frozen in the per-cell execution receipts.",
        "",
        "| Club | Static Loft (°) | Lie at Zero Lean (°) | Reference Speed (m/s) | Attack Angle (°) | Assumed Head Mass (kg) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ("driver", "seven_iron", "pitching_wedge"):
        if name not in clubs:
            continue
        club = clubs[name]
        lines.append(
            f"| {name} | {club['loft_deg']:g} | {club['lie_deg']:g} | {club['club_speed_mps']:g} | {club['attack_angle_deg']:g} | {club['clubhead_mass_kg']:.3f} |"
        )
    lines.extend(
        (
            "",
            "The static loft/lie citations and every assumed head property are listed in the [Scientific Design](DESIGN.md). The score model treats carry endpoints as final lie on hypothetical flat course geometry.",
            "",
        )
    )
    return lines


def _anchor_lines(primary: dict, *, n_shots: int = 10_000) -> list[str]:
    anchors = [
        cell
        for cell in primary["cells"]
        if cell["config"]["club"]["shaft_lean_deg"] == 0
        and cell["config"]["curve_scale"] == 1
        and cell["config"]["noise"]["standard_deviations"][0] == 2
        and _profile(cell["cell"]) == "combined2"
    ]
    lines = [
        "### Illustrative Full-Noise Flight Outcomes",
        "",
        "The following three club cells use zero lean, curve scale 1, shaft-twist SD 2°, and the combined-2 profile. Each named pattern has its own nominal face, path and speed solved before noise. Mean carry, spread and RMSE are conditional on supported shots; this is one selected stratum, not the entire grid.",
        "",
        f"| Club | Pattern | Mean Carry (m) | Downrange SD (m) | Lateral SD (m) | Target RMSE (m) | Supported / {n_shots:,} |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for cell in anchors:
        for pattern in PATTERNS:
            if cell["status"] != "completed":
                lines.append(
                    f"| {cell['config']['club']['club_id']} | {pattern} | Unavailable | — | — | — | 0 |"
                )
                continue
            item = cell["summary"][pattern]
            lines.append(
                f"| {cell['config']['club']['club_id']} | {pattern} | {item['mean_carry_m']:.2f} | {item['downrange_sd_m']:.2f} | {item['lateral_sd_m']:.2f} | {item['target_rmse_m']:.2f} | {item['supported_n']:,} |"
            )
    lines.extend(
        (
            "",
            "| Club | Pattern | Nominal Face (° Right) | Nominal Path (° Right) | Nominal Speed (m/s) | Matched Target X (m) |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        )
    )
    for cell in anchors:
        if cell["status"] != "completed":
            continue
        for pattern in PATTERNS:
            nominal = cell["nominals"][pattern]
            lines.append(
                f"| {cell['config']['club']['club_id']} | {pattern} | {nominal['face_deg']:+.3f} | {nominal['path_deg']:+.3f} | {nominal['speed_mps']:.3f} | {cell['target_x_m']:.2f} |"
            )
    lines.extend(
        (
            "",
            "### Long-and-Short Directional Check",
            "",
            "A closed-face or left-side miss is not assumed to be long. The selected combined-2/2° cells directly report mean downrange error and all four simulated endpoint quadrants. Signs are relative to the common physical target; fractions are conditional on supported shots.",
            "",
            "| Club | Pattern | Mean X Error (m) | Left-Long | Left-Short | Right-Long | Right-Short |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        )
    )
    for cell in anchors:
        if cell["status"] != "completed":
            continue
        for pattern in PATTERNS:
            stats = cell["summary"][pattern]["statistics"]
            q = stats["quadrant_fractions"]
            lines.append(
                f"| {cell['config']['club']['club_id']} | {pattern} | {stats['mean_downrange_error_m']:+.2f} | {q['left_long']:.3f} | {q['left_short']:.3f} | {q['right_long']:.3f} | {q['right_short']:.3f} |"
            )
    return lines + [""]


def _selected_comparison_lines(primary: dict) -> list[str]:
    cells = [
        cell
        for cell in primary["cells"]
        if cell["config"]["club"]["shaft_lean_deg"] == 0
        and _profile(cell["cell"]) in ("shaft_only", "combined2")
    ]
    lines = [
        "### Zero-Lean Curve and Variability Comparison",
        "",
        "This predeclared subset varies shaft-twist SD (1°/2°), curve strength (1×/2×), and isolated versus combined delivery errors for each club. Differences are matched shot by shot against Straight inside each cell. The common paired count may be smaller than every pattern's individual supported count. SG is per simulated opportunity; multiply by 100 for the same hypothetical effect over 100 identical opportunities, not per round.",
        "",
        "| Club | Profile | Twist SD (°) | Curve | Common Paired n | Draw Δ SG [95%] | Fade Δ SG [95%] |",
        "| --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for cell in cells:
        config = cell["config"]
        descriptor = [
            config["club"]["club_id"],
            _profile(cell["cell"]),
            f"{config['noise']['standard_deviations'][0]:g}",
            f"{config['curve_scale']:g}",
        ]
        if cell["status"] != "completed":
            lines.append(
                "| "
                + " | ".join(
                    [
                        *descriptor,
                        "Unavailable",
                        "Nominal Match Unavailable (See Inventory)",
                        "—",
                    ]
                )
                + " |"
            )
            continue
        paired = cell["paired_comparisons"]
        sg = cell.get("course_scoring", {}).get("paired_benefit_vs_straight", {})
        lines.append(
            "| "
            + " | ".join(
                [
                    *descriptor,
                    f"{paired['paired_n']:,}",
                    _interval(sg.get("Draw")),
                    _interval(sg.get("Fade")),
                ]
            )
            + " |"
        )
    lines.extend(
        (
            "",
            "For the same selected combined-2, 2° shaft variation, curve scale 1 cells, paired radial error gives the geometric endpoint comparison independently of the hypothetical score:",
            "",
            "| Club | Common Paired n | Draw Δ Radial Error (m) [95%] | Fade Δ Radial Error (m) [95%] |",
            "| --- | ---: | --- | --- |",
        )
    )
    for cell in cells:
        config = cell["config"]
        if (
            _profile(cell["cell"]) != "combined2"
            or config["noise"]["standard_deviations"][0] != 2
            or config["curve_scale"] != 1
        ):
            continue
        if cell["status"] != "completed":
            lines.append(
                f"| {config['club']['club_id']} | Unavailable | Nominal Match Unavailable (See Inventory) | — |"
            )
            continue
        paired = cell["paired_comparisons"]
        radial = paired["mean_radial_error_difference_vs_straight"]
        lines.append(
            f"| {config['club']['club_id']} | {paired['paired_n']:,} | {_interval(radial.get('Draw'))} | {_interval(radial.get('Fade'))} |"
        )
    return lines + [""]


def _sensitivity_lines(sensitivity: dict[str, dict]) -> list[str]:
    lines = ["## Course-Dimension Sensitivity", ""]
    if not sensitivity:
        return lines + [
            "Course-dimension sensitivity was not computed from the saved final cells.",
            "",
        ]
    lines.extend(
        (
            "Each case re-scores the same validated, commonly supported endpoints from lean-0, curve-scale-1, shaft-SD-1°, combined-2 cells; no new flights are simulated. Driver dimensions are fairway half-width and hole distance. Iron and wedge dimensions are green radius. Changes in score with course dimensions do not change the flight distribution.",
            "",
            "| Club | Scenario | Paired n | Draw SG − Straight [95%] | Fade SG − Straight [95%] |",
            "| --- | --- | ---: | --- | --- |",
        )
    )
    for cell_name, result in sorted(sensitivity.items()):
        population = result["population"]
        for scenario, scored in sorted(result["scenarios"].items()):
            paired = scored.get("paired_benefit_vs_straight", {})
            lines.append(
                f"| {cell_name.split('_lean', 1)[0]} | {scenario} | {population['common_supported_n']:,} | "
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
        "The complete paired comparison inventory, including every unavailable pair, is in [REPORT_LEAN.csv](REPORT_LEAN.csv). The compact table below shows combined-2, 2° shaft variation, curve scale 1, for each club. Positive differences mean the later lean is higher; paired SG and radial error have different desirable signs.",
        "",
        "| Club | Lean Pair (°) | Common n | Pattern | Δ Radial Error (m) [95%] | Δ SG [95%] |",
        "| --- | --- | ---: | --- | --- | --- |",
    ]
    for pair in pairs:
        if (
            pair["profile"] != "combined2"
            or pair["rotation_sd_deg"] != 2
            or pair["curve_scale"] != 1
        ):
            continue
        descriptor = [
            pair["club_id"],
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


def render_lean_inventory(lean: dict) -> str:
    """Preserve every across-lean pair and physical pattern in a CSV appendix."""
    fields = (
        "status",
        "reason",
        "club",
        "profile",
        "shaft_rotation_sd_deg",
        "curve_scale",
        "lean_before_deg",
        "lean_after_deg",
        "target_x_m",
        "total_per_population",
        "common_supported_n",
        "pattern",
        "radial_error_difference_m",
        "radial_lower_95",
        "radial_upper_95",
        "course_sg_difference",
        "course_sg_lower_95",
        "course_sg_upper_95",
    )
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for pair in lean["pairs"]:
        for pattern in PATTERNS:
            metric = pair.get("patterns", {}).get(pattern, {})
            radial = metric.get("radial_error_difference_m", {})
            sg = metric.get("course_sg_difference", {})
            writer.writerow(
                {
                    "status": pair["status"],
                    "reason": pair.get("reason", ""),
                    "club": pair["club_id"],
                    "profile": pair["profile"],
                    "shaft_rotation_sd_deg": pair["rotation_sd_deg"],
                    "curve_scale": pair["curve_scale"],
                    "lean_before_deg": pair["leans_deg"][0],
                    "lean_after_deg": pair["leans_deg"][1],
                    "target_x_m": pair["target_x_m"],
                    "total_per_population": pair["total_per_physical_population"],
                    "common_supported_n": pair.get("common_supported_n", ""),
                    "pattern": pattern,
                    "radial_error_difference_m": radial.get("estimate", ""),
                    "radial_lower_95": radial.get("lower_95", ""),
                    "radial_upper_95": radial.get("upper_95", ""),
                    "course_sg_difference": sg.get("estimate", ""),
                    "course_sg_lower_95": sg.get("lower_95", ""),
                    "course_sg_upper_95": sg.get("upper_95", ""),
                }
            )
    return stream.getvalue()


def render_final_report(
    primary: dict, control: dict, lean: dict, sensitivity: dict[str, dict]
) -> str:
    """Describe only integrity-checked final cells; never infer unavailable outcomes."""
    primary_total, primary_done, primary_unavailable = _count_matrix(primary, "primary")
    control_total, control_done, control_unavailable = _count_matrix(control, "control")
    cells_digest = hashlib.sha256(
        render_cell_inventory(primary, control).encode()
    ).hexdigest()
    lean_digest = hashlib.sha256(render_lean_inventory(lean).encode()).hexdigest()
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
        "[Phase One](../README.md) included both a fixed-delivered-loft face-error experiment and earlier shaft-rotation geometric controls. Those historical outputs remain separate. The original face-angle SD is not interchangeable with this phase's primitive shaft-twist SD. This phase pitches the complete club, matches raw nominal landings, adds path/speed/strike noise, and uses the independently reviewed two-body incremental contact law. Numeric phase-one and phase-two effects must not be pooled as if they were the same experiment.",
        "",
    ]
    unavailable_example = next(
        (cell for cell in primary["cells"] if cell["status"] == "unavailable"),
        None,
    )
    if unavailable_example is not None:
        lines.extend(
            [
                f"One unavailable planned cell is `{unavailable_example['cell']}`: `{unavailable_example.get('reason', 'unspecified')}`. The complete inventory preserves every reason.",
                "",
            ]
        )
    lines.extend(_experimental_control_lines(primary))
    lines.extend(
        [
            "## Primary Within-Lean Results",
            "",
            "The source-bound [Primary Matrix Plan](final_v1/matrix_plan.json) and per-cell manifests identify every input and output. Every pattern in every cell, including explicit unavailable cases, appears in the [Complete Cell Inventory](REPORT_CELLS.csv). Supported-shot statistics and course scores are conditional whenever a cell has unsupported deliveries. Curves are compared against Straight on common supported shot indices within each cell, rather than pooling different target distances.",
            "",
        ]
    )
    lines.extend(_profile_control_lines(primary))
    lines.extend(_anchor_lines(primary))
    lines.extend(_selected_comparison_lines(primary))
    lines.extend(
        (
            "### Matched Flight and Dispersion Graphics",
            "",
            "The [Full Figure Inventory](final_figures_v1/availability.csv) records every available panel. These direct comparison figures use shared axes and the same within-lean target; the flight shape and dispersion must be read alongside supported counts and longitudinal error.",
            "",
            "![Driver Matched-Pattern Overhead Flights](final_figures_v1/details/driver_lean0_curve1_rotation2_combined2/overhead.png)",
            "",
            "![Driver Matched-Pattern Landing Density](final_figures_v1/details/driver_lean0_curve1_rotation2_combined2/density.png)",
            "",
            "![Driver Draw: Shaft-Rotation Variability Comparison](final_figures_v1/comparisons/driver/combined2/Draw/shaft_rotation_sd.png)",
            "",
            "![7-Iron Draw: Curve-Strength Comparison](final_figures_v1/comparisons/seven_iron/combined2/Draw/curve_scale.png)",
            "",
            "![Pitching Wedge Fade: Shaft-Rotation Variability Comparison](final_figures_v1/comparisons/pitching_wedge/combined2/Fade/shaft_rotation_sd.png)",
            "",
        )
    )
    lines.extend(_effect_summary(primary, "Primary Matrix"))
    lines.extend(
        (
            "",
            "## Across-Lean Control Cell Accounting",
            "",
            "The [Control Matrix Plan](control_final_v1/matrix_plan.json) fixes target distance to the zero-lean reference. Unavailable cells are retained and named in the complete CSV inventory. This does not by itself identify a pure lean effect because the optimizer rematches delivery inputs. Driver lean can shorten or eliminate a physically airborne nominal; all comparisons are conditioned on successful nominal matching under the stated bounds.",
            "",
        )
    )
    lines.extend(_effect_summary(control, "Control Matrix"))
    lines.extend(_lean_lines(lean))
    lines.extend(_sensitivity_lines(sensitivity))
    lines.extend(
        (
            "## Reproducibility and Limitations",
            "",
            f"Primary plan SHA-256: `{primary['plan_sha256']}`. Control plan SHA-256: `{control['plan_sha256']}`. Source and native-binary identities, numerical settings, seeds, and artifact hashes are frozen in each `run_start.json`, `receipt.json`, and `manifest.json`; the cell loader validates them before aggregation. Review the [Independent Validation Record](validation/README.md) and [Final Run Acceptance](FINAL_RUN_ACCEPTANCE.md).",
            "",
            f"Companion inventory SHA-256: `REPORT_CELLS.csv` `{cells_digest}`; `REPORT_LEAN.csv` `{lean_digest}`. These appendices preserve all planned and unavailable rows while the main report selects readable comparisons.",
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


def validate_final_plans(
    primary: dict, control: dict, *, common_targets: dict[str, float]
) -> None:
    """Require the exact accepted primary/control factorial protocols."""
    if (
        primary.get("stage") != "final"
        or primary.get("planned_cells") != 324
        or len(primary.get("cells", [])) != 324
    ):
        raise ValueError("primary final protocol requires 324 cells")
    if (
        control.get("stage") != "final"
        or control.get("planned_cells") != 72
        or len(control.get("cells", [])) != 72
    ):
        raise ValueError("control final protocol requires 72 cells")
    from src.tools.shot_pattern_analysis.matched_report import (
        validate_original_matrix_protocol,
    )

    validate_original_matrix_protocol(primary, protocol="original_primary")
    validate_original_matrix_protocol(
        control,
        protocol="original_control",
        common_targets=common_targets,
    )


def _write_atomic(destination: Path, content: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w", encoding="utf-8", dir=destination.parent, delete=False
    ) as handle:
        staged = Path(handle.name)
        handle.write(content)
    try:
        os.replace(staged, destination)
    finally:
        staged.unlink(missing_ok=True)


def build_final_report(
    primary_root: Path, control_root: Path, destination: Path
) -> Path:
    """Validate saved matrices, then atomically publish editable Markdown."""
    from src.tools.shot_pattern_analysis.matched_lean_control import analyze_across_lean
    from src.tools.shot_pattern_analysis.matched_report import (
        load_matrix_report,
        load_reviewed_common_targets,
    )
    from src.tools.shot_pattern_analysis.matched_sensitivity import (
        analyze_saved_sensitivity,
    )

    primary_root, control_root, destination = map(
        Path, (primary_root, control_root, destination)
    )
    primary = load_matrix_report(primary_root)
    control = load_matrix_report(control_root)
    common_targets, _ = load_reviewed_common_targets(primary_root.parent)
    validate_final_plans(primary, control, common_targets=common_targets)
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
    _write_atomic(
        destination.with_name("REPORT_CELLS.csv"),
        render_cell_inventory(primary, control),
    )
    _write_atomic(destination.with_name("REPORT_LEAN.csv"), render_lean_inventory(lean))
    _write_atomic(destination, rendered)
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
