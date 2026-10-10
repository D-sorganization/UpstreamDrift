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
import json
import os
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any


PATTERNS = ("Straight", "Draw", "Fade")

from src.tools.shot_pattern_analysis.matched_formatting import (
    format_sg_interval_record,
)
from .final_report_inputs import (
    assemble_qualified_interpretation,
    profile_from_cell as _profile,
    verified_original_figure_links,
    verified_original_figure_inventory,
)


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


def _interval(data: Mapping[str, Any] | None) -> str:
    return format_sg_interval_record(data)


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
        f"Each completed cell uses {reference['n_shots']:,} indexed errors per physical pattern with seed `{reference.get('seed', 'see cell plan')}`. Nominal matching accepts raw X/Y residual at most {reference.get('match_tolerance_m', 'see cell plan')} m. Under original protocols (protocol original_primary/1 and original_control/1), nominal speed is searched within 75–125% of each listed reference speed; no post-simulation aiming rotation is applied. Flight timestep, contact resolution, runtime backend and exact source hashes are frozen in the per-cell execution receipts.",
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


def _resolved_selected_comparison_lines(resolved_cells: list[dict]) -> list[str]:
    from .report_tables import resolved_selection_lines

    return resolved_selection_lines(resolved_cells)


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


def _bounds_control_lines(bounds_control: Mapping[str, Any] | None) -> list[str]:
    from .report_tables import keyed_effect_table as render_effect_table

    lines = [
        "### Fixed-Target Across-Lean Bounds Controls",
        "",
        "The fixed-target across-lean controls evaluate whether adjusting shaft lean from 0° to 5° alters relative curve scoring when nominal target distance is held constant. These comparisons evaluate solver bounds adapted after original search; original unavailable cells remain unchanged in the complete matrix accounting. Rematched speed/face/path prevents causal claims.",
        "",
        "Under original protocols (protocol original_primary/1 and original_control/1), nominal speed is searched within 75–125% of reference speed. To resolve the reviewed 5°-lean fixed-target nominal search, the driver control supplement (protocol driver-control-speed-bounds/1) expanded nominal speed search to 75–130%. Additionally, qualified supplemental protocols (driver-control-speed-bounds/1, pw-control-face-bounds/1, and pw-primary-face-bounds/1) evaluated delivery solutions within an explicit absolute face box of [-6.0°, +6.0°]. These adapted bounds were not preregistered in the original factorial protocol and do not imply physical impossibility outside them.",
        "",
    ]
    if bounds_control is None:
        lines.extend(["No qualified bounds-control comparison is available.", ""])
        return lines
    effects = bounds_control.get("effects", [])
    if effects:
        lines.extend([render_effect_table(effects), ""])
    unavailable = bounds_control.get("unavailable", [])
    for item in unavailable:
        lines.append(f"- {item}")
    if unavailable:
        lines.append("")
    if not effects and not unavailable:
        lines.extend(["No qualified bounds-control comparison is available.", ""])
    return lines


def _conclusions_and_synthesis_lines(interpretation: dict) -> list[str]:
    from .report_synthesis import synthesis_lines

    return synthesis_lines(interpretation)


def _qualified_interpretation_lines(interpretation: dict) -> list[str]:
    """Render only prevalidated cell, pair, and asset references."""
    from src.tools.shot_pattern_analysis.matched_interpretation import (
        render_cell_table,
        render_geometry_table,
        render_scoring_table,
    )

    from .report_tables import keyed_effect_table as render_effect_table

    supplement = interpretation["supplement"]
    if supplement.get("status") != "qualified":
        raise ValueError("additive supplement must be qualified before publication")
    lines = [
        "### Condition-Resolved Carry and Dispersion",
        "",
        "Each row names the actual cell. Mean carry, signed X/Y bias, spread, and RMSE use that pattern's individually supported endpoints. The shared nominal target removes initial range mismatch within a cell; random errors can still shift realized carry. Primary lean targets can differ, while fixed-target controls rematch face, path, and speed and do not isolate a causal distance or lean effect.",
        "",
        render_cell_table(interpretation["selected_cells"]),
        "",
        "### Delivered Face, Loft, and Simulated Directional Outcomes",
        "",
        render_geometry_table(interpretation["selected_cells"]),
        "",
        "### Absolute Course Scoring by Actual Context",
        "",
        render_scoring_table(interpretation["scores"]),
        "",
        "### Direct Curve and Shaft-Variability Interactions",
        "",
        "Only reviewed same-index six-population paired contrasts appear here. A shaft-twist SD of 1° or 2° is an assumed primitive input, not a measured face-angle SD or a player precision estimate. Positive interaction means the named curve gained relative to Straight in condition B versus A; this can differ from a lateral-width ranking because target bias, downrange error, and lie scoring also matter.",
        "",
        render_effect_table(interpretation["effects"]),
        "",
    ]
    lines.extend(_bounds_control_lines(interpretation.get("bounds_control")))
    if "resolved_selected_cells" in interpretation:
        lines.extend(_conclusions_and_synthesis_lines(interpretation))
    lines.extend(
        [
            "### Verified Academic and Social Figures",
            "",
            "Figure links below come from source-bound inventories and include their actual protocol and population type. Individual cell densities use individually supported endpoints, while paired densities use the shared six-population supported-index intersection; displayed point overlays and overhead paths are sampled, as labeled in each image. A 95% highest-density region is not a confidence interval for the mean.",
            "",
        ]
    )
    for figure in interpretation["figure_links"]:
        path = Path(figure["relative_path"])
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("figure link must remain relative to the report")
        label = (
            f"{figure['club'].replace('_', ' ').title()} "
            f"{figure['kind'].replace('_', ' ').title()} "
            f"({figure['presentation'].title()}, {figure['source_protocol']}, "
            f"{figure['source_kind'].replace('_', ' ')})"
        )
        if figure.get("physical_pattern"):
            label += f" · {figure['physical_pattern']}"
        if figure.get("comparison_kind"):
            label += " · " + str(figure["comparison_kind"]).replace(
                "_", " "
            ).title().replace("Sd", "SD")
            label += " · " + " → ".join(figure["source_cells"])
        lines.append(f"- [{label}]({path.as_posix()})")
    if not interpretation["figure_links"]:
        lines.append("No qualified figure asset is available for these selected cells.")
    for inventory_path in interpretation.get("figure_inventories", []):
        relative = Path(inventory_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("figure inventory link must remain relative")
        lines.append(
            f"- [Complete Original Figure Availability: {relative.parent.name}]({relative.as_posix()})"
        )
    lines.extend(
        (
            "",
            "### Separately Qualified Bounds Supplement",
            "",
            f"The additive supplement inventories {supplement['unique_executed_cells']} unique planned cells and verified {supplement['verified_control_aliases']} control aliases without duplicating those Monte Carlo shots. It is an adapted bounds protocol after the original bounded-search audit, not part of the original factorial preregistration. Its original-versus-supplement results retain explicit versioned protocol identities, common supported indices, and unavailable conditions. [Full Paired Comparison Inventory]("
            + supplement["comparison_inventory"]
            + ") and [Supplement Figure Inventory]("
            + supplement["figure_inventory"]
            + ") contain every reviewed result and asset.",
            "",
            f"Supplement publication receipt SHA-256: `{supplement['receipt_sha256']}`. "
            f"Parent numerical pipeline receipt SHA-256: `{supplement.get('publication_identity', {}).get('parent_receipt_sha256', supplement['receipt_sha256'])}`.",
            "",
        )
    )
    return lines


def render_final_report(
    primary: dict,
    control: dict,
    lean: dict,
    sensitivity: dict[str, dict],
    *,
    interpretation: dict | None = None,
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
        "### Deterministic Head-Assumption Sensitivity",
        "",
        "The [Deterministic Head-Assumption Sensitivity](DESIGN.md#deterministic-head-assumption-sensitivity) section of the scientific design, supported by the [Saved Sensitivity Summary](validation/independent_head_sensitivity_summary.json) and [Full Execution Receipt](validation/independent_head_sensitivity.json), documents a separate deterministic evaluation of off-center impact sensitivity. The chosen ±20% perturbations to clubhead parameters (CG depth, principal inertias, face radii) and contact-law comparison (energetic restitution versus Newton velocity restitution) are model sensitivities, not measured parameter uncertainty. Narrow Monte Carlo confidence intervals quantify only random delivery draw sampling error and do not validate impact model accuracy or off-center delivery realism.",
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
    if interpretation is None:
        lines.extend(_anchor_lines(primary))
    else:
        lines.extend(_qualified_interpretation_lines(interpretation))
    lines.extend(_selected_comparison_lines(primary))
    if interpretation is not None and "resolved_selected_cells" in interpretation:
        lines.extend(
            _resolved_selected_comparison_lines(
                interpretation["resolved_selected_cells"]
            )
        )
    if interpretation is None:
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


def _report_source_hashes() -> dict[str, str]:
    """Bind the public interpretation, qualification, and builder implementations."""
    repository = Path(__file__).resolve().parents[3]
    sources = (
        Path(__file__).resolve(),
        repository / "scripts/research/shot_pattern_analysis/final_report_inputs.py",
        repository / "scripts/research/shot_pattern_analysis/report_synthesis.py",
        repository / "scripts/research/shot_pattern_analysis/report_tables.py",
        repository / "scripts/research/shot_pattern_analysis/supplement_publication.py",
        repository
        / "scripts/research/shot_pattern_analysis/supplement_parent_verifier.py",
        repository / "src/tools/shot_pattern_analysis/matched_interpretation.py",
        repository / "src/tools/shot_pattern_analysis/matched_formatting.py",
        repository / "src/tools/shot_pattern_analysis/matched_report.py",
        repository / "src/tools/shot_pattern_analysis/matched_lean_control.py",
        repository / "src/tools/shot_pattern_analysis/matched_sensitivity.py",
        repository / "src/tools/shot_pattern_analysis/matched_supplement_pipeline.py",
        repository / "src/tools/shot_pattern_analysis/matched_supplement_report.py",
        repository / "src/tools/shot_pattern_analysis/matched_supplement_figures.py",
    )
    return {
        str(path.relative_to(repository)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sources
    }


def final_publication_receipt(
    *,
    output_paths: tuple[Path, Path, Path],
    source_sha256: dict[str, str],
    primary_plan_sha256: str,
    control_plan_sha256: str,
    figure_manifest_sha256: dict[str, str],
    supplement_receipt_sha256: str,
    supplement_publication_identity: dict | None = None,
) -> dict:
    """Bind every editable report output to its validated input identities."""
    if len({path.parent.resolve() for path in output_paths}) != 1:
        raise ValueError("publication outputs must share one directory")
    for path in output_paths:
        if not path.is_file():
            raise ValueError(f"publication output is missing: {path.name}")
    return {
        "schema": "matched-final-publication-receipt/1",
        "source_sha256": dict(source_sha256),
        "primary_plan_sha256": primary_plan_sha256,
        "control_plan_sha256": control_plan_sha256,
        "figure_manifest_sha256": dict(figure_manifest_sha256),
        "supplement_receipt_sha256": supplement_receipt_sha256,
        "supplement_publication_identity": supplement_publication_identity,
        "output_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in output_paths
        },
    }


def assert_publication_inputs_unchanged(
    *,
    primary_root: Path,
    primary: dict,
    control_root: Path,
    control: dict,
    primary_figures: Path,
    control_figures: Path,
    figure_manifest_sha256: dict[Path, str],
    supplement_dir: Path,
    supplement_receipt_sha256: str,
    source_sha256: dict[str, str],
    report_root: Path,
) -> None:
    """Revalidate long-running aggregation inputs immediately before writing."""
    from .supplement_publication import load_verified_supplement_publication

    for root, report in ((primary_root, primary), (control_root, control)):
        assert_matrix_inputs_unchanged(root, report)
    if _report_source_hashes() != source_sha256:
        raise RuntimeError("report interpretation source changed during assembly")
    if any(
        hashlib.sha256((root / "figure_manifest.json").read_bytes()).hexdigest()
        != digest
        for root, digest in figure_manifest_sha256.items()
    ):
        raise RuntimeError("figure manifest changed during report assembly")
    for root, report, protocol in (
        (primary_figures, primary, "original_primary"),
        (control_figures, control, "original_control"),
    ):
        verified_original_figure_links(
            root, report, report_root=report_root, protocol=protocol
        )
        verified_original_figure_inventory(
            root, report, report_root=report_root, protocol=protocol
        )
    verified = load_verified_supplement_publication(supplement_dir)
    if verified["publication_identity"]["receipt_sha256"] != supplement_receipt_sha256:
        raise RuntimeError("supplement receipt changed during report assembly")
    if _report_source_hashes() != source_sha256:
        raise RuntimeError(
            "report interpretation source changed during final validation"
        )


def load_supplement_at_stable_receipt(directory: Path) -> tuple[dict, str]:
    """Bind validated values to the same receipt observed before their load."""
    from .supplement_publication import load_verified_supplement_publication

    supplement = load_verified_supplement_publication(directory)
    return supplement, supplement["publication_identity"]["receipt_sha256"]


def build_final_report(
    primary_root: Path,
    control_root: Path,
    destination: Path,
    *,
    supplement_dir: Path | None = None,
    primary_figures: Path | None = None,
    control_figures: Path | None = None,
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
    report_root = destination.parent.resolve()
    supplement_dir = Path(supplement_dir or report_root / "supplement_report_v1")
    primary_figures = Path(primary_figures or report_root / "final_figures_v1")
    control_figures = Path(control_figures or report_root / "control_final_figures_v1")
    sources_before = _report_source_hashes()
    figure_manifests_before = {
        root: hashlib.sha256((root / "figure_manifest.json").read_bytes()).hexdigest()
        for root in (primary_figures, control_figures)
    }
    primary = load_matrix_report(primary_root)
    control = load_matrix_report(control_root)
    supplement, supplement_receipt_before = load_supplement_at_stable_receipt(
        supplement_dir
    )
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
    interpretation = assemble_qualified_interpretation(
        primary_root,
        primary,
        control_root,
        control,
        supplement,
        primary_figures=primary_figures,
        control_figures=control_figures,
        supplement_dir=supplement_dir,
        report_root=report_root,
    )
    rendered = render_final_report(
        primary, control, lean, sensitivity, interpretation=interpretation
    )
    assert_publication_inputs_unchanged(
        primary_root=primary_root,
        primary=primary,
        control_root=control_root,
        control=control,
        primary_figures=primary_figures,
        control_figures=control_figures,
        figure_manifest_sha256=figure_manifests_before,
        supplement_dir=supplement_dir,
        supplement_receipt_sha256=supplement_receipt_before,
        source_sha256=sources_before,
        report_root=report_root,
    )
    cell_inventory_path = destination.with_name("REPORT_CELLS.csv")
    lean_inventory_path = destination.with_name("REPORT_LEAN.csv")
    _write_atomic(cell_inventory_path, render_cell_inventory(primary, control))
    _write_atomic(lean_inventory_path, render_lean_inventory(lean))
    _write_atomic(destination, rendered)
    publication_receipt = final_publication_receipt(
        output_paths=(destination, cell_inventory_path, lean_inventory_path),
        source_sha256=sources_before,
        primary_plan_sha256=primary["plan_sha256"],
        control_plan_sha256=control["plan_sha256"],
        figure_manifest_sha256={
            root.name: digest for root, digest in figure_manifests_before.items()
        },
        supplement_receipt_sha256=interpretation["supplement"]["receipt_sha256"],
        supplement_publication_identity=interpretation["supplement"].get(
            "publication_identity"
        ),
    )
    _write_atomic(
        destination.with_name("REPORT_PUBLICATION_RECEIPT.json"),
        json.dumps(publication_receipt, indent=2, sort_keys=True) + "\n",
    )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("primary_root", type=Path)
    parser.add_argument("control_root", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--supplement-dir", type=Path)
    parser.add_argument("--primary-figures", type=Path)
    parser.add_argument("--control-figures", type=Path)
    arguments = parser.parse_args()
    print(
        build_final_report(
            arguments.primary_root,
            arguments.control_root,
            arguments.destination,
            supplement_dir=arguments.supplement_dir,
            primary_figures=arguments.primary_figures,
            control_figures=arguments.control_figures,
        )
    )


if __name__ == "__main__":
    main()
