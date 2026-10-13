"""Bounded narrative derived from the exact selected, qualified study records."""

from __future__ import annotations

import math
from collections import Counter

from src.tools.shot_pattern_analysis.matched_formatting import format_sg_value

CLUBS = ("driver", "seven_iron", "pitching_wedge")
PATTERNS = ("Straight", "Draw", "Fade")


def validate_resolved_selection(cells: list[dict]) -> None:
    """Require the actual 24-cell selection and coherent scoring populations."""
    expected = {
        f"{club}_lean0_curve{curve}_rotation{sd}_{profile}"
        for club in CLUBS
        for curve in (1, 2)
        for sd in (1, 2)
        for profile in ("shaft_only", "combined2")
    }
    if len(cells) != 24 or {c["cell"] for c in cells} != expected:
        raise ValueError("resolved selection requires 24 unique exact cell identities")
    for cell in cells:
        cfg = cell["config"]
        club = cfg["club"]["club_id"]
        profile = cell["cell"].split("_rotation", 1)[1].split("_", 1)[1]
        name = (
            f"{club}_lean0_curve{cfg['curve_scale']:g}_rotation"
            f"{cfg['noise']['standard_deviations'][0]:g}_{profile}"
        )
        if name != cell["cell"] or cfg["club"]["shaft_lean_deg"] != 0:
            raise ValueError("resolved config disagrees with selected identity")
        supplemental = club == "pitching_wedge" and cfg["curve_scale"] == 2
        protocol = "pw-primary-face-bounds/1" if supplemental else "original_primary/1"
        if cell.get("source_protocol") != protocol:
            raise ValueError("resolved selection has incorrect source protocol")
        if supplemental and (
            cfg.get("match_face_bounds_deg") != [-6.0, 6.0]
            or cfg.get("match_speed_factors") != [0.75, 1.25]
        ):
            raise ValueError("resolved supplement bounds differ from reviewed protocol")
        if cell.get("status") != "completed":
            raise ValueError("resolved selection contains uncompleted cell")
        score = cell["course_scoring"]
        total = cfg["n_shots"]
        n = score["paired_n"]
        if total != 10000 or score["total_n"] != total or not 2 <= n <= total:
            raise ValueError("resolved scoring population count is invalid")
        if score["status"] not in ("available", "conditional_only"):
            raise ValueError("resolved selection scoring is unavailable")
        for pattern in PATTERNS:
            geom = cell["summary"][pattern]
            if geom["total_n"] != total or not n <= geom["supported_n"] <= total:
                raise ValueError("scored population exceeds individual supported count")
        for pattern in ("Draw", "Fade"):
            interval = score["paired_benefit_vs_straight"][pattern]
            values = [interval[key] for key in ("lower_95", "estimate", "upper_95")]
            if not all(math.isfinite(x) for x in values) or values != sorted(values):
                raise ValueError("invalid selected scoring interval")


def synthesis_lines(interpretation: dict) -> list[str]:
    """Summarize actual values without asserting a universal preferred shape."""
    cells = interpretation.get("resolved_selected_cells", [])
    validate_resolved_selection(cells)
    lines = [
        "### Conclusions and Geometric Mechanism Synthesis",
        "",
        "The following descriptive ranges and interval-direction counts cover only the "
        "eight selected zero-lean conditions per club (curve ×1/×2, shaft-twist SD "
        "1°/2°, shaft-only/combined2). Positive SG favors the named curve over Straight. "
        "Ranges summarize cell point estimates, not pooled confidence intervals. "
        "Shared draws make these cells dependent; interval directions are pointwise "
        "Monte Carlo statements, not empirical or multiplicity-adjusted discoveries.",
        "",
        "| Club | Pattern | Selected SG Estimate Range (Strokes) | Positive / Negative / Zero-Spanning 95% Intervals | Common-Three Scored n Range / 10,000 |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for club in CLUBS:
        rows = [c for c in cells if c["config"]["club"]["club_id"] == club]
        counts = [c["course_scoring"]["paired_n"] for c in rows]
        for pattern in ("Draw", "Fade"):
            intervals = [
                c["course_scoring"]["paired_benefit_vs_straight"][pattern] for c in rows
            ]
            estimates = [item["estimate"] for item in intervals]
            directions = Counter(
                "positive"
                if item["lower_95"] > 0
                else "negative"
                if item["upper_95"] < 0
                else "spans"
                for item in intervals
            )
            lines.append(
                f"| {club} | {pattern} | {format_sg_value(min(estimates))} to "
                f"{format_sg_value(max(estimates))} | {directions['positive']} / "
                f"{directions['negative']} / {directions['spans']} | "
                f"{min(counts):,}–{max(counts):,} / 10,000 |"
            )
    lines += [
        "",
        "The relative scoring pattern depends on club and delivery-noise assumptions; "
        "these selected rows do not establish a universal best shape. An interval spanning "
        "zero does not establish equivalence. The direct paired interaction tables above "
        "answer SD 1→2 and curve ×1→×2 on the same six-population intersection; their "
        "intervals must not be obtained by subtracting the marginal intervals summarized here.",
        "",
        "For the Straight, curve ×1, shaft-only subset below, nominal targets are matched "
        "before noise. The realized downrange errors and correlations use each pattern's "
        "individual supported endpoints; SD changes need not preserve mean carry.",
        "",
        "| Club | Shaft-Twist SD (°) | Lateral/Downrange Correlation | Mean Downrange Error (m) | Individual Supported / Attempted |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for club in CLUBS:
        for sd in (1, 2):
            cell = next(
                c
                for c in cells
                if c["cell"] == f"{club}_lean0_curve1_rotation{sd}_shaft_only"
            )
            item = cell["summary"]["Straight"]
            stats = item["statistics"]
            corr = stats["corr_lateral_downrange_error"]
            bias = stats["mean_downrange_error_m"]
            if not all(math.isfinite(x) for x in (corr, bias)):
                raise ValueError("nonfinite selected geometry statistic")
            lines.append(
                f"| {club} | {sd} | {corr:+.4f} | {bias:+.3f} | {item['supported_n']:,} / {item['total_n']:,} |"
            )
    lines += [
        "",
        "Positive lateral coordinates mean right. A positive lateral/downrange "
        "correlation associates rightward errors with longer downrange endpoints; a "
        "negative correlation associates leftward errors with longer endpoints. This "
        "describes the selected simulated population, not a causal face-angle attribution.",
        "",
        "At zero shaft twist, delivered loft is L − lean and the rigid geometry gives "
        "d(loft)/d(face) = cot(lie) cos(L − lean)/cos(L), positive for these setups. "
        "Opening the face by shaft rotation locally increases delivered loft; closing "
        "locally decreases it. Fixed primitive twist SD approximately preserves the "
        "first-order loft SD across lean while face SD changes. Launch, speed, spin and "
        "aerodynamics determine whether that loft change increases carry. See the "
        "[Scientific Design](DESIGN.md#frames-and-rigid-delivery-geometry) and the existing editable "
        "[Shot-Pattern Reference](../shot_pattern_analysis.tex).",
        "",
        "Fixed-target controls remove nominal range mismatch, but include rematched "
        "face, path and speed and retain conditional support. No causal fraction of "
        "dispersion or scoring is assigned to distance, lean or curvature. Narrow "
        "Monte Carlo intervals exclude assumed head-property and contact-law errors "
        "and unmeasured golfer variability.",
        "",
    ]
    return lines
