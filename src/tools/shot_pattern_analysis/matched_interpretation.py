"""Pure, evidence-bound summaries for the matched shot-pattern report.

Inputs are already integrity-checked by ``matched_report``.  This module does
not compare separately conditioned means or infer a causal distance effect.
"""

from __future__ import annotations

import math
import hashlib
import csv
import gzip
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Mapping

from .matched_report import load_validated_cell
from .provenance import manifest_files_match


PATTERNS = ("Straight", "Draw", "Fade")
_QUADRANTS = ("left_long", "left_short", "right_long", "right_short")


def _finite(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _delivered_means(rows: Iterable[Mapping[str, Any]]) -> dict[str, float | int]:
    values = [
        (
            _finite(row["face_deg"], "delivered face"),
            _finite(row["loft_deg"], "delivered loft"),
        )
        for row in rows
    ]
    if not values:
        raise ValueError("delivered geometry requires supported rows")
    return {
        "mean_delivered_face_deg": fmean(face for face, _ in values),
        "mean_delivered_loft_deg": fmean(loft for _, loft in values),
        "delivered_mean_n": len(values),
    }


def _face_loft_covariance(diagnostics: Mapping[str, Any]) -> float:
    fields = diagnostics["fields"]
    covariance = diagnostics["covariance"]
    face = fields.index("face_deg")
    loft = fields.index("loft_deg")
    return _finite(covariance[face][loft], "face-loft covariance")


def describe_cell(
    cell: Mapping[str, Any],
    *,
    delivered_rows: Mapping[str, Iterable[Mapping[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Summarize a validated cell without treating unattempted rows as misses.

    Optional delivered rows must be supported rows from a manifest-verified CSV.
    Their mean population is reported separately from the endpoint population.
    """
    status = cell["status"]
    if status not in ("completed", "unavailable", "not_run"):
        raise ValueError("unknown cell status")
    config = cell["config"]
    planned = int(config["n_shots"])
    if planned <= 0:
        raise ValueError("planned population must be positive")
    result: dict[str, Any] = {
        "cell": cell["cell"],
        "club_id": config["club"]["club_id"],
        "shaft_lean_deg": _finite(config["club"]["shaft_lean_deg"], "shaft lean"),
        "status": status,
        "planned_n_per_pattern": planned,
        "attempted_n_per_pattern": planned if status == "completed" else 0,
        "patterns": {},
    }
    if status != "completed":
        result["reason"] = cell.get("reason", "")
        result["patterns"] = {
            name: {"status": "Not Attempted", "supported_n": None} for name in PATTERNS
        }
        return result
    target = _finite(cell["target_x_m"], "target x")
    result["target_x_m"] = target
    for name in PATTERNS:
        item = cell["summary"][name]
        statistics = item["statistics"]
        quadrants = statistics["quadrant_fractions"]
        if any(key not in quadrants for key in _QUADRANTS):
            raise ValueError("all four endpoint quadrants are required")
        supported = int(item["supported_n"])
        attempted = int(item["total_n"])
        if attempted != planned or not 0 <= supported <= attempted:
            raise ValueError("endpoint population disagrees with plan")
        interval = tuple(
            _finite(v, "support interval") for v in item["supported_fraction_exact95"]
        )
        if len(interval) != 2 or not 0 <= interval[0] <= interval[1] <= 1:
            raise ValueError("invalid support interval")
        nominal = cell["nominals"][name]
        pattern: dict[str, Any] = {
            "status": "Completed",
            "supported_n": supported,
            "attempted_n": attempted,
            "supported_fraction_ci95": interval,
            "nominal_face_deg": _finite(nominal["face_deg"], "nominal face"),
            "nominal_path_deg": _finite(nominal["path_deg"], "nominal path"),
            "nominal_speed_mps": _finite(nominal["speed_mps"], "nominal speed"),
            "mean_carry_m": _finite(item["mean_carry_m"], "mean carry"),
            "mean_downrange_error_m": _finite(
                statistics["mean_downrange_error_m"], "downrange bias"
            ),
            "mean_lateral_error_m": _finite(
                statistics["mean_lateral_error_m"], "lateral bias"
            ),
            "downrange_sd_m": _finite(item["downrange_sd_m"], "downrange spread"),
            "lateral_sd_m": _finite(item["lateral_sd_m"], "lateral spread"),
            "target_rmse_m": _finite(item["target_rmse_m"], "target RMSE"),
            "quadrant_fractions": {
                key: _finite(quadrants[key], key) for key in _QUADRANTS
            },
            "face_loft_covariance_deg2": _face_loft_covariance(
                item["joint_delivery_outcome_diagnostics"]
            ),
            "mean_delivered_face_deg": None,
            "mean_delivered_loft_deg": None,
            "delivered_mean_n": None,
        }
        if delivered_rows is not None:
            if name not in delivered_rows:
                raise ValueError(f"missing delivered rows for {name}")
            pattern.update(_delivered_means(delivered_rows[name]))
            if pattern["delivered_mean_n"] != supported:
                raise ValueError(
                    "delivered geometry population differs from supported endpoints"
                )
        result["patterns"][name] = pattern
    return result


def describe_saved_cell(directory: Path) -> dict[str, Any]:
    """Read supported delivered geometry only from a manifest-validated cell.

    The compressed CSV is covered by the cell manifest checked before and after
    reading. Unavailable cells have no attempted CSV and remain Not Attempted.
    """
    directory = Path(directory)
    cell = {**load_validated_cell(directory), "cell": directory.name}
    if cell["status"] != "completed":
        return describe_cell(cell)
    rows: dict[str, list[dict[str, str]]] = {name: [] for name in PATTERNS}
    with gzip.open(directory / "shots.csv.gz", "rt", newline="") as stream:
        for row in csv.DictReader(stream):
            name = row["pattern"]
            if name not in rows:
                raise ValueError("saved shot has an unknown physical pattern")
            if row["status"] == "supported":
                rows[name].append(row)
            elif not row["status"]:
                raise ValueError("saved shot has an empty support status")
    if not manifest_files_match(directory / "manifest.json"):
        raise ValueError("cell manifest changed during delivered-geometry reading")
    return describe_cell(cell, delivered_rows=rows)


def describe_scoring(cell: Mapping[str, Any]) -> dict[str, Any]:
    """Expose absolute SG and the scored population separately from paired SG."""
    if cell["status"] != "completed":
        return {"status": "Not Attempted", "reason": cell.get("reason", "")}
    scoring = cell.get("course_scoring", {})
    if scoring.get("status") not in ("available", "conditional_only"):
        return {"status": "Unavailable", "reason": scoring.get("reason", "")}
    paired = int(scoring["paired_n"])
    total = int(scoring["total_n"])
    individual = {name: int(scoring["patterns"][name]["n"]) for name in PATTERNS}
    if not 0 <= paired <= min(individual.values()) or any(
        not 0 <= value <= total for value in individual.values()
    ):
        raise ValueError("course scoring populations disagree")
    context = {
        key: scoring.get(key)
        for key in (
            "scenario_id",
            "start_lie",
            "start_distance_m",
            "hole_distance_m",
            "fairway_half_width_m",
            "green_radius_m",
            "target_x_m",
        )
    }
    return {
        "status": scoring["status"],
        "context": context,
        "paired_n": paired,
        "total_n_per_pattern": total,
        "patterns": {
            name: {
                "individual_supported_n": individual[name],
                "absolute_mean_strokes_gained": _finite(
                    scoring["patterns"][name]["mean_strokes_gained"],
                    "mean strokes gained",
                ),
            }
            for name in PATTERNS
        },
        "interpretation": "Historical-tour baseline; conditional simulated endpoint score, not observed player strokes gained.",
    }


def describe_effect_modification(pair: Mapping[str, Any]) -> dict[str, Any]:
    """Interpret only a qualified same-index six-population paired contrast."""
    if pair.get("schema") != "matched-explicit-source-pair/1":
        raise ValueError("explicit source pair schema required")
    if pair.get("conditioning") != "same_indices_supported_in_all_six_populations":
        raise ValueError("six-population common-index conditioning required")
    kind = pair.get("comparison_kind")
    if kind not in (
        "curve_strength",
        "shaft_rotation_sd",
        "across_lean",
        "speed_bound",
    ):
        raise ValueError("unsupported comparison kind")
    status = pair.get("status")
    if status not in ("available", "conditional_only", "unavailable"):
        raise ValueError("unknown paired status")
    common = int(pair["common_supported_n"])
    total = int(pair["total_n_per_population"])
    if not 0 <= common <= total:
        raise ValueError("invalid common six-population count")
    effects = pair.get("course_scoring_effect_modification", {})
    result = {
        "comparison_kind": kind,
        "status": status,
        "common_six_n": common,
        "total_n_per_population": total,
        "source_protocols": tuple(pair.get("source_protocols", ())),
        "draw_minus_straight_change": effects.get("Draw"),
        "fade_minus_straight_change": effects.get("Fade"),
        "inference_limit": (
            "pointwise Monte Carlo intervals on shared seeded errors; cells are correlated, "
            "not independent replications or multiplicity-adjusted discoveries. "
            "Nominal target matching removes initial range mismatch only. The paired "
            "contrast also changes matched delivery and does not isolate a causal distance effect."
        ),
    }
    return result


def select_verified_assets(
    root: Path,
    entries: Iterable[Mapping[str, Any]],
    figure_manifest: Mapping[str, Any],
    *,
    club: str,
    kind: str,
    presentation: str,
) -> list[dict[str, Any]]:
    """Select only hash-bound indexed assets for one club and presentation."""
    if presentation not in ("academic", "social"):
        raise ValueError("unknown figure presentation")
    if kind not in ("density", "overhead", "comparison"):
        raise ValueError("unknown figure kind")
    outputs = figure_manifest["outputs"]
    cells = figure_manifest["cell_manifest_sha256"]
    root = Path(root).resolve()
    selected = []
    for entry in entries:
        if entry.get("club") != club or entry.get("status") != "completed":
            continue
        referenced_cells = (
            (entry["first_cell"], entry["second_cell"])
            if kind == "comparison"
            else (entry["cell"],)
        )
        if any(cell not in cells for cell in referenced_cells):
            raise ValueError(
                "figure references a cell absent from the inventoried plan"
            )
        path_field = "png_overhead" if kind == "overhead" else "png"
        relative_value = entry.get(path_field)
        if not relative_value:
            continue
        relative = Path(str(relative_value))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("asset path must remain inside its inventory root")
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"indexed asset is missing: {relative}")
        expected = outputs.get(relative.as_posix())
        if (
            expected is None
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected
        ):
            raise ValueError(f"indexed asset hash differs: {relative}")
        selected.append(
            {
                "kind": kind,
                "presentation": presentation,
                "path": path,
            }
        )
    return selected


def render_cell_table(cells: Iterable[Mapping[str, Any]]) -> str:
    """Compact conditional outcome table for selected, labeled cells."""
    lines = [
        "| Club / Lean | Pattern | Mean Carry / Target X (m) | X Bias / SD (m) | Y Bias / SD (m) | Target RMSE (m) | Supported / Attempted [95% CI] |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for cell in cells:
        label = f"{cell['club_id']} / {cell['shaft_lean_deg']:g}°"
        for name in PATTERNS:
            item = cell["patterns"][name]
            if item["status"] != "Completed":
                lines.append(f"| {label} | {name} | Not Attempted | — | — | — | — |")
                continue
            lines.append(
                f"| {label} | {name} | {item['mean_carry_m']:.2f} / {cell['target_x_m']:.2f} | "
                f"{item['mean_downrange_error_m']:+.2f} / {item['downrange_sd_m']:.2f} | "
                f"{item['mean_lateral_error_m']:+.2f} / {item['lateral_sd_m']:.2f} | "
                f"{item['target_rmse_m']:.2f} | "
                f"{item['supported_n']:,} / {item['attempted_n']:,} "
                f"[{item['supported_fraction_ci95'][0] * 100:.1f}%, "
                f"{item['supported_fraction_ci95'][1] * 100:.1f}%] |"
            )
    return "\n".join(lines)


def render_effect_table(effects: Iterable[Mapping[str, Any]]) -> str:
    """Show direct paired curve interactions, never marginal mean subtraction."""
    lines = [
        "| Contrast | Curved Pattern | B − A Relative SG (95% MC Interval) | Common Six / Attempted | Protocols |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for effect in effects:
        protocols = " → ".join(effect["source_protocols"])
        for name, key in (
            ("Draw", "draw_minus_straight_change"),
            ("Fade", "fade_minus_straight_change"),
        ):
            value = effect[key]
            interval = "Unavailable"
            if isinstance(value, Mapping):
                estimate = _finite(value["estimate"], "paired effect")
                lower = _finite(value["lower_95"], "paired effect interval")
                upper = _finite(value["upper_95"], "paired effect interval")
                if lower > upper:
                    raise ValueError("paired effect interval is reversed")
                interval = f"{estimate:+.4f} [{lower:+.4f}, {upper:+.4f}]"
            lines.append(
                f"| {effect['comparison_kind']} | {name} | {interval} | "
                f"{effect['common_six_n']:,} / {effect['total_n_per_population']:,} | "
                f"{protocols} |"
            )
    lines.extend(
        (
            "",
            "B − A is the change in (curved − Straight) SG on the same six-population "
            "supported-index intersection. It is not a causal distance contribution. "
            "Intervals are pointwise Monte Carlo sampling intervals; shared seeded "
            "errors correlate cells and do not validate player performance.",
        )
    )
    return "\n".join(lines)


def render_scoring_table(
    named_scores: Iterable[tuple[str, Mapping[str, Any]]],
) -> str:
    """Report hypothetical course context and absolute conditional SG means."""
    lines = [
        "| Cell | Course Context | Pattern | Absolute Mean SG | Individual Scored / Attempted | Common Paired / Attempted |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for cell_name, score in named_scores:
        if score["status"] not in ("available", "conditional_only"):
            lines.append(f"| {cell_name} | {score['status']} | — | — | — | — |")
            continue
        context = score["context"]
        details = [f"{context['start_lie']}; {context['scenario_id']}"]
        if context["start_distance_m"] is not None:
            details.append(f"start {float(context['start_distance_m']):.1f} m")
        if context["hole_distance_m"] is not None:
            details.append(f"hole {float(context['hole_distance_m']):.1f} m")
        if context["fairway_half_width_m"] is not None:
            details.append(
                f"fairway half-width {float(context['fairway_half_width_m']):.1f} m"
            )
        if context["green_radius_m"] is not None:
            details.append(f"green radius {float(context['green_radius_m']):.1f} m")
        course = "; ".join(details)
        total = int(score["total_n_per_pattern"])
        for name in PATTERNS:
            item = score["patterns"][name]
            lines.append(
                f"| {cell_name} | {course} | {name} | "
                f"{item['absolute_mean_strokes_gained']:+.4f} | "
                f"{item['individual_supported_n']:,} / {total:,} | "
                f"{score['paired_n']:,} / {total:,} |"
            )
    lines.extend(
        (
            "",
            "Absolute SG is a conditional simulated endpoint score against a historical-tour "
            "baseline. Driver tee and iron/wedge approach contexts are different "
            "opportunities. The individual and common paired populations are distinct; "
            "no values here are measured player strokes gained or per-round effects.",
        )
    )
    return "\n".join(lines)


def render_geometry_table(cells: Iterable[Mapping[str, Any]]) -> str:
    """Separate delivered geometry covariance from simulated endpoint quadrants."""
    selected = list(cells)
    lines = [
        "| Club / Lean | Pattern | Nominal Face / Path (°) | Mean Delivered Face / Loft (°) | Face–Loft Covariance (°²) | Mean Population |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for cell in selected:
        label = f"{cell['club_id']} / {cell['shaft_lean_deg']:g}°"
        for name in PATTERNS:
            item = cell["patterns"][name]
            if item["status"] != "Completed":
                lines.append(f"| {label} | {name} | Not Attempted | — | — | — |")
                continue
            face, loft = (
                item["mean_delivered_face_deg"],
                item["mean_delivered_loft_deg"],
            )
            delivered = (
                "Unavailable"
                if face is None or loft is None
                else f"{face:+.2f} / {loft:.2f}"
            )
            mean_n = (
                "—"
                if item["delivered_mean_n"] is None
                else f"{item['delivered_mean_n']:,}"
            )
            lines.append(
                f"| {label} | {name} | {item['nominal_face_deg']:+.2f} / "
                f"{item['nominal_path_deg']:+.2f} | {delivered} | "
                f"{item['face_loft_covariance_deg2']:+.3f} | {mean_n} |"
            )
    lines.extend(
        (
            "",
            "| Club / Lean | Pattern | Left-Long | Left-Short | Right-Long | Right-Short |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        )
    )
    for cell in selected:
        label = f"{cell['club_id']} / {cell['shaft_lean_deg']:g}°"
        for name in PATTERNS:
            item = cell["patterns"][name]
            if item["status"] != "Completed":
                lines.append(f"| {label} | {name} | Not Attempted | — | — | — |")
                continue
            quadrants = item["quadrant_fractions"]
            lines.append(
                f"| {label} | {name} | "
                + " | ".join(f"{quadrants[key]:.3f}" for key in _QUADRANTS)
                + " |"
            )
    lines.extend(
        (
            "",
            "The rigid geometry predicts face/loft coupling for these stated club setups. "
            "Quadrants are simulated endpoint frequencies conditional on supported shots; "
            "they do not identify a closed-face mechanism or establish that left shots "
            "travel longer. Path, strike, speed and spin can also change the outcome.",
        )
    )
    return "\n".join(lines)
