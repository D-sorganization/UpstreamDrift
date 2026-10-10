"""Validate completed matrix bundles and export a compact comparison overview.

This module reads saved evidence only. It never invokes a flight or scoring run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

PATTERNS = ("Straight", "Draw", "Fade")
CLUBS = ("driver", "seven_iron", "pitching_wedge")
DELIVERY_MODES = ("fixed_loft", "shaft_rotation")
REQUIRED_FILES = {
    "shots.csv",
    "summary.json",
    "receipt.json",
    "overhead_flight.png",
    "dispersion.png",
    "dispersion_equal_range.png",
    "strokes_gained.json",
    "strokes_gained_baseline.json",
    "run_start.json",
}
ROW_FIELDS = (
    "scenario",
    "club_id",
    "delivery_mode",
    "face_sd_deg",
    "curve_scale",
    "pattern",
    "shots_per_pattern",
    "lateral_sd_m",
    "mean_carry_m",
    "carry_sd_m",
    "aimed_rmse_m",
    "aimed_hit_rate",
    "corr_lateral_downrange_error",
    "p95_radial_target_error_m",
    "equal_range_lateral_sd_m",
    "mean_strokes_gained",
    "paired_delta_sg_vs_straight",
    "paired_delta_sg_lower_95",
    "paired_delta_sg_upper_95",
)
PATTERN_COLORS = {"Draw": "#ffbf69", "Fade": "#64d9ff"}


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _expected_cells() -> tuple[tuple[str, dict[str, Any]], ...]:
    # Import lazily: validating this module or using its CLI help never imports
    # the simulation, plotting, or scoring implementation.
    from src.tools.shot_pattern_analysis.matrix import corrected_matrix

    return tuple((name, asdict(config)) for name, config in corrected_matrix())


def _check_bundle(name: str, config: dict[str, Any], root: Path) -> dict[str, Any]:
    bundle = root / name
    manifest_path = bundle / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"matrix cell {name} has no completed manifest")
    manifest = _load_json(manifest_path)
    if manifest.get("scenario") != name or manifest.get("config") != config:
        raise ValueError(f"matrix manifest config does not match corrected cell {name}")
    hashes = manifest.get("files_sha256")
    if not isinstance(hashes, dict) or not REQUIRED_FILES.issubset(hashes):
        raise ValueError(f"matrix manifest is incomplete for {name}")
    for filename, expected_hash in hashes.items():
        if Path(filename).name != filename or not isinstance(expected_hash, str):
            raise ValueError(
                f"manifest contains an invalid artifact reference for {name}"
            )
        path = bundle / filename
        if not path.is_file():
            raise ValueError(f"manifest artifact is missing for {name}: {filename}")
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(f"manifest hash mismatch for {name}: {filename}")

    run_start = _load_json(bundle / "run_start.json")
    receipt = _load_json(bundle / "receipt.json")
    if run_start.get("config") != config:
        raise ValueError(f"run-start config does not match completed cell {name}")
    source_hashes = run_start.get("source_sha256")
    binary_hash = run_start.get("native_binary_sha256")
    if (
        not isinstance(source_hashes, dict)
        or not source_hashes
        or any(
            not isinstance(value, str) or len(value) != 64
            for value in source_hashes.values()
        )
        or not isinstance(binary_hash, str)
        or len(binary_hash) != 64
        or receipt.get("source_unchanged_during_run") is not True
        or receipt.get("execution_source_sha256") != source_hashes
        or receipt.get("execution_native_binary_sha256") != binary_hash
    ):
        raise ValueError(f"source-unchanged execution evidence is invalid for {name}")

    summary = _load_json(bundle / "summary.json")
    if summary.get("analysis_status", "current") != "current":
        raise ValueError(f"superseded or non-current analysis bundle: {name}")
    if summary.get("config") != config:
        raise ValueError(f"summary config does not match completed cell {name}")
    if set(summary.get("patterns", {})) != set(PATTERNS):
        raise ValueError(f"summary patterns are incomplete for {name}")
    if any(
        not math.isclose(float(summary["patterns"][pattern].get("n", -1)), 10_000)
        for pattern in PATTERNS
    ):
        raise ValueError(f"each pattern must contain exactly 10,000 shots in {name}")
    if config["n_shots"] != 10_000:
        raise ValueError(f"matrix cell is not the frozen 10,000-shot design: {name}")

    scoring = summary.get("course_scoring")
    if not isinstance(scoring, dict) or scoring.get("status") != "available":
        raise ValueError(f"source-backed course scoring is unavailable for {name}")
    if scoring.get("club_id") != config["club_id"]:
        raise ValueError(f"scoring club does not match matrix cell {name}")
    if scoring.get("source_backed_status") != "available":
        raise ValueError(f"source-backed scoring evidence is unavailable for {name}")
    scoring_artifact = _load_json(bundle / "strokes_gained.json")
    if scoring_artifact != scoring:
        raise ValueError(f"saved scoring artifact differs from summary in {name}")
    baseline_artifact = _load_json(bundle / "strokes_gained_baseline.json")
    if baseline_artifact.get("baseline", {}).get("table_sha256") != scoring.get(
        "baseline", {}
    ).get("table_sha256"):
        raise ValueError(f"scoring baseline artifact does not match report in {name}")
    if set(scoring.get("patterns", {})) != set(PATTERNS):
        raise ValueError(f"scoring pattern coverage is incomplete for {name}")
    if any(scoring["patterns"][pattern].get("n") != 10_000 for pattern in PATTERNS):
        raise ValueError(f"scoring does not cover all shots in {name}")

    for section, section_patterns in (
        ("landing_dispersion", summary.get("landing_dispersion")),
        (
            "equal_range_diagnostic",
            summary.get("equal_range_diagnostic", {}).get("patterns"),
        ),
    ):
        if not isinstance(section_patterns, dict) or set(section_patterns) != set(
            PATTERNS
        ):
            raise ValueError(f"{section} coverage is incomplete for {name}")
    return summary


def load_validated_matrix(matrix_root: Path) -> list[dict[str, Any]]:
    """Load all 24 completed cells after verifying configuration and evidence."""
    matrix_root = Path(matrix_root)
    if not matrix_root.is_dir():
        raise ValueError(f"matrix root is not a directory: {matrix_root}")
    cells = _expected_cells()
    expected_names = {name for name, _ in cells}
    actual_names = {path.name for path in matrix_root.iterdir() if path.is_dir()}
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        unexpected = sorted(actual_names - expected_names)
        raise ValueError(
            f"expected exactly 24 complete matrix cells; missing={missing}, unexpected={unexpected}"
        )
    summaries = [_check_bundle(name, config, matrix_root) for name, config in cells]

    # Comparison contracts require a shared club/delivery baseline. Compare
    # each four-cell uncertainty block, not unlike clubs against one another.
    from src.tools.shot_pattern_analysis.comparison import (
        validate_comparison,
        validate_scoring_comparison,
    )

    for club in CLUBS:
        for mode in DELIVERY_MODES:
            block = [
                summary
                for summary in summaries
                if summary["config"]["club_id"] == club
                and summary["config"]["delivery_mode"] == mode
            ]
            if len(block) != 4:
                raise ValueError(f"expected four uncertainty cells for {club}/{mode}")
            validate_comparison(block)
            validate_scoring_comparison(block)
    baseline_hashes = {
        summary["course_scoring"]["baseline"]["table_sha256"] for summary in summaries
    }
    if len(baseline_hashes) != 1:
        raise ValueError(
            "All matrix cells must use the same source-backed scoring benchmark"
        )
    return summaries


def _statistics_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        config = summary["config"]
        scoring = summary["course_scoring"]
        paired = scoring["paired_benefit_vs_straight"]
        for pattern in PATTERNS:
            flight = summary["patterns"][pattern]
            landing = summary["landing_dispersion"][pattern]
            equal_range = summary["equal_range_diagnostic"]["patterns"][pattern]
            effect = paired.get(pattern)
            rows.append(
                {
                    "scenario": summary["config"]["club_id"]
                    + "_"
                    + config["delivery_mode"]
                    + f"_sd{config['face_sd_deg']:g}_scale{config['curve_scale']:g}",
                    "club_id": config["club_id"],
                    "delivery_mode": config["delivery_mode"],
                    "face_sd_deg": config["face_sd_deg"],
                    "curve_scale": config["curve_scale"],
                    "pattern": pattern,
                    "shots_per_pattern": int(flight["n"]),
                    "lateral_sd_m": float(flight["aimed_lateral_sd_m"]),
                    "mean_carry_m": float(flight["mean_carry_m"]),
                    "carry_sd_m": float(flight["carry_sd_m"]),
                    "aimed_rmse_m": float(flight["aimed_target_rmse_m"]),
                    "aimed_hit_rate": float(flight["aimed_target_hit_fraction"]),
                    "corr_lateral_downrange_error": float(
                        landing["corr_lateral_downrange_error"]
                    ),
                    "p95_radial_target_error_m": float(
                        landing["p95_radial_target_error_m"]
                    ),
                    "equal_range_lateral_sd_m": float(equal_range["lateral_sd_m"]),
                    "mean_strokes_gained": float(
                        scoring["patterns"][pattern]["mean_strokes_gained"]
                    ),
                    "paired_delta_sg_vs_straight": (
                        0.0 if effect is None else float(effect["estimate"])
                    ),
                    "paired_delta_sg_lower_95": (
                        0.0 if effect is None else float(effect["lower_95"])
                    ),
                    "paired_delta_sg_upper_95": (
                        0.0 if effect is None else float(effect["upper_95"])
                    ),
                }
            )
    return rows


def _write_statistics(
    rows: list[dict[str, Any]],
    output_dir: Path,
    *,
    summaries: list[dict[str, Any]],
    matrix_root: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "overview_statistics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=ROW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    json_path = output_dir / "overview_statistics.json"
    payload = {
        "title": "Shot Pattern Matrix Overview",
        "row_count": len(rows),
        "rows": rows,
        "provenance": {
            "validated_matrix_cells": len(summaries),
            "shots_per_pattern": 10_000,
            "source_unchanged_for_every_run": True,
            "scoring_source": "Saved course_scoring results cross-checked against strokes_gained.json",
            "baseline": summaries[0]["course_scoring"]["baseline"],
            "manifest_sha256_by_scenario": {
                name: hashlib.sha256(
                    (matrix_root / name / "manifest.json").read_bytes()
                ).hexdigest()
                for name, _config in _expected_cells()
            },
        },
        "interpretation": {
            "positive_paired_strokes_gained": "favors the curved pattern vs. Straight",
            "physics": "model-conditional carry-only endpoint predictions",
            "driver_scoring": "historical benchmark for a hypothetical tee shot to fairway or rough",
            "iron_scoring": "historical benchmark for a hypothetical approach to green or rough",
            "uncertainty": "paired Monte Carlo intervals; excludes model and parameter uncertainty",
        },
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return {"statistics_csv": csv_path, "statistics_json": json_path}


def _plot_scoring_forest(
    rows: list[dict[str, Any]], *, delivery_mode: str, output: Path
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    club_titles = {
        "driver": "Driver Tee Outcomes",
        "seven_iron": "7-Iron Approach Outcomes",
        "pitching_wedge": "Pitching Wedge Approach Outcomes",
    }
    cell_order = ((1.0, 1.0), (2.0, 1.0), (1.0, 2.0), (2.0, 2.0))
    selected = [row for row in rows if row["delivery_mode"] == delivery_mode]
    minimum = min(row["paired_delta_sg_lower_95"] for row in selected)
    maximum = max(row["paired_delta_sg_upper_95"] for row in selected)
    padding = max(0.02, (maximum - minimum) * 0.08)
    x_limits = (minimum - padding, maximum + padding)
    navy, foreground = "#101c30", "#eff5fc"
    with plt.rc_context(
        {
            "figure.facecolor": navy,
            "axes.facecolor": navy,
            "text.color": foreground,
            "axes.labelcolor": foreground,
            "xtick.color": foreground,
            "ytick.color": foreground,
            "font.size": 12,
        }
    ):
        fig, axes = plt.subplots(1, 3, figsize=(19.2, 10.8), dpi=100, sharex=True)
        for ax, club in zip(axes, CLUBS, strict=True):
            club_rows = [row for row in selected if row["club_id"] == club]
            ax.axvline(0.0, color="#f3f6fa", alpha=0.8, linewidth=1.2)
            ax.set_xlim(*x_limits)
            ax.set_ylim(-0.6, 3.6)
            ax.set_yticks(range(4))
            ax.set_yticklabels(
                [f"SD {sd:g}°, Curve {scale:g}×" for sd, scale in cell_order]
            )
            ax.tick_params(axis="y", length=0)
            ax.invert_yaxis()
            ax.set_title(club_titles[club], fontsize=15, fontweight="bold")
            ax.grid(axis="x", alpha=0.18)
            ax.spines[["top", "right", "left"]].set_visible(False)
            for position, (sd, scale) in enumerate(cell_order):
                for offset, pattern in ((-0.13, "Draw"), (0.13, "Fade")):
                    row = next(
                        row
                        for row in club_rows
                        if row["face_sd_deg"] == sd
                        and row["curve_scale"] == scale
                        and row["pattern"] == pattern
                    )
                    estimate = row["paired_delta_sg_vs_straight"]
                    lower = row["paired_delta_sg_lower_95"]
                    upper = row["paired_delta_sg_upper_95"]
                    ax.errorbar(
                        estimate,
                        position + offset,
                        xerr=[[estimate - lower], [upper - estimate]],
                        fmt="o",
                        color=PATTERN_COLORS[pattern],
                        markersize=6,
                        capsize=3,
                        linewidth=1.5,
                    )
        fig.suptitle(
            f"Paired Model-Conditional Strokes-Gained Effects — {delivery_mode.replace('_', ' ').title()}",
            fontsize=23,
            fontweight="bold",
            y=0.96,
        )
        fig.supxlabel(
            "Paired Strokes Gained vs. Straight (Strokes; Positive Is Better)", y=0.14
        )
        fig.legend(
            handles=_pattern_legend_handles(),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.91),
            ncol=2,
            frameon=False,
        )
        fig.text(
            0.5,
            0.065,
            "Driver: Hypothetical Tee Shot on a 400 m Hole to Fairway/Rough. 7-Iron and PW: Approach to Green/Rough.",
            ha="center",
            fontsize=11,
        )
        fig.text(
            0.5,
            0.03,
            "Historical Tour Benchmark • Carry-Only • Model-Conditional, Not Player-Score Predictions • 95% Monte Carlo Intervals",
            ha="center",
            fontsize=11,
        )
        fig.subplots_adjust(left=0.09, right=0.98, bottom=0.22, top=0.80, wspace=0.25)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=100)
        plt.close(fig)
    return output


def _pattern_legend_handles() -> list[Any]:
    """Build the one shared two-entry legend used by both forest plots."""
    from matplotlib.lines import Line2D

    return [
        Line2D(
            [0],
            [0],
            color=color,
            marker="o",
            linestyle="none",
            markersize=7,
            label=pattern,
        )
        for pattern, color in PATTERN_COLORS.items()
    ]


def export_overview(matrix_root: Path, output_dir: Path) -> dict[str, Path]:
    """Export validated 72-row summary data and two delivery-mode SG forests."""
    summaries = load_validated_matrix(matrix_root)
    rows = _statistics_rows(summaries)
    if len(rows) != 72:
        raise ValueError(f"expected 72 club-pattern rows, received {len(rows)}")
    paths = _write_statistics(
        rows,
        Path(output_dir),
        summaries=summaries,
        matrix_root=Path(matrix_root),
    )
    paths["fixed_loft_forest_png"] = _plot_scoring_forest(
        rows,
        delivery_mode="fixed_loft",
        output=Path(output_dir) / "strokes_gained_fixed_loft.png",
    )
    paths["shaft_rotation_forest_png"] = _plot_scoring_forest(
        rows,
        delivery_mode="shaft_rotation",
        output=Path(output_dir) / "strokes_gained_shaft_rotation.png",
    )
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and summarize a completed 24-cell shot-pattern matrix."
    )
    parser.add_argument("matrix_root", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = export_overview(args.matrix_root, args.output_dir)
    for path in paths.values():
        sys.stdout.write(f"{path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
