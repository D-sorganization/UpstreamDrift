"""Audit saved pilot rows independently; no flight simulation or effect inference."""

from __future__ import annotations

import argparse
import collections
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from src.tools.launch_monitor_model import (
    CourseStateColumnsV1,
    StrokesGainedRequestV1,
    analyze_source_backed_strokes_gained,
)
from src.tools.shot_pattern_analysis.noise import (
    NOISE_FIELDS,
    NoiseSpec,
    sample_paired_noise,
)
from src.tools.shot_pattern_analysis.scoring import (
    YARDS_PER_METRE,
    build_broadie_approx_baseline,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pilot", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    plan = json.loads((args.pilot / "matrix_plan.json").read_text())
    counts = collections.Counter()
    reasons = collections.Counter()
    failures = []
    maximum = dict.fromkeys(
        ("statistics", "covariance", "noise", "nominal_residual", "direct_api_sg"), 0.0
    )
    hashes_checked = 0
    rows_checked = 0
    api_rows = 0
    cell_hashes = {}
    baseline = build_broadie_approx_baseline()
    for key, config in plan["cells"].items():
        folder = args.pilot / key
        try:
            summary = json.loads((folder / "summary.json").read_text())
            manifest = json.loads((folder / "manifest.json").read_text())
            for name, digest in manifest["files_sha256"].items():
                actual = hashlib.sha256((folder / name).read_bytes()).hexdigest()
                assert actual == digest, (key, name, "hash")
                hashes_checked += 1
            cell_hashes[key] = hashlib.sha256(
                (folder / "manifest.json").read_bytes()
            ).hexdigest()
            assert summary["config"] == config
            start = json.loads((folder / "run_start.json").read_text())
            assert start["config"] == config and start["execution"] == plan["execution"]
            counts[summary["status"]] += 1
            if summary["status"] == "unavailable":
                reasons[summary["reason"]] += 1
                assert (
                    summary["shot_rows"] == 0 and not (folder / "shots.csv.gz").exists()
                )
                continue
            receipt = json.loads((folder / "receipt.json").read_text())
            assert (
                receipt["execution"] == plan["execution"]
                and receipt["source_unchanged_during_run"]
            )
            assert not summary["aim_adjustment_applied"]
            with gzip.open(folder / "shots.csv.gz", "rt") as handle:
                rows = list(csv.DictReader(handle))
            rows_checked += len(rows)
            assert len(rows) == 3 * config["n_shots"] == receipt["shot_rows"]
            noise = sample_paired_noise(
                NoiseSpec(
                    tuple(config["noise"]["standard_deviations"]),
                    config["noise"]["correlation"],
                ),
                n_shots=config["n_shots"],
                seed=config["seed"],
            )
            api_data = []
            api_names = []
            for pattern in ("Straight", "Draw", "Fade"):
                selected = [row for row in rows if row["pattern"] == pattern]
                assert [int(row["shot_index"]) for row in selected] == list(
                    range(config["n_shots"])
                )
                nominal = summary["nominals"][pattern]
                residual = math.hypot(
                    nominal["outcome"]["carry_x_m"] - summary["target_x_m"],
                    nominal["outcome"]["carry_y_m"],
                )
                maximum["nominal_residual"] = max(maximum["nominal_residual"], residual)
                assert (
                    residual <= config["match_tolerance_m"]
                    and abs(residual - nominal["residual_m"]) < 1e-12
                )
                offsets = np.array(
                    [
                        0.0,
                        nominal["path_deg"],
                        nominal["speed_mps"],
                        0.0,
                        0.0,
                        config["club"]["shaft_lean_deg"],
                        config["club"]["attack_angle_deg"],
                    ]
                )
                observed = np.array(
                    [[float(row[field]) for field in NOISE_FIELDS] for row in selected]
                )
                maximum["noise"] = max(
                    maximum["noise"],
                    float(np.max(np.abs(observed - (noise + offsets)))),
                )
                supported = [row for row in selected if row["status"] == "supported"]
                saved = summary["summary"][pattern]
                assert (
                    len(supported) == saved["supported_n"]
                    and len(selected) == saved["total_n"]
                )
                xy = np.array(
                    [
                        [float(row["carry_x_m"]), float(row["carry_y_m"])]
                        for row in supported
                    ]
                )
                error = xy - np.array([summary["target_x_m"], 0.0])
                radial = np.linalg.norm(error, axis=1)
                actual = {
                    "lateral_sd_m": float(np.std(xy[:, 1], ddof=1)),
                    "downrange_sd_m": float(np.std(xy[:, 0], ddof=1)),
                    "mean_carry_m": float(np.linalg.norm(xy, axis=1).mean()),
                    "target_rmse_m": float(np.sqrt(np.mean(radial**2))),
                    "target_containment_fraction": float(
                        np.mean(radial <= config["target_radius_m"])
                    ),
                }
                maximum["statistics"] = max(
                    maximum["statistics"],
                    *(abs(value - saved[field]) for field, value in actual.items()),
                )
                diagnostics = saved["joint_delivery_outcome_diagnostics"]
                matrix = np.array(
                    [
                        [float(row[field]) for field in diagnostics["fields"]]
                        for row in supported
                    ]
                )
                maximum["covariance"] = max(
                    maximum["covariance"],
                    float(
                        np.max(
                            np.abs(
                                np.cov(matrix, rowvar=False)
                                - np.array(diagnostics["covariance"])
                            )
                        )
                    ),
                )
                if key.endswith("lean0_curve1_rotation1_combined2"):
                    driver = config["club"]["club_id"] == "driver"
                    for row in supported:
                        x, y = float(row["carry_x_m"]), float(row["carry_y_m"])
                        distance = (
                            math.hypot(400.0 - x, y)
                            if driver
                            else math.hypot(x - summary["target_x_m"], y)
                        )
                        primary = (
                            abs(y) <= 15.0
                            if driver
                            else distance <= config["target_radius_m"]
                        )
                        api_data.append(
                            {
                                "start_lie": "tee" if driver else "fairway",
                                "finish_lie": ("fairway" if driver else "green")
                                if primary
                                else "rough",
                                "start_distance": (
                                    400.0 if driver else summary["target_x_m"]
                                )
                                * YARDS_PER_METRE,
                                "finish_distance": distance * YARDS_PER_METRE,
                                "context": "standard",
                                "target": "hole",
                            }
                        )
                        api_names.append(pattern)
            if api_data:
                common = {
                    "context_column": "context",
                    "target_column": "target",
                    "distance_unit": "yd",
                }
                request = StrokesGainedRequestV1(
                    start=CourseStateColumnsV1(
                        lie_column="start_lie",
                        distance_column="start_distance",
                        **common,
                    ),
                    finish=CourseStateColumnsV1(
                        lie_column="finish_lie",
                        distance_column="finish_distance",
                        **common,
                    ),
                    min_samples=1,
                )
                scored = analyze_source_backed_strokes_gained(
                    pd.DataFrame(api_data), baseline, request
                )
                assert scored.exclusions.total_excluded == 0
                api_rows += len(scored.row_results)
                for pattern in ("Straight", "Draw", "Fade"):
                    mean = float(
                        np.mean(
                            [
                                row.strokes_gained
                                for row in scored.row_results
                                if api_names[row.source_index] == pattern
                            ]
                        )
                    )
                    maximum["direct_api_sg"] = max(
                        maximum["direct_api_sg"],
                        abs(
                            mean
                            - summary["course_scoring"]["patterns"][pattern][
                                "mean_strokes_gained"
                            ]
                        ),
                    )
        except (AssertionError, KeyError, ValueError, OSError) as exc:
            failures.append({"cell": key, "error": repr(exc)})
    limits = {
        "statistics": 1e-10,
        "covariance": 1e-8,
        "noise": 1e-12,
        "nominal_residual": 0.05,
        "direct_api_sg": 1e-12,
    }
    result = {
        "scope": "Pilot integrity only; no final scientific effect inference",
        "plan_sha256": hashlib.sha256(
            (args.pilot / "matrix_plan.json").read_bytes()
        ).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "counts": dict(counts),
        "unavailable_reasons": dict(reasons),
        "planned_cells": len(plan["cells"]),
        "rows_checked": rows_checked,
        "manifest_files_checked": hashes_checked,
        "direct_api_rows": api_rows,
        "maxima": maximum,
        "limits": limits,
        "failures": failures,
        "pass": not failures
        and all(maximum[k] <= limit for k, limit in limits.items()),
        "cell_manifest_sha256": cell_hashes,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    sys.stdout.write(
        json.dumps(
            {k: v for k, v in result.items() if k != "cell_manifest_sha256"}, indent=2
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
