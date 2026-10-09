"""Cell-bounded final-data audit; no flights and no inference from partial runs.

Keep final evidence and acceptance files immutable. This script writes only its
requested audit receipt; a partial receipt can never report final acceptance.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest

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
from src.tools.shot_pattern_analysis.scoring_cache import source_backed_score_cache

AUDIT_VERSION = "independent-saved-data/2"
PATTERNS = ("Straight", "Draw", "Fade")
LIMITS = {
    "input": 1e-12,
    "summary": 1e-10,
    "covariance": 1e-8,
    "correlation": 1e-10,
    "paired": 1e-9,
    "scoring": 1e-12,
    "direct_api": 1e-12,
    "bootstrap": 1e-8,
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(maxima, kind, actual, expected):
    difference = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
    maxima[kind] = max(maxima[kind], difference)
    assert math.isfinite(difference) and difference <= LIMITS[kind], (
        kind,
        difference,
        LIMITS[kind],
    )


def covariance_check(frame, saved, maxima):
    for field in ("covariance", "correlation", "pair_counts"):
        recorded = np.asarray(saved[field], dtype=float)
        assert np.allclose(recorded, recorded.T, atol=1e-12, rtol=0, equal_nan=True)
    values = frame[saved["fields"]].to_numpy(float)
    assert saved["n"] == len(frame)
    for i in range(values.shape[1]):
        for j in range(i, values.shape[1]):
            valid = np.isfinite(values[:, i]) & np.isfinite(values[:, j])
            a, b = values[valid, i], values[valid, j]
            assert saved["pair_counts"][i][j] == len(a)
            if len(a) < 2:
                assert (
                    saved["covariance"][i][j] is None
                    and saved["correlation"][i][j] is None
                )
                continue
            compare(
                maxima,
                "covariance",
                np.cov(a, b, ddof=1)[0, 1],
                saved["covariance"][i][j],
            )
            if np.std(a) > 1e-14 and np.std(b) > 1e-14:
                compare(
                    maxima,
                    "correlation",
                    np.corrcoef(a, b)[0, 1],
                    saved["correlation"][i][j],
                )
            else:
                assert saved["correlation"][i][j] is None


def score_arrays(frames, common, summary, caches, baseline, maxima, direct):
    config = summary["config"]
    driver = config["club"]["club_id"] == "driver"
    start = 400.0 if driver else summary["target_x_m"]
    cachekey = (driver, start)
    if cachekey not in caches:
        caches[cachekey] = source_backed_score_cache(
            baseline,
            start_lie="tee" if driver else "fairway",
            start_distance_yards=start * YARDS_PER_METRE,
            finish_lies=("fairway", "rough") if driver else ("green", "rough"),
        )
    cache = caches[cachekey]
    scores = {}
    api_rows = []
    api_expected = []
    for pattern, frame in frames.items():
        xy = frame.loc[common, ["carry_x_m", "carry_y_m"]].to_numpy(float)
        distance = (
            np.hypot(400.0 - xy[:, 0], xy[:, 1])
            if driver
            else np.hypot(xy[:, 0] - summary["target_x_m"], xy[:, 1])
        )
        primary = (
            np.abs(xy[:, 1]) <= 15.0
            if driver
            else distance <= config["target_radius_m"]
        )
        lies = np.where(primary, "fairway" if driver else "green", "rough")
        values = np.empty(len(common))
        for lie in set(lies):
            mask = lies == lie
            values[mask] = cache.score_many(lie, distance[mask] * YARDS_PER_METRE)
        scores[pattern] = values
        stored = summary["course_scoring"]["patterns"][pattern]
        assert stored["n"] == len(common)
        compare(maxima, "scoring", values.mean(), stored["mean_strokes_gained"])
        compare(
            maxima,
            "summary",
            primary.mean(),
            stored["fairway_fraction" if driver else "green_fraction"],
        )
        if direct:
            selected = np.unique(
                np.r_[
                    np.linspace(0, len(common) - 1, min(64, len(common)), dtype=int),
                    np.argmin(distance),
                    np.argmax(distance),
                ]
            )
            for i in selected:
                api_rows.append(
                    {
                        "start_lie": "tee" if driver else "fairway",
                        "finish_lie": str(lies[i]),
                        "start_distance": start * YARDS_PER_METRE,
                        "finish_distance": distance[i] * YARDS_PER_METRE,
                        "context": "standard",
                        "target": "hole",
                    }
                )
                api_expected.append(values[i])
    if api_rows:
        columns = {
            "context_column": "context",
            "target_column": "target",
            "distance_unit": "yd",
        }
        request = StrokesGainedRequestV1(
            start=CourseStateColumnsV1(
                lie_column="start_lie", distance_column="start_distance", **columns
            ),
            finish=CourseStateColumnsV1(
                lie_column="finish_lie", distance_column="finish_distance", **columns
            ),
            min_samples=1,
        )
        result = analyze_source_backed_strokes_gained(
            pd.DataFrame(api_rows), baseline, request
        )
        assert result.exclusions.total_excluded == 0 and len(result.row_results) == len(
            api_rows
        )
        for row in result.row_results:
            compare(
                maxima, "direct_api", row.strokes_gained, api_expected[row.source_index]
            )
    return scores, len(api_rows)


def bootstrap_check(differences, saved, seed, maxima):
    """Independent batched paired resampling, sharing rows across all metrics."""
    matrix = np.column_stack(differences)
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(25):
        indices = rng.integers(0, len(matrix), size=(20, len(matrix)))
        means.append(matrix[indices].mean(axis=1))
    bounds = np.quantile(np.concatenate(means), [0.025, 0.975], axis=0)
    for i, item in enumerate(saved):
        compare(maxima, "bootstrap", bounds[:, i], [item["lower_95"], item["upper_95"]])


def audit_cell(folder, config, execution, caches, baseline, maxima):
    summary = json.loads((folder / "summary.json").read_text())
    manifest = json.loads((folder / "manifest.json").read_text())
    for name, digest in manifest["files_sha256"].items():
        assert sha(folder / name) == digest, (folder.name, name, "hash")
    start = json.loads((folder / "run_start.json").read_text())
    assert (
        start["config"] == config == summary["config"]
        and start["execution"] == execution
    )
    result = {
        "status": summary["status"],
        "manifest_sha256": sha(folder / "manifest.json"),
        "files_checked": len(manifest["files_sha256"]),
    }
    if summary["status"] == "unavailable":
        assert summary["shot_rows"] == 0 and not (folder / "shots.csv.gz").exists()
        return {**result, "reason": summary["reason"], "rows": 0}
    assert (
        summary["status"] == "completed" and summary["aim_adjustment_applied"] is False
    )
    receipt = json.loads((folder / "receipt.json").read_text())
    assert (
        receipt["execution"] == execution
        and receipt["source_unchanged_during_run"]
        and receipt["all_inputs_retained"]
    )
    data = pd.read_csv(folder / "shots.csv.gz", float_precision="round_trip")
    assert set(data["status"]) <= {
        "supported",
        "off_face",
        "delivery_out_of_range",
        "delivery_nonfinite",
        "delivery_nonapproaching",
        "nonairborne_launch",
    }
    n = config["n_shots"]
    assert len(data) == 3 * n == receipt["shot_rows"]
    assert set(data["pattern"]) == set(PATTERNS)
    noise = sample_paired_noise(
        NoiseSpec(**config["noise"]), n_shots=n, seed=config["seed"]
    )
    input_info = summary["input_diagnostics"]
    assert input_info["fields"] == list(NOISE_FIELDS) and input_info["paired_inputs"]
    compare(
        maxima,
        "covariance",
        np.cov(noise, rowvar=False),
        input_info["realized_covariance"],
    )
    compare(
        maxima,
        "summary",
        np.std(noise, axis=0, ddof=1),
        input_info["realized_standard_deviations"],
    )
    frames = {}
    supports = {}
    radii = {}
    target = summary["target_x_m"]
    radius = config["target_radius_m"]
    for pattern in PATTERNS:
        frame = data.loc[data["pattern"] == pattern].set_index("shot_index")
        assert frame.index.tolist() == list(range(n))
        nominal = summary["nominals"][pattern]
        residual = math.hypot(
            nominal["outcome"]["carry_x_m"] - target, nominal["outcome"]["carry_y_m"]
        )
        assert residual <= config["match_tolerance_m"]
        compare(maxima, "summary", residual, nominal["residual_m"])
        offsets = [
            0.0,
            nominal["path_deg"],
            nominal["speed_mps"],
            0.0,
            0.0,
            config["club"]["shaft_lean_deg"],
            config["club"]["attack_angle_deg"],
        ]
        compare(
            maxima, "input", frame[list(NOISE_FIELDS)].to_numpy(float), noise + offsets
        )
        valid = frame["status"] == "supported"
        good = frame.loc[valid]
        saved = summary["summary"][pattern]
        assert saved["total_n"] == n and saved["supported_n"] == len(good)
        assert (
            dict(Counter(frame.loc[~valid, "status"])) == saved["unsupported_reasons"]
        )
        compare(maxima, "summary", len(good) / n, saved["supported_fraction"])
        interval = binomtest(len(good), n).proportion_ci(
            confidence_level=0.95, method="exact"
        )
        compare(
            maxima,
            "summary",
            [interval.low, interval.high],
            saved["supported_fraction_exact95"],
        )
        assert saved["statistics_conditioning"] == (
            "all_samples" if len(good) == n else "supported_airborne_only"
        )
        frames[pattern] = frame
        supports[pattern] = set(good.index)
        if len(good) < 2:
            assert saved["statistics"] is None
            continue
        xy = good[["carry_x_m", "carry_y_m"]].to_numpy(float)
        error = xy - [target, 0.0]
        radial = np.linalg.norm(error, axis=1)
        actual = {
            "lateral_sd_m": np.std(xy[:, 1], ddof=1),
            "downrange_sd_m": np.std(xy[:, 0], ddof=1),
            "mean_carry_m": np.hypot(*xy.T).mean(),
            "target_rmse_m": np.sqrt(np.mean(radial**2)),
            "target_containment_fraction": np.mean(radial <= radius),
        }
        for field, value in actual.items():
            compare(maxima, "summary", value, saved[field])
        stats = saved["statistics"]
        compare(
            maxima,
            "summary",
            np.quantile(radial, [0.5, 0.95]),
            [stats["median_radial_target_error_m"], stats["p95_radial_target_error_m"]],
        )
        compare(
            maxima,
            "summary",
            error.mean(axis=0),
            [stats["mean_downrange_error_m"], stats["mean_lateral_error_m"]],
        )
        covariance_check(good, saved["joint_delivery_outcome_diagnostics"], maxima)
    common = sorted(set.intersection(*supports.values()))
    paired = summary["paired_comparisons"]
    scoring = summary["course_scoring"]
    for saved in (paired, scoring):
        assert saved["paired_n"] == len(common) and saved["total_n"] == n
        assert saved["conditioning"] == (
            "all_samples" if len(common) == n else "common_supported_indices_only"
        )
    result.update(
        rows=len(data),
        supported_rows=sum(map(len, supports.values())),
        paired_n=len(common),
        total_n=n,
    )
    if len(common) < 2:
        assert paired["status"] == scoring["status"] == "unavailable"
        return result
    selected = (
        "_curve1_rotation1_combined2" in folder.name
        or "_curve2_rotation2_shaft_only" in folder.name
        or "_curve2_rotation2_shaft_strike" in folder.name
    )
    differences = []
    intervals = []
    for name, frame in frames.items():
        radii[name] = np.hypot(
            frame.loc[common, "carry_x_m"].to_numpy() - target,
            frame.loc[common, "carry_y_m"].to_numpy(),
        )
    for metric, transform in (
        ("mean_radial_error_difference_vs_straight", lambda x: x),
        ("mean_squared_target_error_difference_m2", lambda x: x * x),
        ("containment_fraction_difference", lambda x: (x <= radius).astype(float)),
    ):
        for name in ("Draw", "Fade"):
            delta = transform(radii[name]) - transform(radii["Straight"])
            saved = paired[metric][name]
            compare(maxima, "paired", delta.mean(), saved["estimate"])
            differences.append(delta)
            intervals.append(saved)
    if scoring["status"] in ("available", "conditional_only"):
        assert scoring["status"] == (
            "available" if len(common) == n else "conditional_only"
        )
        assert scoring["baseline"]["table_sha256"] == baseline.table_sha256
        if len(common) < n:
            assert (
                scoring["full_population_status"]
                == "unavailable_unsupported_terminal_lies"
            )
        scores, count = score_arrays(
            frames, common, summary, caches, baseline, maxima, selected
        )
        result["direct_api_rows"] = count
        for name in ("Draw", "Fade"):
            delta = scores[name] - scores["Straight"]
            saved = scoring["paired_benefit_vs_straight"][name]
            compare(maxima, "scoring", delta.mean(), saved["estimate"])
            differences.append(delta)
            intervals.append(saved)
    else:
        assert scoring["status"] == "unavailable" and scoring.get("reason")
        result["scoring_unavailable_reason"] = scoring["reason"]
    if selected:
        bootstrap_check(differences, intervals, config["seed"], maxima)
    result["paired_comparisons"] = paired
    result["course_scoring_status"] = scoring["status"]
    result["unsupported_reasons"] = {
        name: summary["summary"][name]["unsupported_reasons"] for name in PATTERNS
    }
    result["bootstrap_checked"] = selected
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--max-cells", type=int)
    parser.add_argument("--supplement-stage", choices=("pilot", "final"))
    parser.add_argument("--alias-references", type=Path)
    parser.add_argument("--supplement-acceptance", type=Path)
    args = parser.parse_args()
    from docs.research.shot_pattern_analysis.phase2.validation.supplement_audit_protocol import (
        load_json_strict,
        validate_supplement_plan,
        validate_control_aliases,
        validate_execution_acceptance,
    )

    plan = load_json_strict(args.root / "matrix_plan.json")
    evidence = Path(__file__).resolve().parents[1]
    protocol_review = None
    if "supplement_protocol" in plan:
        protocol_review = validate_supplement_plan(
            plan, args.supplement_stage, evidence
        )
    else:
        assert args.supplement_stage is None and args.alias_references is None
        assert args.supplement_acceptance is None
        assert len(plan["cells"]) in (72, 324), "unexpected factorial plan size"
    acceptance_review = None
    if protocol_review is not None and args.supplement_stage == "final":
        acceptance_review = validate_execution_acceptance(
            plan, args.supplement_acceptance, evidence
        )
    alias_review = None
    if args.alias_references is not None:
        alias_review = validate_control_aliases(
            args.alias_references, args.supplement_stage, evidence
        )
    repo = Path(__file__).resolve().parents[5]
    checked_sources = (
        "src/tools/shot_pattern_analysis/noise.py",
        "src/tools/shot_pattern_analysis/scoring.py",
        "src/tools/shot_pattern_analysis/scoring_cache.py",
        "vendor/ud-tools/src/shared/python/launch_monitor/strokes_gained.py",
    )
    for name in checked_sources:
        assert sha(repo / name) == plan["execution"]["source_sha256"][name], name
    maxima = dict.fromkeys(LIMITS, 0.0)
    results: dict[str, Any] = {}
    failures = []
    caches: dict[Any, Any] = {}
    baseline = build_broadie_approx_baseline()
    for key, config in plan["cells"].items():
        if args.max_cells is not None and len(results) >= args.max_cells:
            break
        folder = args.root / key
        if not (folder / "manifest.json").is_file():
            continue
        try:
            results[key] = audit_cell(
                folder, config, plan["execution"], caches, baseline, maxima
            )
        except (AssertionError, KeyError, ValueError, OSError, TypeError) as exc:
            failures.append({"cell": key, "error": repr(exc)})
    complete = len(results) == len(plan["cells"]) and not failures
    receipt = {
        "scope": "Independent per-plan saved-data audit; partial runs do not establish final results",
        "audit_version": AUDIT_VERSION,
        "data_stage": plan["stage"],
        "protocol_review": protocol_review,
        "alias_review": alias_review,
        "acceptance_review": acceptance_review,
        "protocol_auditor_sha256": sha(
            Path(__file__).with_name("supplement_audit_protocol.py")
        ),
        "execution": plan["execution"],
        "plan_sha256": sha(args.root / "matrix_plan.json"),
        "script_sha256": sha(Path(__file__)),
        "audit_dependency_sha256": {name: sha(repo / name) for name in checked_sources},
        "planned_cells": len(plan["cells"]),
        "audited_cells": len(results),
        "complete": complete,
        "pass": complete and not failures,
        "partial_checks_pass": not failures,
        "maxima": maxima,
        "limits": LIMITS,
        "failures": failures,
        "cells": results,
    }
    args.output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    sys.stdout.write(
        json.dumps(
            {k: v for k, v in receipt.items() if k not in ("cells", "execution")},
            indent=2,
        )
        + "\n"
    )
    if failures or (not complete and not args.allow_partial):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
