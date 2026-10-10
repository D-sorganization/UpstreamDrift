"""Hypothetical driver tee and iron approach scoring of corrected shot bundles."""

from __future__ import annotations

import math
import csv
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from .scoring import (
    SOURCE_PDF_SHA256,
    YARDS_PER_METRE,
    build_broadie_approx_baseline,
    load_pattern_endpoints,
)
from .scoring_cache import source_backed_score_cache
from .uncertainty import paired_mean_difference


def score_course_endpoints(
    endpoints: dict[str, list[tuple[int, float, float]]],
    *,
    club_id: str,
    target_x_m: float,
    hole_distance_m: float = 400.0,
    fairway_half_width_m: float = 15.0,
    green_radius_m: float = 15.0,
    seed: int = 20261008,
) -> dict:
    """Score paired aimed endpoints on an explicitly hypothetical flat hole.

    Driver: 400 m tee start, corridor ±15 m; fairway or rough at carry.
    Irons/PW: fairway start at nominal straight carry; 15 m circular green,
    rough elsewhere. Every endpoint is a carry-only stop, without roll,
    hazards, elevation, or directional course features.
    """
    if club_id not in {"driver", "seven_iron", "pitching_wedge"}:
        raise ValueError("club_id must be a named illustrative preset")
    dimensions = (target_x_m, hole_distance_m, fairway_half_width_m, green_radius_m)
    if not all(math.isfinite(value) and value > 0 for value in dimensions):
        raise ValueError("course dimensions must be finite and positive")
    if set(endpoints) != {"Straight", "Draw", "Fade"}:
        raise ValueError("all three paired patterns are required")
    indices = [index for index, _, _ in endpoints["Straight"]]
    if (
        len(indices) < 2
        or len(indices) != len(set(indices))
        or any(not isinstance(index, int) or index < 0 for index in indices)
    ):
        raise ValueError("at least two distinct paired shot indices are required")
    for shots in endpoints.values():
        if [index for index, _, _ in shots] != indices:
            raise ValueError("patterns must share ordered shot indices")
        if any(not math.isfinite(x) or not math.isfinite(y) for _, x, y in shots):
            raise ValueError("endpoints must be finite")

    driver = club_id == "driver"
    start_lie = "tee" if driver else "fairway"
    start_distance_m = hole_distance_m if driver else target_x_m
    finish_lies = ("fairway", "rough") if driver else ("green", "rough")
    baseline = build_broadie_approx_baseline()
    try:
        cache = source_backed_score_cache(
            baseline,
            start_lie=start_lie,
            start_distance_yards=start_distance_m * YARDS_PER_METRE,
            finish_lies=finish_lies,
        )
        scores: dict[str, np.ndarray] = {}
        patterns: dict[str, dict[str, float | int]] = {}
        for name, shots in endpoints.items():
            xy = np.array([(x, y) for _, x, y in shots], dtype=float)
            if driver:
                fair = np.abs(xy[:, 1]) <= fairway_half_width_m
                primary_lie = "fairway"
                primary_name = "fairway_fraction"
                distances = np.hypot(hole_distance_m - xy[:, 0], xy[:, 1])
            else:
                distances = np.hypot(xy[:, 0] - target_x_m, xy[:, 1])
                fair = distances <= green_radius_m
                primary_lie = "green"
                primary_name = "green_fraction"
            distances_yards = distances * YARDS_PER_METRE
            shot_scores = np.empty(len(shots))
            for lie, mask in ((primary_lie, fair), ("rough", ~fair)):
                if bool(np.any(mask)):
                    shot_scores[mask] = cache.score_many(lie, distances_yards[mask])
            scores[name] = shot_scores
            patterns[name] = {
                "n": len(shots),
                "mean_strokes_gained": float(np.mean(shot_scores)),
                primary_name: float(np.mean(fair)),
            }
    except ValueError as exc:
        raise ValueError(
            f"course state outside published baseline support: {exc}"
        ) from exc
    paired = {
        name: paired_mean_difference(scores[name], scores["Straight"], seed=seed)
        for name in ("Draw", "Fade")
    }
    return {
        "status": "available",
        "scenario_id": "tee_fairway_rough" if driver else "approach_green_rough",
        "scenario": (
            f"Historical PGA tee shot on a flat {hole_distance_m:g} m hole "
            f"with a {2 * fairway_half_width_m:g} m fairway corridor"
            if driver
            else "Historical PGA approach to a centered circular green with rough outside"
        ),
        "club_id": club_id,
        "start_lie": start_lie,
        "start_distance_m": start_distance_m,
        "target_x_m": target_x_m,
        "hole_distance_m": hole_distance_m if driver else None,
        "fairway_half_width_m": fairway_half_width_m if driver else None,
        "green_radius_m": green_radius_m if not driver else None,
        "source_backed_status": "available",
        "evaluation": "Public Tools API scored each baseline knot; endpoint scores are exact piecewise-linear interpolation of those returned values",
        "api_evaluated_states": cache.api_evaluated_states,
        "scored_shots": sum(len(shots) for shots in endpoints.values()),
        "interpolation_tolerance_strokes": 1e-12,
        "baseline": {
            "id": baseline.baseline_id,
            "version": baseline.version,
            "table_sha256": baseline.table_sha256,
            "source_url": baseline.source_url,
        },
        "patterns": patterns,
        "paired_benefit_vs_straight": paired,
        "limitations": [
            "2003-2010 PGA benchmark and approximate reconciled putting fit; not a current or player-specific estimate.",
            "Flat hypothetical geometry; landing is treated as final lie with no roll, hazards, or recovery obstruction.",
            "Bootstrap intervals include Monte Carlo sampling only, not club delivery, course, or model uncertainty.",
        ],
    }


def score_corrected_bundle(output_dir: Path) -> Path:
    """Score a completed corrected-club bundle without new flight simulation."""
    output_dir = Path(output_dir)
    summary_path = output_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    club_id = summary["config"]["club_id"]
    target = float(summary["target_x_m"])
    seed = int(summary["config"]["seed"])
    endpoints = load_pattern_endpoints(output_dir / "shots.csv")

    def sensitivity(**changes: float) -> dict:
        try:
            return score_course_endpoints(
                endpoints, club_id=club_id, target_x_m=target, seed=seed, **changes
            )
        except ValueError as exc:
            return {"status": "unavailable", "reason": str(exc)}

    def compact(scenario: dict) -> dict:
        if scenario["status"] != "available":
            return scenario
        return {
            "status": "available",
            "scenario": scenario["scenario"],
            "patterns": scenario["patterns"],
            "paired_benefit_vs_straight": scenario["paired_benefit_vs_straight"],
            "api_evaluated_states": scenario["api_evaluated_states"],
        }

    try:
        report = score_course_endpoints(
            endpoints, club_id=club_id, target_x_m=target, seed=seed
        )
        if club_id == "driver":
            report["hole_distance_sensitivity_m"] = {
                f"{hole:g}": compact(
                    report if hole == 400.0 else sensitivity(hole_distance_m=hole)
                )
                for hole in (350.0, 400.0, 450.0)
            }
            report["fairway_width_sensitivity_m"] = {
                f"{2 * half:g}": compact(
                    report if half == 15.0 else sensitivity(fairway_half_width_m=half)
                )
                for half in (10.0, 15.0, 20.0)
            }
        else:
            report["green_radius_sensitivity_m"] = {
                f"{radius:g}": compact(
                    report if radius == 15.0 else sensitivity(green_radius_m=radius)
                )
                for radius in (10.0, 15.0, 20.0)
            }
        sensitivity_reports = (
            [
                item
                for key, item in report["hole_distance_sensitivity_m"].items()
                if key != "400"
            ]
            + [
                item
                for key, item in report["fairway_width_sensitivity_m"].items()
                if key != "30"
            ]
            if club_id == "driver"
            else [
                item
                for key, item in report["green_radius_sensitivity_m"].items()
                if key != "15"
            ]
        )
        report["total_api_evaluated_states"] = report["api_evaluated_states"] + sum(
            item.get("api_evaluated_states", 0) for item in sensitivity_reports
        )
    except ValueError as exc:
        report = {"status": "unavailable", "reason": str(exc), "club_id": club_id}
    baseline = build_broadie_approx_baseline()
    baseline_path = output_dir / "strokes_gained_baseline.json"
    baseline_path.write_text(
        json.dumps(
            {
                "baseline": baseline.model_dump(mode="json"),
                "source_pdf_sha256": SOURCE_PDF_SHA256,
                "derivation": "Broadie 2011 Appendix A Table 9 factual tee/fairway/rough knots; approximate reconciled putting benchmark",
            },
            indent=2,
        )
        + "\n"
    )
    report["baseline_artifact"] = baseline_path.name
    report_path = output_dir / "strokes_gained.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    summary["course_scoring"] = report
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    receipt_path = output_dir / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    repo_root = Path(__file__).resolve().parents[3]
    provider_root = repo_root / "vendor/ud-tools"
    provider_source = (
        provider_root / "src/shared/python/launch_monitor/strokes_gained.py"
    )
    provider_commit = subprocess.run(
        ["git", "-C", str(provider_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    receipt["scoring_postprocess"] = {
        "method": "Public Tools API at baseline knots; exact piecewise-linear interpolation of returned scores at endpoints",
        "api_evaluated_states": report.get("total_api_evaluated_states", 0),
        "scored_shots": report.get("scored_shots", 0),
        "baseline_artifact_sha256": hashlib.sha256(
            baseline_path.read_bytes()
        ).hexdigest(),
        "source_pdf_sha256": SOURCE_PDF_SHA256,
        "scoring_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "scoring_cache_source_sha256": hashlib.sha256(
            (Path(__file__).parent / "scoring_cache.py").read_bytes()
        ).hexdigest(),
        "baseline_builder_source_sha256": hashlib.sha256(
            (Path(__file__).parent / "scoring.py").read_bytes()
        ).hexdigest(),
        "provider_commit": provider_commit,
        "provider_source_sha256": hashlib.sha256(
            provider_source.read_bytes()
        ).hexdigest(),
    }
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return report_path
