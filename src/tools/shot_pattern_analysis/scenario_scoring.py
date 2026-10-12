"""Hypothetical driver tee and iron approach scoring of corrected shot bundles."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .course_score_vectors import score_course_endpoint_vectors
from .scoring import (
    SOURCE_PDF_SHA256,
    YARDS_PER_METRE,
    build_broadie_approx_baseline,
    load_pattern_endpoints,
)
from .provenance import sha256_file
from .scoring_cache import source_backed_score_cache
from .uncertainty import (
    DEFAULT_BOOTSTRAP_SEED,
    PUBLICATION_RESAMPLES,
    paired_mean_difference,
)


def score_course_endpoints(
    endpoints: dict[str, list[tuple[int, float, float]]],
    *,
    club_id: str,
    target_x_m: float,
    hole_distance_m: float = 400.0,
    fairway_half_width_m: float = 15.0,
    green_radius_m: float = 15.0,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    resamples: int = PUBLICATION_RESAMPLES,
) -> dict:
    """Score paired aimed endpoints on an explicitly hypothetical flat hole.

    Driver: 400 m tee start, corridor ±15 m; fairway or rough at carry.
    Irons/PW: fairway start at nominal straight carry; 15 m circular green,
    rough elsewhere. Every endpoint is a carry-only stop, without roll,
    hazards, elevation, or directional course features.
    """
    vectors = score_course_endpoint_vectors(
        endpoints,
        club_id=club_id,
        target_x_m=target_x_m,
        hole_distance_m=hole_distance_m,
        fairway_half_width_m=fairway_half_width_m,
        green_radius_m=green_radius_m,
    )
    driver = club_id == "driver"
    patterns: dict[str, dict[str, float | int]] = {
        name: {
            "n": len(shots),
            "mean_strokes_gained": float(np.mean(vectors.scores[name])),
            vectors.primary_fraction_name: vectors.primary_fraction[name],
        }
        for name, shots in endpoints.items()
    }
    paired = {
        name: paired_mean_difference(
            vectors.scores[name],
            vectors.scores["Straight"],
            seed=seed,
            resamples=resamples,
        )
        for name in ("Draw", "Fade")
    }
    return {
        "status": "available",
        "scenario_id": vectors.scenario_id,
        "scenario": (
            f"Historical PGA tee shot on a flat {hole_distance_m:g} m hole "
            f"with a {2 * fairway_half_width_m:g} m fairway corridor"
            if driver
            else "Historical PGA approach to a centered circular green with rough outside"
        ),
        "club_id": club_id,
        "start_lie": vectors.start_lie,
        "start_distance_m": vectors.start_distance_m,
        "target_x_m": target_x_m,
        "hole_distance_m": hole_distance_m if driver else None,
        "fairway_half_width_m": fairway_half_width_m if driver else None,
        "green_radius_m": green_radius_m if not driver else None,
        "source_backed_status": "available",
        "evaluation": "Public Tools API scored each baseline knot; endpoint scores are exact piecewise-linear interpolation of those returned values",
        "api_evaluated_states": vectors.api_evaluated_states,
        "scored_shots": sum(len(shots) for shots in endpoints.values()),
        "interpolation_tolerance_strokes": 1e-12,
        "baseline": vectors.baseline,
        "patterns": patterns,
        "paired_benefit_vs_straight": paired,
        "limitations": [
            "2003-2010 PGA benchmark and approximate reconciled putting fit; not a current or player-specific estimate.",
            "Flat hypothetical geometry; landing is treated as final lie with no roll, hazards, or recovery obstruction.",
            "Bootstrap intervals include Monte Carlo sampling only, not club delivery, course, or model uncertainty.",
        ],
    }


def _add_sensitivities(
    report: dict, endpoints: dict, *, club_id: str, target: float, seed: int
) -> None:
    """Add supported course-geometry sensitivity results to a scored report."""

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


def score_corrected_bundle(output_dir: Path) -> Path:
    """Score a completed corrected-club bundle without new flight simulation."""
    output_dir = Path(output_dir)
    summary_path = output_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    club_id = summary["config"]["club_id"]
    target = float(summary["target_x_m"])
    seed = int(summary["config"]["seed"])
    endpoints = load_pattern_endpoints(output_dir / "shots.csv")
    try:
        report = score_course_endpoints(
            endpoints, club_id=club_id, target_x_m=target, seed=seed
        )
        _add_sensitivities(report, endpoints, club_id=club_id, target=target, seed=seed)
    except ValueError as exc:
        report = {"status": "unavailable", "reason": str(exc), "club_id": club_id}
    # Read and check everything that can fail before the first write, so a
    # failure leaves the bundle exactly as it was.
    baseline = build_broadie_approx_baseline()
    baseline_path = output_dir / "strokes_gained_baseline.json"
    baseline_bytes = (
        json.dumps(
            {
                "baseline": baseline.model_dump(mode="json"),
                "source_pdf_sha256": SOURCE_PDF_SHA256,
                "derivation": "Broadie 2011 Appendix A Table 9 factual tee/fairway/rough knots; approximate reconciled putting benchmark",
            },
            indent=2,
        )
        + "\n"
    ).encode("utf-8")
    report["baseline_artifact"] = baseline_path.name
    report_path = output_dir / "strokes_gained.json"
    summary["course_scoring"] = report
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
        "baseline_artifact_sha256": hashlib.sha256(baseline_bytes).hexdigest(),
        "source_pdf_sha256": SOURCE_PDF_SHA256,
        "scoring_source_sha256": sha256_file(Path(__file__)),
        "scoring_cache_source_sha256": sha256_file(
            Path(__file__).parent / "scoring_cache.py"
        ),
        "baseline_builder_source_sha256": sha256_file(
            Path(__file__).parent / "scoring.py"
        ),
        "provider_commit": provider_commit,
        "provider_source_sha256": sha256_file(provider_source),
    }
    _replace_files_atomically(
        {
            baseline_path: baseline_bytes,
            report_path: _json_bytes(report),
            summary_path: _json_bytes(summary),
            receipt_path: _json_bytes(receipt),
        }
    )
    return report_path


def _json_bytes(payload: dict) -> bytes:
    return (json.dumps(payload, indent=2) + "\n").encode("utf-8")


def _replace_files_atomically(contents: dict[Path, bytes]) -> None:
    """Stage every file first, then swap each into place with ``os.replace``.

    A failure while staging leaves every destination untouched.
    """
    staged: list[tuple[Path, Path]] = []
    try:
        for destination, data in contents.items():
            handle, staging_name = tempfile.mkstemp(
                prefix=f".{destination.name}.", dir=destination.parent
            )
            staged.append((Path(staging_name), destination))
            with os.fdopen(handle, "wb") as stream:
                stream.write(data)
        for temporary, destination in staged:
            os.replace(temporary, destination)
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)
