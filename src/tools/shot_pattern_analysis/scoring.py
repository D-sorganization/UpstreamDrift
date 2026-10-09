"""Source-backed, explicitly approximate historical PGA approach scenario."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.tools.launch_monitor_model import (
    CourseStateColumnsV1,
    ExpectedStrokesBaselineV2,
    ExpectedStrokesStateV2,
    StrokesGainedRequestV1,
    analyze_source_backed_strokes_gained,
    baseline_table_sha256,
)

from .uncertainty import paired_mean_difference

SOURCE_URL = "https://www.columbia.edu/~mnb2/broadie/Assets/strokes_gained_pga_broadie_20110408.pdf"
SOURCE_PDF_SHA256 = "f22c305ae47f3f2973882d16852b6f43dbde577c5e1c11b12bcb19d2956699e7"
YARDS_PER_METRE = 1.0936132983377078
_TEE_TABLE = {
    100: 2.92,
    120: 2.99,
    140: 2.97,
    160: 2.99,
    180: 3.05,
    200: 3.12,
    220: 3.17,
    240: 3.25,
    260: 3.45,
    280: 3.65,
    300: 3.71,
    320: 3.79,
    340: 3.86,
    360: 3.92,
    380: 3.96,
    400: 3.99,
    420: 4.02,
    440: 4.08,
    460: 4.17,
    480: 4.28,
    500: 4.41,
    520: 4.54,
    540: 4.65,
    560: 4.74,
    580: 4.79,
    600: 4.82,
}
_FAIRWAY_TABLE = {
    10: 2.18,
    20: 2.40,
    30: 2.52,
    40: 2.60,
    50: 2.66,
    60: 2.70,
    70: 2.72,
    80: 2.75,
    90: 2.77,
    100: 2.80,
    120: 2.85,
    140: 2.91,
    160: 2.98,
    180: 3.08,
    200: 3.19,
    220: 3.32,
    240: 3.45,
    260: 3.58,
    280: 3.69,
    300: 3.78,
    320: 3.84,
    340: 3.88,
    360: 3.95,
    380: 4.03,
    400: 4.11,
    420: 4.19,
    440: 4.27,
    460: 4.34,
    480: 4.42,
    500: 4.50,
    520: 4.58,
    540: 4.66,
    560: 4.74,
    580: 4.82,
    600: 4.89,
}
_ROUGH_TABLE = {
    10: 2.34,
    20: 2.59,
    30: 2.70,
    40: 2.78,
    50: 2.87,
    60: 2.91,
    70: 2.93,
    80: 2.96,
    90: 2.99,
    100: 3.02,
    120: 3.08,
    140: 3.15,
    160: 3.23,
    180: 3.31,
    200: 3.42,
    220: 3.53,
    240: 3.64,
    260: 3.74,
    280: 3.83,
    300: 3.90,
    320: 3.95,
    340: 4.02,
    360: 4.11,
    380: 4.21,
    400: 4.30,
    420: 4.40,
    440: 4.49,
    460: 4.58,
    480: 4.68,
    500: 4.77,
    520: 4.87,
    540: 4.96,
    560: 5.06,
    580: 5.15,
    600: 5.25,
}
_HOLE_RADIUS_YD = 2.125 / 36


class ScoringUnavailable(ValueError):
    """A valid flight analysis lies outside the optional scoring scenario."""


def _normal_cdf(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def putt_probabilities(distance_yards: float) -> tuple[float, float]:
    """Return paper Eq. 5 p1 and an anchor-reconciled approximation to p3.

    The paper prints Eq. 6 coefficients in an order that yields invalid
    negative p3 near 33 ft. The reordered expression below uses feet and
    reproduces its stated 40-ft/10% and 33-ft/two-putt anchors. It is an
    approximation, not an exact transcription of the published fit.
    """
    if not math.isfinite(distance_yards) or distance_yards < 0:
        raise ValueError("putt distance must be finite and nonnegative")
    sigma_angle = math.radians(1.46)
    sigma_distance = 0.057
    target_past_yards = 0.5
    holeout_depth_yards = 2 / 3
    critical_angle = math.atan2(_HOLE_RADIUS_YD, distance_yards)
    direction = 2 * _normal_cdf(critical_angle / sigma_angle) - 1
    denominator = sigma_distance * (distance_yards + target_past_yards)
    distance = _normal_cdf((holeout_depth_yards - target_past_yards) / denominator)
    distance -= _normal_cdf(-target_past_yards / denominator)
    one_putt = direction * distance
    feet = 3 * distance_yards
    exponent = 5.49 - 0.106 * feet + 0.000563 * feet * feet
    three_putt = 1 / (1 + math.exp(exponent)) - 0.00398
    # Independently fitted curves can overlap by a tiny amount at short range.
    # Bound the residual probability assigned to three putts by 1 - p1.
    three_putt = min(three_putt, 1 - one_putt)
    if not (0 <= one_putt <= 1 and 0 <= three_putt <= 1 and one_putt + three_putt <= 1):
        raise ValueError("putting approximation outside probability bounds")
    return one_putt, three_putt


def green_expected_strokes(distance_yards: float) -> float:
    """Equation 7 for an unholed ball, with the approximate three-putt curve."""
    if not math.isfinite(distance_yards) or distance_yards < 0:
        raise ValueError("putt distance must be finite and nonnegative")
    if distance_yards == 0:
        return 1.0  # The endpoint is an unholed tap-in, not a made shot.
    one_putt, three_putt = putt_probabilities(distance_yards)
    return 2 - one_putt + three_putt


def build_broadie_approx_baseline() -> ExpectedStrokesBaselineV2:
    """Build a digest-checked benchmark from Table 9 and Eqs. 5–7."""
    states = [
        ExpectedStrokesStateV2(
            lie="tee",
            context="standard",
            target="hole",
            distance_yards=float(distance),
            expected_strokes=strokes,
        )
        for distance, strokes in _TEE_TABLE.items()
    ]
    states += [
        ExpectedStrokesStateV2(
            lie="fairway",
            context="standard",
            target="hole",
            distance_yards=float(distance),
            expected_strokes=strokes,
        )
        for distance, strokes in _FAIRWAY_TABLE.items()
    ]
    states += [
        ExpectedStrokesStateV2(
            lie="rough",
            context="standard",
            target="hole",
            distance_yards=float(distance),
            expected_strokes=strokes,
        )
        for distance, strokes in _ROUGH_TABLE.items()
    ]
    states += [
        ExpectedStrokesStateV2(
            lie="green",
            context="standard",
            target="hole",
            distance_yards=feet / 3,
            expected_strokes=green_expected_strokes(feet / 3),
        )
        for feet in (0, 0.17708333333333334, *range(1, 76))
    ]
    points = tuple(states)
    return ExpectedStrokesBaselineV2(
        baseline_id="broadie-2011-historical-tour-approx",
        version="table9-full-plus-anchor-reconciled-putting/2",
        source_url=SOURCE_URL,
        license="Factual values transcribed for analysis; no publication license asserted",
        table_sha256=baseline_table_sha256(points),
        states=points,
    )


def score_approach_endpoints(
    endpoints: dict[str, list[tuple[int, float, float]]],
    *,
    target_x_m: float,
    green_radius_m: float,
    seed: int = 20261008,
) -> dict:
    """Score aimed carry endpoints as an approach to a centered circular green.

    The ball is assumed to stop at carry; all area outside the green is rough.
    This hypothetical course has no roll, hazards, slope, or wind.
    """
    if (
        not math.isfinite(target_x_m)
        or target_x_m <= 0
        or not math.isfinite(green_radius_m)
        or green_radius_m <= 0
    ):
        raise ScoringUnavailable(
            "target distance and green radius must be positive finite"
        )
    start_yards = target_x_m * YARDS_PER_METRE
    if not min(_FAIRWAY_TABLE) <= start_yards <= max(_FAIRWAY_TABLE):
        raise ScoringUnavailable(
            "start distance is outside the published fairway benchmark range"
        )
    if set(endpoints) != {"Straight", "Draw", "Fade"}:
        raise ValueError("all three patterns are required")
    if any(len(samples) < 2 for samples in endpoints.values()):
        raise ValueError("each pattern requires at least two scored endpoints")
    rows = []
    shot_keys = []
    for pattern, samples in endpoints.items():
        for index, x, y in samples:
            if not all(math.isfinite(v) for v in (x, y)) or index < 0:
                raise ValueError(
                    "endpoints must have nonnegative indices and finite coordinates"
                )
            distance_m = math.hypot(x - target_x_m, y)
            rows.append(
                {
                    "start_lie": "fairway",
                    "start_context": "standard",
                    "finish_lie": "green" if distance_m <= green_radius_m else "rough",
                    "finish_context": "standard",
                    "target": "hole",
                    "start_distance_yd": start_yards,
                    "finish_distance_yd": distance_m * YARDS_PER_METRE,
                }
            )
            shot_keys.append((pattern, index))
    request = StrokesGainedRequestV1(
        start=CourseStateColumnsV1(
            lie_column="start_lie",
            context_column="start_context",
            target_column="target",
            distance_column="start_distance_yd",
            distance_unit="yd",
        ),
        finish=CourseStateColumnsV1(
            lie_column="finish_lie",
            context_column="finish_context",
            target_column="target",
            distance_column="finish_distance_yd",
            distance_unit="yd",
        ),
        min_samples=1,
    )
    baseline = build_broadie_approx_baseline()
    result = analyze_source_backed_strokes_gained(pd.DataFrame(rows), baseline, request)
    if result.exclusions.total_excluded:
        raise ScoringUnavailable(
            f"{result.exclusions.total_excluded} approach endpoints outside benchmark support"
        )
    scores: dict[str, dict[int, float]] = {name: {} for name in endpoints}
    for row in result.row_results:
        pattern, index = shot_keys[row.source_index]
        if index in scores[pattern]:
            raise ValueError("duplicate pattern/shot index")
        scores[pattern][index] = row.strokes_gained
    paired_indices = sorted(scores["Straight"])
    if any(sorted(scores[name]) != paired_indices for name in scores):
        raise ValueError("patterns must share the same shot indices")
    means = {
        name: {
            "n": len(values),
            "mean_strokes_gained": float(np.mean(list(values.values()))),
        }
        for name, values in scores.items()
    }
    paired = {
        name: paired_mean_difference(
            [scores[name][i] for i in paired_indices],
            [scores["Straight"][i] for i in paired_indices],
            seed=seed,
        )
        for name in ("Draw", "Fade")
    }
    return {
        "scenario": "Historical PGA Tour approach from fairway to a circular green; carry-only stop",
        "status": "available",
        "source_backed_status": result.status,
        "evaluation": "Every endpoint scored by the source-backed public Tools API",
        "api_scored_shots": len(result.row_results),
        "baseline": {
            "id": baseline.baseline_id,
            "version": baseline.version,
            "table_sha256": baseline.table_sha256,
            "source_url": baseline.source_url,
        },
        "target_distance_m": target_x_m,
        "green_radius_m": green_radius_m,
        "patterns": means,
        "paired_benefit_vs_straight": paired,
        "limitations": [
            "Historical 2003-2010 PGA Tour benchmark is not player-specific.",
            "Putting fit reconciles a printed coefficient/unit inconsistency using published anchors; it is approximate.",
            "Circular green and surrounding rough are hypothetical; carry endpoints have no roll or hazards.",
            "Confidence intervals describe Monte Carlo sampling only, not physics, course, or benchmark uncertainty.",
        ],
    }


def score_saved_bundle(output_dir: Path) -> Path:
    """Add the approach scenario to a completed shot bundle without re-simulation."""
    output_dir = Path(output_dir)
    summary_path = output_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    if summary.get("config", {}).get("club_id") in {
        "driver",
        "seven_iron",
        "pitching_wedge",
    }:
        from .scenario_scoring import score_corrected_bundle

        return score_corrected_bundle(output_dir)
    endpoints: dict[str, list[tuple[int, float, float]]] = {
        name: [] for name in ("Straight", "Draw", "Fade")
    }
    with (output_dir / "shots.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            endpoints[row["pattern"]].append(
                (
                    int(row["shot_index"]),
                    float(row["aimed_x_m"]),
                    float(row["aimed_y_m"]),
                )
            )
    try:
        report = score_approach_endpoints(
            endpoints,
            target_x_m=float(summary["target_x_m"]),
            green_radius_m=float(summary["config"]["target_radius_m"]),
            seed=int(summary["config"]["seed"]),
        )
    except ScoringUnavailable as exc:
        report = {
            "status": "unavailable",
            "reason": str(exc),
            "scenario": "Historical PGA Tour approach to a circular green",
        }
        return _write_report(output_dir, summary_path, summary, report)
    sensitivity = {}
    for radius in (10.0, 15.0, 20.0):
        scenario = (
            report
            if radius == float(summary["config"]["target_radius_m"])
            else _score_sensitivity_radius(
                endpoints,
                float(summary["target_x_m"]),
                radius,
                int(summary["config"]["seed"]),
            )
        )
        sensitivity[f"{radius:g}"] = (
            {"status": "unavailable", "reason": scenario["reason"]}
            if scenario["status"] == "unavailable"
            else {
                "patterns": scenario["patterns"],
                "paired_benefit_vs_straight": scenario["paired_benefit_vs_straight"],
            }
        )
    report["green_radius_sensitivity_m"] = sensitivity
    baseline = build_broadie_approx_baseline()
    baseline_path = output_dir / "strokes_gained_baseline.json"
    baseline_path.write_text(
        json.dumps(
            {
                "baseline": baseline.model_dump(mode="json"),
                "source_pdf_sha256": SOURCE_PDF_SHA256,
                "derivation": {
                    "fairway_and_rough": "Published Appendix A Table 9 factual benchmark values, yards",
                    "green": "Published Eq. 5 and Eq. 7 with anchor-reconciled approximate Eq. 6, sampled every foot through 75 ft",
                    "putting_anchor_check": "8 ft one-putt about 50%; 33 ft expected putts about 2; 40 ft three-putt about 10%",
                    "license_note": "Source publication license not asserted by this artifact",
                },
            },
            indent=2,
        )
        + "\n"
    )
    report["baseline_artifact"] = baseline_path.name
    path = _write_report(output_dir, summary_path, summary, report)
    receipt_path = output_dir / "receipt.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        receipt["scoring_postprocess"] = {
            "method": "Full public source-backed Tools API for each endpoint and green-radius sensitivity",
            "scoring_source_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "baseline_artifact_sha256": hashlib.sha256(
                baseline_path.read_bytes()
            ).hexdigest(),
            "source_pdf_sha256": SOURCE_PDF_SHA256,
        }
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return path


def _score_sensitivity_radius(
    endpoints: dict[str, list[tuple[int, float, float]]],
    target_x_m: float,
    radius_m: float,
    seed: int,
) -> dict:
    try:
        return score_approach_endpoints(
            endpoints, target_x_m=target_x_m, green_radius_m=radius_m, seed=seed
        )
    except ScoringUnavailable as exc:
        return {"status": "unavailable", "reason": str(exc)}


def _write_report(
    output_dir: Path, summary_path: Path, summary: dict, report: dict
) -> Path:
    path = output_dir / "strokes_gained.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    summary["approach_scoring"] = report
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return path
