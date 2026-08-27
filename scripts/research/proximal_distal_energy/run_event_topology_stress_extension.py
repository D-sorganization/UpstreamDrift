"""Run the preregistered adaptive Phase B stress-to-failure extension.

Phase B was registered on issue #9125 only after Phase A retained topology
through the 1% dimensionless scale.  Its five-level stop rule is fixed before
execution.  These artificial scales are not human variability estimates.
"""

from __future__ import annotations

from argparse import ArgumentParser
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from scripts.research.proximal_distal_energy.numeric_evidence import (
    canonicalize_published_numbers,
)
from scripts.research.proximal_distal_energy.run_event_topology_robustness import (
    ADEQUACY,
    ARRAY_PATH as PHASE_A_ARRAY_PATH,
    BASE_DT_S,
    COMMON_HORIZON_S,
    CONTROL_SCALES_NM,
    GUARD_OFFSET_SCALE,
    REGISTERED_DELAYS_S,
    REGISTERED_SEED,
    REPORT_PATH as PHASE_A_REPORT_PATH,
    SCHEMA_VERSION as PHASE_A_SCHEMA_VERSION,
    STATE_SCALES,
    ScenarioCase,
    run_scenario,
)

DATA_DIR = ROOT / "docs/research/proximal_distal_energy_transfer/data"
REPORT_PATH = DATA_DIR / "event_topology_stress_extension.json"
ARRAY_PATH = DATA_DIR / "event_topology_stress_extension.npz"
SCHEMA_VERSION = "proximal-distal-event-topology-stress-extension/v1"
PREREGISTRATION_COMMENT = (
    "https://github.com/D-sorganization/UpstreamDrift/issues/9125"
    "#issuecomment-5431171920"
)
INFERENCE_BOUNDARY = (
    "This adaptive Phase B maps topology failure under fixed synthetic "
    "dimensionless stress levels after Phase A retained topology through 1%. "
    "It does not estimate human motor noise, fatigue, strength, injury, skill, "
    "target accuracy, controller superiority, or coaching guidance."
)


def registered_stress_scenarios() -> tuple[ScenarioCase, ...]:
    """Return the fixed preregistered Phase B scale ladder and stop rule."""

    return tuple(
        ScenarioCase(name, fraction, 192)
        for name, fraction in (
            ("fraction_0p02", 0.02),
            ("fraction_0p05", 0.05),
            ("fraction_0p1", 0.10),
            ("fraction_0p2", 0.20),
            ("fraction_0p5", 0.50),
        )
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_identity() -> dict[str, Any]:
    phase_a = json.loads(PHASE_A_REPORT_PATH.read_text(encoding="utf-8"))
    if phase_a.get("schema_version") != PHASE_A_SCHEMA_VERSION:
        raise ValueError("Phase A topology report is not qualified")
    source_paths = (
        Path(
            "scripts/research/proximal_distal_energy/run_event_topology_robustness.py"
        ),
        Path(
            "scripts/research/proximal_distal_energy/"
            "run_event_topology_stress_extension.py"
        ),
    )
    return {
        "required_phase_a_schema": PHASE_A_SCHEMA_VERSION,
        "required_phase_a_report_sha256": _sha256(PHASE_A_REPORT_PATH),
        "required_phase_a_array_sha256": _sha256(PHASE_A_ARRAY_PATH),
        "preregistration_comment": PREREGISTRATION_COMMENT,
        "source_sha256": {
            path.as_posix(): _sha256(ROOT / path) for path in source_paths
        },
    }


def build_evidence() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Execute every fixed Phase B level and retain full-precision arrays."""

    scenario_records: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {
        "registered_delays_s": np.asarray(REGISTERED_DELAYS_S),
        "registered_scale_fractions": np.asarray(
            [scenario.scale_fraction for scenario in registered_stress_scenarios()]
        ),
    }
    for scenario in registered_stress_scenarios():
        record, scenario_arrays = run_scenario(scenario)
        scenario_records.append(record)
        arrays.update(scenario_arrays)
    summaries = [
        summary
        for scenario in scenario_records
        for summary in scenario["delay_summaries"]
    ]
    report = {
        "schema_version": SCHEMA_VERSION,
        "source_identity": _source_identity(),
        "registration": {
            "phase": "adaptive_phase_b_stress_to_failure",
            "trigger": "phase_a_all_topologies_preserved_through_fraction_0p01",
            "preregistration_comment": PREREGISTRATION_COMMENT,
            "fixed_stop_rule_completed": True,
            "base_dt_s": BASE_DT_S,
            "common_horizon_s": COMMON_HORIZON_S,
            "delays_s": list(REGISTERED_DELAYS_S),
            "seed": REGISTERED_SEED,
            "state_scales": list(STATE_SCALES),
            "control_scales_nm": list(CONTROL_SCALES_NM),
            "guard_offset_scale": GUARD_OFFSET_SCALE,
            "required_independent_pairs": ADEQUACY.required_independent_pairs,
            "confidence": ADEQUACY.confidence,
            "maximum_interval_half_width": ADEQUACY.maximum_interval_half_width,
            "perturbations_are_human_noise_estimates": False,
        },
        "scenarios": scenario_records,
        "qualification": {
            "all_summaries_adequate": all(
                summary["adequacy_passed"] for summary in summaries
            ),
            "first_registered_scale_with_topology_loss": _first_loss(scenario_records),
            "human_or_strategy_ranking_available": False,
        },
        "availability": {
            "human_motor_noise": "unavailable",
            "fatigue_interpretation": "unavailable",
            "target_accuracy": "unavailable",
            "controller_ranking": "suppressed",
            "coaching_recommendation": "unavailable",
        },
        "inference_boundary": INFERENCE_BOUNDARY,
    }
    validate_report(report)
    return canonicalize_published_numbers(report), arrays


def _first_loss(scenarios: list[dict[str, Any]]) -> float | None:
    for scenario in scenarios:
        if any(
            summary["preserved_pair_count"] < summary["independent_pair_count"]
            for summary in scenario["delay_summaries"]
        ):
            return float(scenario["scale_fraction"])
    return None


def validate_report(report: dict[str, Any]) -> dict[str, int]:
    """Fail closed on stop-rule, adequacy, or adaptive-phase provenance drift."""

    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected Phase B topology schema")
    registration = report.get("registration", {})
    if registration.get("preregistration_comment") != PREREGISTRATION_COMMENT:
        raise ValueError("Phase B preregistration identity drifted")
    if registration.get("fixed_stop_rule_completed") is not True:
        raise ValueError("Phase B fixed stop rule must be completed")
    scenarios = report.get("scenarios", [])
    expected = registered_stress_scenarios()
    if [item.get("name") for item in scenarios] != [item.name for item in expected]:
        raise ValueError("Phase B scale ladder is incomplete")
    for record, scenario in zip(scenarios, expected, strict=True):
        if record.get("scale_fraction") != scenario.scale_fraction:
            raise ValueError("Phase B scale identity drifted")
        summaries = record.get("delay_summaries", [])
        if len(summaries) != len(REGISTERED_DELAYS_S):
            raise ValueError("Phase B delay matrix is incomplete")
        for summary in summaries:
            if summary.get("independent_pair_count") != 96:
                raise ValueError("Phase B pair count drifted")
            if summary.get("adequacy_passed") is not True:
                if (
                    summary.get("preservation_fraction") is not None
                    or summary.get("preservation_interval") is not None
                ):
                    raise ValueError("inadequate Phase B summary published probability")
    if report.get("source_identity") != _source_identity():
        raise ValueError("Phase B source identity is stale")
    boundary = str(report.get("inference_boundary", "")).lower()
    for phrase in ("adaptive phase b", "synthetic", "human motor noise", "coaching"):
        if phrase not in boundary:
            raise ValueError(f"Phase B inference boundary must retain '{phrase}'")
    return {
        "scenario_count": len(scenarios),
        "delay_count": len(REGISTERED_DELAYS_S),
        "retained_outcome_count": sum(
            scenario.replicate_count * len(REGISTERED_DELAYS_S) for scenario in expected
        ),
    }


def write_evidence() -> None:
    """Write deterministic Phase B JSON and compressed raw arrays."""

    report, arrays = build_evidence()
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    np.savez_compressed(ARRAY_PATH, **arrays)


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("write", "validate"), nargs="?", default="validate"
    )
    args = parser.parse_args()
    if args.mode == "write":
        write_evidence()
        print(f"wrote {REPORT_PATH}")
        print(f"wrote {ARRAY_PATH}")
        return
    registered = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    print(json.dumps(validate_report(registered), sort_keys=True))


if __name__ == "__main__":
    main()
