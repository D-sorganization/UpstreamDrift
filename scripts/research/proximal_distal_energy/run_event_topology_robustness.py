"""Run the registered global event-topology robustness study for issue #9125.

Perturbation magnitudes are dimensionless synthetic stress-test scenarios
relative to previously registered state, control, and guard scales. They are
not estimates of human motor noise, fatigue, strength, injury, or skill.
"""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from scripts.research.proximal_distal_energy.event_robustness_noise import (
    RobustnessNoiseConfig,
    generate_common_random_perturbations,
)
from scripts.research.proximal_distal_energy.event_robustness_study import (
    DelayNoiseTopologyResult,
    evaluate_delay_noise_topology,
)
from scripts.research.proximal_distal_energy.event_robustness_summary import (
    TopologyAdequacyConfig,
    summarize_topology_by_delay,
)
from scripts.research.proximal_distal_energy.event_topology_robustness import (
    DelayContinuationConfig,
    DelayInterpolation,
    EventTopologyStatus,
)
from scripts.research.proximal_distal_energy.numeric_evidence import (
    canonicalize_published_numbers,
)
from scripts.research.proximal_distal_energy.torque_programs import (
    restrain_then_drive_program,
)
from scripts.research.proximal_distal_energy.trajectory_control_authority import (
    GuardCrossingConfig,
)
from src.shared.python.simulation_backends import GolfModelParams

DATA_DIR = ROOT / "docs/research/proximal_distal_energy_transfer/data"
REPORT_PATH = DATA_DIR / "event_topology_robustness.json"
ARRAY_PATH = DATA_DIR / "event_topology_robustness.npz"
PARENT_REPORT_PATH = DATA_DIR / "bounded_event_reachability.json"
PARENT_ARRAY_PATH = DATA_DIR / "bounded_event_reachability.npz"
SCHEMA_VERSION = "proximal-distal-event-topology-robustness/v1"
BASE_DT_S = 2e-3
SOURCE_PROGRAM_HORIZON_S = 0.40
COMMON_HORIZON_S = 0.60
INITIAL_STATE = (-2.2, -1.57, 0.0, 0.0)
STATE_SCALES = (math.pi, math.pi, 10.0, 10.0)
CONTROL_SCALES_NM = (100.0, 100.0)
GUARD_OFFSET_SCALE = math.pi
REGISTERED_DELAYS_S = tuple(round(index * 0.02, 2) for index in range(11))
REGISTERED_SEED = 9125
ADEQUACY = TopologyAdequacyConfig(
    required_independent_pairs=96,
    maximum_interval_half_width=0.10,
    confidence=0.95,
)
GUARD = GuardCrossingConfig(
    guard_gradient=(1.0, 1.0, 0.0, 0.0),
    guard_tolerance=1e-10,
    time_tolerance_s=1e-12,
    transversality_threshold=1e-8,
)
STATUS_CODES = {status.value: index for index, status in enumerate(EventTopologyStatus)}
INFERENCE_BOUNDARY = (
    "These are synthetic model-scenario perturbations of one analytical "
    "double-pendulum trajectory and one geometric event surface. They do not "
    "establish human motor noise, fatigue, strength, injury, skill, controller "
    "superiority, or coaching guidance."
)


@dataclass(frozen=True, slots=True)
class ScenarioCase:
    """One dimensionless perturbation-scale scenario."""

    name: str
    scale_fraction: float
    replicate_count: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("scenario name must be nonempty")
        if not math.isfinite(self.scale_fraction) or self.scale_fraction < 0.0:
            raise ValueError("scale_fraction must be finite and nonnegative")
        if (
            isinstance(self.replicate_count, bool)
            or not isinstance(self.replicate_count, int)
            or self.replicate_count < 2
            or self.replicate_count % 2 != 0
        ):
            raise ValueError("replicate_count must be a positive even integer")


def registered_scenarios() -> tuple[ScenarioCase, ...]:
    """Return the immutable zero and dimensionless stress-test ladder."""

    return (
        ScenarioCase("zero", 0.0, 4),
        ScenarioCase("fraction_0p001", 0.001, 192),
        ScenarioCase("fraction_0p005", 0.005, 192),
        ScenarioCase("fraction_0p01", 0.01, 192),
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_identity() -> dict[str, Any]:
    parent = json.loads(PARENT_REPORT_PATH.read_text(encoding="utf-8"))
    if parent.get("schema_version") != "proximal-distal-bounded-event-reachability/v1":
        raise ValueError("parent bounded-event report is not qualified")
    source_paths = (
        Path("scripts/research/proximal_distal_energy/event_topology_robustness.py"),
        Path("scripts/research/proximal_distal_energy/event_robustness_noise.py"),
        Path("scripts/research/proximal_distal_energy/event_robustness_study.py"),
        Path("scripts/research/proximal_distal_energy/event_robustness_summary.py"),
        Path(
            "scripts/research/proximal_distal_energy/run_event_topology_robustness.py"
        ),
    )
    return {
        "required_parent_issue": 9124,
        "required_parent_report_schema": parent["schema_version"],
        "required_parent_report_sha256": _sha256(PARENT_REPORT_PATH),
        "required_parent_array_sha256": _sha256(PARENT_ARRAY_PATH),
        "source_sha256": {
            path.as_posix(): _sha256(ROOT / path) for path in source_paths
        },
    }


def _noise_config(scenario: ScenarioCase) -> RobustnessNoiseConfig:
    fraction = scenario.scale_fraction
    return RobustnessNoiseConfig(
        seed=REGISTERED_SEED,
        replicate_count=scenario.replicate_count,
        initial_state_sd=tuple(fraction * value for value in STATE_SCALES),
        command_sd_nm=tuple(fraction * value for value in CONTROL_SCALES_NM),
        guard_offset_sd=fraction * GUARD_OFFSET_SCALE,
    )


def _delay_config() -> DelayContinuationConfig:
    return DelayContinuationConfig(
        delays_s=REGISTERED_DELAYS_S,
        common_horizon_s=COMMON_HORIZON_S,
        interpolation=DelayInterpolation.LINEAR_NODAL,
        prehistory_control=(0.0, 0.0),
        posthistory_control=(0.0, 0.0),
    )


def _registered_controls() -> np.ndarray:
    return restrain_then_drive_program(60.0, 15.0, 10.0, 0.10).controls(
        round(SOURCE_PROGRAM_HORIZON_S / BASE_DT_S), BASE_DT_S
    )


def _summary_record(summary: Any) -> dict[str, Any]:
    return {
        "delay_s": summary.delay_s,
        "topology_counts": dict(summary.topology_counts),
        "independent_pair_count": summary.independent_pair_count,
        "preserved_pair_count": summary.preserved_pair_count,
        "adequacy_passed": summary.adequacy_passed,
        "preservation_fraction": summary.preservation_fraction,
        "preservation_interval": (
            list(summary.preservation_interval)
            if summary.preservation_interval is not None
            else None
        ),
    }


def _result_arrays(
    scenario: ScenarioCase,
    result: DelayNoiseTopologyResult,
    perturbations: Any,
) -> dict[str, np.ndarray]:
    delay_count = len(REGISTERED_DELAYS_S)
    replicate_count = scenario.replicate_count
    max_crossings = max(
        (item.topology.crossing_count for item in result.outcomes), default=0
    )
    event_slots = max(1, max_crossings)
    status = np.empty((delay_count, replicate_count), dtype=np.int16)
    crossing_count = np.empty((delay_count, replicate_count), dtype=np.int16)
    event_time = np.full((delay_count, replicate_count, event_slots), np.nan)
    transversality = np.full_like(event_time, np.nan)
    direction = np.zeros_like(event_time, dtype=np.int8)
    for delay_index in range(delay_count):
        retained = result.outcomes[
            delay_index * replicate_count : (delay_index + 1) * replicate_count
        ]
        for replicate_index, item in enumerate(retained):
            status[delay_index, replicate_index] = STATUS_CODES[
                item.topology.status.value
            ]
            crossing_count[delay_index, replicate_index] = item.topology.crossing_count
            for event_index, event in enumerate(item.topology.events):
                event_time[delay_index, replicate_index, event_index] = event.time_s
                transversality[delay_index, replicate_index, event_index] = (
                    event.transversality_per_s
                )
                direction[delay_index, replicate_index, event_index] = (
                    1 if event.direction.value == "negative_to_nonnegative" else -1
                )
    prefix = scenario.name
    return {
        f"{prefix}_initial_state_delta": perturbations.initial_state_delta,
        f"{prefix}_command_delta_nm": perturbations.command_delta_nm,
        f"{prefix}_guard_offset_delta": perturbations.guard_offset_delta,
        f"{prefix}_status_code": status,
        f"{prefix}_crossing_count": crossing_count,
        f"{prefix}_event_time_s": event_time,
        f"{prefix}_transversality_per_s": transversality,
        f"{prefix}_direction_code": direction,
    }


def run_scenario(
    scenario: ScenarioCase,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    noise = _noise_config(scenario)
    delay = _delay_config()
    output_count = round(delay.common_horizon_s / BASE_DT_S)
    perturbations = generate_common_random_perturbations(
        noise, control_sample_count=output_count
    )
    result = evaluate_delay_noise_topology(
        params=GolfModelParams.default(),
        initial_state=INITIAL_STATE,
        controls=_registered_controls(),
        dt_s=BASE_DT_S,
        guard=GUARD,
        delay_config=delay,
        perturbations=perturbations,
    )
    summaries = summarize_topology_by_delay(result, config=ADEQUACY)
    record = {
        "name": scenario.name,
        "scale_fraction": scenario.scale_fraction,
        "replicate_count": scenario.replicate_count,
        "independent_pair_count": scenario.replicate_count // 2,
        "initial_state_sd": list(noise.initial_state_sd),
        "command_sd_nm": list(noise.command_sd_nm),
        "guard_offset_sd": noise.guard_offset_sd,
        "delay_summaries": [_summary_record(summary) for summary in summaries],
    }
    return record, _result_arrays(scenario, result, perturbations)


def build_evidence() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Run the registered campaign and return portable and raw evidence."""

    scenario_records: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {
        "registered_delays_s": np.asarray(REGISTERED_DELAYS_S),
        "state_scales": np.asarray(STATE_SCALES),
        "control_scales_nm": np.asarray(CONTROL_SCALES_NM),
    }
    for scenario in registered_scenarios():
        record, scenario_arrays = run_scenario(scenario)
        scenario_records.append(record)
        arrays.update(scenario_arrays)
    nonzero_summaries = [
        summary
        for scenario in scenario_records[1:]
        for summary in scenario["delay_summaries"]
    ]
    report = {
        "schema_version": SCHEMA_VERSION,
        "source_identity": _source_identity(),
        "registration": {
            "base_dt_s": BASE_DT_S,
            "source_program_horizon_s": SOURCE_PROGRAM_HORIZON_S,
            "common_horizon_s": COMMON_HORIZON_S,
            "delays_s": list(REGISTERED_DELAYS_S),
            "delay_interpolation": DelayInterpolation.LINEAR_NODAL.value,
            "prehistory_control_nm": [0.0, 0.0],
            "posthistory_control_nm": [0.0, 0.0],
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
            "all_nonzero_summaries_adequate": all(
                summary["adequacy_passed"] for summary in nonzero_summaries
            ),
            "controller_or_strategy_ranking_available": False,
        },
        "availability": {
            "human_motor_noise": "unavailable",
            "fatigue_interpretation": "unavailable",
            "controller_ranking": "suppressed",
            "coaching_recommendation": "unavailable",
        },
        "inference_boundary": INFERENCE_BOUNDARY,
    }
    validate_report(report)
    return canonicalize_published_numbers(report), arrays


def validate_report(report: dict[str, Any]) -> dict[str, int]:
    """Fail closed on registration, adequacy, or inference-boundary drift."""

    scenarios = report.get("scenarios", [])
    for scenario in scenarios:
        summaries = scenario.get("delay_summaries", [])
        for summary in summaries:
            adequate = summary.get("adequacy_passed") is True
            fraction = summary.get("preservation_fraction")
            interval = summary.get("preservation_interval")
            if not adequate and (fraction is not None or interval is not None):
                raise ValueError("inadequate summary cannot publish probability fields")
            if adequate:
                if fraction is None or interval is None or len(interval) != 2:
                    raise ValueError(
                        "adequate summary must publish its bounded interval"
                    )
                half_width = (interval[1] - interval[0]) / 2.0
                if (
                    summary.get("independent_pair_count", 0)
                    < ADEQUACY.required_independent_pairs
                    or half_width > ADEQUACY.maximum_interval_half_width
                ):
                    raise ValueError("adequacy publication gate is inconsistent")
    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected event-topology robustness schema")
    registration = report.get("registration", {})
    if registration.get("delays_s") != list(REGISTERED_DELAYS_S):
        raise ValueError("registered delay schedule drifted")
    if registration.get("base_dt_s") != BASE_DT_S:
        raise ValueError("registered integration step drifted")
    if registration.get("common_horizon_s") != COMMON_HORIZON_S:
        raise ValueError("registered common horizon drifted")
    expected = registered_scenarios()
    if [item.get("name") for item in scenarios] != [item.name for item in expected]:
        raise ValueError("registered scenario matrix is incomplete")
    for record, scenario in zip(scenarios, expected, strict=True):
        if record.get("replicate_count") != scenario.replicate_count:
            raise ValueError("registered replicate count drifted")
        if len(record.get("delay_summaries", [])) != len(REGISTERED_DELAYS_S):
            raise ValueError("registered delay matrix is incomplete")
    if report.get("source_identity") != _source_identity():
        raise ValueError("event-topology robustness source identity is stale")
    availability = report.get("availability", {})
    expected_availability = {
        "human_motor_noise": "unavailable",
        "fatigue_interpretation": "unavailable",
        "controller_ranking": "suppressed",
        "coaching_recommendation": "unavailable",
    }
    if availability != expected_availability:
        raise ValueError("unqualified interpretations must remain unavailable")
    boundary = str(report.get("inference_boundary", "")).lower()
    for phrase in (
        "synthetic model-scenario",
        "human motor noise",
        "fatigue",
        "coaching",
    ):
        if phrase not in boundary:
            raise ValueError(f"inference boundary must retain '{phrase}'")
    return {
        "scenario_count": len(scenarios),
        "delay_count": len(REGISTERED_DELAYS_S),
        "retained_outcome_count": sum(
            scenario.replicate_count * len(REGISTERED_DELAYS_S) for scenario in expected
        ),
    }


def write_evidence() -> None:
    """Write deterministic portable JSON and full-precision NPZ evidence."""

    report, arrays = build_evidence()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
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


__all__ = [
    "ADEQUACY",
    "ARRAY_PATH",
    "BASE_DT_S",
    "COMMON_HORIZON_S",
    "REGISTERED_DELAYS_S",
    "REPORT_PATH",
    "SCHEMA_VERSION",
    "ScenarioCase",
    "build_evidence",
    "registered_scenarios",
    "run_scenario",
    "validate_report",
    "write_evidence",
]


if __name__ == "__main__":
    main()
