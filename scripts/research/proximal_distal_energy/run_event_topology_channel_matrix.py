"""Run the preregistered #9125 Phase C channel and numerical controls."""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
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
    CommonRandomPerturbations,
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
from scripts.research.proximal_distal_energy.event_topology_channel_controls import (
    ChannelMask,
    apply_channel_mask,
    event_metric_records,
    mask_common_random_perturbations,
    registered_channel_masks,
)
from scripts.research.proximal_distal_energy.event_topology_robustness import (
    DelayContinuationConfig,
    DelayInterpolation,
    EventTopologyStatus,
    GlobalEventTopology,
    evaluate_delay_continuation,
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
REPORT_PATH = DATA_DIR / "event_topology_channel_matrix.json"
ARRAY_PATH = DATA_DIR / "event_topology_channel_matrix.npz"
PARENT_REPORT_PATH = DATA_DIR / "event_topology_stress_extension.json"
PARENT_ARRAY_PATH = DATA_DIR / "event_topology_stress_extension.npz"
BOUNDED_REPORT_PATH = DATA_DIR / "bounded_event_reachability.json"
SCHEMA_VERSION = "proximal-distal-event-topology-channel-matrix/v1"
PREREGISTRATION_COMMENT = (
    "https://github.com/D-sorganization/UpstreamDrift/issues/9125#"
    "issuecomment-5431439586"
)
BASE_DT_S = 0.002
SOURCE_PROGRAM_HORIZON_S = 0.40
COMMON_HORIZON_S = 0.60
INITIAL_STATE = (-2.2, -1.57, 0.0, 0.0)
STATE_SCALES = (math.pi, math.pi, 10.0, 10.0)
CONTROL_SCALES_NM = (100.0, 100.0)
GUARD_OFFSET_SCALE = math.pi
NOISE_SCALE_FRACTION = 0.01
REPLICATE_COUNT = 192
REGISTERED_SEED = 9125
REGISTERED_DELAYS_S = tuple(round(index * 0.02, 2) for index in range(11))
ADEQUACY = TopologyAdequacyConfig(96, 0.10, 0.95)
GUARD = GuardCrossingConfig(
    guard_gradient=(1.0, 1.0, 0.0, 0.0),
    guard_tolerance=1e-10,
    time_tolerance_s=1e-12,
    transversality_threshold=1e-8,
)
STATUS_CODES = {status.value: index for index, status in enumerate(EventTopologyStatus)}
INFERENCE_BOUNDARY = (
    "Phase C uses synthetic open-loop generalized-coordinate channel masks. "
    "They are not anatomical isolation, human motor-noise, fatigue, strength, "
    "injury, strategy-ranking, or coaching evidence."
)


def registered_step_sizes_s() -> tuple[float, ...]:
    """Return the fixed physical integration-step refinement ladder."""

    return (0.001, 0.002, 0.004)


def registered_horizons_s() -> tuple[float, ...]:
    """Return the fixed undelayed global-search horizon controls."""

    return (0.40, 0.60, 0.80)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_identity() -> dict[str, Any]:
    parent = json.loads(PARENT_REPORT_PATH.read_text(encoding="utf-8"))
    if (
        parent.get("schema_version")
        != "proximal-distal-event-topology-stress-extension/v1"
    ):
        raise ValueError("Phase B topology evidence is not qualified")
    source_paths = (
        Path(
            "scripts/research/proximal_distal_energy/event_topology_channel_controls.py"
        ),
        Path(
            "scripts/research/proximal_distal_energy/"
            "run_event_topology_channel_matrix.py"
        ),
    )
    return {
        "required_parent_schema": parent["schema_version"],
        "required_parent_report_sha256": _sha256(PARENT_REPORT_PATH),
        "required_parent_array_sha256": _sha256(PARENT_ARRAY_PATH),
        "source_sha256": {
            path.as_posix(): _sha256(ROOT / path) for path in source_paths
        },
    }


def _controls(dt_s: float, mask: ChannelMask) -> np.ndarray:
    count = round(SOURCE_PROGRAM_HORIZON_S / dt_s)
    controls = restrain_then_drive_program(60.0, 15.0, 10.0, 0.10).controls(count, dt_s)
    return apply_channel_mask(controls, mask)


def _delay_config(
    horizon_s: float, delays_s: tuple[float, ...]
) -> DelayContinuationConfig:
    return DelayContinuationConfig(
        delays_s=delays_s,
        common_horizon_s=horizon_s,
        interpolation=DelayInterpolation.LINEAR_NODAL,
        prehistory_control=(0.0, 0.0),
        posthistory_control=(0.0, 0.0),
    )


def _noise_config() -> RobustnessNoiseConfig:
    fraction = NOISE_SCALE_FRACTION
    return RobustnessNoiseConfig(
        seed=REGISTERED_SEED,
        replicate_count=REPLICATE_COUNT,
        initial_state_sd=tuple(fraction * value for value in STATE_SCALES),
        command_sd_nm=tuple(fraction * value for value in CONTROL_SCALES_NM),
        guard_offset_sd=fraction * GUARD_OFFSET_SCALE,
    )


def _topology_record(topology: GlobalEventTopology) -> dict[str, Any]:
    return {
        "status": topology.status.value,
        "crossing_count": topology.crossing_count,
        "events": event_metric_records(topology, GolfModelParams.default()),
        "numerical_message": topology.message,
    }


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


def _map_arrays(
    mask: ChannelMask,
    result: DelayNoiseTopologyResult,
    perturbations: CommonRandomPerturbations,
) -> dict[str, np.ndarray]:
    delay_count = len(REGISTERED_DELAYS_S)
    max_crossings = max(
        (outcome.topology.crossing_count for outcome in result.outcomes), default=0
    )
    event_slots = max(1, max_crossings)
    shape = (delay_count, REPLICATE_COUNT)
    status = np.empty(shape, dtype=np.int16)
    crossing_count = np.empty(shape, dtype=np.int16)
    event_time = np.full((*shape, event_slots), np.nan)
    event_state = np.full((*shape, event_slots, 4), np.nan)
    event_speed = np.full((*shape, event_slots), np.nan)
    transversality = np.full((*shape, event_slots), np.nan)
    direction = np.zeros((*shape, event_slots), dtype=np.int8)
    for delay_index in range(delay_count):
        retained = result.outcomes[
            delay_index * REPLICATE_COUNT : (delay_index + 1) * REPLICATE_COUNT
        ]
        for replicate_index, outcome in enumerate(retained):
            topology = outcome.topology
            status[delay_index, replicate_index] = STATUS_CODES[topology.status.value]
            crossing_count[delay_index, replicate_index] = topology.crossing_count
            for event_index, metrics in enumerate(
                event_metric_records(topology, GolfModelParams.default())
            ):
                event_time[delay_index, replicate_index, event_index] = metrics[
                    "event_time_s"
                ]
                event_state[delay_index, replicate_index, event_index] = metrics[
                    "event_state"
                ]
                event_speed[delay_index, replicate_index, event_index] = metrics[
                    "clubhead_speed_m_s"
                ]
                transversality[delay_index, replicate_index, event_index] = metrics[
                    "transversality_per_s"
                ]
                direction[delay_index, replicate_index, event_index] = (
                    1 if metrics["direction"] == "negative_to_nonnegative" else -1
                )
    prefix = f"channel_{mask.name}"
    return {
        f"{prefix}_mask": np.asarray(mask.values),
        f"{prefix}_command_delta_nm": perturbations.command_delta_nm,
        f"{prefix}_status_code": status,
        f"{prefix}_crossing_count": crossing_count,
        f"{prefix}_event_time_s": event_time,
        f"{prefix}_event_state": event_state,
        f"{prefix}_clubhead_speed_m_s": event_speed,
        f"{prefix}_transversality_per_s": transversality,
        f"{prefix}_direction_code": direction,
    }


def _channel_map(
    mask: ChannelMask,
    perturbations: CommonRandomPerturbations,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    masked = mask_common_random_perturbations(perturbations, mask)
    result = evaluate_delay_noise_topology(
        params=GolfModelParams.default(),
        initial_state=INITIAL_STATE,
        controls=_controls(BASE_DT_S, mask),
        dt_s=BASE_DT_S,
        guard=GUARD,
        delay_config=_delay_config(COMMON_HORIZON_S, REGISTERED_DELAYS_S),
        perturbations=masked,
    )
    summaries = summarize_topology_by_delay(result, config=ADEQUACY)
    return (
        {
            "channel": mask.name,
            "mask": list(mask.values),
            "nominal_by_delay": [
                {
                    "delay_s": outcome.delay_s,
                    **_topology_record(outcome.topology),
                }
                for outcome in result.nominal.outcomes
            ],
            "noise_scale_fraction": NOISE_SCALE_FRACTION,
            "replicate_count": REPLICATE_COUNT,
            "delay_summaries": [_summary_record(summary) for summary in summaries],
        },
        _map_arrays(mask, result, masked),
    )


def _step_refinement_records(mask: ChannelMask) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for dt_s in registered_step_sizes_s():
        result = evaluate_delay_continuation(
            params=GolfModelParams.default(),
            initial_state=INITIAL_STATE,
            controls=_controls(dt_s, mask),
            dt_s=dt_s,
            guard=GUARD,
            config=_delay_config(COMMON_HORIZON_S, REGISTERED_DELAYS_S),
        )
        records.append(
            {
                "channel": mask.name,
                "dt_s": dt_s,
                "zero_delay_control_residual_nm": result.zero_delay_control_residual,
                "outcomes": [
                    {
                        "delay_s": outcome.delay_s,
                        **_topology_record(outcome.topology),
                    }
                    for outcome in result.outcomes
                ],
            }
        )
    return records


def _horizon_records(mask: ChannelMask) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for horizon_s in registered_horizons_s():
        result = evaluate_delay_continuation(
            params=GolfModelParams.default(),
            initial_state=INITIAL_STATE,
            controls=_controls(BASE_DT_S, mask),
            dt_s=BASE_DT_S,
            guard=GUARD,
            config=_delay_config(horizon_s, (0.0,)),
        )
        records.append(
            {
                "channel": mask.name,
                "horizon_s": horizon_s,
                **_topology_record(result.outcomes[0].topology),
            }
        )
    return records


def _topology_identity(outcome: dict[str, Any]) -> tuple[Any, ...]:
    return (
        outcome["status"],
        outcome["crossing_count"],
        tuple(event["direction"] for event in outcome["events"]),
    )


def _event_residuals(
    reference: dict[str, Any], candidate: dict[str, Any]
) -> tuple[float, float, float, float]:
    if _topology_identity(reference) != _topology_identity(candidate):
        return (math.inf, math.inf, math.inf, math.inf)
    time_residual = 0.0
    state_residual = 0.0
    speed_residual = 0.0
    transversality_residual = 0.0
    for left, right in zip(reference["events"], candidate["events"], strict=True):
        time_residual = max(
            time_residual,
            abs(float(left["event_time_s"]) - float(right["event_time_s"])),
        )
        state_residual = max(
            state_residual,
            float(
                np.max(
                    np.abs(
                        np.asarray(left["event_state"], dtype=float)
                        - np.asarray(right["event_state"], dtype=float)
                    )
                )
            ),
        )
        speed_residual = max(
            speed_residual,
            abs(float(left["clubhead_speed_m_s"]) - float(right["clubhead_speed_m_s"])),
        )
        transversality_residual = max(
            transversality_residual,
            abs(
                float(left["transversality_per_s"])
                - float(right["transversality_per_s"])
            ),
        )
    return time_residual, state_residual, speed_residual, transversality_residual


def summarize_step_refinement(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compare every fixed physical step to the registered base-step replay."""

    summaries: list[dict[str, Any]] = []
    channels = list(dict.fromkeys(str(record["channel"]) for record in records))
    for channel in channels:
        grouped = {
            float(record["dt_s"]): record
            for record in records
            if record["channel"] == channel
        }
        reference = grouped[BASE_DT_S]["outcomes"]
        identities_match = True
        residuals = [0.0, 0.0, 0.0, 0.0]
        for dt_s in registered_step_sizes_s():
            candidate = grouped[dt_s]["outcomes"]
            for left, right in zip(reference, candidate, strict=True):
                identities_match &= _topology_identity(left) == _topology_identity(
                    right
                )
                residuals = [
                    max(current, observed)
                    for current, observed in zip(
                        residuals, _event_residuals(left, right), strict=True
                    )
                ]
        summaries.append(
            {
                "channel": channel,
                "topology_identity_all_match": identities_match,
                "maximum_event_time_residual_s": residuals[0],
                "maximum_event_state_residual": residuals[1],
                "maximum_clubhead_speed_residual_m_s": residuals[2],
                "maximum_transversality_residual_per_s": residuals[3],
            }
        )
    return summaries


def summarize_horizon_controls(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Type original-horizon truncation separately from expanded support."""

    summaries: list[dict[str, Any]] = []
    channels = list(dict.fromkeys(str(record["channel"]) for record in records))
    for channel in channels:
        grouped = {
            float(record["horizon_s"]): record
            for record in records
            if record["channel"] == channel
        }
        expanded_stable = _topology_identity(grouped[0.60]) == _topology_identity(
            grouped[0.80]
        )
        original_differs = _topology_identity(grouped[0.40]) != _topology_identity(
            grouped[0.60]
        )
        summaries.append(
            {
                "channel": channel,
                "expanded_horizon_identity_stable": expanded_stable,
                "original_horizon_differs": original_differs,
                "interpretation": (
                    "original_horizon_truncation" if original_differs else "no_change"
                ),
            }
        )
    return summaries


def _parent_outcome_separation() -> dict[str, Any]:
    parent = json.loads(BOUNDED_REPORT_PATH.read_text(encoding="utf-8"))
    trials = parent["continuation_trials"]
    return {
        "source_schema": parent["schema_version"],
        "source_sha256": _sha256(BOUNDED_REPORT_PATH),
        "trial_count": len(trials),
        "solver_status_counts": dict(Counter(item["solver_status"] for item in trials)),
        "feasibility_status_counts": dict(
            Counter(item["replay_feasibility_status"] for item in trials)
        ),
        "constraint_status_counts": dict(
            Counter(item["replay_constraint_status"] for item in trials)
        ),
        "maximum_target_residual": max(
            float(item["maximum_target_residual"]) for item in trials
        ),
        "fields_are_source_linked_not_recomputed": True,
    }


def build_evidence() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Execute the complete fixed Phase C matrix and retain raw outcomes."""

    output_count = round(COMMON_HORIZON_S / BASE_DT_S)
    perturbations = generate_common_random_perturbations(
        _noise_config(), control_sample_count=output_count
    )
    maps: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {
        "registered_delays_s": np.asarray(REGISTERED_DELAYS_S),
        "shared_initial_state_delta": perturbations.initial_state_delta,
        "shared_guard_offset_delta": perturbations.guard_offset_delta,
    }
    step_records: list[dict[str, Any]] = []
    horizon_records: list[dict[str, Any]] = []
    for mask in registered_channel_masks():
        record, retained = _channel_map(mask, perturbations)
        maps.append(record)
        arrays.update(retained)
        step_records.extend(_step_refinement_records(mask))
        horizon_records.extend(_horizon_records(mask))
    step_summary = summarize_step_refinement(step_records)
    horizon_summary = summarize_horizon_controls(horizon_records)
    zero_command_nonzero_count = int(
        np.count_nonzero(arrays["channel_zero_command_delta_nm"])
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "source_identity": _source_identity(),
        "registration": {
            "preregistration_comment": PREREGISTRATION_COMMENT,
            "base_dt_s": BASE_DT_S,
            "common_horizon_s": COMMON_HORIZON_S,
            "step_sizes_s": list(registered_step_sizes_s()),
            "horizons_s": list(registered_horizons_s()),
            "delays_s": list(REGISTERED_DELAYS_S),
            "seed": REGISTERED_SEED,
            "replicate_count": REPLICATE_COUNT,
            "independent_pair_count": REPLICATE_COUNT // 2,
            "noise_scale_fraction": NOISE_SCALE_FRACTION,
            "fixed_stop_rule_completed": True,
        },
        "channel_maps": maps,
        "step_refinement": step_records,
        "step_refinement_summary": step_summary,
        "horizon_controls": horizon_records,
        "horizon_control_summary": horizon_summary,
        "parent_bounded_outcomes": _parent_outcome_separation(),
        "outcome_separation": {
            "topology_and_event_kinematics": "retained_in_phase_c",
            "feasibility_target_error_bounds_objective": "source_linked_from_phase_9124",
            "work_power": "unavailable_without_independent_quadrature_contract",
            "speed_cannot_rescue_failed_topology_or_parent_feasibility": True,
        },
        "qualification": {
            "all_step_topology_identities_stable": all(
                item["topology_identity_all_match"] for item in step_summary
            ),
            "expanded_horizon_topology_stable": all(
                item["expanded_horizon_identity_stable"] for item in horizon_summary
            ),
            "original_horizon_truncation_channels": [
                item["channel"]
                for item in horizon_summary
                if item["original_horizon_differs"]
            ],
            "zero_authority_command_nonzero_count": zero_command_nonzero_count,
            "preservation_is_success_probability": False,
        },
        "availability": {
            "work_power": "unavailable",
            "human_motor_noise": "unavailable",
            "anatomical_channel_attribution": "unavailable",
            "controller_ranking": "suppressed",
            "coaching_recommendation": "unavailable",
        },
        "inference_boundary": INFERENCE_BOUNDARY,
    }
    validate_report(report)
    return canonicalize_published_numbers(report), arrays


def validate_report(
    report: dict[str, Any], *, verify_source: bool = True
) -> dict[str, int]:
    """Fail closed on incomplete matrices, provenance, or inference promotion."""

    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected Phase C topology schema")
    registration = report.get("registration", {})
    expected_registration = {
        "preregistration_comment": PREREGISTRATION_COMMENT,
        "base_dt_s": BASE_DT_S,
        "common_horizon_s": COMMON_HORIZON_S,
        "step_sizes_s": list(registered_step_sizes_s()),
        "horizons_s": list(registered_horizons_s()),
        "fixed_stop_rule_completed": True,
    }
    if any(
        registration.get(key) != value for key, value in expected_registration.items()
    ):
        raise ValueError("Phase C registration drifted")
    if report.get("availability", {}).get("work_power") != "unavailable":
        raise ValueError("Phase C cannot promote work/power")
    if report.get("availability", {}).get("coaching_recommendation") != "unavailable":
        raise ValueError("Phase C cannot promote coaching guidance")
    names = [mask.name for mask in registered_channel_masks()]
    if [item.get("channel") for item in report.get("channel_maps", [])] != names:
        raise ValueError("Phase C channel map is incomplete")
    step_keys = {
        (item.get("channel"), item.get("dt_s"))
        for item in report.get("step_refinement", [])
    }
    if step_keys != {
        (name, dt_s) for name in names for dt_s in registered_step_sizes_s()
    }:
        raise ValueError("Phase C step-refinement matrix is incomplete")
    horizon_keys = {
        (item.get("channel"), item.get("horizon_s"))
        for item in report.get("horizon_controls", [])
    }
    if horizon_keys != {
        (name, horizon_s) for name in names for horizon_s in registered_horizons_s()
    }:
        raise ValueError("Phase C horizon-control matrix is incomplete")
    qualification = report.get("qualification", {})
    if qualification.get("all_step_topology_identities_stable") is not True:
        raise ValueError("Phase C step topology is not stable")
    if qualification.get("expanded_horizon_topology_stable") is not True:
        raise ValueError("Phase C expanded-horizon topology is not stable")
    if qualification.get("zero_authority_command_nonzero_count") != 0:
        raise ValueError("zero authority acquired command noise")
    if qualification.get("preservation_is_success_probability") is not False:
        raise ValueError("topology preservation cannot be promoted as success")
    boundary = str(report.get("inference_boundary", "")).lower()
    for phrase in ("synthetic", "not anatomical", "coaching"):
        if phrase not in boundary:
            raise ValueError(f"Phase C inference boundary must retain '{phrase}'")
    if verify_source and report.get("source_identity") != _source_identity():
        raise ValueError("Phase C source identity is stale")
    return {
        "channel_count": len(names),
        "step_control_count": len(step_keys),
        "horizon_control_count": len(horizon_keys),
        "retained_noise_outcome_count": len(names)
        * len(REGISTERED_DELAYS_S)
        * REPLICATE_COUNT,
    }


def write_evidence() -> None:
    """Write deterministic Phase C JSON and compressed full-precision arrays."""

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
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    print(json.dumps(validate_report(report), sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "BASE_DT_S",
    "COMMON_HORIZON_S",
    "PREREGISTRATION_COMMENT",
    "build_evidence",
    "registered_horizons_s",
    "registered_step_sizes_s",
    "summarize_horizon_controls",
    "summarize_step_refinement",
    "validate_report",
    "write_evidence",
]
