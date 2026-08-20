"""Build the fail-closed structural-corner headline propagation plan."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.research.proximal_distal_energy.articulated_atlas_authority import (
    ArticulatedAtlasAuthority,
)
from scripts.research.proximal_distal_energy.articulated_ground_atlas import (
    ArticulatedGroundAtlasConfig,
    SOURCE_PATHS as GROUND_SOURCE_PATHS,
)
from scripts.research.proximal_distal_energy.articulated_scaled_authority import (
    load_scaled_authority,
)
from scripts.research.proximal_distal_energy.articulated_shaft_atlas import (
    ArticulatedShaftAtlasConfig,
    SOURCE_PATHS as SHAFT_SOURCE_PATHS,
)

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "docs/research/proximal_distal_energy_transfer/data"
CAMPAIGN = DATA / "articulated_structural_authority_campaign.json"
DEFAULT_OUTPUT = DATA / "articulated_structural_propagation_plan.json"
SOURCE_PATHS = tuple(
    dict.fromkeys(
        (
            "scripts/research/proximal_distal_energy/articulated_atlas_authority.py",
            "scripts/research/proximal_distal_energy/articulated_scaled_authority.py",
            "scripts/research/proximal_distal_energy/articulated_structural_authority_campaign.py",
            "scripts/research/proximal_distal_energy/articulated_structural_propagation_plan.py",
            "tests/research/test_articulated_structural_propagation_plan.py",
            *SHAFT_SOURCE_PATHS,
            *GROUND_SOURCE_PATHS,
        )
    )
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _serialized(record: dict[str, Any]) -> bytes:
    return (json.dumps(record, indent=2) + "\n").encode("utf-8")


def _scientific_configuration(config: Any) -> dict[str, Any]:
    """Exclude operational parallelism from the scientific design bind."""

    configuration = asdict(config)
    configuration.pop("worker_count")
    return configuration


def _planned_failures(
    authority: ArticulatedAtlasAuthority,
    states: tuple[tuple[int, int], ...],
) -> list[dict[str, int | str]]:
    requested = set(states)
    return [
        dict(failure)
        for failure in authority.selected_failures()
        if (int(failure["case_index"]), int(failure["phase_index"])) in requested
    ]


def _corner_plan(
    row: dict[str, Any],
    shaft: ArticulatedShaftAtlasConfig,
    ground: ArticulatedGroundAtlasConfig,
    data_directory: Path,
) -> dict[str, Any]:
    scaled = load_scaled_authority(
        data_directory / row["record_artifact"],
        data_directory / row["array_artifact"],
    )
    authority = ArticulatedAtlasAuthority.from_scaled(scaled)
    if authority.authority_sha256 != row["authority_sha256"]:
        raise RuntimeError("campaign and authority artifact digests do not match")
    if shaft.case_indices != ground.case_indices or (
        shaft.sample_indices != ground.sample_indices
    ):
        raise ValueError("shaft and ground atlases must use the same registered states")
    states = tuple(
        (case, phase) for case in shaft.case_indices for phase in shaft.sample_indices
    )
    feasible = authority.feasible_states(shaft.case_indices, shaft.sample_indices)
    failures = _planned_failures(authority, states)
    if len(feasible) + len(failures) != len(states):
        raise RuntimeError("every requested state must be feasible or retained failed")
    shaft_headline_cells = (
        len(feasible) * 2 * len(shaft.forward.time_steps_s) * 2 * len(shaft.horizons_s)
    )
    ground_headline_cells = (
        len(feasible)
        * 2
        * len(ground.forward.time_steps_s)
        * 2
        * len(ground.horizons_s)
    )
    return {
        "corner_id": row["corner_id"],
        "status": "ready_with_retained_failure" if failures else "ready",
        "requested_state_count": len(states),
        "feasible_state_count": len(feasible),
        "retained_failures": failures,
        "feasible_states": [list(state) for state in feasible],
        "expected_shaft_trajectory_count": len(feasible)
        * len(shaft.activations)
        * 2
        * len(shaft.forward.time_steps_s)
        * 2,
        "expected_ground_trajectory_count": len(feasible)
        * (len(ground.ground_activations) + len(ground.control_names))
        * 2
        * len(ground.forward.time_steps_s)
        * 2,
        "expected_shaft_headline_cell_count": shaft_headline_cells,
        "expected_ground_headline_cell_count": ground_headline_cells,
        "authority": authority.provenance_record(),
    }


def build_structural_propagation_plan(
    campaign_path: Path = CAMPAIGN,
    *,
    data_directory: Path = DATA,
) -> dict[str, Any]:
    """Bind every registered structural authority to both headline designs."""

    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    if campaign.get("status") != "complete" or len(campaign.get("corners", [])) != 7:
        raise RuntimeError("the seven-corner authority campaign must be complete")
    shaft = ArticulatedShaftAtlasConfig()
    ground = ArticulatedGroundAtlasConfig()
    corners = [
        _corner_plan(row, shaft, ground, data_directory) for row in campaign["corners"]
    ]
    design = json.loads(
        json.dumps(
            {
                "case_indices": list(shaft.case_indices),
                "phase_indices": list(shaft.sample_indices),
                "shaft_configuration": _scientific_configuration(shaft),
                "ground_configuration": _scientific_configuration(ground),
                "parallelism": "worker_count is operational and excluded from the scientific design digest",
            }
        )
    )
    design_sha = hashlib.sha256(
        json.dumps(design, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    acceptance = {
        "nominal_reproduction": {
            "shaft_matched_cell_count": 126,
            "shaft_total_cell_count": 384,
            "ground_matched_cell_count": 0,
            "ground_total_cell_count": 384,
        },
        "required_controls": [
            "both native engines",
            "velocity reversal",
            "time-step refinement",
            "pathway killswitches",
            "unchanged load-work matching",
            "inconsistent authority-model scaling must fail closed",
            "deliberately infeasible joint-limit state must remain classified",
        ],
        "failure_policy": "retain every planned or dynamic failure without imputation",
        "invalidators": [
            "nominal headline counts do not reproduce",
            "a planned state is missing or duplicated",
            "an authority, model, source, configuration, or result digest differs",
            "a registered numerical, parity, power, energy, domain, or control gate fails",
            "a failed state or branch is silently omitted or replaced",
        ],
        "interpretation": "engineering OAT sensitivity only; no population, human, causal coaching, or universal performance inference",
    }
    analysis = {
        "cell_identity_fields": [
            "case_index",
            "phase_index",
            "velocity_factor",
            "time_step_s",
            "engine",
            "horizon_s",
        ],
        "per_corner_estimands": [
            "matched count and fraction",
            "matched final-speed-difference range and sign counts",
            "matching cells entered, exited, and persistent versus nominal",
            "corner-minus-nominal speed difference on persistent common support",
            "load- and work-match error distributions",
            "retained state, branch, and gate failure classes",
        ],
        "support_rule": "never compare corner outcome ranges as paired effects without persistent common matching support",
        "zero_nominal_ground_rule": "a corner admitting cells when nominal ground admits 0/384 is support emergence, not evidence of paired ground-pathway benefit",
        "count_rule": "matched-count movement diagnoses conditioning-set sensitivity, not outcome direction or causal benefit",
        "outcome_absolute_resolution_tolerance_m_s": 0.001,
        "resolution_rule": "classify a persistent-support corner-minus-nominal outcome change as resolved only when its magnitude exceeds the maximum declared tolerance, two-engine discrepancy, and fine-versus-coarse time-step discrepancy; otherwise report unresolved, not no effect",
        "oat_secant_rule": "report low-to-nominal and nominal-to-high engineering secants separately; do not label either a derivative or population effect",
        "secant_definition": "divide each persistent-common-support outcome change by the registered scale-factor span for that one-sided interval",
        "secant_boundary": "secants use engineering bounds and are not comparable parameter-importance measures across axes",
        "nonmonotonicity_rule": "retain opposing or materially unequal one-sided secants as nonmonotonic engineering sensitivity rather than averaging them",
        "interaction_rule": "one-at-a-time corners do not estimate higher-order parameter interactions",
        "multiplicity": "report all registered OAT corners descriptively; do not select favorable corners or assign confirmatory p-values",
    }
    evidence_contract = {
        "schema_version": "articulated-structural-propagation/v1",
        "checkpoint_identity_fields": [
            "corner_id",
            "authority_sha256",
            "scales",
            "model_sha256",
            "atlas_source_sha256",
            "scientific_configuration_sha256",
            "state_slot",
            "state",
            "pathway",
            "branch_kind",
            "branch_slot",
        ],
        "required_cell_arrays": {
            pathway: [
                "cell_identity",
                "matched_load_work",
                "matched_final_speed_difference_m_s",
                "load_match_relative_error",
                "work_match_relative_error",
                "gate_status",
                "failure_class",
                "two_engine_speed_difference_discrepancy_m_s",
                "time_step_speed_difference_discrepancy_m_s",
                "resolution_threshold_m_s",
                "resolved_outcome_change",
            ]
            for pathway in ("shaft", "ground")
        },
        "matching_metric_semantics": {
            "shaft": {
                "comparison": "coupled versus rigid",
                "load": "peak station force",
                "work": "terminal dissipated work",
            },
            "ground": {
                "comparison": "coupled versus fixed",
                "load": "peak grip force",
                "work": "terminal total dissipated work",
            },
        },
        "required_axis_summary_fields": [
            "axis_name",
            "low_scale",
            "nominal_scale",
            "high_scale",
            "low_to_nominal_secant_m_s_per_unit_scale",
            "nominal_to_high_secant_m_s_per_unit_scale",
            "nonmonotonic_classification",
        ],
        "write_policy": "write checkpoints and the final record atomically",
        "completion_policy": "complete only after all seven corners are accounted for",
        "partial_record_policy": "an in-progress or partial record must not qualify as release evidence",
        "resolution_boundary": "the 0.001 m/s floor is a preregistered synthetic numerical interpretation threshold, not device accuracy or human measurement uncertainty",
    }
    contract_sha = hashlib.sha256(
        json.dumps(
            {
                "design_sha256": design_sha,
                "acceptance": acceptance,
                "analysis": analysis,
                "evidence_contract": evidence_contract,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": "articulated-structural-propagation-plan/v1",
        "status": "ready",
        "authority_campaign_sha256": _sha256(campaign_path),
        "design_sha256": design_sha,
        "contract_sha256": contract_sha,
        "design": design,
        "acceptance": acceptance,
        "analysis": analysis,
        "evidence_contract": evidence_contract,
        "corners": corners,
        "source_sha256": {path: _sha256(ROOT / path) for path in SOURCE_PATHS},
        "limitations": {
            "scope": "engineering OAT corners, not a participant distribution",
            "evidence": "execution plan only; no headline result is implied",
            "human_inference": "none",
        },
    }


def validate_structural_propagation_plan(
    plan_path: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    """Reject a committed plan that differs from current governed inputs."""

    expected = build_structural_propagation_plan()
    observed_bytes = plan_path.read_bytes()
    if observed_bytes != _serialized(expected):
        raise RuntimeError("committed structural propagation plan is stale or altered")
    return expected


def write_structural_propagation_plan(
    plan_path: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    """Atomically replace the governed propagation plan."""

    record = build_structural_propagation_plan()
    temporary = plan_path.with_suffix(plan_path.suffix + ".tmp")
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_bytes(_serialized(record))
    temporary.replace(plan_path)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("write", "validate"), nargs="?", default="write"
    )
    args = parser.parse_args()
    if args.command == "validate":
        validate_structural_propagation_plan()
    else:
        write_structural_propagation_plan()


if __name__ == "__main__":
    main()
