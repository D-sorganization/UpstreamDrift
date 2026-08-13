"""Register the arm--wrist allocation and transmission audit slice."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ARTICLE = ROOT / "docs" / "research" / "proximal_distal_energy_transfer"
REGISTRY = ARTICLE / "data" / "claim_audit_registry.json"
INVENTORY = ARTICLE / "data" / "claim_candidate_inventory.json"
CHAPTER = (
    "docs/research/proximal_distal_energy_transfer/chapters/"
    "_ch06bc_torque_allocation_preload.qmd"
)
ARTIFACTS = [
    "docs/research/proximal_distal_energy_transfer/data/torque_allocation_preload_study.json",
    "docs/research/proximal_distal_energy_transfer/data/torque_allocation_preload_study.npz",
    "scripts/research/proximal_distal_energy/torque_allocation_preload.py",
    "scripts/research/proximal_distal_energy/run_torque_allocation_preload_study.py",
    "tests/research/test_torque_allocation_preload.py",
    "tests/research/test_torque_allocation_preload_evidence.py",
]


def _claim(
    claim_id: str,
    candidate_ids: list[str],
    statement: str,
    classification: str,
    status: str,
    domain: str,
    boundary: str,
    falsifier: str,
) -> dict[str, object]:
    return {
        "claim_id": claim_id,
        "candidate_ids": candidate_ids,
        "statement": statement,
        "classification": classification,
        "published_status": status,
        "audit_status": "allocation_closure_transmission_sensitivity_and_boundaries_checked",
        "source_locations": [],
        "evidence_artifacts": ARTIFACTS,
        "model_domain": domain,
        "uncertainty_boundary": boundary,
        "competing_explanations": [
            "changed trajectory rather than allocation",
            "alternative cost weights",
            "unmeasured bilateral internal force",
            "continuous biological compliance rather than a dead zone",
            "subject-specific activation and tissue properties",
        ],
        "negative_controls": [
            "same state and matched club task",
            "direct-moment plus force-couple closure",
            "relaxed versus preloaded initialization",
            "continuous finite preparation history",
            "zero-dead-zone equivalence",
            "participant-held-out measurement protocol",
        ],
        "falsifier": falsifier,
        "adjudication": (
            "All chapter passages were checked against regenerated allocation and "
            "transmission evidence, original-source boundaries, explicit numerical "
            "equivalence, temporal resolution, and the governed human-data gate."
        ),
        "reviewer": "Codex technical audit",
        "last_verified_on": "2026-08-13",
    }


def main() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    candidates = [
        candidate
        for candidate in inventory["candidates"]
        if candidate["source_path"] == CHAPTER
    ]
    ids = [candidate["candidate_id"] for candidate in candidates]
    by_id = {candidate["candidate_id"]: candidate for candidate in candidates}
    groups = {
        "PD-CLAIM-200": ids[:6],
        "PD-CLAIM-201": ids[6:10],
        "PD-CLAIM-202": ids[10:13],
        "PD-CLAIM-203": ids[13:19],
        "PD-CLAIM-204": ids[19:23],
        "PD-CLAIM-205": ids[23:26],
        "PD-CLAIM-206": ids[26:],
    }
    claims = [
        _claim(
            "PD-CLAIM-200",
            groups["PD-CLAIM-200"],
            "Direct wrist moment and the moment of separated hand forces are mechanically distinguishable but not uniquely recoverable from club kinematics; current EMG and grip literature motivates rather than identifies allocation.",
            "mechanism_identifiability_and_literature_boundary",
            "partly_supported_mechanics_biological_allocation_unidentified",
            "Same-state club mechanics plus observational literature.",
            "Scapular or wrist EMG does not establish hand-force direction, joint torque, or a unique club-moment pathway.",
            "Kinematics alone uniquely recover bilateral allocation or cited EMG is presented as torque/pathway proof.",
        ),
        _claim(
            "PD-CLAIM-201",
            groups["PD-CLAIM-201"],
            "Minimum-norm proximal and wrist control subspaces and their convex mixtures reproduce the same 8 N m same-state club task, with direct wrist moment plus grip-force couple closing to machine precision.",
            "matched_task_allocation_equivalence",
            "supported_for_declared_same_state_task",
            "Planar constrained two-arm model at 19 geometries and 21 allocation fractions.",
            "The proximal subspace is generalized shoulder/elbow torque and is not a measured scapular action.",
            "Task error exceeds 1e-10 N m or moment partition fails closure.",
        ),
        _claim(
            "PD-CLAIM-202",
            groups["PD-CLAIM-202"],
            "Task-equivalent allocations have geometry-dependent hand-force and generalized-torque demands, so no allocation optimum exists without an explicitly weighted multi-objective cost.",
            "geometry_and_objective_dependent_internal_demand",
            "supported_for_declared_unweighted_metrics",
            "Synthetic planar allocation surface.",
            "RMS hand force and Euclidean generalized-torque norm are not physiological effort, safety, comfort, or accuracy measures.",
            "One allocation minimizes every preregistered objective across geometry or the paper asserts a universal optimum.",
        ),
        _claim(
            "PD-CLAIM-203",
            groups["PD-CLAIM-203"],
            "The transmission experiment defines slack narrowly as zero transmitted torque in a synthetic rotational dead zone and compares persistent-direction and complete-role-reversal histories without state reset.",
            "operational_dead_zone_and_history_contract",
            "supported_as_phenomenological_model_definition",
            "Two abstract first-order dead-zone channels.",
            "The parameters are manufactured and the preparation interval is not an anatomical backswing.",
            "The model is described as literal tendon, wrist, grip, or scapular backlash.",
        ),
        _claim(
            "PD-CLAIM-204",
            groups["PD-CLAIM-204"],
            "At the declared positive dead zone, persistent direction preserves transmission and lowers torque-error impulse; zero-sample occupancies are resolution-bounded, and zero dead zone makes the programs equivalent.",
            "conditional_preload_continuity_with_equivalence_region",
            "conditional_on_declared_dead_zone_family",
            "Four dead zones by three time constants with a 0.1 ms temporal grid.",
            "All nine positive-dead-zone cases favor persistence, all three zero-dead-zone cases are equivalent, and no biological dead zone has been identified.",
            "The advantage persists when the defining dead zone is removed, reverses within the registered grid, or temporal precision exceeds its one-step bracket.",
        ),
        _claim(
            "PD-CLAIM-205",
            groups["PD-CLAIM-205"],
            "Persistent arm drive has only a conditional advantage under the backlash-like transmission law; wrist-led preparation may trade proximal load, elastic storage, or accuracy in ways the current model does not price.",
            "critical_strategy_comparison_boundary",
            "no_universal_strategy_supported",
            "Comparison of two extreme command allocations in reduced models.",
            "Neither extreme maps uniquely to anatomy or includes calibrated elasticity, neural control, fatigue, injury, or accuracy.",
            "Either extreme is presented as universally superior or as a coaching prescription.",
        ),
        _claim(
            "PD-CLAIM-206",
            groups["PD-CLAIM-206"],
            "Bilateral six-axis wrenches, pressure, constrained inverse dynamics, stiffness/delay, EMG, shaft, launch, and participant holdout are required to test the allocation and preload predictions; those human results remain unexecuted.",
            "registered_measurement_and_human_falsification_gate",
            "untested_no_governed_human_data",
            "Frozen prospective measurement plan.",
            "EMG timing alone and net club wrench are insufficient to identify the bilateral pathway.",
            "Measured allocation forces or delays fail the prediction, held-out benefit disappears, or human support is claimed without the registered modalities.",
        ),
    ]
    claim_by_candidate = {
        candidate_id: claim_id
        for claim_id, candidate_ids in groups.items()
        for candidate_id in candidate_ids
    }
    registry["candidate_reviews"] = [
        review
        for review in registry["candidate_reviews"]
        if review["candidate_id"] not in claim_by_candidate
        and not set(review["claim_ids"]).intersection(groups)
    ]
    registry["candidate_reviews"].extend(
        {
            "candidate_id": candidate_id,
            "disposition": "material_claims_mapped",
            "claim_ids": [claim_id],
            "rationale": (
                "This passage participates in the registered allocation, "
                "transmission, literature-boundary, or human-falsification audit."
            ),
            "reviewer": "Codex technical audit",
            "last_verified_on": "2026-08-13",
        }
        for candidate_id, claim_id in claim_by_candidate.items()
    )
    registry["claims"] = [
        claim for claim in registry["claims"] if claim["claim_id"] not in groups
    ] + claims
    for claim in claims:
        claim["source_locations"] = [
            f"{by_id[candidate_id]['source_path']}:{by_id[candidate_id]['line_start']}"
            for candidate_id in claim["candidate_ids"]
        ]
    for item in registry["release_claim_inventory"]:
        if item["release_claim_key"] in {
            "arm_wrist_allocation_equivalence",
            "preload_continuity_advantage",
        }:
            item["audit_state"] = "reviewed"
        if item["release_claim_key"] in {
            "scapular_or_muscle_strategy_identification",
            "universal_control_strategy",
        }:
            item["audit_state"] = "reviewed_as_unsupported"
    registry["paper"]["source_digest"] = inventory["source_digest"]
    registry["audit_scope"]["current_scope"] = (
        "The complete candidate audit remains in progress. The torque-allocation "
        "and transmission-preload chapter is fully adjudicated, including explicit "
        "zero-dead-zone equivalence, temporal-resolution bounds, literature limits, "
        "and the governed human-measurement gate."
    )
    REGISTRY.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
