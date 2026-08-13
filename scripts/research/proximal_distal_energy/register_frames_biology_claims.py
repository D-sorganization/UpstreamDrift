"""Register the frame, reduced-biological, and human-boundary audit slice."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ARTICLE = ROOT / "docs" / "research" / "proximal_distal_energy_transfer"
REGISTRY = ARTICLE / "data" / "claim_audit_registry.json"
INVENTORY = ARTICLE / "data" / "claim_candidate_inventory.json"
FRAME_CHAPTER = (
    "docs/research/proximal_distal_energy_transfer/chapters/"
    "_ch07b_frames_biology_engines.qmd"
)
HUMAN_CHAPTER = (
    "docs/research/proximal_distal_energy_transfer/chapters/"
    "_ch06e_experimental_protocol.qmd"
)


def _claim(
    claim_id: str,
    candidate_ids: list[str],
    statement: str,
    classification: str,
    published_status: str,
    artifacts: list[str],
    model_domain: str,
    uncertainty: str,
    falsifier: str,
) -> dict[str, object]:
    return {
        "claim_id": claim_id,
        "candidate_ids": candidate_ids,
        "statement": statement,
        "classification": classification,
        "published_status": published_status,
        "audit_status": "frames_biology_refinement_and_human_boundary_checked",
        "source_locations": [],
        "evidence_artifacts": artifacts,
        "model_domain": model_domain,
        "uncertainty_boundary": uncertainty,
        "competing_explanations": [
            "coordinate or reference-point convention error",
            "reduced muscle-law structural error",
            "generic uncalibrated parameters",
            "timestep sensitivity",
            "missing synchronized human modalities",
        ],
        "negative_controls": [
            "proper rotation and reference transport",
            "Jacobian virtual work",
            "matched-moment activation null space",
            "timestep refinement",
            "participant-level holdout",
            "synthetic-data boundary",
        ],
        "falsifier": falsifier,
        "adjudication": (
            "Every passage in the frame/biological chapter and experimental "
            "protocol chapter was reviewed against regenerated evidence and the "
            "verified #8555/#8556 external state."
        ),
        "reviewer": "Codex technical audit",
        "last_verified_on": "2026-08-13",
    }


def main() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    by_path: dict[str, list[dict[str, object]]] = {FRAME_CHAPTER: [], HUMAN_CHAPTER: []}
    by_id: dict[str, dict[str, object]] = {}
    for candidate in inventory["candidates"]:
        by_id[candidate["candidate_id"]] = candidate
        if candidate["source_path"] in by_path:
            by_path[candidate["source_path"]].append(candidate)

    frame_ids = [candidate["candidate_id"] for candidate in by_path[FRAME_CHAPTER]]
    human_ids = [candidate["candidate_id"] for candidate in by_path[HUMAN_CHAPTER]]
    groups = {
        "PD-CLAIM-194": frame_ids[:12],
        "PD-CLAIM-195": frame_ids[12:17],
        "PD-CLAIM-196": frame_ids[17:23],
        "PD-CLAIM-197": frame_ids[23:30],
        "PD-CLAIM-198": frame_ids[30:],
        "PD-CLAIM-199": human_ids,
    }
    artifacts = [
        "docs/research/proximal_distal_energy_transfer/data/advanced_biological_bridge.json",
        "docs/research/proximal_distal_energy_transfer/data/advanced_biological_bridge.npz",
        "scripts/research/proximal_distal_energy/advanced_biological_bridge.py",
        "scripts/research/proximal_distal_energy/run_advanced_biological_bridge.py",
        "tests/research/test_advanced_biological_bridge.py",
    ]
    claims = [
        _claim(
            "PD-CLAIM-194",
            groups["PD-CLAIM-194"],
            "Proper rotations, reference-point transport, and Jacobian transpose mapping preserve physical power to floating-point tolerance under the declared wrench/twist convention.",
            "frame_reference_and_virtual_work_identity",
            "supported_to_declared_numerical_tolerance",
            artifacts,
            "Deterministic algebraic three-dimensional samples.",
            "Closure verifies implementation identities, not measured accuracy or model validity.",
            "Any properly paired transformation changes power beyond 1e-11 W.",
        ),
        _claim(
            "PD-CLAIM-195",
            groups["PD-CLAIM-195"],
            "A net generalized moment does not uniquely identify muscle forces; a reduced agonist-antagonist family closes the same 10 N m moment while internal-force proxies vary.",
            "reduced_muscle_redundancy_identifiability",
            "supported_at_reduced_hill_type_tier",
            artifacts,
            "Generic isometric Hill-type pair with constant moment arms.",
            "The stiffness and energy values are proxies and do not identify anatomy, effort, injury, or preferred co-contraction.",
            "Moment closure fails or the internal-force family is unique under the stated underdetermined map.",
        ),
        _claim(
            "PD-CLAIM-196",
            groups["PD-CLAIM-196"],
            "Reduced activation and series-force states make preparation history observable, but the persistent-direction advantage is below one percent at the published step and its magnitude is not timestep converged.",
            "reduced_preparation_history_with_numerical_boundary",
            "direction_stable_magnitude_nonconverged",
            artifacts,
            "Generic four-channel first-order activation and series-force model.",
            "No scapular mechanism, passive wrist, physiological benefit, or performance effect is established.",
            "The sign disappears on the registered step grid or a quantitative physiological claim is made from the nonconverged magnitude.",
        ),
        _claim(
            "PD-CLAIM-197",
            groups["PD-CLAIM-197"],
            "The engine ladder separates executed reduced results from proposed Drake, OpenSim, and MyoSuite validation; five adapter round trips test coordinate representation only.",
            "engine_capability_and_representation_boundary",
            "supported_for_coordinate_representation_only",
            artifacts,
            "Lightweight adapter encode/decode plus declared engine roles.",
            "No five-engine dynamics, contact, anatomy, or human parity is established.",
            "Optional capability or adapter parity is presented as executed dynamics or human validation.",
        ),
        _claim(
            "PD-CLAIM-198",
            groups["PD-CLAIM-198"],
            "The next executable tiers require dynamic-state parity, subject-scaled paths, activation-driven replay, multi-objective optimization, and governed synchronized measurement before biological fitting.",
            "explicit_open_research_and_falsification_boundary",
            "explicitly_untested_next_tiers",
            artifacts,
            "Repository-managed research plan.",
            "Actionability does not imply that any listed result has been executed.",
            "An unexecuted tier is described as completed evidence.",
        ),
        _claim(
            "PD-CLAIM-199",
            groups["PD-CLAIM-199"],
            "The participant-held-out bilateral-wrench protocol is implemented, but no qualifying governed human dataset was found; all human hypotheses remain untested and synthetic or literature-only substitutes are prohibited.",
            "governed_human_acquisition_boundary",
            "untested_no_governed_human_data",
            [
                "docs/research/proximal_distal_energy_transfer/data/experimental_protocol_v1.json",
                "docs/research/proximal_distal_energy_transfer/data/experimental_protocol_dry_run.json",
                "docs/research/proximal_distal_energy_transfer/EXPERIMENTAL_FALSIFICATION_PROTOCOL.md",
                "scripts/research/proximal_distal_energy/experimental_protocol.py",
                "tests/research/test_experimental_falsification_protocol.py",
            ],
            "Frozen acquisition and analysis contract; no governed participant outcomes.",
            "Public instrumented-grip papers are design references, not participant-level data authorities satisfying the contract.",
            "Human support is claimed without qualifying governed held-out data or synthetic/digitized evidence is substituted.",
        ),
    ]

    claim_by_candidate = {
        candidate_id: claim_id
        for claim_id, candidate_ids in groups.items()
        for candidate_id in candidate_ids
    }
    retained = [
        review
        for review in registry["candidate_reviews"]
        if review["candidate_id"] not in claim_by_candidate
    ]
    for candidate_id, claim_id in claim_by_candidate.items():
        retained.append(
            {
                "candidate_id": candidate_id,
                "disposition": "material_claims_mapped",
                "claim_ids": [claim_id],
                "rationale": (
                    "This passage participates in the registered frame, reduced "
                    "biological, engine-boundary, or governed human-acquisition audit."
                ),
                "reviewer": "Codex technical audit",
                "last_verified_on": "2026-08-13",
            }
        )
    registry["candidate_reviews"] = retained
    registry["claims"] = [
        claim for claim in registry["claims"] if claim["claim_id"] not in groups
    ] + claims
    for claim in registry["claims"]:
        claim["source_locations"] = [
            f"{by_id[candidate_id]['source_path']}:{by_id[candidate_id]['line_start']}"
            for candidate_id in claim["candidate_ids"]
        ]
    for item in registry["release_claim_inventory"]:
        if item["release_claim_key"] in {
            "reference_frame_power_invariance",
            "muscle_redundancy_same_moment",
            "canonical_pose_adapter_round_trip",
        }:
            item["audit_state"] = "reviewed"
        if item["release_claim_key"] in {
            "scapular_or_muscle_strategy_identification",
            "human_experimental",
            "human_torso_velocity_strategy",
            "drake_opensim_myosuite_human_validation",
        }:
            item["audit_state"] = "reviewed_as_unsupported_or_untested"
    registry["paper"]["source_digest"] = inventory["source_digest"]
    registry["audit_scope"]["current_scope"] = (
        "The complete candidate audit remains in progress. The frame, reduced "
        "biological, engine-boundary, and governed human-protocol chapters are "
        "fully adjudicated; #8555 is merged, while #8556 remains open at its "
        "external bilateral-wrench acquisition boundary."
    )
    REGISTRY.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
