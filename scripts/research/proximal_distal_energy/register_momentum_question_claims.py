"""Adjudicate the critical momentum-transfer question chapter."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ARTICLE = ROOT / "docs/research/proximal_distal_energy_transfer"
REGISTRY = ARTICLE / "data/claim_audit_registry.json"
INVENTORY = ARTICLE / "data/claim_candidate_inventory.json"
CHAPTER = "docs/research/proximal_distal_energy_transfer/chapters/_ch08b_momentum_transfer_questions.qmd"
ARTIFACTS = [
    CHAPTER,
    "docs/research/proximal_distal_energy_transfer/MOMENTUM_TRANSFER_QUESTION_PROGRAM.md",
    "docs/research/proximal_distal_energy_transfer/data/momentum_transfer_question_registry.json",
    "docs/research/proximal_distal_energy_transfer/data/momentum_transfer_experiment_registry.json",
]


def main() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    candidates = [
        item for item in inventory["candidates"] if item["source_path"] == CHAPTER
    ]
    by_line = {item["line_start"]: item for item in candidates}
    groups = {
        "PD-CLAIM-242": [by_line[line]["candidate_id"] for line in (3, 12, 22)],
        "PD-CLAIM-243": [by_line[14]["candidate_id"]],
        "PD-CLAIM-244": [by_line[line]["candidate_id"] for line in (28, 38, 41)],
        "PD-CLAIM-245": [by_line[line]["candidate_id"] for line in (53, 63)],
        "PD-CLAIM-246": [by_line[line]["candidate_id"] for line in (71, 82)],
        "PD-CLAIM-247": [by_line[line]["candidate_id"] for line in (88, 108, 115)],
        "PD-CLAIM-248": [by_line[123]["candidate_id"]],
    }
    definitions = {
        "PD-CLAIM-242": (
            "The seven-question map reports model- and estimand-bounded answers rather than a universal transfer fraction, release time, or human technique.",
            "question_map_scope",
            "supported_as_scientific_governance",
            "The map summarizes existing adjudicated evidence and prospective tests; it is not new experimental evidence.",
            "A conditional entry is presented as a universal or human conclusion.",
        ),
        "PD-CLAIM-243": (
            "The seven-question table retains observable-specific drift attribution and bounded current answers after adding the geometry and recovery studies.",
            "critical_question_answer_map",
            "model_and_estimand_bounded",
            "A table entry is a synthesis of registered evidence rather than an independent experiment.",
            "A conditional model entry is presented as a universal or human conclusion.",
        ),
        "PD-CLAIM-244": (
            "Force--velocity and relative-link angles provide exact power/projection gates, while signed grip separation and transverse differential force determine bilateral couple sign and zeros across declared planar and spatial controls.",
            "geometry_gate_atlas",
            "supported_through_declared_spatial_mechanism_controls",
            "The identities and model controls do not establish feasible or preferred subject-scaled human geometry.",
            "An orthogonal, coincident, axial, reversed, or proper-frame control fails beyond registered tolerance.",
        ),
        "PD-CLAIM-245": (
            "Casting, timing demand, and self-correction require measurable event, sensitivity, and perturbation-recovery definitions; state-triggered synthetic performance does not establish reduced human timing demand.",
            "operational_timing_and_recovery_contract",
            "registered_and_human_unresolved",
            "Observer, recovery, and participant experiments have not been completed.",
            "Repeatability or open-loop sensitivity is relabeled self-correction or human timing economy.",
        ),
        "PD-CLAIM-246": (
            "A paired 15-case planar screen found sustained half-error recovery in 13--20 percent of cases, with no decisive policy separation; transient threshold crossing is not self-correction.",
            "trajectory_recovery_screen",
            "no_policy_recovery_advantage_established",
            "The result is limited to a small engineering envelope and a simple delayed observer; it does not estimate human correction.",
            "A policy advantage emerges under expanded preregistered attraction-region, external-load, or participant-held-out tests.",
        ),
        "PD-CLAIM-247": (
            "Energy- and approximate-work-controlled planar evidence rejects maximizing proximal velocity as a general rule at those tiers, while five distinct slack classes require separate state, energy, and measurement tests.",
            "nonmonotonic_velocity_and_typed_slack_boundary",
            "proximal_rule_rejected_at_planar_tiers_slack_mostly_open",
            "The acceleration intervention is pointwise, no work-matched pair is also load matched, and only a phenomenological transmission dead zone has been directly exercised; no human optimum is identified.",
            "A planar reversal is promoted to a human optimum or one slack class is inferred from another.",
        ),
        "PD-CLAIM-248": (
            "Synthetic studies can falsify declared model mechanisms and qualify software but cannot establish coaching or human-control strategies; governed participant-held-out bilateral-wrench data remain required.",
            "human_evidence_boundary",
            "human_validation_blocked",
            "No qualifying governed participant-level dataset is available.",
            "Synthetic traces are substituted for the registered human stage.",
        ),
    }
    old_ids = set(groups)
    mapped_ids = {candidate for members in groups.values() for candidate in members}
    registry["candidate_reviews"] = [
        review
        for review in registry["candidate_reviews"]
        if review["candidate_id"] not in mapped_ids
        and not set(review["claim_ids"]).intersection(old_ids)
    ]
    registry["claims"] = [
        claim for claim in registry["claims"] if claim["claim_id"] not in old_ids
    ]
    by_id = {item["candidate_id"]: item for item in candidates}
    for claim_id, members in groups.items():
        statement, classification, status, boundary, falsifier = definitions[claim_id]
        registry["claims"].append(
            {
                "claim_id": claim_id,
                "candidate_ids": members,
                "statement": statement,
                "classification": classification,
                "published_status": status,
                "audit_status": "evidence_boundary_and_falsification_contract_checked",
                "source_locations": [
                    f"{CHAPTER}:{by_id[item]['line_start']}" for item in members
                ],
                "evidence_artifacts": ARTIFACTS,
                "model_domain": "Cross-tier synthesis and prospective experiment governance.",
                "uncertainty_boundary": boundary,
                "competing_explanations": [
                    "model-tier dependence",
                    "estimand choice",
                    "matching rule",
                    "measurement uncertainty",
                ],
                "negative_controls": [
                    "same-state killswitch",
                    "geometry null and reversal",
                    "matched work and state",
                    "participant holdout",
                ],
                "falsifier": falsifier,
                "adjudication": "Reconciled to the existing claim audit, machine-readable question registry, and prospective experiment contract without promoting model evidence to human evidence.",
                "reviewer": "Codex technical audit",
                "last_verified_on": "2026-08-13",
            }
        )
        registry["candidate_reviews"].extend(
            {
                "candidate_id": item,
                "disposition": "material_claims_mapped",
                "claim_ids": [claim_id],
                "rationale": "This passage states or bounds a registered critical-question conclusion.",
                "reviewer": "Codex technical audit",
                "last_verified_on": "2026-08-13",
            }
            for item in members
        )
    registry["paper"]["source_digest"] = inventory["source_digest"]
    registry["audit_scope"]["completion_status"] = "complete"
    registry["audit_scope"]["current_scope"] = (
        "The complete 945-candidate paper inventory is adjudicated. The critical-question chapter maps fifteen candidates to seven bounded claims and retains all human-data gates."
    )
    REGISTRY.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
