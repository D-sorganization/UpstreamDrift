"""Re-adjudicate the expanded proximal rate and acceleration chapter."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ARTICLE = ROOT / "docs/research/proximal_distal_energy_transfer"
REGISTRY = ARTICLE / "data/claim_audit_registry.json"
INVENTORY = ARTICLE / "data/claim_candidate_inventory.json"
CHAPTER = "docs/research/proximal_distal_energy_transfer/chapters/_ch03d_shoulder_velocity_transfer.qmd"
ARTIFACTS = [
    CHAPTER,
    "docs/research/proximal_distal_energy_transfer/data/shoulder_velocity_transfer_study.json",
    "docs/research/proximal_distal_energy_transfer/data/shoulder_velocity_transfer_study.npz",
    "docs/research/proximal_distal_energy_transfer/data/shoulder_velocity_strategy_study.json",
    "docs/research/proximal_distal_energy_transfer/data/shoulder_velocity_strategy_study.npz",
    "docs/research/proximal_distal_energy_transfer/data/proximal_acceleration_transfer_study.json",
    "docs/research/proximal_distal_energy_transfer/data/proximal_acceleration_transfer_study.npz",
]


def main() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    abstract_candidate = next(
        item["candidate_id"]
        for item in inventory["candidates"]
        if item["source_path"]
        == "docs/research/proximal_distal_energy_transfer/proximal_distal_energy_transfer.qmd"
        and item["line_start"] == 10
    )
    superseded_abstract_candidate = "PD-CAND-384556ac26fc8724"
    for claim in registry["claims"]:
        claim["candidate_ids"] = [
            abstract_candidate if item == superseded_abstract_candidate else item
            for item in claim["candidate_ids"]
        ]
    for review in registry["candidate_reviews"]:
        if review["candidate_id"] == superseded_abstract_candidate:
            review["candidate_id"] = abstract_candidate
    candidates = [
        item for item in inventory["candidates"] if item["source_path"] == CHAPTER
    ]
    by_line = {item["line_start"]: item for item in candidates}
    line_groups = {
        "PD-CLAIM-081": (5, 16),
        "PD-CLAIM-082": (31, 44),
        "PD-CLAIM-083": (54, 58, 73),
        "PD-CLAIM-084": (92, 96, 113, 123, 127, 138),
        "PD-CLAIM-249": (144, 153, 157),
        "PD-CLAIM-085": (173, 182),
        "PD-CLAIM-086": (187,),
        "PD-CLAIM-250": (201,),
        "PD-CLAIM-087": (210, 214),
        "PD-CLAIM-088": (225, 229),
        "PD-CLAIM-235": (240, 256, 264, 275),
        "PD-CLAIM-236": (285, 289, 301, 310, 315),
        "PD-CLAIM-237": (327, 338),
        "PD-CLAIM-238": (342, 354),
        "PD-CLAIM-089": (364, 367, 391),
        "PD-CLAIM-090": (401, 405, 416),
        "PD-CLAIM-091": (429, 432, 445),
        "PD-CLAIM-092": (459, 474),
    }
    groups = {
        claim_id: [by_line[line]["candidate_id"] for line in lines]
        for claim_id, lines in line_groups.items()
    }
    statements = {
        "PD-CLAIM-081": "The reduced coordinate is fixed-hub first-link rate, not anatomical shoulder or thorax velocity, and cited human associations do not identify causal transfer or bilateral wrench allocation.",
        "PD-CLAIM-082": "The declared drift-control acceleration, interface-force, and same-point power decomposition closes pointwise but is not passive human motion.",
        "PD-CLAIM-083": "Relative-rate, absolute-rate, and total-kinetic-energy matching are distinct counterfactuals; the energy-matched quadratic is nondegenerate only after the zero-energy transition state.",
        "PD-CLAIM-084": "The 126-case rate atlas is phase- and matching-rule-dependent; exact energy matching removes the large monotonic pre-impact trend and retains reversals and adverse power regions.",
        "PD-CLAIM-249": "The 45-case identical-state acceleration intervention closes exactly and shows that interface-power response and club-angular-acceleration response can have opposite signs before impact while required proximal torque and power vary materially.",
        "PD-CLAIM-085": "The forward experiment retains all 60 timing and proximal-drive programs, including 34 that miss the registered impact event.",
        "PD-CLAIM-086": "Among 26 valid programs, release-state proximal rate has adverse descriptive speed and braking associations after a declared regression, without causal or human identification.",
        "PD-CLAIM-250": "Eight approximate net- and positive-actuator-work-matched pairs all favor the lower-rate member, but reused pairs and the absence of a simultaneously load-matched pair prevent a joint causal estimate.",
        "PD-CLAIM-087": "The highest-speed sampled program and fixed-step refinement retain exact interface-work closure while exposing a large modeled net interface load.",
        "PD-CLAIM-088": "The sampled speed, braking, and peak-force Pareto screen is weakly discriminating and does not identify unique human strategies.",
        "PD-CLAIM-235": "The rotating-base tier is a finite-inertia, constrained bilateral planar mechanism with explicit contact-power and energy checks.",
        "PD-CLAIM-236": "The rotating-base result depends on rate-matching contract and retains speed, braking, force-couple, and load outcomes rather than a universal torso-speed benefit.",
        "PD-CLAIM-237": "Exact same-state torso, arm, and wrist killswitches produce nonmonotonic channel effects in the registered program.",
        "PD-CLAIM-238": "Coincident and reversed moment-arm controls verify the bilateral couple geometry while preserving the human-evidence boundary.",
        "PD-CLAIM-089": "The chapter states testable design hypotheses rather than instructions and requires actuator, interface, club-energy, and external-work accounting.",
        "PD-CLAIM-090": "Reduced models cannot identify bilateral hand or anatomical torso strategy, and sampled optima require explicit load and robustness costs.",
        "PD-CLAIM-091": "A human test requires synchronized declared-frame kinematics, bilateral identified wrenches, impact outcomes, participant holdout, and preregistered falsifiers.",
        "PD-CLAIM-092": "The expanded deterministic evidence contains 126 rate cases, 45 acceleration cases, and 60 forward programs and supports only model-bounded counterexamples to a standalone proximal-rate rule.",
    }
    old_ids = set(groups)
    mapped = {candidate for members in groups.values() for candidate in members}
    if mapped != {item["candidate_id"] for item in candidates}:
        raise RuntimeError("proximal dose candidate map is incomplete or overlapping")
    registry["candidate_reviews"] = [
        review
        for review in registry["candidate_reviews"]
        if review["candidate_id"] not in mapped
        and not set(review["claim_ids"]).intersection(old_ids)
    ]
    registry["claims"] = [
        claim for claim in registry["claims"] if claim["claim_id"] not in old_ids
    ]
    by_id = {item["candidate_id"]: item for item in candidates}
    for claim_id, members in groups.items():
        registry["claims"].append(
            {
                "claim_id": claim_id,
                "candidate_ids": members,
                "statement": statements[claim_id],
                "classification": "mechanistic_model_evidence_and_boundary",
                "published_status": "supported_at_declared_synthetic_tiers",
                "audit_status": "evidence_boundary_and_falsification_contract_checked",
                "source_locations": [
                    f"{CHAPTER}:{by_id[item]['line_start']}" for item in members
                ],
                "evidence_artifacts": ARTIFACTS,
                "model_domain": "Declared planar pointwise and forward mechanism tiers, with a separate rotating-base bilateral tier.",
                "uncertainty_boundary": "No anatomical, subject-scaled, causal human, or universal coaching inference is supported.",
                "competing_explanations": [
                    "stored kinetic energy",
                    "relative-state matching",
                    "actuator work",
                    "interface load",
                    "timing and valid-impact selection",
                ],
                "negative_controls": [
                    "energy matching",
                    "same-state acceleration intervention",
                    "approximate work matching",
                    "same-state channel killswitch",
                    "geometry null and reversal",
                ],
                "falsifier": "The registered closure, null, sign, work, load, spatial, or participant-held-out control contradicts the stated model-bounded result.",
                "adjudication": "Reconciled to executable artifacts without promoting pointwise or synthetic evidence to a human strategy.",
                "reviewer": "Codex technical audit",
                "last_verified_on": "2026-08-13",
            }
        )
        registry["candidate_reviews"].extend(
            {
                "candidate_id": item,
                "disposition": "material_claims_mapped",
                "claim_ids": [claim_id],
                "rationale": "This passage states or bounds a registered proximal rate or acceleration result.",
                "reviewer": "Codex technical audit",
                "last_verified_on": "2026-08-13",
            }
            for item in members
        )
    registry["paper"]["source_digest"] = inventory["source_digest"]
    registry["audit_scope"]["completion_status"] = "complete"
    registry["audit_scope"]["current_scope"] = (
        "The complete 945-candidate paper inventory is adjudicated, including exact energy matching, actuator-work accounting, and identical-state acceleration interventions."
    )
    REGISTRY.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
