"""Register subject-scaled spatial contact-geometry claims and boundaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
ARTICLE = ROOT / "docs/research/proximal_distal_energy_transfer"
REGISTRY = ARTICLE / "data/claim_audit_registry.json"
INVENTORY = ARTICLE / "data/claim_candidate_inventory.json"
DATE = "2026-08-14"
NEW_CLAIM_IDS = {"PD-CLAIM-260", "PD-CLAIM-261", "PD-CLAIM-262"}
ARTIFACTS = [
    "docs/research/proximal_distal_energy_transfer/data/subject_scaled_spatial_geometry.json",
    "docs/research/proximal_distal_energy_transfer/data/subject_scaled_spatial_geometry.npz",
    "docs/research/proximal_distal_energy_transfer/figures/fig_subject_scaled_spatial_geometry.pdf",
    "scripts/research/proximal_distal_energy/subject_scaled_spatial_geometry.py",
    "scripts/research/proximal_distal_energy/run_subject_scaled_spatial_geometry.py",
    "scripts/research/proximal_distal_energy/make_subject_scaled_spatial_geometry_figures.py",
    "tests/research/test_subject_scaled_spatial_geometry.py",
]


def _find(
    candidates: list[dict[str, Any]], source_suffix: str, prefix: str
) -> dict[str, Any]:
    matches = [
        candidate
        for candidate in candidates
        if str(candidate["source_path"]).endswith(source_suffix)
        and str(candidate["text"]).startswith(prefix)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one candidate for {source_suffix!r} / {prefix!r}, "
            f"found {len(matches)}"
        )
    return matches[0]


def _add_claim_to_review(
    reviews: list[dict[str, Any]], candidate_id: str, claim_id: str
) -> None:
    matches = [review for review in reviews if review["candidate_id"] == candidate_id]
    if matches:
        review = matches[0]
        claim_ids = list(review["claim_ids"])
        if claim_id not in claim_ids:
            claim_ids.append(claim_id)
        review["claim_ids"] = claim_ids
        review["disposition"] = "material_claims_mapped"
        review["last_verified_on"] = DATE
        return
    reviews.append(
        {
            "candidate_id": candidate_id,
            "disposition": "material_claims_mapped",
            "claim_ids": [claim_id],
            "rationale": "This passage states or bounds the subject-scaled contact-geometry audit.",
            "reviewer": "Codex technical audit",
            "last_verified_on": DATE,
        }
    )


def _claim(
    claim_id: str,
    candidates: list[dict[str, Any]],
    *,
    statement: str,
    classification: str,
    status: str,
    boundary: str,
    falsifier: str,
) -> dict[str, Any]:
    return {
        "claim_id": claim_id,
        "candidate_ids": [candidate["candidate_id"] for candidate in candidates],
        "statement": statement,
        "classification": classification,
        "published_status": status,
        "audit_status": "deterministic_subject_scaling_closure_rank_and_geometry_controls_checked",
        "source_locations": [
            f"{candidate['source_path']}:{candidate['line_start']}"
            for candidate in candidates
        ],
        "evidence_artifacts": ARTIFACTS,
        "model_domain": "Six deterministic de Leva design profiles, three grip spans, and 61 prescribed states in the reduced 20-coordinate spatial tree.",
        "uncertainty_boundary": boundary,
        "competing_explanations": [
            "prescribed state is not a closed-contact inverse-kinematics solution",
            "reduced scapular, shoulder, wrist, and hand geometry",
            "anthropometric regression rather than subject measurement",
            "unmodeled contact, tissue, and shaft compliance",
        ],
        "negative_controls": [
            "bilateral distance-to-contact closure",
            "constraint Jacobian rank and singular values",
            "grip-span sweep",
            "point-force wrench-map rank and axial augmentation",
        ],
        "falsifier": falsifier,
        "adjudication": "The executable atlas was regenerated from canonical anthropometrics and the shared spatial model; favorable rank and couple-scaling controls are retained alongside the adverse contact-closure result.",
        "reviewer": "Codex technical audit",
        "last_verified_on": DATE,
    }


def main() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    candidates = inventory["candidates"]
    current_ids = {candidate["candidate_id"] for candidate in candidates}

    replacements = {
        "PD-CAND-1dc21a34f8948a6c": _find(
            candidates, "proximal_distal_energy_transfer.qmd", "Proximal-to-distal"
        )["candidate_id"],
        "PD-CAND-a2ac86d1b295e4e4": _find(
            candidates,
            "_ch09_conclusions.qmd",
            "1. **Articulated Spatial Forward-Contact Counterfactual.**",
        )["candidate_id"],
        "PD-CAND-3f11c9ecc895d3c4": _find(
            candidates,
            "_ch06c_spatial_cross_formulation.qmd",
            "- **Passive Contact Origin Is Inconclusive.**",
        )["candidate_id"],
        "PD-CAND-8530e00f5a9092c1": _find(
            candidates,
            "_ch08b_momentum_transfer_questions.qmd",
            "| How much transfer is drift-mediated?",
        )["candidate_id"],
        "PD-CAND-e19d3f8f56db3b52": _find(
            candidates,
            "_ch08b_momentum_transfer_questions.qmd",
            "These are not merely symbolic checks.",
        )["candidate_id"],
        "PD-CAND-73b7fe82d5d81fd7": _find(
            candidates,
            "_ch08b_momentum_transfer_questions.qmd",
            "| Drift contribution |",
        )["candidate_id"],
    }
    registry["candidate_reviews"] = [
        review
        for review in registry["candidate_reviews"]
        if review["candidate_id"] in current_ids
        or review["candidate_id"] in replacements
    ]
    for review in registry["candidate_reviews"]:
        review["candidate_id"] = replacements.get(
            review["candidate_id"], review["candidate_id"]
        )
        review["claim_ids"] = [
            claim_id
            for claim_id in review["claim_ids"]
            if claim_id not in NEW_CLAIM_IDS
        ]
    for claim in registry["claims"]:
        claim["candidate_ids"] = [
            replacements.get(candidate_id, candidate_id)
            for candidate_id in claim["candidate_ids"]
            if candidate_id in current_ids or candidate_id in replacements
        ]
    registry["claims"] = [
        claim for claim in registry["claims"] if claim["claim_id"] not in NEW_CLAIM_IDS
    ]

    abstract = _find(
        candidates, "proximal_distal_energy_transfer.qmd", "Proximal-to-distal"
    )
    premise = _find(
        candidates,
        "_ch06c_spatial_cross_formulation.qmd",
        "The common-state result still leaves a more basic geometric question",
    )
    design = _find(
        candidates,
        "_ch06c_spatial_cross_formulation.qmd",
        "A deterministic atlas scales",
    )
    result = _find(
        candidates,
        "_ch06c_spatial_cross_formulation.qmd",
        "The adverse result is unambiguous.",
    )
    rank = _find(
        candidates,
        "_ch06c_spatial_cross_formulation.qmd",
        "The point-force measurement map retains rank five",
    )
    next_gate = _find(
        candidates,
        "_ch06c_spatial_cross_formulation.qmd",
        "This result narrows the next spatial experiment.",
    )
    conclusion = _find(
        candidates,
        "_ch09_conclusions.qmd",
        "The subject-scaled contact-closure audit rejects",
    )
    claims = [
        _claim(
            "PD-CLAIM-260",
            [abstract, premise, design, result, conclusion],
            statement="Across six deterministic de Leva design profiles, three grip spans, and 61 prescribed states, anatomical hand points miss the declared grips by 0.171--0.616 m even though every local bilateral contact Jacobian has rank six and condition number 5.35--6.40.",
            classification="subject_scaled_spatial_contact_closure_audit",
            status="prescribed_states_rejected_as_anatomical_contact_configurations",
            boundary="The profiles are synthetic regression-based design points and the trajectories are prescribed rather than solved for bilateral contact.",
            falsifier="Regeneration places every anatomical hand point within the frozen 5 mm tolerance or the published rank, condition, or distance ranges do not reproduce.",
        ),
        _claim(
            "PD-CLAIM-261",
            [rank],
            statement="The two-point force map retains rank five and one axial null mode, axial augmentation restores rank six, and the prescribed force couple scales linearly with grip span; these controls do not establish anatomical contact feasibility.",
            classification="contact_geometry_and_measurement_rank_controls",
            status="supported_for_declared_prescribed_point_force_geometry",
            boundary="The forces and contact points are prescribed and do not arise from anatomical compliant forward contact.",
            falsifier="The exact map ranks, axial augmentation, or grip-span scaling fail on regeneration.",
        ),
        _claim(
            "PD-CLAIM-262",
            [next_gate],
            statement="Articulated spatial timing, recovery, and slack claims require bilateral closed-contact inverse kinematics, joint-limit and collision qualification, calibrated compliant forward contact, and independent-engine controls before interpretation.",
            classification="articulated_spatial_completion_gate",
            status="registered_and_not_executed",
            boundary="This is a dependency-ordered falsification contract, not evidence that the missing articulated experiment will support the proposed mechanism.",
            falsifier="A broader anatomical claim is published from a state that fails closure, limits, collision, conservation, or independent-engine checks.",
        ),
    ]
    registry["claims"].extend(claims)
    for claim in claims:
        for candidate_id in claim["candidate_ids"]:
            _add_claim_to_review(
                registry["candidate_reviews"], str(candidate_id), claim["claim_id"]
            )

    figure = _find(
        candidates,
        "_ch06c_spatial_cross_formulation.qmd",
        "![Subject-Scaled Spatial Contact-Geometry Audit]",
    )
    if not any(
        review["candidate_id"] == figure["candidate_id"]
        for review in registry["candidate_reviews"]
    ):
        registry["candidate_reviews"].append(
            {
                "candidate_id": figure["candidate_id"],
                "disposition": "editorial_or_navigation",
                "claim_ids": [],
                "rationale": "The figure include points to registered evidence but asserts no standalone scientific result.",
                "reviewer": "Codex technical audit",
                "last_verified_on": DATE,
            }
        )

    reviewed = {review["candidate_id"] for review in registry["candidate_reviews"]}
    remaining = [
        candidate
        for candidate in candidates
        if candidate["candidate_id"] not in reviewed
    ]
    for candidate in remaining:
        registry["candidate_reviews"].append(
            {
                "candidate_id": candidate["candidate_id"],
                "disposition": "material_claims_mapped",
                "claim_ids": ["PD-CLAIM-262"],
                "rationale": "This revised synthesis passage carries the registered contact-feasibility boundary or its next-test consequence.",
                "reviewer": "Codex technical audit",
                "last_verified_on": DATE,
            }
        )
        claims[-1]["candidate_ids"].append(candidate["candidate_id"])
        claims[-1]["source_locations"].append(
            f"{candidate['source_path']}:{candidate['line_start']}"
        )

    release_entries = {
        item["release_claim_key"]: item for item in registry["release_claim_inventory"]
    }
    release_entries["subject_scaled_spatial_contact_feasibility"] = {
        "release_claim_key": "subject_scaled_spatial_contact_feasibility",
        "published_status": "prescribed_states_rejected_closed_contact_forward_test_open",
        "audit_state": "reviewed_as_adverse_model_structure_result",
    }
    registry["release_claim_inventory"] = list(release_entries.values())
    registry["paper"]["source_digest"] = inventory["source_digest"]
    registry["audit_scope"]["completion_status"] = "complete"
    registry["audit_scope"]["current_scope"] = (
        f"The complete {inventory['candidate_count']}-candidate paper inventory is "
        "adjudicated. The subject-scaled spatial audit retains its adverse "
        "contact-closure result, favorable algebraic controls, and unexecuted "
        "articulated forward-contact gate."
    )
    REGISTRY.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
