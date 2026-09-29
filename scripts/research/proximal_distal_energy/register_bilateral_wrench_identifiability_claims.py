"""Register the bilateral-wrench structural-identifiability claims."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ARTICLE = ROOT / "docs/research/proximal_distal_energy_transfer"
REGISTRY = ARTICLE / "data/claim_audit_registry.json"
INVENTORY = ARTICLE / "data/claim_candidate_inventory.json"
CHAPTER = (
    "docs/research/proximal_distal_energy_transfer/chapters/_ch05_two_hand_wrench.qmd"
)
APPENDIX = "docs/research/proximal_distal_energy_transfer/chapters/_appendices.qmd"
DATE = "2026-08-14"
CLAIM_IDS = {"PD-CLAIM-254", "PD-CLAIM-255", "PD-CLAIM-256"}
ARTIFACTS = [
    "docs/research/proximal_distal_energy_transfer/data/bilateral_wrench_identifiability_study.json",
    "docs/research/proximal_distal_energy_transfer/figures/fig_bilateral_wrench_identifiability.pdf",
    "scripts/research/proximal_distal_energy/bilateral_wrench_identifiability.py",
    "scripts/research/proximal_distal_energy/run_bilateral_wrench_identifiability_study.py",
    "tests/research/test_bilateral_wrench_identifiability.py",
    "tests/research/test_bilateral_wrench_identifiability_evidence.py",
]


def _claim(
    claim_id: str,
    candidates: list[dict[str, object]],
    *,
    statement: str,
    classification: str,
    status: str,
    boundary: str,
    falsifier: str,
) -> dict[str, object]:
    return {
        "claim_id": claim_id,
        "candidate_ids": [str(candidate["candidate_id"]) for candidate in candidates],
        "statement": statement,
        "classification": classification,
        "published_status": status,
        "audit_status": "analytic_rank_manufactured_null_rotation_and_span_controls_checked",
        "source_locations": [
            f"{candidate['source_path']}:{candidate['line_start']}"
            for candidate in candidates
        ],
        "evidence_artifacts": ARTIFACTS,
        "model_domain": "Instantaneous linear map from two separated three-dimensional hand contacts to one net club wrench.",
        "uncertainty_boundary": boundary,
        "competing_explanations": [
            "free hand moments",
            "distributed contact pressure",
            "contact-center migration",
            "sensor cross-talk and synchronization error",
            "constitutive assumptions used to select one allocation",
        ],
        "negative_controls": [
            "manufactured equal-and-opposite axial null vector",
            "one-scalar axial measurement augmentation",
            "three proper spatial rotations",
            "forty-nine grip-span geometries",
            "explicit force and moment measurement scaling",
        ],
        "falsifier": falsifier,
        "adjudication": "The analytic map, numerical SVD, manufactured null mode, augmentation, spatial rotations, and neutral human boundary reconcile with the committed JSON and tests.",
        "reviewer": "Codex technical audit",
        "last_verified_on": DATE,
    }


def main() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    current_ids = {
        str(candidate["candidate_id"]) for candidate in inventory["candidates"]
    }
    registry["candidate_reviews"] = [
        review
        for review in registry["candidate_reviews"]
        if review["candidate_id"] in current_ids
        and not set(review["claim_ids"]).intersection(CLAIM_IDS)
    ]
    registry["claims"] = [
        claim for claim in registry["claims"] if claim["claim_id"] not in CLAIM_IDS
    ]

    reviewed = {review["candidate_id"] for review in registry["candidate_reviews"]}
    new_chapter = [
        candidate
        for candidate in inventory["candidates"]
        if candidate["source_path"] == CHAPTER
        and candidate["candidate_id"] not in reviewed
    ]
    if len(new_chapter) != 8:
        raise ValueError(f"expected 8 new chapter candidates, found {len(new_chapter)}")
    groups = {
        "PD-CLAIM-254": new_chapter[:4],
        "PD-CLAIM-255": new_chapter[4:6],
        "PD-CLAIM-256": new_chapter[6:],
    }
    claims = [
        _claim(
            "PD-CLAIM-254",
            groups["PD-CLAIM-254"],
            statement="Two separated three-axis point forces map to a net club wrench with rank five and one invisible equal-and-opposite axial force mode; point forces alone cannot create a midpoint free moment along the separation axis.",
            classification="bilateral_point_force_structural_identifiability",
            status="supported_analytically_and_numerically",
            boundary="Structural rank is exact for declared point contacts; it does not establish that the null-mode force exists in a golfer.",
            falsifier="The manufactured axial vector produces a nonzero net wrench, or the distinct-contact map does not have rank five.",
        ),
        _claim(
            "PD-CLAIM-255",
            groups["PD-CLAIM-255"],
            statement="The full bilateral six-axis hand-wrench map has rank six and nullity six; one axial scalar closes only the point-force rank gap, so net club wrench cannot uniquely recover individual full hand-wrench allocation.",
            classification="bilateral_full_wrench_measurement_boundary",
            status="structurally_unidentifiable_from_net_wrench",
            boundary="Additional constitutive assumptions may select an allocation but do not create measurement information; practical sensor qualification remains open.",
            falsifier="Net wrench uniquely recovers all twelve bilateral components without additional measurements or assumptions.",
        ),
        _claim(
            "PD-CLAIM-256",
            groups["PD-CLAIM-256"],
            statement="Grip span changes the normalized nonzero conditioning while proper rotation preserves structural rank and singular values; no muscle, scapular, intentional, or human strategy is identified.",
            classification="geometry_conditioning_and_scope_boundary",
            status="supported_for_declared_scaling_with_human_validation_open",
            boundary="Condition values depend on declared force/moment scaling and exclude noise, cross-talk, contact migration, tissue dynamics, and subject variability.",
            falsifier="Rank changes under a consistent proper rotation, the span trend fails under the declared scaling, or the result is promoted to a biological strategy.",
        ),
    ]
    registry["claims"].extend(claims)
    for claim_id, candidates in groups.items():
        registry["candidate_reviews"].extend(
            {
                "candidate_id": candidate["candidate_id"],
                "disposition": "material_claims_mapped",
                "claim_ids": [claim_id],
                "rationale": "This passage states or bounds the bilateral-wrench identifiability result.",
                "reviewer": "Codex technical audit",
                "last_verified_on": DATE,
            }
            for candidate in candidates
        )

    appendix_candidates = [
        candidate
        for candidate in inventory["candidates"]
        if candidate["source_path"] == APPENDIX
        and candidate["candidate_id"] not in reviewed
    ]
    if len(appendix_candidates) > 1:
        raise ValueError(
            f"expected at most 1 changed appendix candidate, found {len(appendix_candidates)}"
        )
    if appendix_candidates:
        registry["candidate_reviews"].append(
            {
                "candidate_id": appendix_candidates[0]["candidate_id"],
                "disposition": "editorial_or_navigation",
                "claim_ids": [],
                "rationale": "This artifact inventory points to evidence but asserts no standalone scientific result.",
                "reviewer": "Codex technical audit",
                "last_verified_on": DATE,
            }
        )

    release_entries = {
        item["release_claim_key"]: item for item in registry["release_claim_inventory"]
    }
    release_entries["individual_hand_allocation_from_net_wrench"] = {
        "release_claim_key": "individual_hand_allocation_from_net_wrench",
        "published_status": "structurally_unidentifiable",
        "audit_state": "reviewed_as_structurally_unidentifiable",
    }
    release_entries["bilateral_human_wrench_validation"] = {
        "release_claim_key": "bilateral_human_wrench_validation",
        "published_status": "untested",
        "audit_state": "reviewed_as_untested",
    }
    registry["release_claim_inventory"] = list(release_entries.values())
    registry["paper"]["source_digest"] = inventory["source_digest"]
    registry["paper"]["baseline_commit"] = "a4b53e39b"
    for collection in registry["research_collections"]:
        collection["last_check"] = DATE
    registry["audit_scope"]["completion_status"] = "complete"
    registry["audit_scope"]["current_scope"] = (
        "The complete 970-candidate paper inventory is adjudicated. The bilateral-wrench extension adds three bounded structural-identifiability claims while retaining practical-sensor and human-data gates."
    )
    REGISTRY.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
