"""Register the screening-state rigid-refinement extension and boundaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
ARTICLE = ROOT / "docs/research/proximal_distal_energy_transfer"
REGISTRY = ARTICLE / "data/claim_audit_registry.json"
INVENTORY = ARTICLE / "data/claim_candidate_inventory.json"
DATE = "2026-08-28"
CLAIM_IDS = {"PD-CLAIM-333", "PD-CLAIM-334", "PD-CLAIM-335"}
CHAPTER = "_ch08b_momentum_transfer_questions.qmd"
ARTIFACTS = [
    "docs/research/proximal_distal_energy_transfer/data/articulated_rigid_refinement_plan.json",
    "docs/research/proximal_distal_energy_transfer/data/articulated_rigid_refinement/summary.json",
    "docs/research/proximal_distal_energy_transfer/figures/fig_rigid_refinement_screening_states.pdf",
    "scripts/research/proximal_distal_energy/articulated_rigid_refinement_plan.py",
    "scripts/research/proximal_distal_energy/articulated_rigid_refinement_launcher.py",
    "scripts/research/proximal_distal_energy/make_rigid_refinement_figure.py",
    "tests/research/test_articulated_rigid_refinement_plan.py",
    "tests/research/test_articulated_rigid_refinement_evidence.py",
    "tests/research/test_make_rigid_refinement_figure.py",
]


def _find(candidates: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    matches = [
        candidate
        for candidate in candidates
        if str(candidate["source_path"]).endswith(CHAPTER)
        and str(candidate["text"]).startswith(prefix)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one rigid-refinement candidate for {prefix!r}")
    return matches[0]


def _claim(
    claim_id: str,
    candidates: list[dict[str, Any]],
    statement: str,
    classification: str,
    uncertainty_boundary: str,
    falsifier: str,
) -> dict[str, Any]:
    return {
        "claim_id": claim_id,
        "candidate_ids": [candidate["candidate_id"] for candidate in candidates],
        "statement": statement,
        "classification": classification,
        "published_status": "promotion_withheld",
        "audit_status": "rigid_refinement_extension_checked",
        "adjudication_outcome": "supported",
        "source_locations": [
            f"{candidate['source_path']}:{candidate['line_start']}"
            for candidate in candidates
        ],
        "evidence_artifacts": ARTIFACTS,
        "model_domain": (
            "Eighteen declared synthetic screening states, nominal and "
            "high-damping rigid-contact variants, three unseen time steps, a "
            "five-millisecond horizon, and two registered native engines."
        ),
        "uncertainty_boundary": uncertainty_boundary,
        "competing_explanations": [
            "state-dependent quadrature error rather than physical work",
            "semi-implicit transient behavior rather than asymptotic convergence",
            "synthetic source-state geometry rather than a participant distribution",
            "unavailable native parity rather than cross-engine agreement",
        ],
        "negative_controls": [
            "all 18 previously declared screening states",
            "high-damping comparison at every state",
            "step sizes disjoint from the post-result pilot",
            "unchanged 0.800 refinement ceiling",
            "typed native-engine absence retained as a failed promotion gate",
        ],
        "falsifier": falsifier,
        "adjudication": (
            "The complete checkpoint inventory retains all three state-dependent "
            "failures, the original smoke failure, the excluded pilot, and "
            "unavailable native parity without threshold changes."
        ),
        "reviewer": "Codex technical audit",
        "last_verified_on": DATE,
    }


def main() -> None:
    """Replace this extension's claims and reconcile candidate reviews."""

    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    candidates = inventory["candidates"]
    selected = {
        "design": _find(candidates, "The original rigid smoke failed"),
        "result": _find(candidates, "All 216 registered cases"),
        "figure": _find(candidates, "![Rigid Refinement Across Screening States"),
        "boundary": _find(candidates, "This extension does not support"),
    }
    valid_ids = {candidate["candidate_id"] for candidate in candidates}
    registry["claims"] = [
        claim for claim in registry["claims"] if claim["claim_id"] not in CLAIM_IDS
    ]
    for claim in registry["claims"]:
        claim["candidate_ids"] = [
            candidate_id
            for candidate_id in claim.get("candidate_ids", [])
            if candidate_id in valid_ids
        ]
    claims = [
        _claim(
            "PD-CLAIM-333",
            [selected["design"]],
            "The rigid-refinement extension prospectively evaluates the complete declared screening matrix for the two original failing variants at step sizes disjoint from the disclosed pilot.",
            "prospective_rigid_refinement_design",
            "The state set is synthetic, the horizon is five milliseconds, and the development pilot is not confirmatory evidence.",
            "The committed manifest, state set, variants, steps, threshold, or pilot exclusion differs from the published contract.",
        ),
        _claim(
            "PD-CLAIM-334",
            [selected["result"]],
            "All 108 MuJoCo cases satisfy individual closure tolerances, but three state--variant groups fail the frozen work-refinement gate; all 108 Pinocchio cases remain typed unavailable.",
            "state_dependent_rigid_refinement_result",
            "This is numerical-attribution evidence, not a physical-transfer, human, equipment, or coaching result.",
            "A reproduced checkpoint changes a retained residual or status, suppresses a failed group, treats native absence as parity, or promotes the campaign.",
        ),
        _claim(
            "PD-CLAIM-335",
            [selected["boundary"]],
            "The screening extension does not support a blanket claim that finer steps resolve the original work-refinement failure and requires an independently registered integrator or quadrature comparison.",
            "rigid_refinement_inference_boundary",
            "The three failures may reflect integrator, quadrature, event, or short-horizon state dependence and do not identify a physical mechanism.",
            "The paper presents the extension as convergence, cross-engine parity, a physical mechanism, or a human strategy without the missing prospective evidence.",
        ),
    ]
    registry["claims"].extend(claims)
    reviews = {
        review["candidate_id"]: review
        for review in registry["candidate_reviews"]
        if review["candidate_id"] in valid_ids
    }
    for claim in claims:
        for candidate_id in claim["candidate_ids"]:
            reviews[candidate_id] = {
                "candidate_id": candidate_id,
                "disposition": "material_claims_mapped",
                "claim_ids": [claim["claim_id"]],
                "rationale": "This passage states or bounds the rigid-refinement extension.",
                "reviewer": "Codex technical audit",
                "last_verified_on": DATE,
            }
    figure = selected["figure"]
    reviews[figure["candidate_id"]] = {
        "candidate_id": figure["candidate_id"],
        "disposition": "editorial_or_navigation",
        "claim_ids": [],
        "rationale": "The figure include visualizes governed evidence without adding a standalone claim.",
        "reviewer": "Codex technical audit",
        "last_verified_on": DATE,
    }
    registry["candidate_reviews"] = list(reviews.values())
    registry["paper"]["source_digest"] = inventory["source_digest"]
    registry["audit_scope"]["current_scope"] = (
        "The complete paper inventory is adjudicated. The rigid-refinement "
        "extension retains three state-dependent failures and unavailable native "
        "parity; independent integration and governed human validation remain open."
    )
    REGISTRY.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
