"""Register the articulated contact-projection qualification claims."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
ARTICLE = ROOT / "docs/research/proximal_distal_energy_transfer"
REGISTRY = ARTICLE / "data/claim_audit_registry.json"
INVENTORY = ARTICLE / "data/claim_candidate_inventory.json"
DATE = "2026-08-15"
CLAIM_IDS = {"PD-CLAIM-277", "PD-CLAIM-278", "PD-CLAIM-279"}
ARTIFACTS = [
    "docs/research/proximal_distal_energy_transfer/data/articulated_contact_projection.json",
    "docs/research/proximal_distal_energy_transfer/data/articulated_contact_projection.npz",
    "docs/research/proximal_distal_energy_transfer/figures/fig_articulated_contact_projection.pdf",
    "scripts/research/proximal_distal_energy/articulated_contact_projection.py",
    "scripts/research/proximal_distal_energy/run_articulated_contact_projection.py",
    "scripts/research/proximal_distal_energy/make_articulated_contact_projection_figure.py",
    "scripts/research/proximal_distal_energy/register_articulated_contact_projection_claims.py",
    "tests/research/test_articulated_contact_projection.py",
]


def _find(candidates: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    matches = [
        candidate
        for candidate in candidates
        if str(candidate["source_path"]).endswith(
            "_ch06c_spatial_cross_formulation.qmd"
        )
        and str(candidate["text"]).startswith(prefix)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one contact-projection candidate for {prefix!r}")
    return matches[0]


def _claim(
    claim_id: str,
    candidates: list[dict[str, Any]],
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
        "audit_status": "articulated_contact_projection_power_geometry_and_initial_acceleration_checked",
        "source_locations": [
            f"{candidate['source_path']}:{candidate['line_start']}"
            for candidate in candidates
        ],
        "evidence_artifacts": ARTIFACTS,
        "model_domain": (
            "Six synthetic profiles, three grip spans, 13 closed states per "
            "case, 20 scalar coordinates, finite point contact, and two native engines."
        ),
        "uncertainty_boundary": boundary,
        "competing_explanations": [
            "shared idealized Kelvin-Voigt contact law",
            "same-state acceleration rather than forward integration",
            "engineering segment and contact parameters",
            "common analytical contact Jacobian projection",
        ],
        "negative_controls": [
            "zero-preload closed-state control",
            "exact action-reaction and virtual-power identities",
            "non-positive damping power",
            "coincident and reversed moment-arm controls",
            "all 234 closed states retained",
        ],
        "falsifier": falsifier,
        "adjudication": (
            "All registered states pass the same-state contact-projection gates. "
            "The result is restricted to generalized force and initial acceleration."
        ),
        "reviewer": "Codex technical audit",
        "last_verified_on": DATE,
    }


def main() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    candidates = inventory["candidates"]
    selected = {
        "design": _find(candidates, "The next gate adds finite bilateral"),
        "equation": _find(candidates, "For each hand $i$"),
        "methods": _find(candidates, "where $J_g$ and $J_h$"),
        "figure": _find(candidates, "![Articulated Contact Projection"),
        "result": _find(candidates, "All 234 states pass."),
        "boundary": _find(
            candidates, "This closes the same-state contact-projection question"
        ),
    }
    claims = [
        _claim(
            "PD-CLAIM-277",
            [selected["design"], selected["equation"], selected["methods"]],
            "Finite bilateral Kelvin-Voigt forces are projected through the hand and club Jacobians at all 234 closed states without direct club actuation.",
            "subject_scaled_articulated_contact_projection_design",
            "complete_for_declared_same_state_matrix",
            "The contact law, geometry, and perturbation are engineering values rather than participant or equipment calibration.",
            "Any force is prescribed independently of achieved contact state, action-reaction or virtual power fails, or a closed state is omitted.",
        ),
        _claim(
            "PD-CLAIM-278",
            [selected["result"]],
            "All 234 states pass contact, passivity, geometry-control, and native MuJoCo-Pinocchio initial-acceleration gates.",
            "subject_scaled_articulated_contact_projection_result",
            "supported_at_declared_same_state_initial_acceleration_tier",
            "The result is one same-state acceleration evaluation and contains no forward trajectory or accumulated work.",
            "Any reproduced state exceeds a registered residual or native acceleration tolerance.",
        ),
        _claim(
            "PD-CLAIM-279",
            [selected["boundary"]],
            "The same-state contact-projection result does not establish forward contact, contact loss, delivery, anatomy, equipment, muscle action, human transfer, slack benefit, timing economy, or coaching strategy.",
            "subject_scaled_articulated_contact_projection_inference_boundary",
            "explicitly_bounded",
            "Forward integration, calibrated structure/contact, ground, and governed human evidence remain open.",
            "A trajectory, human, equipment, or strategy claim is attributed to this initial-acceleration gate alone.",
        ),
    ]
    registry["claims"] = [
        claim for claim in registry["claims"] if claim["claim_id"] not in CLAIM_IDS
    ] + claims
    valid_ids = {candidate["candidate_id"] for candidate in candidates}
    reviews = {
        review["candidate_id"]: review
        for review in registry["candidate_reviews"]
        if review["candidate_id"] in valid_ids
    }
    for candidate in selected.values():
        reviews[candidate["candidate_id"]] = {
            "candidate_id": candidate["candidate_id"],
            "disposition": "material_claims_mapped",
            "claim_ids": [],
            "rationale": "This passage states or bounds the articulated contact-projection gate.",
            "reviewer": "Codex technical audit",
            "last_verified_on": DATE,
        }
    for claim in claims:
        for candidate_id in claim["candidate_ids"]:
            reviews[candidate_id]["claim_ids"].append(claim["claim_id"])
    reviews[selected["figure"]["candidate_id"]].update(
        disposition="editorial_or_navigation",
        rationale="The figure include points to registered evidence without a standalone claim.",
    )
    registry["candidate_reviews"] = list(reviews.values())
    entries = {
        item["release_claim_key"]: item for item in registry["release_claim_inventory"]
    }
    entries["subject_scaled_articulated_contact_projection"] = {
        "release_claim_key": "subject_scaled_articulated_contact_projection",
        "published_status": "same_state_bilateral_contact_projection_and_initial_acceleration_qualified",
        "audit_state": "reviewed_as_same_state_articulated_contact_projection_result",
    }
    registry["release_claim_inventory"] = list(entries.values())
    registry["audit_scope"]["current_scope"] = (
        "The complete paper inventory is adjudicated. Articulated bilateral "
        "contact projection and native initial acceleration pass at all 234 "
        "closed states. Forward articulated contact and governed human validation remain open."
    )
    registry["paper"]["source_digest"] = inventory["source_digest"]
    REGISTRY.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
