"""Register the native articulated-inertia qualification claims."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
ARTICLE = ROOT / "docs/research/proximal_distal_energy_transfer"
REGISTRY = ARTICLE / "data/claim_audit_registry.json"
INVENTORY = ARTICLE / "data/claim_candidate_inventory.json"
DATE = "2026-08-15"
CLAIM_IDS = {"PD-CLAIM-274", "PD-CLAIM-275", "PD-CLAIM-276"}
ARTIFACTS = [
    "docs/research/proximal_distal_energy_transfer/data/articulated_inertia_cross_engine.json",
    "docs/research/proximal_distal_energy_transfer/data/articulated_inertia_cross_engine.npz",
    "docs/research/proximal_distal_energy_transfer/figures/fig_articulated_inertia_cross_engine.pdf",
    "scripts/research/proximal_distal_energy/articulated_inertia_cross_engine.py",
    "scripts/research/proximal_distal_energy/run_articulated_inertia_cross_engine.py",
    "scripts/research/proximal_distal_energy/make_articulated_inertia_cross_engine_figure.py",
    "scripts/research/proximal_distal_energy/register_articulated_inertia_claims.py",
    "tests/research/test_articulated_inertia_cross_engine.py",
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
        raise ValueError(f"expected one articulated candidate beginning {prefix!r}")
    return matches[0]


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
        "audit_status": "native_articulated_mass_bias_inverse_dynamics_and_positive_definiteness_checked",
        "source_locations": [
            f"{candidate['source_path']}:{candidate['line_start']}"
            for candidate in candidates
        ],
        "evidence_artifacts": ARTIFACTS,
        "model_domain": (
            "Six synthetic profiles, three grip spans, 13 closed states per "
            "case, 20 scalar coordinates, and two native rigid-body engines."
        ),
        "uncertainty_boundary": boundary,
        "competing_explanations": [
            "shared idealized spherical-body inertia specification",
            "common-state comparison rather than trajectory integration",
            "broad reduced-tree geometry and engineering joint bounds",
            "finite-difference velocity and acceleration estimates",
        ],
        "negative_controls": [
            "independent engine model assembly",
            "mass-matrix symmetry gate",
            "positive-definiteness eigenvalue gate",
            "mass, bias, and inverse-dynamics operators checked separately",
            "all 234 closed states retained",
        ],
        "falsifier": falsifier,
        "adjudication": (
            "All registered states pass the common-state native dynamics gates. "
            "The result is restricted to articulated rigid-body operator transport."
        ),
        "reviewer": "Codex technical audit",
        "last_verified_on": DATE,
    }


def _build_claims(
    candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    selected = {
        "design": _find(candidates, "The reduced forward-contact result cannot"),
        "methods": _find(candidates, "MuJoCo independently evaluates"),
        "figure": _find(candidates, "![Native Articulated Dynamics"),
        "result": _find(
            candidates,
            "All 234 states pass the preregistered mass-matrix",
        ),
        "boundary": _find(candidates, "This closes an engine-transport question"),
    }
    claims = [
        _claim(
            "PD-CLAIM-274",
            [selected["design"], selected["methods"]],
            statement=(
                "The articulated common-state gate independently assembles the "
                "same subject-scaled 20-coordinate tree in native MuJoCo and "
                "robotics Pinocchio at all 234 closed states."
            ),
            classification="subject_scaled_articulated_inertia_design",
            status="complete_for_declared_common_state_matrix",
            boundary=(
                "Segment geometry and spherical inertias are engineering design "
                "values rather than participant-specific anatomy."
            ),
            falsifier=(
                "Any closed state is omitted, model dimensions or joint order "
                "differ, or an engine does not use its named native operators."
            ),
        ),
        _claim(
            "PD-CLAIM-275",
            [selected["result"]],
            statement=(
                "All 234 states pass native mass-matrix, bias-force, "
                "inverse-dynamics, symmetry, and positive-definiteness gates."
            ),
            classification="subject_scaled_articulated_inertia_result",
            status="supported_at_declared_common_state_rigid_body_tier",
            boundary=(
                "The result tests rigid-body operator agreement and does not "
                "advance contact or a delivery trajectory."
            ),
            falsifier=(
                "Any reproduced state exceeds a registered relative tolerance, "
                "loses symmetry, or has a nonpositive mass-matrix eigenvalue."
            ),
        ),
        _claim(
            "PD-CLAIM-276",
            [selected["boundary"]],
            statement=(
                "Native articulated inertia parity does not establish forward "
                "contact, anatomy, grip or shaft behavior, muscles, human "
                "transfer, timing economy, slack benefit, or coaching strategy."
            ),
            classification="subject_scaled_articulated_inertia_inference_boundary",
            status="explicitly_bounded",
            boundary=(
                "Compliant bilateral contact, detailed anatomy, equipment, ground, "
                "delivery, and governed human evidence remain open."
            ),
            falsifier=(
                "A contact, anatomy, equipment, muscle, delivery, or human claim "
                "is attributed to common-state inertia parity alone."
            ),
        ),
    ]
    return claims, selected


def _reconcile(
    registry: dict[str, Any],
    inventory: dict[str, Any],
    claims: list[dict[str, Any]],
    selected: dict[str, dict[str, Any]],
) -> None:
    registry["claims"] = [
        claim for claim in registry["claims"] if claim["claim_id"] not in CLAIM_IDS
    ]
    inventory_candidate_ids = {
        candidate["candidate_id"] for candidate in inventory["candidates"]
    }
    reviews = {
        review["candidate_id"]: review
        for review in registry["candidate_reviews"]
        if review["candidate_id"] in inventory_candidate_ids
    }
    for candidate in selected.values():
        reviews.setdefault(
            candidate["candidate_id"],
            {
                "candidate_id": candidate["candidate_id"],
                "disposition": "material_claims_mapped",
                "claim_ids": [],
                "rationale": (
                    "This passage states or bounds the native articulated-inertia gate."
                ),
                "reviewer": "Codex technical audit",
                "last_verified_on": DATE,
            },
        )
    for review in reviews.values():
        review["claim_ids"] = [
            claim_id for claim_id in review["claim_ids"] if claim_id not in CLAIM_IDS
        ]
    for claim in claims:
        for candidate_id in claim["candidate_ids"]:
            review = reviews[candidate_id]
            review["disposition"] = "material_claims_mapped"
            review["claim_ids"] = sorted(set(review["claim_ids"]) | {claim["claim_id"]})
            review["last_verified_on"] = DATE
    reviews[selected["figure"]["candidate_id"]].update(
        disposition="editorial_or_navigation",
        claim_ids=[],
        rationale=(
            "The figure include points to registered evidence but asserts no "
            "standalone result."
        ),
    )
    registry["candidate_reviews"] = list(reviews.values())
    registry["claims"].extend(claims)
    entries = {
        item["release_claim_key"]: item for item in registry["release_claim_inventory"]
    }
    entries["subject_scaled_articulated_inertia"] = {
        "release_claim_key": "subject_scaled_articulated_inertia",
        "published_status": (
            "native_common_state_mass_bias_and_inverse_dynamics_qualified"
        ),
        "audit_state": "reviewed_as_common_state_articulated_dynamics_result",
    }
    registry["release_claim_inventory"] = list(entries.values())
    registry["audit_scope"]["current_scope"] = (
        "The complete paper inventory is adjudicated. Native MuJoCo and "
        "Pinocchio articulated mass, bias, and inverse dynamics agree at all "
        "234 closed states. Forward articulated contact, calibrated equipment, "
        "full delivery, and governed human validation remain open."
    )
    registry["paper"]["source_digest"] = inventory["source_digest"]


def main() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    claims, selected = _build_claims(inventory["candidates"])
    _reconcile(registry, inventory, claims, selected)
    REGISTRY.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
