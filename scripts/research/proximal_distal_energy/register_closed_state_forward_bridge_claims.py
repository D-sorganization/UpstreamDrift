"""Register the closed-state forward-contact bridge claims and boundaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
ARTICLE = ROOT / "docs/research/proximal_distal_energy_transfer"
REGISTRY = ARTICLE / "data/claim_audit_registry.json"
INVENTORY = ARTICLE / "data/claim_candidate_inventory.json"
DATE = "2026-08-14"
CLAIM_IDS = {"PD-CLAIM-268", "PD-CLAIM-269", "PD-CLAIM-270"}
ARTIFACTS = [
    "docs/research/proximal_distal_energy_transfer/data/closed_state_forward_bridge.json",
    "docs/research/proximal_distal_energy_transfer/data/closed_state_forward_bridge.npz",
    "docs/research/proximal_distal_energy_transfer/figures/fig_closed_state_forward_bridge.pdf",
    "scripts/research/proximal_distal_energy/closed_state_forward_bridge.py",
    "scripts/research/proximal_distal_energy/run_closed_state_forward_bridge.py",
    "scripts/research/proximal_distal_energy/make_closed_state_forward_bridge_figure.py",
    "scripts/research/proximal_distal_energy/register_closed_state_forward_bridge_claims.py",
    "tests/research/test_closed_state_forward_bridge.py",
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
        raise ValueError(f"expected one candidate beginning {prefix!r}")
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
    model_domain: str,
) -> dict[str, Any]:
    return {
        "claim_id": claim_id,
        "candidate_ids": [candidate["candidate_id"] for candidate in candidates],
        "statement": statement,
        "classification": classification,
        "published_status": status,
        "audit_status": "deterministic_mapping_constitutive_and_cross_engine_controls_checked",
        "source_locations": [
            f"{candidate['source_path']}:{candidate['line_start']}"
            for candidate in candidates
        ],
        "evidence_artifacts": ARTIFACTS,
        "model_domain": model_domain,
        "uncertainty_boundary": boundary,
        "competing_explanations": [
            "finite-difference velocity error",
            "reduced hand-carriage dynamics",
            "uncalibrated engineering contact parameters",
            "short integration horizon",
        ],
        "negative_controls": [
            "position and velocity closure",
            "exact zero-preload contact",
            "action-reaction and damping passivity",
            "independent-engine state-digest and trajectory comparison",
        ],
        "falsifier": falsifier,
        "adjudication": "The deterministic artifacts were regenerated in native MuJoCo and Pinocchio and retain the short-horizon reduced-model boundary.",
        "reviewer": "Codex technical audit",
        "last_verified_on": DATE,
    }


def _build_claims(candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    design = _find(candidates, "The closed configurations now enter")
    mapping = _find(candidates, "All 234 position mappings retain")
    forward = _find(candidates, "A spanning subset advances")
    boundary = _find(candidates, "This is an initialization and short-horizon")
    next_gate = _find(candidates, "The next spatial experiment must retain")
    claims = [
        _claim(
            "PD-CLAIM-268",
            [design, mapping],
            statement="All 234 subject-scaled closed states map into engine-neutral forward-contact coordinates with maximum position and velocity closure errors of 1.16e-10 m and 1.29 mm/s, and exact closure creates zero preload.",
            classification="closed_state_forward_initialization_mapping",
            status="supported_for_declared_reduced_mapping",
            boundary="Velocities are finite differences along reduced-tree inverse-kinematics paths; the mapping adds neither anatomical dynamics nor equipment calibration.",
            falsifier="Any mapped state exceeds a registered closure tolerance or develops nonzero force at exact closure.",
            model_domain="All 234 reduced-tree closed configurations under one constant initial rigid transformation.",
        ),
        _claim(
            "PD-CLAIM-269",
            [forward],
            statement="MuJoCo and Pinocchio receive identical digested states and pass trajectory, contact-wrench, and normalized-energy gates for all 54 cases in the 4 ms initialization audit.",
            classification="closed_state_short_horizon_cross_engine_forward_audit",
            status="supported_for_declared_reduced_short_horizon_subset",
            boundary="Finite-mass hand carriages replace articulated arms after initialization, and 4 ms is not a downswing or delivery simulation.",
            falsifier="An engine pair receives a different digest or fails any registered comparison gate.",
            model_domain="Six profiles, three grip spans, and early, middle, and late phases in two reduced native solvers.",
        ),
        _claim(
            "PD-CLAIM-270",
            [boundary, next_gate],
            statement="The bridge removes an initialization gap but does not establish calibrated equipment, articulated anatomy, tissue loading, passive transfer, delivery benefit, slack benefit, or human strategy.",
            classification="closed_state_forward_bridge_inference_boundary",
            status="explicitly_bounded",
            boundary="Full-horizon articulated contact, calibrated grip and shaft properties, adverse-load controls, and governed human validation remain open.",
            falsifier="A mechanism, delivery, anatomy, or coaching conclusion is attributed to this initialization audit alone.",
            model_domain="Coordinate mapping, constitutive controls, and 4 ms reduced forward initialization only.",
        ),
    ]
    figure = _find(
        candidates,
        "![Closed Subject States Enter Independent Forward Solvers Without Preload]",
    )
    return claims, str(figure["candidate_id"])


def _reconcile_reviews(
    registry: dict[str, Any], claims: list[dict[str, Any]], figure_id: str
) -> None:
    registry["claims"] = [
        claim for claim in registry["claims"] if claim["claim_id"] not in CLAIM_IDS
    ]
    for claim in registry["claims"]:
        if figure_id in claim["candidate_ids"]:
            index = claim["candidate_ids"].index(figure_id)
            claim["candidate_ids"].pop(index)
            claim["source_locations"].pop(index)
    reviews = {
        review["candidate_id"]: review for review in registry["candidate_reviews"]
    }
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
    reviews[figure_id].update(
        disposition="editorial_or_navigation",
        claim_ids=[],
        rationale="The figure include points to registered evidence but asserts no standalone result.",
        reviewer="Codex technical audit",
        last_verified_on=DATE,
    )
    registry["claims"].extend(claims)


def _update_release(registry: dict[str, Any]) -> None:
    entries = {
        item["release_claim_key"]: item for item in registry["release_claim_inventory"]
    }
    entries["subject_scaled_closed_contact_feasibility"]["published_status"] = (
        "reduced_tree_closed_contact_screen_and_short_forward_initialization_passed"
    )
    entries["closed_state_forward_initialization"] = {
        "release_claim_key": "closed_state_forward_initialization",
        "published_status": "supported_for_234_mappings_and_54_short_cross_engine_cases",
        "audit_state": "reviewed_as_short_horizon_reduced_model_result",
    }
    registry["release_claim_inventory"] = list(entries.values())
    registry["audit_scope"]["current_scope"] = (
        "The complete paper inventory is adjudicated. Subject-scaled closed contact "
        "passes its reduced-tree screen, and all 234 states map into a 54-case "
        "short-horizon two-engine initialization audit. Calibrated articulated "
        "forward contact and governed human validation remain unexecuted."
    )


def main() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    claims, figure_id = _build_claims(inventory["candidates"])
    _reconcile_reviews(registry, claims, figure_id)
    _update_release(registry)
    registry["paper"]["source_digest"] = inventory["source_digest"]
    REGISTRY.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
