"""Register bounded articulated forward-contact claims and boundaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
ARTICLE = ROOT / "docs/research/proximal_distal_energy_transfer"
REGISTRY = ARTICLE / "data/claim_audit_registry.json"
INVENTORY = ARTICLE / "data/claim_candidate_inventory.json"
DATE = "2026-08-15"
CLAIM_IDS = {"PD-CLAIM-280", "PD-CLAIM-281", "PD-CLAIM-282"}
ARTIFACTS = [
    "docs/research/proximal_distal_energy_transfer/data/articulated_forward_contact.json",
    "docs/research/proximal_distal_energy_transfer/data/articulated_forward_contact.npz",
    "docs/research/proximal_distal_energy_transfer/figures/fig_articulated_forward_contact.pdf",
    "scripts/research/proximal_distal_energy/articulated_forward_contact.py",
    "scripts/research/proximal_distal_energy/run_articulated_forward_contact.py",
    "scripts/research/proximal_distal_energy/make_articulated_forward_contact_figure.py",
    "scripts/research/proximal_distal_energy/register_articulated_forward_contact_claims.py",
    "tests/research/test_articulated_forward_contact.py",
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
        raise ValueError(f"expected one articulated-forward candidate for {prefix!r}")
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
        "audit_status": "bounded_articulated_forward_contact_refinement_and_adverse_controls_checked",
        "source_locations": [
            f"{candidate['source_path']}:{candidate['line_start']}"
            for candidate in candidates
        ],
        "evidence_artifacts": ARTIFACTS,
        "model_domain": (
            "Six synthetic profiles, three grip spans, three closed-state phases, "
            "seven registered variants, three time steps, two native engines, "
            "and a five-millisecond forward horizon."
        ),
        "uncertainty_boundary": boundary,
        "competing_explanations": [
            "bilateral idealized attachment rather than unilateral contact",
            "short right-censored horizon",
            "engineering stiffness, damping, and retention threshold",
            "shared analytical contact projection and semi-implicit integrator",
        ],
        "negative_controls": [
            "zero-preload branch",
            "initial-velocity reversal",
            "one-factor stiffness and damping perturbations",
            "three-step refinement",
            "independent native MuJoCo and Pinocchio operators",
        ],
        "falsifier": falsifier,
        "adjudication": (
            "The registered bounded trajectories are reported with explicit "
            "right-censoring and no human or coaching inference."
        ),
        "reviewer": "Codex technical audit",
        "last_verified_on": DATE,
    }


def main() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    candidates = inventory["candidates"]
    selected = {
        "design": _find(candidates, "The forward gate advances eighteen"),
        "methods": _find(candidates, "Each trajectory begins from an achieved"),
        "figure": _find(candidates, "![Bounded Articulated Bilateral-Attachment"),
        "result": _find(candidates, "All registered five-millisecond trajectories"),
        "boundary": _find(candidates, "This is a right-censored synthetic result"),
    }
    claims = [
        _claim(
            "PD-CLAIM-280",
            [selected["design"], selected["methods"]],
            "A preregistered cross-profile, phase, adverse-control, refinement, and inertia-and-bias transport matrix advances the articulated bilateral-attachment model for five milliseconds without active torque or direct club actuation.",
            "bounded_articulated_forward_contact_design",
            "complete_for_declared_bounded_matrix",
            "The attachments, parameters, states, and retention threshold are synthetic engineering constructs.",
            "A trajectory is omitted, an unregistered active drive is applied, or force is clipped without a typed event.",
        ),
        _claim(
            "PD-CLAIM-281",
            [selected["result"]],
            "The registered five-millisecond articulated trajectories pass attachment-retention, power, work-energy, refinement, and MuJoCo-Pinocchio inertia-and-bias transport gates while sharing their contact law and state update.",
            "bounded_articulated_forward_contact_result",
            "supported_at_declared_five_millisecond_synthetic_tier",
            "The result is right-censored at five milliseconds and is not calibrated to anatomy, equipment, ground reaction, or human data.",
            "Any reproduced trajectory exceeds a registered gate or refinement fails to reduce the worst energy residual.",
        ),
        _claim(
            "PD-CLAIM-282",
            [selected["boundary"]],
            "The bounded bilateral-attachment result does not establish late-downswing persistence, unilateral contact or slack, distributed grip or shaft behavior, muscle action, human transfer, timing economy, or coaching strategy.",
            "bounded_articulated_forward_contact_inference_boundary",
            "explicitly_bounded",
            "Longer horizons, typed unilateral slack, calibrated distributed structures, ground coupling, and governed bilateral human wrenches remain open.",
            "A human, equipment, slack, late-downswing, or coaching claim is attributed to this five-millisecond synthetic screen alone.",
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
            "rationale": "This passage states or bounds the bounded articulated forward gate.",
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
    entries["bounded_articulated_forward_contact"] = {
        "release_claim_key": "bounded_articulated_forward_contact",
        "published_status": "five_millisecond_bilateral_attachment_forward_gate_qualified",
        "audit_state": "reviewed_as_right_censored_synthetic_forward_result",
    }
    registry["release_claim_inventory"] = list(entries.values())
    registry["audit_scope"]["current_scope"] = (
        "The complete paper inventory is adjudicated. The articulated bilateral "
        "attachment tier passes its registered five-millisecond forward gates. "
        "Longer typed-slack tiers and governed human validation remain open."
    )
    registry["paper"]["source_digest"] = inventory["source_digest"]
    REGISTRY.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
