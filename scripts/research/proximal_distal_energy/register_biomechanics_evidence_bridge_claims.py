"""Register the generated biomechanics evidence bridge in the claim audit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
ARTICLE = ROOT / "docs/research/proximal_distal_energy_transfer"
REGISTRY = ARTICLE / "data/claim_audit_registry.json"
INVENTORY = ARTICLE / "data/claim_candidate_inventory.json"
SOURCE_REGISTER = ARTICLE / "data/biomechanics_source_register.json"
CHAPTER = (
    "docs/research/proximal_distal_energy_transfer/chapters/"
    "_biomechanics_evidence_bridge.qmd"
)
CLAIM_ID = "PD-CLAIM-305"


def register_claims(
    registry: dict[str, Any],
    inventory: dict[str, Any],
    source_register: dict[str, Any],
) -> dict[str, Any]:
    """Return an idempotently adjudicated registry for the generated chapter."""
    candidates = [
        item for item in inventory["candidates"] if item["source_path"] == CHAPTER
    ]
    if len(candidates) != 37:
        raise ValueError(
            "biomechanics evidence chapter candidate count changed; review the "
            f"new inventory before registration: {len(candidates)}"
        )
    navigation = [
        item
        for item in candidates
        if item["text"].startswith("The complete generated reviewer surface")
    ]
    if len(navigation) != 1:
        raise ValueError(
            "reviewer-surface navigation candidate is missing or ambiguous"
        )
    material = [item for item in candidates if item not in navigation]
    candidate_ids = {item["candidate_id"] for item in candidates}
    material_ids = [item["candidate_id"] for item in material]

    registry["claims"] = [
        claim for claim in registry["claims"] if claim["claim_id"] != CLAIM_ID
    ]
    registry["candidate_reviews"] = [
        review
        for review in registry["candidate_reviews"]
        if review["candidate_id"] not in candidate_ids
        and CLAIM_ID not in review["claim_ids"]
    ]
    source_urls = [source["stable_url"] for source in source_register["sources"]]
    registry["claims"].append(
        {
            "claim_id": CLAIM_ID,
            "candidate_ids": material_ids,
            "statement": (
                "The versioned biomechanics evidence bridge records sixteen "
                "independently authored works across sixteen domains and maps "
                "seven measurement modalities to nine mechanisms, nine "
                "transportability dimensions, observable discriminators, "
                "countermodels, and fail-closed data gates; it retains human "
                "validation as externally blocked and bilateral hand-wrench "
                "allocation as structurally unidentified without independent "
                "bilateral measurements."
            ),
            "classification": "governed_source_and_measurement_authority",
            "published_status": "supported_as_audit_and_protocol_contract",
            "audit_status": "sources_claims_digests_and_generated_surfaces_checked",
            "adjudication_outcome": "supported",
            "source_locations": [
                f"{CHAPTER}:{item['line_start']}" for item in material
            ],
            "evidence_artifacts": [
                "docs/research/proximal_distal_energy_transfer/data/biomechanics_source_register.json",
                "docs/research/proximal_distal_energy_transfer/data/biomechanics_evidence_bridge.json",
                "docs/research/proximal_distal_energy_transfer/BIOMECHANICS_EVIDENCE_BRIDGE.md",
                "scripts/research/proximal_distal_energy/biomechanics_source_register.py",
                "scripts/research/proximal_distal_energy/biomechanics_evidence_bridge.py",
                "scripts/research/proximal_distal_energy/biomechanics_evidence_surfaces.py",
                "tests/unit/research/test_biomechanics_source_register.py",
                "tests/unit/research/test_biomechanics_evidence_bridge.py",
                "tests/unit/research/test_biomechanics_evidence_surfaces.py",
                *source_urls,
            ],
            "model_domain": (
                "Source qualification, model-to-measurement mapping, prospective "
                "falsification, identifiability, and transportability governance."
            ),
            "uncertainty_boundary": (
                "The register is not a systematic review, population estimate, "
                "calibrated apparatus result, participant outcome, or evidence "
                "that every qualifying dataset has been located."
            ),
            "competing_explanations": [
                "unlocated or inaccessible participant datasets",
                "apparatus-specific measurement validity",
                "alternative inverse problems and constitutive assumptions",
                "population, equipment, and task transport failure",
            ],
            "negative_controls": [
                "unknown-field rejection",
                "stale-digest rejection",
                "project-authored source exclusion",
                "missing-modality rejection",
                "source-to-claim reciprocity rejection",
                "bilateral-allocation identifiability killswitch",
            ],
            "falsifier": (
                "Any registered source, digest, modality, claim link, generated "
                "surface, or identifiability rule fails validation, or governed "
                "participant evidence contradicts the declared measurement and "
                "transport boundaries."
            ),
            "adjudication": (
                "The paper presents the generated records as an inspectable "
                "measurement and falsification contract, not as new human data "
                "or proof of a preferred technique."
            ),
            "reviewer": "Codex technical audit",
            "last_verified_on": "2026-08-24",
        }
    )
    registry["candidate_reviews"].extend(
        {
            "candidate_id": item["candidate_id"],
            "disposition": "material_claims_mapped",
            "claim_ids": [CLAIM_ID],
            "rationale": (
                "Generated measurement, identifiability, falsification, or "
                "transport record mapped to the bounded evidence-bridge authority."
            ),
            "reviewer": "Codex technical audit",
            "last_verified_on": "2026-08-24",
        }
        for item in material
    )
    registry["candidate_reviews"].append(
        {
            "candidate_id": navigation[0]["candidate_id"],
            "disposition": "editorial_or_navigation",
            "claim_ids": [],
            "rationale": "Links readers to generated authorities without a scientific proposition.",
            "reviewer": "Codex technical audit",
            "last_verified_on": "2026-08-24",
        }
    )
    registry["paper"]["source_digest"] = inventory["source_digest"]
    registry["audit_scope"]["completion_status"] = "complete"
    return registry


def main() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    source_register = json.loads(SOURCE_REGISTER.read_text(encoding="utf-8"))
    result = register_claims(registry, inventory, source_register)
    REGISTRY.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
