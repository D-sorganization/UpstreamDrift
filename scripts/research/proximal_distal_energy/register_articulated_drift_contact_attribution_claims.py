"""Register the articulated same-state drift/contact attribution claims."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
ARTICLE = ROOT / "docs/research/proximal_distal_energy_transfer"
REGISTRY = ARTICLE / "data/claim_audit_registry.json"
INVENTORY = ARTICLE / "data/claim_candidate_inventory.json"
DATE = "2026-08-27"
CLAIM_IDS = {"PD-CLAIM-327", "PD-CLAIM-328", "PD-CLAIM-329"}
SOURCE = "_ch06cd_articulated_attribution.qmd"
ARTIFACTS = [
    "docs/research/proximal_distal_energy_transfer/data/articulated_drift_contact_attribution.json",
    "docs/research/proximal_distal_energy_transfer/data/articulated_drift_contact_attribution.npz",
    "docs/research/proximal_distal_energy_transfer/figures/fig_articulated_drift_contact_attribution.pdf",
    "scripts/research/proximal_distal_energy/articulated_drift_contact_attribution.py",
    "scripts/research/proximal_distal_energy/run_articulated_drift_contact_attribution.py",
    "scripts/research/proximal_distal_energy/make_articulated_drift_contact_attribution_figure.py",
    "scripts/research/proximal_distal_energy/register_articulated_drift_contact_attribution_claims.py",
    "tests/research/test_articulated_drift_contact_attribution.py",
    "tests/research/test_articulated_drift_contact_attribution_evidence.py",
]


def _find(candidates: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    matches = [
        candidate
        for candidate in candidates
        if str(candidate["source_path"]).endswith(SOURCE)
        and str(candidate["text"]).startswith(prefix)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one attribution candidate for {prefix!r}")
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
        "audit_status": "articulated_same_state_drift_contact_attribution_checked",
        "adjudication_outcome": "supported",
        "source_locations": [
            f"{candidate['source_path']}:{candidate['line_start']}"
            for candidate in candidates
        ],
        "evidence_artifacts": ARTIFACTS,
        "model_domain": (
            "Six synthetic profiles, three grip spans, thirteen closed states per "
            "case, twenty generalized coordinates, four contribution classes, "
            "and two native engines without a forward step."
        ),
        "uncertainty_boundary": boundary,
        "competing_explanations": [
            "same-state imposed perturbation rather than forward persistence",
            "synthetic bilateral Kelvin-Voigt contact",
            "engineering segment and contact parameters",
            "signed projection cancellation and denominator choice",
        ],
        "negative_controls": [
            "zero-contact pathway killswitch",
            "zero-velocity bias killswitch",
            "gravity-off configuration killswitch",
            "coincident and reversed moment-arm controls",
            "coordinate-scaling invariance",
            "deliberately corrupted generalized-force sentinel",
        ],
        "falsifier": falsifier,
        "adjudication": (
            "All registered same-state closure, pathway, scaling, corruption, and "
            "native-operator gates pass; interpretation remains pointwise."
        ),
        "reviewer": "Codex technical audit",
        "last_verified_on": DATE,
    }


def main() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    candidates = inventory["candidates"]
    selected = {
        "design": _find(candidates, "The contact-projection gate establishes"),
        "definitions": _find(candidates, "The baseline sets"),
        "observables": _find(candidates, "Two complementary signed observables"),
        "interpretation": _find(candidates, "and the generalized-power contribution"),
        "matrix": _find(candidates, "The atlas reuses"),
        "figure": _find(candidates, "![Articulated Same-State Drift"),
        "gates": _find(candidates, "All registered gates pass."),
        "result": _find(candidates, "Across both engines"),
        "boundary": _find(candidates, "The machine-readable"),
    }
    claims = [
        _claim(
            "PD-CLAIM-327",
            [
                selected["design"],
                selected["definitions"],
                selected["observables"],
                selected["interpretation"],
                selected["matrix"],
                selected["gates"],
            ],
            "The articulated same-state acceleration and generalized power close under a configuration, velocity-dependent bias, bilateral-contact, and zero applied-input decomposition across the registered native-engine atlas.",
            "articulated_same_state_attribution_qualification",
            "supported_for_declared_pointwise_synthetic_matrix",
            "The decomposition is operational and state-specific; it does not identify anatomical force sources or a forward causal effect.",
            "Any registered closure, pathway killswitch, coordinate-scaling, corruption-detection, or cross-engine tolerance fails.",
        ),
        _claim(
            "PD-CLAIM-328",
            [selected["result"]],
            "Contact is positively aligned with total mass-metric acceleration but contributes negative generalized power throughout the registered atlas, so acceleration alignment does not identify positive work or transfer.",
            "articulated_acceleration_power_sign_distinction",
            "supported_for_declared_pointwise_synthetic_matrix",
            "The signs arise under an imposed synthetic perturbation and are neither forward work nor human performance evidence.",
            "Any retained state has nonpositive contact acceleration projection, nonnegative contact power, or inadequate unreported denominators.",
        ),
        _claim(
            "PD-CLAIM-329",
            [selected["boundary"]],
            "The same-state attribution does not establish forward persistence, accumulated impulse or work, biological torque or effort, human transfer, slack management, timing economy, coaching strategy, or safety.",
            "articulated_same_state_attribution_inference_boundary",
            "explicitly_bounded",
            "Matched forward attribution through contact transitions, coupled structures, uncertainty, and adverse loading remains open.",
            "A forward, biological, human, equipment, coaching, or safety claim is attributed to this pointwise gate alone.",
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
    for key, candidate in selected.items():
        reviews[candidate["candidate_id"]] = {
            "candidate_id": candidate["candidate_id"],
            "disposition": "material_claims_mapped",
            "claim_ids": [],
            "rationale": "This passage states or bounds the articulated same-state attribution gate.",
            "reviewer": "Codex technical audit",
            "last_verified_on": DATE,
        }
        if key == "figure":
            reviews[candidate["candidate_id"]].update(
                disposition="editorial_or_navigation",
                rationale="The figure points to registered evidence without a standalone claim.",
            )
    for claim in claims:
        for candidate_id in claim["candidate_ids"]:
            reviews[candidate_id]["claim_ids"].append(claim["claim_id"])
    registry["candidate_reviews"] = list(reviews.values())

    entries = {
        item["release_claim_key"]: item for item in registry["release_claim_inventory"]
    }
    entries["subject_scaled_articulated_drift_contact_attribution"] = {
        "release_claim_key": "subject_scaled_articulated_drift_contact_attribution",
        "published_status": "same_state_configuration_velocity_contact_and_zero_input_attribution_qualified",
        "audit_state": "reviewed_as_pointwise_synthetic_attribution_result",
    }
    registry["release_claim_inventory"] = list(entries.values())
    registry["audit_scope"]["current_scope"] = (
        "The complete paper inventory is adjudicated. Articulated same-state "
        "configuration, velocity, contact, and zero-input attribution passes its "
        "registered pointwise gates. Forward impulse/work and governed human "
        "validation remain open."
    )
    registry["paper"]["source_digest"] = inventory["source_digest"]
    REGISTRY.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
