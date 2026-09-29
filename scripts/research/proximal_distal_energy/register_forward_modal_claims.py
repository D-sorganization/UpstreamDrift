"""Register the reconciled forward modal-shaft chapter claims."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ARTICLE = ROOT / "docs/research/proximal_distal_energy_transfer"
REGISTRY = ARTICLE / "data/claim_audit_registry.json"
INVENTORY = ARTICLE / "data/claim_candidate_inventory.json"
CHAPTER = "docs/research/proximal_distal_energy_transfer/chapters/_ch06bbb_forward_modal_shaft.qmd"
ARTIFACTS = [
    "docs/research/proximal_distal_energy_transfer/data/moving_base_modal_shaft_study.json",
    "docs/research/proximal_distal_energy_transfer/data/moving_base_modal_shaft_study.npz",
    "scripts/research/proximal_distal_energy/moving_base_modal_shaft.py",
    "scripts/research/proximal_distal_energy/run_moving_base_modal_shaft_study.py",
    "tests/research/test_moving_base_modal_shaft.py",
    "tests/research/test_moving_base_modal_shaft_evidence.py",
]


def _claim(
    claim_id, candidate_ids, statement, classification, status, boundary, falsifier
):
    return {
        "claim_id": claim_id,
        "candidate_ids": candidate_ids,
        "statement": statement,
        "classification": classification,
        "published_status": status,
        "audit_status": "machine_authority_reproduced_and_stale_prose_reconciled",
        "source_locations": [],
        "evidence_artifacts": ARTIFACTS,
        "model_domain": "Synthetic planar moving-base, two-hand, distributed modal-shaft model.",
        "uncertainty_boundary": boundary,
        "competing_explanations": [
            "linear-beam model-use failure",
            "synthetic parameter selection",
            "fixed-step integration",
            "planar geometry",
            "unmodeled impact and torsion",
        ],
        "negative_controls": [
            "same-state killswitch",
            "coincident moment arms",
            "reversed moment arms",
            "timestep refinement",
            "one-three-six mode comparison",
            "contact-power closure",
        ],
        "falsifier": falsifier,
        "adjudication": "Every passage was reconciled to regenerated hash-bound JSON/NPZ authority; obsolete run values were replaced and the failed model-use screen was promoted to the claim boundary.",
        "reviewer": "Codex technical audit",
        "last_verified_on": "2026-08-13",
    }


def main() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    candidates = [c for c in inventory["candidates"] if c["source_path"] == CHAPTER]
    ids = [c["candidate_id"] for c in candidates]
    by_id = {c["candidate_id"]: c for c in candidates}
    groups = {
        "PD-CLAIM-223": ids[:3],
        "PD-CLAIM-224": ids[3:7],
        "PD-CLAIM-225": ids[7:13],
        "PD-CLAIM-226": ids[13:16],
        "PD-CLAIM-227": ids[16:19],
        "PD-CLAIM-228": ids[19:24],
        "PD-CLAIM-229": ids[24:],
    }
    claims = [
        _claim(
            "PD-CLAIM-223",
            groups["PD-CLAIM-223"],
            "The study is an open synthetic numerical experiment, not calibrated equipment or human evidence.",
            "scope_and_provenance",
            "supported",
            "No governed equipment or participant measurements enter.",
            "The resource is presented as calibrated or prescriptive.",
        ),
        _claim(
            "PD-CLAIM-224",
            groups["PD-CLAIM-224"],
            "Finite-element modes are transported into one constrained forward KKT solve with rigid-modal inertial coupling.",
            "model_construction",
            "supported_numerically",
            "Planar Euler-Bernoulli assumptions remain.",
            "Modal mass/frequency or KKT contracts fail.",
        ),
        _claim(
            "PD-CLAIM-225",
            groups["PD-CLAIM-225"],
            "Separated grip reactions satisfy force-couple geometry and contact-power identities at recorded tolerances.",
            "mechanical_identity",
            "supported",
            "Achieved reactions are model outputs, not measured forces.",
            "Coincident/reversed controls or power closure fail.",
        ),
        _claim(
            "PD-CLAIM-226",
            groups["PD-CLAIM-226"],
            "The baseline reaches a 0.13476 tip-deflection ratio and fails the preregistered 0.05 linear-beam model-use screen.",
            "model_use_screen",
            "failed_quantitative_small_deflection_inference",
            "The run remains an out-of-domain numerical stress test.",
            "The release calls this quantitatively valid equipment response.",
        ),
        _claim(
            "PD-CLAIM-227",
            groups["PD-CLAIM-227"],
            "Mode truncation is excitation-dependent; three modes agree closely with six for the two declared inputs only.",
            "mode_truncation_screen",
            "conditionally_supported",
            "Six modes are a within-model reference, not truth or impact validation.",
            "The result is generalized to arbitrary loads or impact.",
        ),
        _claim(
            "PD-CLAIM-228",
            groups["PD-CLAIM-228"],
            "A same-state zero-command branch retains a negative force-generated couple for about 25 ms and converges under refinement and geometry controls.",
            "counterfactual_mechanism",
            "supported_for_out_of_screen_synthetic_model",
            "Command removal is not muscle inactivity and amplitudes are not equipment predictions.",
            "Persistence disappears or control identities fail.",
        ),
        _claim(
            "PD-CLAIM-229",
            groups["PD-CLAIM-229"],
            "Only numerical coupling, identities, and qualitative geometry survive; quantitative shaft, human, and coaching claims remain unsupported.",
            "bounded_conclusion",
            "supported_with_rejection",
            "The failed deflection screen is dispositive for linear-beam amplitude inference.",
            "Unsupported quantitative or human claims are restored.",
        ),
    ]
    mapping = {
        candidate: claim for claim, members in groups.items() for candidate in members
    }
    registry["candidate_reviews"] = [
        r
        for r in registry["candidate_reviews"]
        if r["candidate_id"] not in mapping
        and not set(r["claim_ids"]).intersection(groups)
    ]
    registry["candidate_reviews"].extend(
        {
            "candidate_id": c,
            "disposition": "material_claims_mapped",
            "claim_ids": [claim],
            "rationale": "This passage participates in the forward modal-shaft reconciliation.",
            "reviewer": "Codex technical audit",
            "last_verified_on": "2026-08-13",
        }
        for c, claim in mapping.items()
    )
    registry["claims"] = [
        c for c in registry["claims"] if c["claim_id"] not in groups
    ] + claims
    for claim in claims:
        claim["source_locations"] = [
            f"{by_id[c]['source_path']}:{by_id[c]['line_start']}"
            for c in claim["candidate_ids"]
        ]
    registry["paper"]["source_digest"] = inventory["source_digest"]
    registry["audit_scope"]["current_scope"] = (
        "The complete audit remains in progress. The forward modal-shaft chapter is fully reconciled, including a failed model-use screen."
    )
    REGISTRY.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
