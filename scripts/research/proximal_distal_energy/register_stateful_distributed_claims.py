"""Register the stateful distributed-grip campaign and fail-closed boundaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
ARTICLE = ROOT / "docs/research/proximal_distal_energy_transfer"
REGISTRY = ARTICLE / "data/claim_audit_registry.json"
INVENTORY = ARTICLE / "data/claim_candidate_inventory.json"
DATE = "2026-08-28"
CLAIM_IDS = {"PD-CLAIM-330", "PD-CLAIM-331", "PD-CLAIM-332"}
CHAPTER = "_ch08b_momentum_transfer_questions.qmd"
ARTIFACTS = [
    "docs/research/proximal_distal_energy_transfer/data/articulated_stateful_distributed_plan.json",
    "docs/research/proximal_distal_energy_transfer/data/articulated_stateful_distributed_smoke/summary.json",
    "docs/research/proximal_distal_energy_transfer/figures/fig_stateful_distributed_grip_falsification.pdf",
    "scripts/research/proximal_distal_energy/articulated_stateful_friction.py",
    "scripts/research/proximal_distal_energy/articulated_stateful_distributed_grip.py",
    "scripts/research/proximal_distal_energy/articulated_stateful_distributed_forward.py",
    "scripts/research/proximal_distal_energy/articulated_stateful_smoke_evaluator.py",
    "scripts/research/proximal_distal_energy/articulated_stateful_smoke_launcher.py",
    "scripts/research/proximal_distal_energy/articulated_stateful_summary.py",
    "scripts/research/proximal_distal_energy/make_stateful_distributed_figure.py",
    "tests/research/test_articulated_stateful_distributed_grip.py",
    "tests/research/test_articulated_stateful_distributed_forward.py",
    "tests/research/test_articulated_stateful_summary.py",
    "tests/research/test_make_stateful_distributed_figure.py",
]


def _find(candidates: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    matches = [
        candidate
        for candidate in candidates
        if str(candidate["source_path"]).endswith(CHAPTER)
        and str(candidate["text"]).startswith(prefix)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one stateful candidate for {prefix!r}")
    return matches[0]


def _claim(
    claim_id: str,
    candidates: list[dict[str, Any]],
    statement: str,
    classification: str,
    published_status: str,
    uncertainty_boundary: str,
    falsifier: str,
) -> dict[str, Any]:
    return {
        "claim_id": claim_id,
        "candidate_ids": [candidate["candidate_id"] for candidate in candidates],
        "statement": statement,
        "classification": classification,
        "published_status": published_status,
        "audit_status": "stateful_distributed_grip_campaign_checked",
        "adjudication_outcome": "supported",
        "source_locations": [
            f"{candidate['source_path']}:{candidate['line_start']}"
            for candidate in candidates
        ],
        "evidence_artifacts": ARTIFACTS,
        "model_domain": (
            "One synthetic articulated source state at 0.12 seconds, nine stateful "
            "distributed-grip variants, three time steps, a five-millisecond "
            "horizon, and two registered native engines. Twenty-seven MuJoCo cases "
            "completed; twenty-seven Pinocchio cases were typed unavailable."
        ),
        "uncertainty_boundary": uncertainty_boundary,
        "competing_explanations": [
            "operator-split coupling error rather than physical dissipation",
            "synthetic preload release rather than measured tissue state",
            "five-millisecond transient rather than delivery behavior",
            "one source state rather than a geometry or participant distribution",
            "unmatched active constraints rather than a slack benefit",
        ],
        "negative_controls": [
            "frictionless killswitch",
            "low-friction slip probe",
            "high-friction inactive-cap control",
            "low and high tangential stiffness",
            "zero preload",
            "complete source-velocity reversal",
            "fully open contact probe",
            "three-resolution passive-energy and coupling-work refinement",
            "typed native-engine unavailability retained as a failed promotion gate",
        ],
        "falsifier": falsifier,
        "adjudication": (
            "The executed outcomes, including the two frozen refinement failures "
            "and unavailable native parity, are retained without threshold changes "
            "or human, anatomical, equipment, or coaching promotion."
        ),
        "reviewer": "Codex technical audit",
        "last_verified_on": DATE,
    }


def _claims(selected: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        _claim(
            "PD-CLAIM-330",
            [selected["law"], selected["execution"]],
            "A preregistered elastic--perfectly-plastic distributed-grip comparator retains tangential state, Coulomb radial return, slip/opening/projection dissipation, and separate passive-energy and operator-coupling ledgers across the registered matrix.",
            "stateful_distributed_grip_design_and_execution",
            "executed_with_disclosed_runtime_amendment",
            "The comparator is not calibrated skin, finger, tendon, neural, shaft, ground, equipment, or human contact, and only one source state and a five-millisecond horizon were evaluated.",
            "The plan, source state, variants, thresholds, operator split, or retained unavailable execution differs from the published contract.",
        ),
        _claim(
            "PD-CLAIM-331",
            [selected["result"], selected["interpretation"]],
            "All nine completed MuJoCo variants show contracting passive-energy defect, while the frictionless preload-release and low-friction slip variants fail the frozen coupling-work refinement gate; Pinocchio parity is unavailable and promotion remains false.",
            "stateful_distributed_grip_fail_closed_result",
            "promotion_withheld_for_refinement_and_native_parity",
            "The result is a one-state short-horizon software and model diagnostic; speed differences are not matched delivery effects or human evidence.",
            "A reproduced checkpoint changes a registered status or residual, a failed ratio is suppressed or relabeled, unavailable parity is treated as agreement, or promotion becomes true without a prospective passing campaign.",
        ),
        _claim(
            "PD-CLAIM-332",
            [selected["law"], selected["interpretation"]],
            "The stateful campaign does not establish that opening, preload removal, friction, stick--slip behavior, or slack creates clubhead speed or identifies a beneficial human strategy.",
            "stateful_distributed_grip_inference_boundary",
            "explicitly_bounded",
            "Matched work, load, delivery, shaft and ground coupling, subject-specific tissue, governed bilateral wrench data, and participant-held-out outcomes remain open.",
            "Any biological, equipment, coaching, safety, timing-economy, or slack-benefit claim is attributed to this synthetic five-millisecond campaign alone.",
        ),
    ]


def main() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    candidates = inventory["candidates"]
    valid_ids = {candidate["candidate_id"] for candidate in candidates}
    selected = {
        "table": _find(candidates, "| How much transfer is drift-mediated?"),
        "law": _find(candidates, "The next preregistered step"),
        "execution": _find(candidates, "The prospective matrix"),
        "result": _find(candidates, "All nine MuJoCo variants"),
        "figure": _find(candidates, "![Stateful Distributed-Grip"),
        "interpretation": _find(candidates, "The separately integrated"),
    }
    registry["claims"] = [
        claim for claim in registry["claims"] if claim["claim_id"] not in CLAIM_IDS
    ]
    for claim in registry["claims"]:
        claim["candidate_ids"] = [
            candidate_id
            for candidate_id in claim.get("candidate_ids", [])
            if candidate_id in valid_ids
        ]
    table_claim_ids = {
        "PD-CLAIM-243",
        "PD-CLAIM-247",
        "PD-CLAIM-253",
        "PD-CLAIM-273",
        "PD-CLAIM-292",
        "PD-CLAIM-296",
    }
    for claim in registry["claims"]:
        if claim["claim_id"] in table_claim_ids:
            claim["candidate_ids"] = list(
                dict.fromkeys(
                    [*claim["candidate_ids"], selected["table"]["candidate_id"]]
                )
            )
    claims = _claims(selected)
    registry["claims"].extend(claims)
    reviews = {
        review["candidate_id"]: review
        for review in registry["candidate_reviews"]
        if review["candidate_id"] in valid_ids
    }
    table_id = selected["table"]["candidate_id"]
    reviews[table_id] = {
        "candidate_id": table_id,
        "disposition": "material_claims_mapped",
        "claim_ids": sorted(table_claim_ids | CLAIM_IDS),
        "rationale": "The updated question table carries existing bounded answers and the new fail-closed stateful result.",
        "reviewer": "Codex technical audit",
        "last_verified_on": DATE,
    }
    for claim in claims:
        for candidate_id in claim["candidate_ids"]:
            review = reviews.setdefault(
                candidate_id,
                {
                    "candidate_id": candidate_id,
                    "disposition": "material_claims_mapped",
                    "claim_ids": [],
                    "rationale": "This passage states or bounds the stateful distributed-grip campaign.",
                    "reviewer": "Codex technical audit",
                    "last_verified_on": DATE,
                },
            )
            review["claim_ids"] = list(
                dict.fromkeys([*review["claim_ids"], claim["claim_id"]])
            )
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
        "The complete paper inventory is adjudicated. The stateful distributed-grip "
        "campaign retains its refinement failures and unavailable native parity; "
        "calibrated tissue, shaft, ground, delivery, and governed human validation "
        "remain open."
    )
    REGISTRY.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
