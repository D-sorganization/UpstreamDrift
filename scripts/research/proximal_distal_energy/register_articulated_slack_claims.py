"""Register typed articulated slack/contact claims and boundaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
ARTICLE = ROOT / "docs/research/proximal_distal_energy_transfer"
REGISTRY = ARTICLE / "data/claim_audit_registry.json"
INVENTORY = ARTICLE / "data/claim_candidate_inventory.json"
DATE = "2026-08-15"
CLAIM_IDS = {"PD-CLAIM-283", "PD-CLAIM-284", "PD-CLAIM-285"}
ARTIFACTS = [
    "docs/research/proximal_distal_energy_transfer/data/articulated_slack_atlas.json",
    "docs/research/proximal_distal_energy_transfer/data/articulated_slack_atlas.npz",
    "docs/research/proximal_distal_energy_transfer/figures/fig_articulated_slack_atlas.pdf",
    "scripts/research/proximal_distal_energy/articulated_slack_contact.py",
    "scripts/research/proximal_distal_energy/articulated_slack_forward.py",
    "scripts/research/proximal_distal_energy/articulated_slack_atlas.py",
    "scripts/research/proximal_distal_energy/run_articulated_slack_atlas.py",
    "scripts/research/proximal_distal_energy/make_articulated_slack_figure.py",
    "scripts/research/proximal_distal_energy/register_articulated_slack_claims.py",
    "tests/research/test_articulated_slack_contact.py",
    "tests/research/test_articulated_slack_forward.py",
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
        raise ValueError(f"expected one articulated-slack candidate for {prefix!r}")
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
        "audit_status": "typed_articulated_slack_events_and_matched_controls_checked",
        "source_locations": [
            f"{candidate['source_path']}:{candidate['line_start']}"
            for candidate in candidates
        ],
        "evidence_artifacts": ARTIFACTS,
        "model_domain": (
            "Eighteen synthetic articulated states, sixteen natural law/preload/"
            "velocity conditions, two isolated event probes, three time steps, "
            "two native engines, and a five-millisecond horizon."
        ),
        "uncertainty_boundary": boundary,
        "competing_explanations": [
            "preload-energy mismatch rather than slack",
            "short right-censored horizon",
            "memoryless radial law rather than distributed contact or tissue",
            "shared analytical contact law and semi-implicit integrator",
        ],
        "negative_controls": [
            "bilateral and zero-slack tension comparators",
            "common-displacement versus matched-extension preload",
            "initial-velocity reversal",
            "open-to-taut and taut-to-open event probes",
            "three-step refinement and two native dynamics operators",
        ],
        "falsifier": falsifier,
        "adjudication": (
            "The registered event and natural-state results are reported separately "
            "with explicit right-censoring and no human or coaching inference."
        ),
        "reviewer": "Codex technical audit",
        "last_verified_on": DATE,
    }


def _selected_candidates(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "law": _find(candidates, "The next gate replaces the ambiguous word"),
        "design": _find(candidates, "The atlas evaluates eighteen"),
        "figure": _find(candidates, "![Typed Unilateral Slack and Reattachment"),
        "numerics": _find(candidates, "All numerical and cross-engine gates pass"),
        "events": _find(candidates, "The natural-state branches produce no opening"),
        "boundary": _find(candidates, "The result therefore falsifies any claim"),
    }


def _repeated_candidates(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "law_power": _find(candidates, "The loading-only damper is a declared"),
        "chapter_boundary": _find(candidates, "The bounded forward experiments now"),
        "open_claims": _find(candidates, "- **Passive Contact Origin Is Inconclusive"),
        "ladder_boundary": next(
            candidate
            for candidate in candidates
            if str(candidate["source_path"]).endswith("_ch07_model_ladder.qmd")
            and str(candidate["text"]).startswith(
                "This executes independent forward spatial contact"
            )
        ),
        "ladder_summary": next(
            candidate
            for candidate in candidates
            if str(candidate["source_path"]).endswith("_ch07_model_ladder.qmd")
            and str(candidate["text"]).startswith("The discrepancy matrix in")
        ),
        "ladder_list": next(
            candidate
            for candidate in candidates
            if str(candidate["source_path"]).endswith("_ch07_model_ladder.qmd")
            and str(candidate["text"]).startswith(
                "1. interaction force may remain nonzero"
            )
        ),
        "slack_summary": next(
            candidate
            for candidate in candidates
            if str(candidate["source_path"]).endswith(
                "_ch08b_momentum_transfer_questions.qmd"
            )
            and str(candidate["text"]).startswith(
                "This evidence supports only a typed experimental program"
            )
        ),
    }


def _claims(
    selected: dict[str, Any], candidates: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    return [
        _claim(
            "PD-CLAIM-283",
            [selected["law"], selected["design"]],
            "A registered articulated atlas distinguishes bilateral, tension-only, and dead-zone attachment laws under common-displacement, matched-extension, reversal, and isolated event controls.",
            "typed_articulated_slack_design",
            "complete_for_declared_bounded_matrix",
            "The passive laws and parameters are synthetic point-attachment comparators, not tissue or intent.",
            "An open interface transmits force, a law creates compression contrary to its contract, or a preload control is omitted.",
        ),
        _claim(
            "PD-CLAIM-284",
            [selected["numerics"], selected["events"]],
            "The 1,944-trajectory atlas passes power, passivity, work-energy, refinement, native-engine trajectory/force, and active-set parity gates; natural states show no transition before five milliseconds while isolated probes reproduce opening and reattachment.",
            "typed_articulated_slack_result",
            "supported_at_declared_five_millisecond_synthetic_tier",
            "Natural event absence is right-censored; probe events qualify implementation and are not natural-state predictions.",
            "Any reproduced cell exceeds a registered numerical/parity gate or an event probe fails to cross the declared boundary.",
        ),
        _claim(
            "PD-CLAIM-285",
            [selected["boundary"]],
            "Changing the law and preload match can reverse open/taut classification, but the atlas does not establish slack benefit, necessity, intentionality, self-correction, timing economy, delivery advantage, or human strategy.",
            "typed_articulated_slack_inference_boundary",
            "explicitly_bounded",
            "Longer distributed grip/shaft contact, calibrated tissue/friction/ground, matched delivery, and governed human measurements remain open.",
            "A biological, benefit, timing, delivery, or coaching claim is attributed to this bounded synthetic atlas alone.",
        ),
    ]


def _update_reviews(
    registry: dict[str, Any],
    inventory: dict[str, Any],
    selected: dict[str, Any],
    claims: list[dict[str, Any]],
) -> None:
    valid_ids = {candidate["candidate_id"] for candidate in inventory["candidates"]}
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
            "rationale": "This passage states or bounds the typed articulated slack gate.",
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


def _attach_repeated_candidates(
    registry: dict[str, Any],
    inventory: dict[str, Any],
    repeated: dict[str, Any],
) -> None:
    valid_ids = {candidate["candidate_id"] for candidate in inventory["candidates"]}
    for claim in registry["claims"]:
        claim["candidate_ids"] = [
            candidate_id
            for candidate_id in claim.get("candidate_ids", [])
            if candidate_id in valid_ids
        ]
    mapping = {
        "law_power": ("PD-CLAIM-283",),
        "chapter_boundary": (
            "PD-CLAIM-128",
            "PD-CLAIM-262",
            "PD-CLAIM-264",
            "PD-CLAIM-270",
            "PD-CLAIM-285",
        ),
        "open_claims": ("PD-CLAIM-128", "PD-CLAIM-262", "PD-CLAIM-285"),
        "ladder_boundary": ("PD-CLAIM-128", "PD-CLAIM-262", "PD-CLAIM-285"),
        "ladder_summary": ("PD-CLAIM-128", "PD-CLAIM-284", "PD-CLAIM-285"),
        "ladder_list": ("PD-CLAIM-128", "PD-CLAIM-284", "PD-CLAIM-285"),
        "slack_summary": ("PD-CLAIM-253", "PD-CLAIM-285"),
    }
    claims = {claim["claim_id"]: claim for claim in registry["claims"]}
    reviews = {
        review["candidate_id"]: review for review in registry["candidate_reviews"]
    }
    for name, claim_ids in mapping.items():
        candidate = repeated[name]
        candidate_id = candidate["candidate_id"]
        location = f"{candidate['source_path']}:{candidate['line_start']}"
        for claim_id in claim_ids:
            claim = claims[claim_id]
            if candidate_id not in claim["candidate_ids"]:
                claim["candidate_ids"].append(candidate_id)
            claim["source_locations"] = list(
                dict.fromkeys([*claim.get("source_locations", []), location])
            )
        reviews[candidate_id] = {
            "candidate_id": candidate_id,
            "disposition": "material_claims_mapped",
            "claim_ids": list(claim_ids),
            "rationale": "This repeated boundary or model-ladder passage inherits the mapped primary claim limits.",
            "reviewer": "Codex technical audit",
            "last_verified_on": DATE,
        }
    claims["PD-CLAIM-128"]["statement"] = (
        "The discrepancy matrix records explicit branch capabilities rather than "
        "cumulative triangular inheritance; ten bounded findings are supported "
        "somewhere in executed tiers, while calibrated anatomical, distributed, "
        "and independently measured human transport remain untested."
    )
    claims["PD-CLAIM-253"]["statement"] = (
        "A two-excitation constitutive audit separates five slack classes, and a "
        "1,944-trajectory articulated atlas transports three attachment classes "
        "under preload and event controls; neither establishes class identity or benefit."
    )
    claims["PD-CLAIM-262"]["statement"] = (
        "The bounded articulated bilateral and typed unilateral screens advance "
        "closed states through 5 ms with independent-engine controls, but timing, "
        "recovery, and benefit claims require longer calibrated distributed contact."
    )
    registry["candidate_reviews"] = list(reviews.values())


def main() -> None:
    """Write typed-slack claims and release-level disposition."""

    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    selected = _selected_candidates(inventory["candidates"])
    repeated = _repeated_candidates(inventory["candidates"])
    claims = _claims(selected, inventory["candidates"])
    registry["claims"] = [
        claim for claim in registry["claims"] if claim["claim_id"] not in CLAIM_IDS
    ] + claims
    _update_reviews(registry, inventory, selected, claims)
    _attach_repeated_candidates(registry, inventory, repeated)
    entries = {
        item["release_claim_key"]: item for item in registry["release_claim_inventory"]
    }
    entries["typed_articulated_slack"] = {
        "release_claim_key": "typed_articulated_slack",
        "published_status": "five_millisecond_typed_attachment_event_gate_qualified",
        "audit_state": "reviewed_as_right_censored_synthetic_contact_result",
    }
    registry["release_claim_inventory"] = list(entries.values())
    registry["audit_scope"]["current_scope"] = (
        "The complete paper inventory is adjudicated. The articulated bilateral "
        "and typed unilateral attachment tiers pass their registered five-millisecond "
        "gates. Distributed calibrated contact and governed human validation remain open."
    )
    registry["paper"]["source_digest"] = inventory["source_digest"]
    REGISTRY.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
