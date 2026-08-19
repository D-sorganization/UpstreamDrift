"""Register the cross-engine forward-contact validity-horizon claims."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
ARTICLE = ROOT / "docs/research/proximal_distal_energy_transfer"
REGISTRY = ARTICLE / "data/claim_audit_registry.json"
INVENTORY = ARTICLE / "data/claim_candidate_inventory.json"
DATE = "2026-08-14"
CLAIM_IDS = {"PD-CLAIM-271", "PD-CLAIM-272", "PD-CLAIM-273"}
ARTIFACTS = [
    "docs/research/proximal_distal_energy_transfer/data/forward_contact_validity_horizon.json",
    "docs/research/proximal_distal_energy_transfer/data/forward_contact_validity_horizon.npz",
    "docs/research/proximal_distal_energy_transfer/figures/fig_forward_contact_validity_horizon.pdf",
    "scripts/research/proximal_distal_energy/forward_contact_validity_horizon.py",
    "scripts/research/proximal_distal_energy/run_forward_contact_validity_horizon.py",
    "scripts/research/proximal_distal_energy/make_forward_contact_validity_horizon_figure.py",
    "scripts/research/proximal_distal_energy/register_forward_contact_validity_horizon_claims.py",
    "tests/research/test_forward_contact_validity_horizon.py",
]


def _find(candidates: list[dict[str, Any]], suffix: str, prefix: str) -> dict[str, Any]:
    matches = [
        candidate
        for candidate in candidates
        if str(candidate["source_path"]).endswith(suffix)
        and str(candidate["text"]).startswith(prefix)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one {suffix} candidate beginning {prefix!r}")
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
        "audit_status": "paired_cross_engine_horizon_adverse_load_and_energy_controls_checked",
        "source_locations": [
            f"{candidate['source_path']}:{candidate['line_start']}"
            for candidate in candidates
        ],
        "evidence_artifacts": ARTIFACTS,
        "model_domain": "Six synthetic profiles, three grip spans, three phase states, ten one-factor branches, four horizons, and two native engines.",
        "uncertainty_boundary": boundary,
        "competing_explanations": [
            "shared reduced model structure",
            "engineering contact and driver parameters",
            "rigid club and hand-carriage substitution",
            "finite 50 ms observation interval",
        ],
        "negative_controls": [
            "zero driver from the first integration step",
            "contact stiffness and damping extremes",
            "represented hand-mass extremes",
            "timestep halving and doubling",
            "work--energy closure without threshold retuning",
        ],
        "falsifier": falsifier,
        "adjudication": "All registered outcomes were retained. No failure was observed through 50 ms, so persistence is reported as right-censored rather than extrapolated.",
        "reviewer": "Codex technical audit",
        "last_verified_on": DATE,
    }


def _build_claims(
    candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    selected = {
        "abstract": _find(
            candidates, "proximal_distal_energy_transfer.qmd", "Proximal-to-distal"
        ),
        "design": _find(
            candidates,
            "_ch06c_spatial_cross_formulation.qmd",
            "The 4 ms result establishes",
        ),
        "figure": _find(
            candidates,
            "_ch06c_spatial_cross_formulation.qmd",
            "![Cross-Engine Forward-Contact",
        ),
        "result": _find(
            candidates,
            "_ch06c_spatial_cross_formulation.qmd",
            "All 2,160 profile--span",
        ),
        "boundary": _find(
            candidates,
            "_ch06c_spatial_cross_formulation.qmd",
            "No first-failure horizon",
        ),
        "question": _find(
            candidates, "_ch08b_momentum_transfer_questions.qmd", "| How much transfer"
        ),
        "conclusion": _find(
            candidates, "_ch09_conclusions.qmd", "The closed-state validity-horizon"
        ),
    }
    claims = [
        _claim(
            "PD-CLAIM-271",
            [selected["design"]],
            statement="The registered horizon matrix advances every profile--span--phase state through four horizons under nominal conditions and nine one-factor adverse or null branches in paired MuJoCo and Pinocchio runs.",
            classification="closed_state_forward_horizon_design",
            status="complete_for_declared_reduced_matrix",
            boundary="The design varies engineering factors one at a time and does not sample a calibrated joint parameter distribution.",
            falsifier="Any registered case is omitted, silently dropped, receives unequal engine inputs, or lacks a typed outcome.",
        ),
        _claim(
            "PD-CLAIM-272",
            [selected["abstract"], selected["result"], selected["conclusion"]],
            statement="All 2,160 registered horizon cases pass the declared trajectory, wrench, normalized-energy, and work--energy closure gates; no first failure is observed through 50 ms.",
            classification="closed_state_cross_engine_validity_horizon_result",
            status="supported_and_right_censored_at_50_ms",
            boundary="Passing through 50 ms supplies no evidence beyond 50 ms and does not constitute a full delivery or impact simulation.",
            falsifier="Any committed case fails a gate, loses finite closure, or a reproduced result finds an earlier incomplete-pass horizon.",
        ),
        _claim(
            "PD-CLAIM-273",
            [selected["boundary"], selected["question"], selected["conclusion"]],
            statement="The validity-horizon result strengthens only the reduced hand-carriage reference and does not establish articulated anatomy, calibrated equipment, passive human transfer, timing economy, slack benefit, or coaching strategy.",
            classification="closed_state_forward_horizon_inference_boundary",
            status="explicitly_bounded",
            boundary="Articulated subject-scaled contact, distributed grip and shaft calibration, ground coupling, contact loss, delivery, impact, and governed human evidence remain open.",
            falsifier="A model, anatomy, equipment, delivery, or human conclusion is attributed to the reduced 50 ms agreement alone.",
        ),
    ]
    return claims, selected


def _replace_changed_candidates(
    registry: dict[str, Any], selected: dict[str, dict[str, Any]]
) -> None:
    replacements = {
        "PD-CAND-595f13f49a6683a9": selected["abstract"],
        "PD-CAND-54446a28f977e4fa": selected["question"],
    }
    reviews = {
        review["candidate_id"]: review for review in registry["candidate_reviews"]
    }
    for old_id, candidate in replacements.items():
        old_review = reviews.pop(old_id)
        old_review["candidate_id"] = candidate["candidate_id"]
        reviews[candidate["candidate_id"]] = old_review
        for claim in registry["claims"]:
            for index, candidate_id in enumerate(claim["candidate_ids"]):
                if candidate_id == old_id:
                    claim["candidate_ids"][index] = candidate["candidate_id"]
                    claim["source_locations"][index] = (
                        f"{candidate['source_path']}:{candidate['line_start']}"
                    )
    registry["candidate_reviews"] = list(reviews.values())


def _reconcile(
    registry: dict[str, Any],
    claims: list[dict[str, Any]],
    selected: dict[str, dict[str, Any]],
) -> None:
    registry["claims"] = [
        claim for claim in registry["claims"] if claim["claim_id"] not in CLAIM_IDS
    ]
    _replace_changed_candidates(registry, selected)
    reviews = {
        review["candidate_id"]: review for review in registry["candidate_reviews"]
    }
    for candidate in selected.values():
        reviews.setdefault(
            candidate["candidate_id"],
            {
                "candidate_id": candidate["candidate_id"],
                "disposition": "material_claims_mapped",
                "claim_ids": [],
                "rationale": "This passage states or bounds the registered forward-contact validity-horizon result.",
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
            reviews[candidate_id]["disposition"] = "material_claims_mapped"
            reviews[candidate_id]["claim_ids"] = sorted(
                set(reviews[candidate_id]["claim_ids"]) | {claim["claim_id"]}
            )
            reviews[candidate_id]["last_verified_on"] = DATE
    reviews[selected["figure"]["candidate_id"]].update(
        disposition="editorial_or_navigation",
        claim_ids=[],
        rationale="The figure include points to registered evidence but asserts no standalone result.",
    )
    registry["candidate_reviews"] = list(reviews.values())
    registry["claims"].extend(claims)


def _update_release(registry: dict[str, Any]) -> None:
    entries = {
        item["release_claim_key"]: item for item in registry["release_claim_inventory"]
    }
    entries["closed_state_forward_validity_horizon"] = {
        "release_claim_key": "closed_state_forward_validity_horizon",
        "published_status": "no_failure_observed_through_registered_50_ms_reduced_interval",
        "audit_state": "reviewed_as_right_censored_reduced_model_result",
    }
    registry["release_claim_inventory"] = list(entries.values())
    registry["audit_scope"]["current_scope"] = (
        "The complete paper inventory is adjudicated. Closed subject-scaled states "
        "pass a 2,160-case reduced cross-engine horizon and adverse-load map through "
        "50 ms. The no-failure result is right-censored; articulated contact, "
        "calibrated equipment, full delivery, and governed human validation remain open."
    )


def main() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    claims, selected = _build_claims(inventory["candidates"])
    _reconcile(registry, claims, selected)
    _update_release(registry)
    registry["paper"]["source_digest"] = inventory["source_digest"]
    REGISTRY.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
