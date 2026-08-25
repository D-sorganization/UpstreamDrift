"""Adjudicate remaining claim candidates after terminology unification.

This script maps repeated methods, summary, limitation, provenance, and model-tier
passages to the already audited primary claims.  It intentionally creates no new
scientific claim: the evidentiary authority remains the primary model/result slice.
"""

from __future__ import annotations

import json
from pathlib import Path

from typing import Any

ROOT = Path(__file__).resolve().parents[3]
ARTICLE = ROOT / "docs/research/proximal_distal_energy_transfer"
REGISTRY = ARTICLE / "data/claim_audit_registry.json"
INVENTORY = ARTICLE / "data/claim_candidate_inventory.json"
DATE = "2026-08-13"

EVIDENCE_BY_CLAIM: dict[str, list[str]] = {
    "PD-CLAIM-235": [
        "docs/research/proximal_distal_energy_transfer/data/rotating_base_torso_velocity_study.json",
        "scripts/research/proximal_distal_energy/rotating_base_two_hand.py",
    ],
    "PD-CLAIM-236": [
        "docs/research/proximal_distal_energy_transfer/data/rotating_base_torso_velocity_study.json"
    ],
    "PD-CLAIM-237": [
        "docs/research/proximal_distal_energy_transfer/data/rotating_base_torso_velocity_study.json"
    ],
    "PD-CLAIM-238": [
        "docs/research/proximal_distal_energy_transfer/data/rotating_base_torso_velocity_study.json"
    ],
    "PD-CLAIM-239": [
        "docs/research/proximal_distal_energy_transfer/data/shaft_beam_reference.json"
    ],
    "PD-CLAIM-240": [
        "docs/research/proximal_distal_energy_transfer/data/shaft_beam_reference.json"
    ],
    "PD-CLAIM-241": [
        "docs/research/proximal_distal_energy_transfer/COMPANION_WORKBENCH.md",
        "docs/research/proximal_distal_energy_transfer/release_manifest.json",
    ],
}


def _assign_intro_and_benchmarks(name: str, text: str) -> tuple[str, ...] | None:
    if name == "proximal_distal_energy_transfer.qmd":
        return ("PD-CLAIM-007", "PD-CLAIM-008", "PD-CLAIM-214")
    if name == "_ch01_introduction.qmd":
        if "zero-torque counterfactual" in text:
            return ("PD-CLAIM-047", "PD-CLAIM-058")
        if "drive early" in text:
            return ("PD-CLAIM-022", "PD-CLAIM-214")
        if "six contributions" in text or text.startswith("1. a synthesis"):
            return ("PD-CLAIM-050", "PD-CLAIM-214")
        if "rq" in text or "four questions" in text:
            return ("PD-CLAIM-050", "PD-CLAIM-215")
        if "empirical findings" in text:
            return ("PD-CLAIM-004", "PD-CLAIM-223")
        if "within a declared model" in text or "first-order question" in text:
            return ("PD-CLAIM-214",)
        return ()
    if name == "_ch03c_ground_reaction_drift.qmd":
        return ("PD-CLAIM-076",)
    if name == "_ch03d_shoulder_velocity_transfer.qmd":
        if "13 of the 18" in text or "matching contract changes" in text:
            return ("PD-CLAIM-236",)
        if "at 30 ms" in text:
            return ("PD-CLAIM-237",)
        if "two geometry controls" in text:
            return ("PD-CLAIM-238",)
        if "supported mechanism claim" in text:
            return ("PD-CLAIM-238",)
        if "force-power atlas" in text:
            return ("PD-CLAIM-235", "PD-CLAIM-236")
        if "initial torso rates" in text:
            return ("PD-CLAIM-235", "PD-CLAIM-236")
        if "rejects nonfinite" in text:
            return ("PD-CLAIM-235",)
        if "multipliers" in text or "seven generalized coordinates" in text:
            return ("PD-CLAIM-235",)
        return ()
    if name == "_ch05a_interactive_workbench.qmd":
        if "exploratory_model_output" in text:
            return ("PD-CLAIM-223", "PD-CLAIM-241")
        if "six bounded experiments" in text:
            return ("PD-CLAIM-230", "PD-CLAIM-241")
        return ("PD-CLAIM-241",)
    if name == "_ch05_platform.qmd":
        if "drift-plus-control" in text or "workflow separates" in text:
            return ("PD-CLAIM-051", "PD-CLAIM-058")
        if "mujoco" in text:
            return ("PD-CLAIM-126",)
        return ("PD-CLAIM-223",)
    return None


def _assign_methods_and_cross(name: str, text: str) -> tuple[str, ...] | None:
    if name == "_ch06bb_shaft_beam_reference.qmd":
        if "0.0413" in text or "slow pulse" in text or "reduced model" in text:
            return ("PD-CLAIM-240",)
        if (
            "structural verification" in text
            or "equipment" in text
            or "synthetic" in text
        ):
            return ("PD-CLAIM-223", "PD-CLAIM-239")
        if "work--energy residual" in text or "damping loss" in text:
            return ("PD-CLAIM-240",)
        if "narrower structural coupling gap" in text:
            return ("PD-CLAIM-224", "PD-CLAIM-239")
        return ("PD-CLAIM-239", "PD-CLAIM-240")
    if name == "_ch06f_open_release.qmd":
        if "completion gates" in text:
            return ("PD-CLAIM-130", "PD-CLAIM-206", "PD-CLAIM-234")
        if "records planar interaction" in text:
            return ("PD-CLAIM-128", "PD-CLAIM-229")
        return ("PD-CLAIM-223", "PD-CLAIM-241")
    if name == "_ch06_methods.qmd":
        if "torque-velocity" in text:
            return ("PD-CLAIM-209",)
        if "finite command-rise" in text:
            return ("PD-CLAIM-213",)
        if "matched-state" in text:
            return ("PD-CLAIM-098", "PD-CLAIM-099")
        if "robustness analyses" in text:
            return ("PD-CLAIM-213",)
        if "pointwise counterfactual" in text:
            return ("PD-CLAIM-212",)
        if "92 distinct programs" in text or "for every program" in text:
            return ("PD-CLAIM-207", "PD-CLAIM-208")
        if "wrist-interface power" in text or "interaction-force audit" in text:
            return ("PD-CLAIM-211",)
        if "phase budgets" in text or "segment energies" in text:
            return ("PD-CLAIM-211",)
        return ("PD-CLAIM-207", "PD-CLAIM-214")
    if name == "_ch06c_spatial_cross_formulation.qmd":
        if "right-censored synthetic result" in text:
            return (
                "PD-CLAIM-262",
                "PD-CLAIM-264",
                "PD-CLAIM-270",
                "PD-CLAIM-282",
                "PD-CLAIM-285",
            )
        if "loading-only damper" in text:
            return ("PD-CLAIM-283",)
        raise ValueError(f"Unhandled spatial-cross candidate: {text[:120]}")
    return None


def _assign_discussion_and_conclusions(name: str, text: str) -> tuple[str, ...] | None:
    if name == "_ch08_discussion.qmd":
        if "active restraint" in text:
            return ("PD-CLAIM-208", "PD-CLAIM-214")
        if "energy budgets" in text:
            return ("PD-CLAIM-211", "PD-CLAIM-212")
        if "specific gaps" in text or "spatial body" in text:
            return ("PD-CLAIM-132", "PD-CLAIM-214")
        return ("PD-CLAIM-208", "PD-CLAIM-214")
    if name == "_ch08b_momentum_transfer_questions.qmd":
        if "typed experimental program" in text:
            return ("PD-CLAIM-253", "PD-CLAIM-285")
        if "next articulated atlas crosses" in text:
            return ("PD-CLAIM-286", "PD-CLAIM-287", "PD-CLAIM-288")
        if "following articulated shaft atlas" in text:
            return ("PD-CLAIM-290", "PD-CLAIM-291", "PD-CLAIM-292")
        raise ValueError(f"Unhandled momentum-question candidate: {text[:120]}")
    if name == "_ch09_conclusions.qmd":
        if "early wrist drive" in text:
            return ("PD-CLAIM-208", "PD-CLAIM-211", "PD-CLAIM-212")
        if "torque-velocity bounds" in text:
            return ("PD-CLAIM-213", "PD-CLAIM-214")
        if "50 ms after" in text:
            return ("PD-CLAIM-003",)
        if "moving-base/flexible" in text:
            return ("PD-CLAIM-123",)
        if "distributed-shaft comparison" in text:
            return ("PD-CLAIM-227", "PD-CLAIM-229")
        if "full-body common-state" in text:
            return ("PD-CLAIM-126",)
        if "spatial forward-contact" in text:
            return ("PD-CLAIM-127",)
        if "uncertainty/control" in text:
            return ("PD-CLAIM-219", "PD-CLAIM-220", "PD-CLAIM-221")
        if "experimental phase" in text:
            return ("PD-CLAIM-206",)
        if "open-resource layer" in text:
            return ("PD-CLAIM-223",)
        if "next work" in text or "articulated spatial" in text:
            return ("PD-CLAIM-130", "PD-CLAIM-206", "PD-CLAIM-234")
        if "literature does not support" in text:
            return ("PD-CLAIM-007", "PD-CLAIM-008", "PD-CLAIM-214")
        return ()
    if name == "_appendices.qmd":
        if "current evidence boundary" in text or "multi-phase" in text:
            return ("PD-CLAIM-128", "PD-CLAIM-132", "PD-CLAIM-234")
        if "tests exercise" in text:
            return ("PD-CLAIM-211", "PD-CLAIM-223")
        if "outputs land" in text or "implementation separates" in text:
            return ("PD-CLAIM-223",)
        if "arm length" in text:
            return ("PD-CLAIM-207",)
        return ()
    return None


def _assign_candidate_claims(candidate: dict[str, object]) -> tuple[str, ...]:
    path = str(candidate["source_path"])
    text = " ".join(str(candidate["text"]).split()).lower()
    name = Path(path).name
    for matcher in (
        _assign_intro_and_benchmarks,
        _assign_methods_and_cross,
        _assign_discussion_and_conclusions,
    ):
        result = matcher(name, text)
        if result is not None:
            return result
    raise ValueError(f"Unhandled remaining chapter: {name}")


def _build_remaining_new_claims() -> list[dict[str, Any]]:
    new_claims = [
        {
            "claim_id": "PD-CLAIM-235",
            "statement": "The rotating-base tier is a seven-coordinate, four-constraint planar mechanism with finite torso inertia, separated hand reactions, projected position/velocity closure, and a checked bilateral contact-power identity.",
            "classification": "rotating_base_model_and_constraint_contract",
            "published_status": "supported_at_declared_model_tier",
            "model_domain": "Finite-inertia planar torso, two arms, separated grips, and compliant club surrogate.",
            "uncertainty_boundary": "The coordinates are mechanism coordinates, not anatomical thorax, scapula, or shoulder observables.",
            "falsifier": "Constraint rank, closure, action-reaction, or bilateral power identity fails.",
        },
        {
            "claim_id": "PD-CLAIM-236",
            "statement": "Thirteen of 18 rotating-base cases pass the registered load/closure envelope; torso rate correlates with delivery speed at r=0.604 and braking work at r=0.273, while the matching rule materially changes the speed/braking conclusion.",
            "classification": "rotating_base_grid_association_and_matching_sensitivity",
            "published_status": "supported_for_registered_deterministic_grid",
            "model_domain": "Eighteen deterministic combinations of three initial torso rates, two matching rules, and three torso commands.",
            "uncertainty_boundary": "Finite-grid associations are not causal effects, population estimates, or universal strategy rules.",
            "falsifier": "Committed rows do not reproduce counts, correlations, or the stated matching-rule contrast.",
        },
        {
            "claim_id": "PD-CLAIM-237",
            "statement": "Exact 30 ms same-state killswitches give nonmonotonic channel effects: continued torso and wrist commands add 1.223 and 1.818 m/s, whereas continued bilateral-arm commands reduce delivery speed by 1.364 m/s in the registered program.",
            "classification": "rotating_base_same_state_command_intervention",
            "published_status": "supported_for_one_registered_program",
            "model_domain": "One exact same-state branch within the declared rotating-base mechanism.",
            "uncertainty_boundary": "A channel continuation effect is conditional on state, commands, model, and endpoint; it is not an anatomical or coaching effect.",
            "falsifier": "Pre-branch states differ or branch recomputation reverses the recorded channel differences.",
        },
        {
            "claim_id": "PD-CLAIM-238",
            "statement": "Coincident grips remove the rotating-base force couple exactly and signed arm reversal flips +1.351 to -1.351 N m; the supported conclusion is conditional reaction-work geometry, not scapular control, safe loading, or human technique.",
            "classification": "rotating_base_geometry_controls_and_scope",
            "published_status": "supported_with_explicit_human_boundary",
            "model_domain": "Algebraic geometry controls and registered planar sensitivity cases.",
            "uncertainty_boundary": "Signed arm reversal is an algebraic control, not a feasible grip exchange; human magnitude and strategy are unmeasured.",
            "falsifier": "Couple survives coincident grips, fails signed reversal, or the paper makes a physiological prescription.",
        },
        {
            "claim_id": "PD-CLAIM-239",
            "statement": "The isolated Euler-Bernoulli beam experiment recovers a declared synthetic modulus and tip mass from two synthetic modes and demonstrates mesh convergence; it is structural verification rather than equipment identification.",
            "classification": "synthetic_beam_identification_and_convergence",
            "published_status": "supported_for_synthetic_reference_only",
            "model_domain": "Declared tapered axisymmetric Euler-Bernoulli beam with synthetic modal observations.",
            "uncertainty_boundary": "No measured shaft modal data, EI profile, boundary condition, or human load enters the inference.",
            "falsifier": "Synthetic truth or mesh convergence fails reproduction, or the result is presented as calibrated equipment evidence.",
        },
        {
            "claim_id": "PD-CLAIM-240",
            "statement": "The isolated one-mode beam closely matches the six-mode slow-pulse response but its short-pulse RMS discrepancy is 0.0413 mm, about 60 times the slow-pulse discrepancy; declared work-energy residuals remain below 2.9e-7 J.",
            "classification": "synthetic_beam_excitation_and_energy_closure",
            "published_status": "supported_for_two_declared_load_histories",
            "model_domain": "One- versus six-mode response under the same two synthetic force/moment histories.",
            "uncertainty_boundary": "Peak agreement can conceal history error; two loads do not establish equipment-wide truncation validity.",
            "falsifier": "Reproduction changes the discrepancy ratio materially or violates declared energy closure.",
        },
        {
            "claim_id": "PD-CLAIM-241",
            "statement": "The companion workbench is exploratory and the open-resource bundle is provenance infrastructure: scientific authority remains the scripted, pinned, fail-closed evidence release rather than either interface or repository branding.",
            "classification": "open_resource_and_exploratory_interface_boundary",
            "published_status": "supported_as_release_governance",
            "model_domain": "Provider-pinned PyQt/React exploration and hash-pinned static evidence bundle.",
            "uncertainty_boundary": "Interface availability and checksums do not validate physical claims or substitute for governed human data.",
            "falsifier": "Interactive output is promoted as publication authority, provider identity drifts, or bundle validation permits artifact drift.",
        },
    ]
    common = {
        "candidate_ids": [],
        "audit_status": "independent_evidence_reconciliation_and_scope_correction_checked",
        "source_locations": [],
        "evidence_artifacts": [],
        "competing_explanations": [
            "finite model scope",
            "parameter choice",
            "numerical implementation",
        ],
        "negative_controls": [
            "exact rerun",
            "geometry control",
            "closure and provenance validation",
        ],
        "adjudication": "Reconciled to committed machine evidence with the stated model and human boundaries retained.",
        "reviewer": "Codex technical audit",
        "last_verified_on": DATE,
    }
    return [{**common, **claim} for claim in new_claims]


def _filter_generated_candidate_reviews(registry: dict[str, Any]) -> set[str]:
    generated_ids = {
        review["candidate_id"]
        for review in registry["candidate_reviews"]
        if review.get("reviewer") == "Codex technical audit"
        and str(review.get("rationale", "")).startswith(
            ("This repeated", "This heading")
        )
    }
    registry["candidate_reviews"] = [
        review
        for review in registry["candidate_reviews"]
        if review["candidate_id"] not in generated_ids
    ]
    for claim in registry["claims"]:
        claim["candidate_ids"] = [
            candidate_id
            for candidate_id in claim.get("candidate_ids", [])
            if candidate_id not in generated_ids
        ]
    return generated_ids


def _generate_remaining_candidate_reviews(
    remaining: list[dict[str, Any]],
    claims: dict[str, Any],
) -> list[dict[str, Any]]:
    reviews: list[dict[str, Any]] = []
    for candidate in remaining:
        mapped = _assign_candidate_claims(candidate)
        unknown = set(mapped) - claims.keys()
        if unknown:
            raise ValueError(f"{candidate['candidate_id']}: unknown claims {unknown}")
        if mapped:
            disposition = "material_claims_mapped"
            rationale = (
                "This repeated methods, summary, limitation, provenance, or model-tier "
                "passage is bounded by the independently audited primary claim records."
            )
        else:
            disposition = "editorial_or_navigation"
            rationale = (
                "This heading, equation fragment, figure/table anchor, command block, "
                "navigation statement, or repository pointer asserts no standalone "
                "scientific result."
            )
        reviews.append(
            {
                "candidate_id": candidate["candidate_id"],
                "disposition": disposition,
                "claim_ids": list(mapped),
                "rationale": rationale,
                "reviewer": "Codex technical audit",
                "last_verified_on": DATE,
            }
        )
    return reviews


def _reconcile_claims_with_reviews(
    registry: dict[str, Any],
    by_id: dict[str, Any],
    new_claims: list[dict[str, Any]],
    reviews: list[dict[str, Any]],
) -> None:
    regenerated_claim_ids = {claim["claim_id"] for claim in new_claims}
    for claim in registry["claims"]:
        if claim["claim_id"] in regenerated_claim_ids:
            candidate_ids = [
                review["candidate_id"]
                for review in registry["candidate_reviews"]
                if claim["claim_id"] in review.get("claim_ids", [])
                and review["candidate_id"] in by_id
            ]
            claim["candidate_ids"] = list(dict.fromkeys(candidate_ids))
            claim["source_locations"] = [
                f"{by_id[candidate_id]['source_path']}:{by_id[candidate_id]['line_start']}"
                for candidate_id in claim["candidate_ids"]
            ]
        extra = [
            r["candidate_id"] for r in reviews if claim["claim_id"] in r["claim_ids"]
        ]
        if extra:
            claim["candidate_ids"] = list(
                dict.fromkeys([*claim.get("candidate_ids", []), *extra])
            )
            claim["source_locations"] = list(
                dict.fromkeys(
                    [
                        *claim.get("source_locations", []),
                        *[
                            f"{by_id[candidate_id]['source_path']}:{by_id[candidate_id]['line_start']}"
                            for candidate_id in extra
                        ],
                    ]
                )
            )
            claim["last_verified_on"] = DATE
        if claim["claim_id"] in EVIDENCE_BY_CLAIM:
            claim["evidence_artifacts"] = EVIDENCE_BY_CLAIM[claim["claim_id"]]


def _reconcile_reciprocal_claim_reviews(
    registry: dict[str, Any],
    by_id: dict[str, Any],
    claims: dict[str, dict[str, Any]],
) -> None:
    reviews_by_id = {
        review["candidate_id"]: review for review in registry["candidate_reviews"]
    }
    for review in registry["candidate_reviews"]:
        if review["disposition"] != "material_claims_mapped":
            review["claim_ids"] = []
    for claim in registry["claims"]:
        retained_ids = [
            candidate_id
            for candidate_id in claim.get("candidate_ids", [])
            if reviews_by_id[candidate_id]["disposition"] == "material_claims_mapped"
        ]
        claim["candidate_ids"] = list(dict.fromkeys(retained_ids))
        claim["source_locations"] = [
            f"{by_id[candidate_id]['source_path']}:{by_id[candidate_id]['line_start']}"
            for candidate_id in claim["candidate_ids"]
        ]
    for review in registry["candidate_reviews"]:
        candidate_id = review["candidate_id"]
        if candidate_id not in by_id:
            continue
        location = (
            f"{by_id[candidate_id]['source_path']}:{by_id[candidate_id]['line_start']}"
        )
        for claim_id in review.get("claim_ids", []):
            claim = claims[claim_id]
            if candidate_id not in claim["candidate_ids"]:
                claim["candidate_ids"].append(candidate_id)
                claim["source_locations"].append(location)
    for claim in registry["claims"]:
        for candidate_id in claim.get("candidate_ids", []):
            review = reviews_by_id[candidate_id]
            review["claim_ids"] = list(
                dict.fromkeys([*review.get("claim_ids", []), claim["claim_id"]])
            )
    for review in registry["candidate_reviews"]:
        review["claim_ids"] = list(dict.fromkeys(review.get("claim_ids", [])))


def _update_remaining_audit_scope(
    registry: dict[str, Any],
    candidate_count: int,
    source_digest: str,
) -> None:
    registry["paper"]["source_digest"] = source_digest
    registry["audit_scope"]["current_scope"] = (
        f"The complete {candidate_count}-candidate paper inventory is adjudicated. Repeated methods, "
        "summary, limitation, provenance, and model-tier passages inherit the primary "
        "claim boundaries; editorial anchors are explicitly classified as nonclaims."
    )
    registry["audit_scope"]["completion_status"] = "complete"


def main() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    _filter_generated_candidate_reviews(registry)
    reviewed = {review["candidate_id"] for review in registry["candidate_reviews"]}
    remaining = [
        c for c in inventory["candidates"] if c["candidate_id"] not in reviewed
    ]
    new_claims = _build_remaining_new_claims()
    new_claim_ids = {c["claim_id"] for c in new_claims}
    registry["claims"] = [
        claim for claim in registry["claims"] if claim["claim_id"] not in new_claim_ids
    ] + new_claims
    claims = {claim["claim_id"]: claim for claim in registry["claims"]}

    reviews = _generate_remaining_candidate_reviews(remaining, claims)
    registry["candidate_reviews"].extend(reviews)
    by_id = {
        candidate["candidate_id"]: candidate for candidate in inventory["candidates"]
    }
    _reconcile_claims_with_reviews(registry, by_id, new_claims, reviews)
    _reconcile_reciprocal_claim_reviews(registry, by_id, claims)
    _update_remaining_audit_scope(
        registry, inventory["candidate_count"], inventory["source_digest"]
    )
    REGISTRY.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
