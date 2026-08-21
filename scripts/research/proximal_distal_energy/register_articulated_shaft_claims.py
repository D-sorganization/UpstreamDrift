"""Register passive articulated-shaft claims and inference boundaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
ARTICLE = ROOT / "docs/research/proximal_distal_energy_transfer"
REGISTRY = ARTICLE / "data/claim_audit_registry.json"
INVENTORY = ARTICLE / "data/claim_candidate_inventory.json"
DATE = "2026-08-15"
CLAIM_IDS = {"PD-CLAIM-289", "PD-CLAIM-290", "PD-CLAIM-291", "PD-CLAIM-292"}
ARTIFACTS = [
    "docs/research/proximal_distal_energy_transfer/data/articulated_shaft_structural_basis.json",
    "docs/research/proximal_distal_energy_transfer/data/articulated_shaft_structural_basis.npz",
    "docs/research/proximal_distal_energy_transfer/data/articulated_shaft_time_step_diagnostic.json",
    "docs/research/proximal_distal_energy_transfer/data/articulated_shaft_atlas.json",
    "docs/research/proximal_distal_energy_transfer/data/articulated_shaft_atlas.npz",
    "docs/research/proximal_distal_energy_transfer/figures/fig_articulated_shaft_atlas.pdf",
    "scripts/research/proximal_distal_energy/generate_articulated_shaft_structural_basis.py",
    "scripts/research/proximal_distal_energy/articulated_shaft.py",
    "scripts/research/proximal_distal_energy/articulated_shaft_forward.py",
    "scripts/research/proximal_distal_energy/articulated_shaft_atlas.py",
    "scripts/research/proximal_distal_energy/articulated_shaft_time_step_diagnostic.py",
    "scripts/research/proximal_distal_energy/run_articulated_shaft_atlas.py",
    "scripts/research/proximal_distal_energy/run_articulated_shaft_time_step_diagnostic.py",
    "scripts/research/proximal_distal_energy/make_articulated_shaft_figure.py",
    "tests/research/test_articulated_shaft.py",
    "tests/research/test_articulated_shaft_forward.py",
    "tests/research/test_articulated_shaft_atlas.py",
]


def _find(candidates: list[dict[str, Any]], suffix: str, prefix: str) -> dict[str, Any]:
    matches = [
        candidate
        for candidate in candidates
        if str(candidate["source_path"]).endswith(suffix)
        and str(candidate["text"]).startswith(prefix)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one articulated-shaft candidate for {prefix!r}")
    return matches[0]


def _selected(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    chapter = "_ch06c_spatial_cross_formulation.qmd"
    return {
        "abstract": _find(
            candidates,
            "proximal_distal_energy_transfer.qmd",
            "Proximal-to-distal (P→D) sequencing",
        ),
        "law": _find(candidates, chapter, "The next intervention admits shaft"),
        "mass": _find(candidates, chapter, "Here $M_r$ is assembled"),
        "ledger": _find(candidates, chapter, "A first-order gravitational"),
        "basis": _find(candidates, chapter, "The bending basis is not refitted"),
        "design": _find(candidates, chapter, "The registered atlas crosses"),
        "figure": _find(candidates, chapter, "![Passive Articulated Shaft"),
        "numerics": _find(candidates, chapter, "All fine-grid numerical"),
        "matching": _find(candidates, chapter, "The shaft comparison is intentionally"),
        "boundary": _find(
            candidates, chapter, "At this shaft-only tier, the contact fibers remain"
        ),
        "summary": _find(
            candidates,
            "_ch06cb_spatial_cross_tail.qmd",
            "The bounded articulated experiments",
        ),
        "ladder": _find(candidates, "_ch07_model_ladder.qmd", "- the three-link"),
        "next": _find(
            candidates,
            "_ch07_model_ladder.qmd",
            "The next decisive model test replaces",
        ),
        "table": _find(
            candidates,
            "_ch08b_momentum_transfer_questions.qmd",
            "| How much transfer is drift-mediated?",
        ),
        "momentum": _find(
            candidates,
            "_ch08b_momentum_transfer_questions.qmd",
            "The following articulated shaft atlas",
        ),
    }


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
        "audit_status": "articulated_shaft_mechanism_and_boundary_checked",
        "source_locations": [
            f"{candidate['source_path']}:{candidate['line_start']}"
            for candidate in candidates
        ],
        "evidence_artifacts": ARTIFACTS,
        "model_domain": (
            "Twelve synthetic articulated states, four shaft activations, two "
            "initial velocity signs, two bounded fine steps, two native engines, "
            "and nested 4/10/25/50 millisecond summaries from 384 trajectories."
        ),
        "uncertainty_boundary": boundary,
        "competing_explanations": [
            "first-mode truncation and synthetic torsional stiffness",
            "state-dependent inertial coupling rather than universal elastic return",
            "frictionless distributed fibers and post-registered matching",
            "unchanged support pathway and absent impact",
        ],
        "negative_controls": [
            "exact rigid reduction and coordinate killswitches",
            "zero elastic initial state and matched initial total energy",
            "initial-velocity reversal and load/work matching",
            "excluded 1.0 and 0.50 millisecond linear-domain probes",
            "three-level limiting-cell refinement and native inertia-and-bias transport parity",
            "one-mode versus six-mode structural reference",
        ],
        "falsifier": falsifier,
        "adjudication": (
            "The passive elastic pathway is reported with mixed matched outcomes "
            "and without assigning equipment, human, timing, or coaching benefit."
        ),
        "reviewer": "Codex technical audit",
        "last_verified_on": DATE,
    }


def _claims(selected: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        _claim(
            "PD-CLAIM-289",
            [selected["law"], selected["mass"], selected["ledger"]],
            "A passive augmented articulated model couples two tip-normalized bending coordinates and one twist coordinate to native rigid dynamics while retaining a named energy and passivity ledger and exact rigid reduction.",
            "articulated_passive_shaft_formulation",
            "complete_for_declared_linear_synthetic_model",
            "The coordinates use synthetic lumped shaft/head inertia and first-order small-deformation kinematics.",
            "The augmented mass loses positive definiteness in-domain, rigid reduction differs, or elastic storage/damping power fails its identity.",
        ),
        _claim(
            "PD-CLAIM-290",
            [selected["basis"]],
            "The articulated bending coordinate reproduces the frozen 5.2399 Hz first FE mode and the declared tapered-section torsion coordinate is 70.1260 Hz; the six-mode authority shows materially greater one-mode error under short loading.",
            "articulated_shaft_structural_authority",
            "supported_as_synthetic_structural_reference",
            "No physical shaft is identified; the first-mode approximation is not adequate by inheritance for high-frequency loading.",
            "The frozen basis hash is stale, its reconstructed frequency differs from the FE authority, or the reported six-mode discrepancies do not reproduce.",
        ),
        _claim(
            "PD-CLAIM-291",
            [selected["design"], selected["numerics"]],
            "All 384 registered fine-grid trajectories pass linear-domain, power, work-energy refinement, activation, and MuJoCo--Pinocchio gates through 50 ms after two coarser torsion probes fail closed.",
            "articulated_shaft_numerical_result",
            "qualified_through_declared_fifty_millisecond_synthetic_horizon",
            "The horizon is right-censored and excludes calibrated friction, tissue, ground, higher-mode fast response, delivery, and impact.",
            "Any committed cell violates a registered gate, either coarse exclusion does not reproduce, or the limiting-cell residual does not decrease at finer steps.",
        ),
        _claim(
            "PD-CLAIM-292",
            [selected["matching"], selected["boundary"]],
            "Among 126 coupled--rigid cells matched within 5% for peak load and dissipated work, final-speed differences include 82 negative and 44 positive values, rejecting a universal passive-shaft speed benefit in the declared model.",
            "articulated_shaft_matched_outcome_boundary",
            "mixed_state_dependent_outcomes_supported",
            "Post-registered matching is descriptive and cannot identify a causal equipment effect or human strategy.",
            "The match rule or counts fail to reproduce, all matched differences have one sign, or a universal equipment/human conclusion is attributed to this atlas.",
        ),
    ]


def _map_review(
    reviews: dict[str, dict[str, Any]],
    candidate: dict[str, Any],
    claim_ids: tuple[str, ...],
    rationale: str,
) -> None:
    reviews[candidate["candidate_id"]] = {
        "candidate_id": candidate["candidate_id"],
        "disposition": "material_claims_mapped",
        "claim_ids": list(claim_ids),
        "rationale": rationale,
        "reviewer": "Codex technical audit",
        "last_verified_on": DATE,
    }


def _attach(
    claims: dict[str, dict[str, Any]],
    reviews: dict[str, dict[str, Any]],
    candidate: dict[str, Any],
    claim_ids: tuple[str, ...],
) -> None:
    location = f"{candidate['source_path']}:{candidate['line_start']}"
    for claim_id in claim_ids:
        claim = claims[claim_id]
        claim["candidate_ids"] = list(
            dict.fromkeys([*claim.get("candidate_ids", []), candidate["candidate_id"]])
        )
        claim["source_locations"] = list(
            dict.fromkeys([*claim.get("source_locations", []), location])
        )
    _map_review(
        reviews,
        candidate,
        claim_ids,
        "This repeated summary inherits the mapped claim boundaries.",
    )


def main() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    valid_ids = {candidate["candidate_id"] for candidate in inventory["candidates"]}
    registry["claims"] = [
        claim for claim in registry["claims"] if claim["claim_id"] not in CLAIM_IDS
    ]
    for claim in registry["claims"]:
        claim["candidate_ids"] = [
            candidate_id
            for candidate_id in claim.get("candidate_ids", [])
            if candidate_id in valid_ids
        ]
    selected = _selected(inventory["candidates"])
    new_claims = _claims(selected)
    registry["claims"].extend(new_claims)
    claims = {claim["claim_id"]: claim for claim in registry["claims"]}
    reviews = {
        review["candidate_id"]: review
        for review in registry["candidate_reviews"]
        if review["candidate_id"] in valid_ids
    }
    primary = {
        "law": ("PD-CLAIM-289",),
        "mass": ("PD-CLAIM-289",),
        "ledger": ("PD-CLAIM-289",),
        "basis": ("PD-CLAIM-290",),
        "design": ("PD-CLAIM-291",),
        "numerics": ("PD-CLAIM-291",),
        "matching": ("PD-CLAIM-292",),
        "boundary": ("PD-CLAIM-292",),
    }
    for name, claim_ids in primary.items():
        _map_review(
            reviews,
            selected[name],
            claim_ids,
            "This passage states or bounds the passive-shaft gate.",
        )
    figure = selected["figure"]
    reviews[figure["candidate_id"]] = {
        "candidate_id": figure["candidate_id"],
        "disposition": "editorial_or_navigation",
        "claim_ids": [],
        "rationale": "The figure include points to governed evidence without a standalone claim.",
        "reviewer": "Codex technical audit",
        "last_verified_on": DATE,
    }
    repeated = {
        "abstract": (
            "PD-CLAIM-007",
            "PD-CLAIM-008",
            "PD-CLAIM-214",
            "PD-CLAIM-260",
            "PD-CLAIM-263",
            "PD-CLAIM-272",
            "PD-CLAIM-291",
            "PD-CLAIM-292",
        ),
        "summary": ("PD-CLAIM-128", "PD-CLAIM-262", "PD-CLAIM-291", "PD-CLAIM-292"),
        "ladder": ("PD-CLAIM-128", "PD-CLAIM-288", "PD-CLAIM-292"),
        "next": ("PD-CLAIM-128", "PD-CLAIM-292"),
        "table": (
            "PD-CLAIM-243",
            "PD-CLAIM-247",
            "PD-CLAIM-253",
            "PD-CLAIM-273",
            "PD-CLAIM-292",
        ),
        "momentum": ("PD-CLAIM-290", "PD-CLAIM-291", "PD-CLAIM-292"),
    }
    for name, claim_ids in repeated.items():
        _attach(claims, reviews, selected[name], claim_ids)
    claims["PD-CLAIM-128"]["statement"] = (
        "The discrepancy matrix records explicit branch capabilities rather than "
        "cumulative triangular inheritance; twelve bounded findings are supported "
        "somewhere in executed tiers, while calibrated anatomical, shaft, ground, "
        "and independently measured human transport remain untested."
    )
    claims["PD-CLAIM-288"]["statement"] = (
        "Distributed fibers establish discretization sensitivity, and a separate "
        "passive first-mode shaft atlas establishes state-dependent elastic response; "
        "neither establishes physical grip, equipment, timing, human, or strategy benefit."
    )
    registry["candidate_reviews"] = list(reviews.values())
    release = {
        item["release_claim_key"]: item for item in registry["release_claim_inventory"]
    }
    release["articulated_shaft_bending_torsion"] = {
        "release_claim_key": "articulated_shaft_bending_torsion",
        "published_status": "fifty_millisecond_passive_shaft_gate_qualified_with_mixed_matched_outcomes",
        "audit_state": "reviewed_as_synthetic_first_mode_result",
    }
    registry["release_claim_inventory"] = list(release.values())
    registry["audit_scope"]["current_scope"] = (
        "The complete paper inventory is adjudicated. Articulated distributed-grip "
        "and passive first-mode shaft tiers pass registered 50 ms gates, with mixed "
        "load/work-matched shaft outcomes and explicit calibration boundaries."
    )
    registry["paper"]["source_digest"] = inventory["source_digest"]
    registry["claims"] = list(claims.values())
    REGISTRY.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
