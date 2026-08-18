"""Register finite-ground/free-moment claims and adverse inference boundaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
ARTICLE = ROOT / "docs/research/proximal_distal_energy_transfer"
REGISTRY = ARTICLE / "data/claim_audit_registry.json"
INVENTORY = ARTICLE / "data/claim_candidate_inventory.json"
DATE = "2026-08-16"
CLAIM_IDS = {"PD-CLAIM-293", "PD-CLAIM-294", "PD-CLAIM-295", "PD-CLAIM-296"}
ARTIFACTS = [
    "docs/research/proximal_distal_energy_transfer/data/articulated_ground_diagnostic.json",
    "docs/research/proximal_distal_energy_transfer/data/articulated_ground_atlas.json",
    "docs/research/proximal_distal_energy_transfer/data/articulated_ground_atlas.npz",
    "docs/research/proximal_distal_energy_transfer/data/articulated_ground_posthoc_sensitivity.json",
    "docs/research/proximal_distal_energy_transfer/figures/fig_articulated_ground_atlas.pdf",
    "scripts/research/proximal_distal_energy/articulated_ground.py",
    "scripts/research/proximal_distal_energy/articulated_ground_forward.py",
    "scripts/research/proximal_distal_energy/articulated_ground_diagnostic.py",
    "scripts/research/proximal_distal_energy/articulated_ground_atlas.py",
    "scripts/research/proximal_distal_energy/articulated_ground_posthoc_sensitivity.py",
    "scripts/research/proximal_distal_energy/run_articulated_ground_diagnostic.py",
    "scripts/research/proximal_distal_energy/run_articulated_ground_atlas.py",
    "scripts/research/proximal_distal_energy/run_articulated_ground_posthoc_sensitivity.py",
    "scripts/research/proximal_distal_energy/make_articulated_ground_figure.py",
    "tests/research/test_articulated_ground.py",
    "tests/research/test_articulated_ground_forward.py",
    "tests/research/test_articulated_ground_diagnostic.py",
    "tests/research/test_articulated_ground_diagnostic_evidence.py",
    "tests/research/test_articulated_ground_atlas.py",
    "tests/research/test_articulated_ground_atlas_evidence.py",
    "tests/research/test_articulated_ground_posthoc_sensitivity.py",
    "tests/research/test_articulated_ground_reference_invariance.py",
]


def _find(candidates: list[dict[str, Any]], suffix: str, prefix: str) -> dict[str, Any]:
    matches = [
        candidate
        for candidate in candidates
        if str(candidate["source_path"]).endswith(suffix)
        and str(candidate["text"]).startswith(prefix)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one finite-ground candidate for {prefix!r}")
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
        "audit_status": "finite_ground_mechanism_and_adverse_match_checked",
        "source_locations": [
            f"{candidate['source_path']}:{candidate['line_start']}"
            for candidate in candidates
        ],
        "evidence_artifacts": ARTIFACTS,
        "model_domain": (
            "Twelve synthetic articulated states, four base pathways, two signed "
            "club-velocity perturbations, two fine steps, two native engines, and "
            "nested 4/10/25/50 millisecond summaries from 576 trajectories."
        ),
        "uncertainty_boundary": boundary,
        "competing_explanations": [
            "linear bilateral support rather than unilateral foot contact",
            "synthetic stiffness and damping rather than force-plate calibration",
            "initial preload convention and short transient horizon",
            "work-definition choice in coupled-versus-fixed matching",
        ],
        "negative_controls": [
            "exact fixed-base and zero-energy reductions",
            "translation-only and intrinsic-free-moment-only pathways",
            "rigid-shaft and horizontal-restraint-removed controls",
            "center-of-pressure reference reversal",
            "three-level diagnostic refinement and two-engine parity",
            "primary-match failure retained without tolerance widening",
        ],
        "falsifier": falsifier,
        "adjudication": (
            "The finite-base mechanism is reported with an empty primary matched "
            "set; unmatched speed differences are not interpreted as an effect."
        ),
        "reviewer": "Codex technical audit",
        "last_verified_on": DATE,
    }


def _review(
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
    _review(reviews, candidate, claim_ids, "Repeated summary inherits claim bounds.")


def main() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    candidates = inventory["candidates"]
    valid_ids = {candidate["candidate_id"] for candidate in candidates}
    registry["claims"] = [
        claim for claim in registry["claims"] if claim["claim_id"] not in CLAIM_IDS
    ]
    for claim in registry["claims"]:
        claim["candidate_ids"] = [
            candidate_id
            for candidate_id in claim.get("candidate_ids", [])
            if candidate_id in valid_ids
        ]

    chapter = "_ch06ca_articulated_ground.qmd"
    selected = {
        "intro": _find(candidates, chapter, "The finite-base extension supplies"),
        "mass": _find(candidates, chapter, "Only the non-club body tree"),
        "law": _find(candidates, chapter, "The synthetic passive support law"),
        "moment": _find(candidates, chapter, "Its first two generalized components"),
        "pathways": _find(candidates, chapter, "Four exact pathway selections"),
        "initialization": _find(
            candidates, chapter, "Initialization is a model variable"
        ),
        "design": _find(candidates, chapter, "The preregistered atlas uses"),
        "parameters": _find(
            candidates, chapter, "The constitutive values are transparent"
        ),
        "diagnostic": _find(candidates, chapter, "The 42-trajectory initialization"),
        "figure": _find(candidates, chapter, "![Finite Ground Reaction"),
        "numerics": _find(candidates, chapter, "All 576 atlas trajectories"),
        "primary": _find(
            candidates, chapter, "The primary load--work screen is adverse"
        ),
        "posthoc": _find(
            candidates, chapter, "A separately labeled post-hoc sensitivity"
        ),
        "boundary": _find(candidates, chapter, "The support is linear and bilateral"),
    }
    new_claims = [
        _claim(
            "PD-CLAIM-293",
            [selected[key] for key in ("intro", "mass", "law", "moment", "pathways")],
            "A passive finite-base extension couples planar translation and pitch to the non-club articulated tree, exposes ground reaction and intrinsic free moment separately from reference transport, and reduces exactly to the fixed shaft tier.",
            "articulated_finite_ground_formulation",
            "complete_for_declared_linear_bilateral_synthetic_model",
            "The support is planar, linear, bilateral, and synthetic; it is not a calibrated foot--ground model.",
            "Fixed-base reduction, power closure, moment transport, or positive-definite augmented inertia fails in the registered domain.",
        ),
        _claim(
            "PD-CLAIM-294",
            [selected["initialization"], selected["diagnostic"]],
            "All 42 diagnostic traces refine monotonically and agree across native engines, while natural-zero, gravity-only, and conditional-base initializations produce materially different short-transient loads and speeds.",
            "articulated_ground_initialization_sensitivity",
            "qualified_as_synthetic_initialization_diagnostic",
            "Conditional equilibrium balances base coordinates only and none of the initializations estimates a human preload state.",
            "Any diagnostic branch fails refinement/parity or the reported initialization separation does not reproduce.",
        ),
        _claim(
            "PD-CLAIM-295",
            [selected[key] for key in ("design", "parameters", "numerics")],
            "All 576 registered finite-ground trajectories pass declared shaft/base domains, refinement, work--energy, and MuJoCo--Pinocchio parity gates through 50 ms.",
            "articulated_ground_numerical_result",
            "qualified_through_declared_fifty_millisecond_synthetic_horizon",
            "Loads and material values are uncalibrated, the horizon excludes impact, and numerical qualification is not efficacy evidence.",
            "A committed trajectory violates a registered domain, refinement, energy, or native-engine gate.",
        ),
        _claim(
            "PD-CLAIM-296",
            [selected[key] for key in ("primary", "posthoc", "boundary")],
            "None of 384 coupled--fixed cells meets the preregistered 5% peak-load and total-work match; a post-hoc non-ground-work screen admits 60 cells with both speed-difference signs, leaving the isolated delivery effect unidentified.",
            "articulated_ground_adverse_matching_result",
            "primary_match_empty_and_posthoc_sign_mixed",
            "The post-hoc screen cannot replace the primary estimand or support a ground-use, timing, coaching, equipment, or human benefit claim.",
            "A primary match exists under the registered rule, the post-hoc counts fail to reproduce, or unmatched differences are presented as identified effects.",
        ),
    ]
    registry["claims"].extend(new_claims)
    claims = {claim["claim_id"]: claim for claim in registry["claims"]}
    reviews = {
        review["candidate_id"]: review
        for review in registry["candidate_reviews"]
        if review["candidate_id"] in valid_ids
    }
    mapping = {
        "intro": ("PD-CLAIM-293",),
        "mass": ("PD-CLAIM-293",),
        "law": ("PD-CLAIM-293",),
        "moment": ("PD-CLAIM-293",),
        "pathways": ("PD-CLAIM-293",),
        "initialization": ("PD-CLAIM-294",),
        "diagnostic": ("PD-CLAIM-294",),
        "design": ("PD-CLAIM-295",),
        "parameters": ("PD-CLAIM-295",),
        "numerics": ("PD-CLAIM-295",),
        "primary": ("PD-CLAIM-296",),
        "posthoc": ("PD-CLAIM-296",),
        "boundary": ("PD-CLAIM-296",),
    }
    for name, claim_ids in mapping.items():
        _review(reviews, selected[name], claim_ids, "Finite-ground claim or boundary.")
    reviews[selected["figure"]["candidate_id"]] = {
        "candidate_id": selected["figure"]["candidate_id"],
        "disposition": "editorial_or_navigation",
        "claim_ids": [],
        "rationale": "The figure include points to governed evidence.",
        "reviewer": "Codex technical audit",
        "last_verified_on": DATE,
    }

    repeated = [
        (
            _find(
                candidates,
                "proximal_distal_energy_transfer.qmd",
                "Proximal-to-distal (P→D) sequencing",
            ),
            (
                "PD-CLAIM-007",
                "PD-CLAIM-008",
                "PD-CLAIM-214",
                "PD-CLAIM-260",
                "PD-CLAIM-263",
                "PD-CLAIM-272",
                "PD-CLAIM-291",
                "PD-CLAIM-292",
                "PD-CLAIM-293",
                "PD-CLAIM-294",
                "PD-CLAIM-295",
                "PD-CLAIM-296",
            ),
        ),
        (
            _find(
                candidates,
                "_ch06c_spatial_cross_formulation.qmd",
                "At this shaft-only tier",
            ),
            ("PD-CLAIM-292", "PD-CLAIM-293"),
        ),
        (
            _find(candidates, "_ch07_model_ladder.qmd", "The discrepancy matrix"),
            ("PD-CLAIM-128", "PD-CLAIM-296"),
        ),
        (
            _find(
                candidates, "_ch07_model_ladder.qmd", "1. interaction force may remain"
            ),
            ("PD-CLAIM-128", "PD-CLAIM-293", "PD-CLAIM-296"),
        ),
        (
            _find(
                candidates, "_ch07_model_ladder.qmd", "Other claims do not yet survive"
            ),
            ("PD-CLAIM-128", "PD-CLAIM-293", "PD-CLAIM-296"),
        ),
        (
            _find(candidates, "_ch07_model_ladder.qmd", "**Articulated Finite Ground"),
            ("PD-CLAIM-293", "PD-CLAIM-295", "PD-CLAIM-296"),
        ),
        (
            _find(
                candidates,
                "_ch07_model_ladder.qmd",
                "The ladder remains deliberately incomplete",
            ),
            ("PD-CLAIM-128", "PD-CLAIM-293", "PD-CLAIM-296"),
        ),
        (
            _find(
                candidates, "_ch07_model_ladder.qmd", "- the three-link trace remains"
            ),
            ("PD-CLAIM-128", "PD-CLAIM-293", "PD-CLAIM-296"),
        ),
        (
            _find(
                candidates,
                "_ch07_model_ladder.qmd",
                "The next decisive model test replaces",
            ),
            ("PD-CLAIM-128", "PD-CLAIM-296"),
        ),
        (
            _find(
                candidates,
                "_ch08b_momentum_transfer_questions.qmd",
                "| How much transfer is drift-mediated?",
            ),
            (
                "PD-CLAIM-243",
                "PD-CLAIM-247",
                "PD-CLAIM-253",
                "PD-CLAIM-273",
                "PD-CLAIM-296",
            ),
        ),
        (
            _find(
                candidates,
                "_ch09_conclusions.qmd",
                "The articulated finite-base extension",
            ),
            ("PD-CLAIM-293", "PD-CLAIM-294", "PD-CLAIM-295", "PD-CLAIM-296"),
        ),
        (
            _find(
                candidates,
                "_ch09_conclusions.qmd",
                "1. **Articulated Spatial Forward-Contact",
            ),
            ("PD-CLAIM-128", "PD-CLAIM-296"),
        ),
        (
            _find(
                candidates,
                "_ch06f_open_release.qmd",
                "The scientific product is distributed",
            ),
            ("PD-CLAIM-128", "PD-CLAIM-293", "PD-CLAIM-295", "PD-CLAIM-296"),
        ),
        (
            _find(
                candidates,
                "_ch06f_open_release.qmd",
                "The release records planar interaction",
            ),
            (
                "PD-CLAIM-128",
                "PD-CLAIM-214",
                "PD-CLAIM-260",
                "PD-CLAIM-263",
                "PD-CLAIM-272",
                "PD-CLAIM-291",
                "PD-CLAIM-292",
                "PD-CLAIM-293",
                "PD-CLAIM-295",
                "PD-CLAIM-296",
            ),
        ),
        (
            _find(
                candidates,
                "_ch06f_open_release.qmd",
                "Five release-level completion gates",
            ),
            ("PD-CLAIM-128", "PD-CLAIM-296"),
        ),
        (
            _find(
                candidates,
                "_ch06cb_spatial_cross_tail.qmd",
                "The fixed shoulder centers",
            ),
            ("PD-CLAIM-262", "PD-CLAIM-265"),
        ),
        (
            _find(
                candidates,
                "_ch06cb_spatial_cross_tail.qmd",
                "The screen uses the same six",
            ),
            ("PD-CLAIM-262", "PD-CLAIM-265"),
        ),
        (
            _find(
                candidates,
                "_ch06cb_spatial_cross_tail.qmd",
                "No fixed-shoulder arm-only state",
            ),
            ("PD-CLAIM-265", "PD-CLAIM-266"),
        ),
        (
            _find(
                candidates,
                "_ch06cb_spatial_cross_tail.qmd",
                "Both paired contact Jacobians",
            ),
            ("PD-CLAIM-267",),
        ),
        (
            _find(
                candidates,
                "_ch06cb_spatial_cross_tail.qmd",
                "This result advances the model ladder",
            ),
            ("PD-CLAIM-262", "PD-CLAIM-267"),
        ),
        (
            _find(
                candidates,
                "_ch06cb_spatial_cross_tail.qmd",
                "The bounded articulated experiments",
            ),
            (
                "PD-CLAIM-128",
                "PD-CLAIM-288",
                "PD-CLAIM-292",
                "PD-CLAIM-293",
                "PD-CLAIM-295",
                "PD-CLAIM-296",
            ),
        ),
        (
            _find(
                candidates,
                "_ch06cb_spatial_cross_tail.qmd",
                "The executed experiment advances",
            ),
            ("PD-CLAIM-182",),
        ),
        (
            _find(
                candidates,
                "_ch06cb_spatial_cross_tail.qmd",
                "First, the common spatial wrench",
            ),
            ("PD-CLAIM-182",),
        ),
        (
            _find(
                candidates, "_ch06cb_spatial_cross_tail.qmd", "Second, the same model"
            ),
            ("PD-CLAIM-182",),
        ),
        (
            _find(
                candidates,
                "_ch06cb_spatial_cross_tail.qmd",
                "Third, the geometry intervention",
            ),
            ("PD-CLAIM-182",),
        ),
        (
            _find(
                candidates,
                "_ch06cb_spatial_cross_tail.qmd",
                "Four stronger statements remain open",
            ),
            ("PD-CLAIM-183",),
        ),
        (
            _find(
                candidates,
                "_ch06cb_spatial_cross_tail.qmd",
                "- **Passive Contact Origin Is Inconclusive",
            ),
            ("PD-CLAIM-183",),
        ),
        (
            _find(
                candidates, "_ch06cb_spatial_cross_tail.qmd", "The next chapter tests"
            ),
            ("PD-CLAIM-183",),
        ),
        (
            _find(candidates, "_appendices.qmd", "- A multi-phase matched-state"),
            (
                "PD-CLAIM-128",
                "PD-CLAIM-183",
                "PD-CLAIM-293",
                "PD-CLAIM-295",
                "PD-CLAIM-296",
            ),
        ),
    ]
    for candidate, claim_ids in repeated:
        _attach(claims, reviews, candidate, claim_ids)

    availability = _find(
        candidates, "_ch09_conclusions.qmd", "Analysis code, tests, figures"
    )
    reviews[availability["candidate_id"]] = {
        "candidate_id": availability["candidate_id"],
        "disposition": "editorial_or_navigation",
        "claim_ids": [],
        "rationale": "This passage links governed artifacts without adding a result.",
        "reviewer": "Codex technical audit",
        "last_verified_on": DATE,
    }
    scapular_figure = _find(
        candidates,
        "_ch06cb_spatial_cross_tail.qmd",
        "![Scapular Mobility and Bilateral Contact Geometry]",
    )
    reviews[scapular_figure["candidate_id"]] = {
        "candidate_id": scapular_figure["candidate_id"],
        "disposition": "editorial_or_navigation",
        "claim_ids": [],
        "rationale": "The moved figure include does not add a standalone claim.",
        "reviewer": "Codex technical audit",
        "last_verified_on": DATE,
    }

    reciprocal: dict[str, list[str]] = {}
    for claim in claims.values():
        for candidate_id in claim.get("candidate_ids", []):
            reciprocal.setdefault(candidate_id, []).append(claim["claim_id"])
    for candidate_id, review in reviews.items():
        review["claim_ids"] = list(
            dict.fromkeys([*review["claim_ids"], *reciprocal.get(candidate_id, [])])
        )

    candidate_by_id = {candidate["candidate_id"]: candidate for candidate in candidates}
    for claim in claims.values():
        if claim.get("candidate_ids"):
            claim["source_locations"] = [
                f"{candidate_by_id[candidate_id]['source_path']}:"
                f"{candidate_by_id[candidate_id]['line_start']}"
                for candidate_id in claim["candidate_ids"]
            ]

    registry["candidate_reviews"] = list(reviews.values())
    release = {
        item["release_claim_key"]: item for item in registry["release_claim_inventory"]
    }
    release["articulated_ground_free_moment"] = {
        "release_claim_key": "articulated_ground_free_moment",
        "published_status": "fifty_millisecond_finite_ground_gate_qualified_primary_match_empty",
        "audit_state": "reviewed_as_synthetic_finite_base_result_with_adverse_primary_match",
    }
    registry["release_claim_inventory"] = list(release.values())
    registry["audit_scope"]["current_scope"] = (
        "The complete paper inventory is adjudicated. The finite-base ground and "
        "intrinsic-free-moment tier passes registered numerical gates, while its "
        "primary coupled--fixed matching set is empty and no effect is identified."
    )
    registry["paper"]["source_digest"] = inventory["source_digest"]
    registry["claims"] = list(claims.values())
    REGISTRY.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
