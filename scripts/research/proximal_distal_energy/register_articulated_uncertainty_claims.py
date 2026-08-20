"""Register articulated uncertainty and structural-authority paper claims."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
ARTICLE = ROOT / "docs/research/proximal_distal_energy_transfer"
REGISTRY = ARTICLE / "data/claim_audit_registry.json"
INVENTORY = ARTICLE / "data/claim_candidate_inventory.json"
DATE = "2026-08-20"
CLAIM_IDS = {f"PD-CLAIM-{value}" for value in range(297, 306)}
REGISTRATION_ARTIFACT = (
    "scripts/research/proximal_distal_energy/register_articulated_uncertainty_claims.py"
)
UNCERTAINTY_ARTIFACTS = [
    "docs/research/proximal_distal_energy_transfer/data/articulated_uncertainty_study.json",
    "docs/research/proximal_distal_energy_transfer/data/articulated_uncertainty_study.npz",
    "docs/research/proximal_distal_energy_transfer/figures/fig_articulated_uncertainty_screen.pdf",
    "scripts/research/proximal_distal_energy/articulated_uncertainty_study.py",
    REGISTRATION_ARTIFACT,
    "tests/research/test_articulated_uncertainty_study.py",
]
STRUCTURAL_ARTIFACTS = [
    "docs/research/proximal_distal_energy_transfer/data/articulated_structural_authority_campaign.json",
    "docs/research/proximal_distal_energy_transfer/data/articulated_structural_propagation_plan.json",
    "docs/research/proximal_distal_energy_transfer/figures/fig_articulated_structural_authority.pdf",
    "scripts/research/proximal_distal_energy/articulated_scaled_authority.py",
    "scripts/research/proximal_distal_energy/articulated_atlas_authority.py",
    "scripts/research/proximal_distal_energy/articulated_structural_authority_campaign.py",
    "scripts/research/proximal_distal_energy/articulated_structural_propagation_plan.py",
    REGISTRATION_ARTIFACT,
    "tests/research/test_articulated_scaled_authority.py",
    "tests/research/test_articulated_atlas_authority.py",
    "tests/research/test_articulated_structural_authority_campaign.py",
    "tests/research/test_articulated_structural_propagation_plan.py",
]
HEADLINE_DESIGN_ARTIFACTS = [
    "scripts/research/proximal_distal_energy/articulated_headline_uncertainty.py",
    REGISTRATION_ARTIFACT,
    "tests/research/test_articulated_headline_uncertainty.py",
]


def _find(candidates: list[dict[str, Any]], suffix: str, prefix: str) -> dict[str, Any]:
    matches = [
        candidate
        for candidate in candidates
        if str(candidate["source_path"]).endswith(suffix)
        and str(candidate["text"]).startswith(prefix)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one claim candidate for {prefix!r}")
    return matches[0]


def _claim(
    claim_id: str,
    candidates: list[dict[str, Any]],
    *,
    statement: str,
    classification: str,
    status: str,
    audit_status: str,
    artifacts: list[str],
    domain: str,
    boundary: str,
    explanations: list[str],
    controls: list[str],
    falsifier: str,
    adjudication: str,
) -> dict[str, Any]:
    return {
        "claim_id": claim_id,
        "candidate_ids": [candidate["candidate_id"] for candidate in candidates],
        "statement": statement,
        "classification": classification,
        "published_status": status,
        "audit_status": audit_status,
        "source_locations": [
            f"{candidate['source_path']}:{candidate['line_start']}"
            for candidate in candidates
        ],
        "evidence_artifacts": artifacts,
        "model_domain": domain,
        "uncertainty_boundary": boundary,
        "competing_explanations": explanations,
        "negative_controls": controls,
        "falsifier": falsifier,
        "adjudication": adjudication,
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
        "disposition": "material_claims_mapped"
        if claim_ids
        else "editorial_or_navigation",
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
        claim["last_verified_on"] = DATE
    _review(
        reviews,
        candidate,
        claim_ids,
        "Repeated synthesis inherits the mapped primary claim boundaries.",
    )


def _write_registry(record: dict[str, Any]) -> None:
    temporary = REGISTRY.with_suffix(REGISTRY.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    temporary.replace(REGISTRY)


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
    reviews = {
        review["candidate_id"]: review
        for review in registry["candidate_reviews"]
        if review["candidate_id"] in valid_ids
    }

    screen_chapter = "_ch06caa_articulated_uncertainty_screen.qmd"
    screen = {
        "bounds": _find(candidates, screen_chapter, "The articulated qualification"),
        "method": _find(candidates, screen_chapter, "Each of the 40 samples"),
        "result": _find(candidates, screen_chapter, "All 40 registered rows"),
        "prcc": _find(candidates, screen_chapter, "Within that conditional screen"),
        "figure": _find(candidates, screen_chapter, "![Articulated Closed-State"),
        "interpretation": _find(candidates, screen_chapter, "Figure @fig-articulated"),
        "availability": _find(candidates, screen_chapter, "The governed record"),
        "boundary": _find(candidates, screen_chapter, "This screen answers"),
    }
    authority_chapter = "_ch06caaa_structural_authority.qmd"
    authority = {
        "method": _find(candidates, authority_chapter, "The headline uncertainty"),
        "header": _find(candidates, authority_chapter, "| Registered Corner"),
        "table": _find(candidates, authority_chapter, "| Nominal |"),
        "figure": _find(candidates, authority_chapter, "![Structural Authority"),
        "caption": _find(candidates, authority_chapter, "Figure @fig-articulated"),
        "result": _find(candidates, authority_chapter, "The low-height corner"),
        "boundary": _find(candidates, authority_chapter, "The campaign therefore"),
        "availability": _find(
            candidates, authority_chapter, "The checkpointed campaign"
        ),
    }
    headline_chapter = "_ch06cab_articulated_headline_uncertainty.qmd"
    headline = {
        "method": _find(candidates, headline_chapter, "The shaft and finite-ground"),
        "bounds": _find(candidates, headline_chapter, "The design contains"),
        "estimand": _find(candidates, headline_chapter, "The estimand is"),
        "figure": _find(candidates, headline_chapter, "![Articulated Headline"),
        "provenance": _find(candidates, headline_chapter, "Figure @fig-articulated"),
        "boundary": _find(candidates, headline_chapter, "This one-at-a-time design"),
    }

    screen_explanations = [
        "finite deterministic design rather than a sampled population",
        "excitation and contact-law dependence",
        "monotone association rather than causal parameter effect",
        "partial opening rather than robust maintained contact",
    ]
    screen_controls = [
        "deterministic registered Latin-hypercube rows",
        "fresh model and bilateral-closure solve for each row",
        "retained feasibility and contact statuses",
        "finite-response and five-percent energy-closure gates",
    ]
    structural_explanations = [
        "engineering scaling rather than calibrated participant anatomy",
        "inverse-kinematics conditioning near a joint-limit boundary",
        "coarse collision surrogates rather than mesh contact",
        "authority regeneration without dynamic propagation",
    ]
    structural_controls = [
        "all thirteen phase states regenerated at every corner",
        "configuration source and array digest validation",
        "nominal-model substitution rejection",
        "retained infeasible state and denominator",
    ]
    headline_explanations = [
        "one-at-a-time engineering corners rather than a joint distribution",
        "matching-set movement rather than an outcome effect",
        "short synthetic trajectory and constitutive parameter dependence",
        "unpropagated structural authority variation",
    ]
    headline_controls = [
        "complete production atlases rather than reduced surrogates",
        "both native engines and velocity reversal",
        "time-step refinement and pathway killswitches",
        "unchanged load-work matching and retained failures",
    ]
    new_claims = [
        _claim(
            "PD-CLAIM-297",
            [screen["bounds"], screen["method"]],
            statement=(
                "A deterministic 40-row articulated screen perturbs nine declared "
                "engineering inputs, rebuilds and recloses each model, retains every "
                "failure state, and computes PRCCs only on finite response rows."
            ),
            classification="articulated_engineering_uncertainty_design",
            status="registered_and_executed_for_declared_synthetic_screen",
            audit_status="design_rows_closure_status_and_response_contract_checked",
            artifacts=UNCERTAINTY_ARTIFACTS,
            domain=(
                "Forty deterministic rows over height, mass, joint limits, grip, "
                "friction, club properties, and initial velocity in a local "
                "distributed-contact trajectory."
            ),
            boundary=(
                "The ranges are engineering bounds, not participant or equipment "
                "distributions, and the screen does not estimate interactions."
            ),
            explanations=screen_explanations,
            controls=screen_controls,
            falsifier=(
                "A registered row is omitted, does not rebuild and reclose its model, "
                "or a nonfinite response enters the PRCC calculation."
            ),
            adjudication=(
                "The design is treated as a deterministic model screen, not as "
                "population uncertainty or causal identification."
            ),
        ),
        _claim(
            "PD-CLAIM-298",
            [screen["result"]],
            statement=(
                "All 40 registered rows have finite response vectors and remain "
                "within the 5% work-energy gate, but every local trajectory partially "
                "opens, rejecting robust full-contact interpretation for this screen."
            ),
            classification="articulated_uncertainty_adverse_contact_result",
            status="finite_energy_closed_but_contact_domain_adverse",
            audit_status="all_rows_statuses_and_energy_gate_reconciled",
            artifacts=UNCERTAINTY_ARTIFACTS,
            domain="The registered 40-row local articulated uncertainty screen.",
            boundary=(
                "Finite integration after partial opening does not establish maintained "
                "physical grip contact, human strategy, or delivery benefit."
            ),
            explanations=screen_explanations,
            controls=screen_controls,
            falsifier=(
                "Any row is nonfinite, exceeds the declared energy gate, or lacks the "
                "reported partial-opening status."
            ),
            adjudication=(
                "The adverse contact-domain result is retained and conditions every "
                "downstream association."
            ),
        ),
        _claim(
            "PD-CLAIM-299",
            [screen["prcc"], screen["interpretation"]],
            statement=(
                "The largest absolute conditional PRCCs are +0.936 for initial "
                "velocity versus peak force, -0.500 for height versus force couple, "
                "-0.940 for friction versus sliding speed, -0.760 for height versus "
                "transitions, and +0.931 for mass versus the bounded numerical residual."
            ),
            classification="conditional_articulated_prcc_screen",
            status="descriptive_associations_for_registered_finite_rows",
            audit_status="prcc_values_status_conditioning_and_boundaries_checked",
            artifacts=UNCERTAINTY_ARTIFACTS,
            domain="Finite rows of the declared deterministic local screen.",
            boundary=(
                "PRCC magnitudes are neither causal effects nor cross-parameter "
                "importance rankings and have no population interval."
            ),
            explanations=screen_explanations,
            controls=screen_controls,
            falsifier="The committed rows do not reproduce the stated PRCC values.",
            adjudication=(
                "Associations are reported with status conditioning and without causal, "
                "population, performance, or coaching promotion."
            ),
        ),
        _claim(
            "PD-CLAIM-300",
            [screen["boundary"]],
            statement=(
                "The local articulated screen does not propagate structural variation "
                "through the 126/384 shaft or 0/384 ground headline estimands and "
                "therefore cannot establish their structural robustness."
            ),
            classification="articulated_uncertainty_inference_boundary",
            status="explicitly_open_headline_propagation_gate",
            audit_status="local_screen_and_headline_estimands_separated",
            artifacts=[*UNCERTAINTY_ARTIFACTS, *HEADLINE_DESIGN_ARTIFACTS],
            domain="Relationship between the local screen and full headline atlases.",
            boundary="No dynamic structural-corner robustness is claimed.",
            explanations=screen_explanations,
            controls=screen_controls,
            falsifier=(
                "The paper promotes local PRCC or contact-status results as completed "
                "headline uncertainty propagation."
            ),
            adjudication="The missing propagation is recorded as an acceptance boundary.",
        ),
        _claim(
            "PD-CLAIM-301",
            [authority["method"]],
            statement=(
                "A seven-corner structural campaign regenerates all 13 phase states for "
                "four headline cases and digest-binds configuration, sources, states, "
                "failures, arrays, and corner-consistent model identity."
            ),
            classification="structural_corner_authority_contract",
            status="qualified_as_digest_bound_execution_authority",
            audit_status="authority_generation_identity_and_failure_contract_checked",
            artifacts=STRUCTURAL_ARTIFACTS,
            domain=(
                "Nominal and low/high height, body-mass, and joint-limit engineering "
                "corners for cases 0, 8, 9, and 17 over 13 phases."
            ),
            boundary="Authority qualification alone is not a dynamic transfer result.",
            explanations=structural_explanations,
            controls=structural_controls,
            falsifier=(
                "A digest drifts, a selected state disappears, or an atlas substitutes "
                "the nominal model for a scaled authority."
            ),
            adjudication=(
                "The authority is accepted only as model/state provenance for later "
                "dynamic execution."
            ),
        ),
        _claim(
            "PD-CLAIM-302",
            [authority["table"], authority["caption"], authority["result"]],
            statement=(
                "Six structural corners retain 52/52 feasible selected states; the "
                "low-height corner retains 51/52 and one case-0 phase-12 "
                "IK nonconvergence at an effectively zero joint-limit margin."
            ),
            classification="structural_corner_feasibility_result",
            status="qualified_with_one_retained_boundary_failure",
            audit_status="corner_counts_margins_clearances_and_failure_reconciled",
            artifacts=STRUCTURAL_ARTIFACTS,
            domain="The seven registered structural authority corners.",
            boundary=(
                "Engineering feasibility and coarse clearance do not establish human "
                "anatomy, attainable technique, or injury risk."
            ),
            explanations=structural_explanations,
            controls=structural_controls,
            falsifier=(
                "Feasible counts, the retained failure identity, joint-limit margin, or "
                "authority digests fail reproduction."
            ),
            adjudication=(
                "The low-height failure remains in the denominator and is not treated "
                "as a participant classification."
            ),
        ),
        _claim(
            "PD-CLAIM-303",
            [authority["boundary"]],
            statement=(
                "Structural authority regeneration does not establish sensitivity of "
                "either headline; completion requires each feasible scaled authority in "
                "both full atlases with the retained failure and all original controls."
            ),
            classification="structural_headline_propagation_boundary",
            status="registered_but_not_yet_executed",
            audit_status="propagation_acceptance_and_invalidation_contract_checked",
            artifacts=STRUCTURAL_ARTIFACTS,
            domain="Planned propagation from structural authorities to both atlases.",
            boundary="No structural robustness, human mechanism, or strategy is claimed.",
            explanations=structural_explanations,
            controls=structural_controls,
            falsifier=(
                "Nominal anatomy is reused, a retained failure is dropped, or any "
                "registered dynamic or provenance gate fails."
            ),
            adjudication="The unexecuted dynamic propagation remains an explicit gate.",
        ),
        _claim(
            "PD-CLAIM-304",
            [headline["method"], headline["bounds"], headline["provenance"]],
            statement=(
                "A registered 19-corner one-at-a-time campaign repeats affected full "
                "shaft and ground atlases across nine engineering axes while retaining "
                "both engines, reversal, refinement, killswitches, horizons, matching, "
                "and failures."
            ),
            classification="articulated_headline_uncertainty_design",
            status="registered_execution_in_progress",
            audit_status="design_bounds_controls_and_partial_record_boundary_checked",
            artifacts=HEADLINE_DESIGN_ARTIFACTS,
            domain=(
                "Nominal plus low/high grip, shaft, and ground constitutive engineering "
                "corners over the complete production atlases."
            ),
            boundary=(
                "One-at-a-time bounds are not a joint distribution, calibration, or "
                "population uncertainty analysis."
            ),
            explanations=headline_explanations,
            controls=headline_controls,
            falsifier=(
                "A corner uses a reduced surrogate, changes the matching rule, loses a "
                "control, or a partial record is promoted as completion."
            ),
            adjudication=(
                "Only the preregistered design and current in-progress status are "
                "supported; no completed sensitivity result is asserted."
            ),
        ),
        _claim(
            "PD-CLAIM-305",
            [headline["estimand"], headline["boundary"]],
            statement=(
                "Headline matched-count movement diagnoses sensitivity of the "
                "comparability set; it is not an outcome effect, probability, pathway "
                "benefit, parameter interaction, or human/coaching result."
            ),
            classification="articulated_headline_estimand_and_inference_boundary",
            status="explicitly_bounded",
            audit_status="estimand_matching_and_nonpromotion_rules_checked",
            artifacts=HEADLINE_DESIGN_ARTIFACTS,
            domain="Interpretation contract for the registered headline campaign.",
            boundary=(
                "Equipment calibration, participant anatomy/contact, unilateral ground, "
                "impact, delivery, and held-out bilateral-wrench evidence remain open."
            ),
            explanations=headline_explanations,
            controls=headline_controls,
            falsifier=(
                "Matched-count change or emerged ground support is described as a speed, "
                "causal, human, or coaching benefit."
            ),
            adjudication=(
                "The count estimand and every prohibited promotion are stated "
                "separately from eventual outcome analysis."
            ),
        ),
    ]
    registry["claims"].extend(new_claims)
    claims = {claim["claim_id"]: claim for claim in registry["claims"]}

    mapping = {
        "bounds": ("PD-CLAIM-297",),
        "method": ("PD-CLAIM-297",),
        "result": ("PD-CLAIM-298",),
        "prcc": ("PD-CLAIM-299",),
        "interpretation": ("PD-CLAIM-299",),
        "boundary": ("PD-CLAIM-300",),
    }
    for name, claim_ids in mapping.items():
        _review(
            reviews, screen[name], claim_ids, "Articulated screen claim or boundary."
        )
    for name in ("figure", "availability"):
        _review(reviews, screen[name], (), "Figure or governed-artifact pointer.")

    mapping = {
        "method": ("PD-CLAIM-301",),
        "header": (),
        "table": ("PD-CLAIM-302",),
        "figure": (),
        "caption": ("PD-CLAIM-302",),
        "result": ("PD-CLAIM-302",),
        "boundary": ("PD-CLAIM-303",),
        "availability": (),
    }
    for name, claim_ids in mapping.items():
        _review(
            reviews,
            authority[name],
            claim_ids,
            "Structural authority claim, boundary, or evidence pointer.",
        )

    mapping = {
        "method": ("PD-CLAIM-304",),
        "bounds": ("PD-CLAIM-304",),
        "estimand": ("PD-CLAIM-305",),
        "figure": (),
        "provenance": ("PD-CLAIM-304",),
        "boundary": ("PD-CLAIM-305",),
    }
    for name, claim_ids in mapping.items():
        _review(
            reviews,
            headline[name],
            claim_ids,
            "Headline uncertainty design, boundary, or figure pointer.",
        )

    repeated = [
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
                "PD-CLAIM-297",
                "PD-CLAIM-298",
                "PD-CLAIM-300",
                "PD-CLAIM-301",
                "PD-CLAIM-302",
                "PD-CLAIM-303",
            ),
        ),
        (
            _find(candidates, "_ch09_conclusions.qmd", "The subject-scaled contact"),
            ("PD-CLAIM-260", "PD-CLAIM-263", "PD-CLAIM-264"),
        ),
        (
            _find(
                candidates, "_ch09_conclusions.qmd", "The subject-scaled articulated"
            ),
            tuple(f"PD-CLAIM-{value}" for value in range(287, 297)),
        ),
        (
            _find(candidates, "_ch09_conclusions.qmd", "A seven-corner structural"),
            ("PD-CLAIM-301", "PD-CLAIM-302", "PD-CLAIM-303"),
        ),
        (
            _find(candidates, "_ch09_conclusions.qmd", "The open-resource layer"),
            ("PD-CLAIM-223", "PD-CLAIM-241", "PD-CLAIM-303"),
        ),
        (
            _find(
                candidates,
                "_ch09_conclusions.qmd",
                "1. **Complete Structural-Corner",
            ),
            ("PD-CLAIM-128", "PD-CLAIM-206", "PD-CLAIM-234", "PD-CLAIM-303"),
        ),
    ]
    for candidate, claim_ids in repeated:
        _attach(claims, reviews, candidate, claim_ids)
    navigation = _find(
        candidates, "_ch09_conclusions.qmd", "The canonical implementation roadmap"
    )
    _review(reviews, navigation, (), "This is an issue-tracker navigation statement.")

    claim_273 = claims["PD-CLAIM-273"]
    claim_273["statement"] = (
        "The 50 ms validity-horizon result strengthens the reduced hand-carriage "
        "reference; the articulated point-contact tier remains qualified only through "
        "5 ms, while later distributed-grip, shaft, and ground tiers separately reach "
        "50 ms. None establishes calibrated anatomy/equipment or human strategy."
    )
    claim_273["uncertainty_boundary"] = (
        "The tiers use different contact, shaft, support, and state contracts; their "
        "horizon results cannot be pooled into a calibrated human persistence claim."
    )
    claim_273["last_verified_on"] = DATE

    reciprocal: dict[str, list[str]] = {}
    for claim in claims.values():
        for candidate_id in claim.get("candidate_ids", []):
            reciprocal.setdefault(candidate_id, []).append(claim["claim_id"])
    for candidate_id, review in reviews.items():
        review["claim_ids"] = list(
            dict.fromkeys([*review["claim_ids"], *reciprocal.get(candidate_id, [])])
        )

    registry["candidate_reviews"] = list(reviews.values())
    registry["claims"] = list(claims.values())
    registry["paper"]["source_digest"] = inventory["source_digest"]
    registry["audit_scope"]["current_scope"] = (
        "The complete 1,092-candidate paper inventory is adjudicated. The local "
        "articulated screen is conditional on partial opening, structural authorities "
        "retain one low-height failure, and both structural propagation and the "
        "19-corner headline campaign remain open without human or coaching promotion."
    )
    registry["audit_scope"]["completion_status"] = "complete"
    _write_registry(registry)


if __name__ == "__main__":
    main()
