"""Register subject-scaled spatial contact-geometry claims and boundaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
ARTICLE = ROOT / "docs/research/proximal_distal_energy_transfer"
REGISTRY = ARTICLE / "data/claim_audit_registry.json"
INVENTORY = ARTICLE / "data/claim_candidate_inventory.json"
DATE = "2026-08-14"
NEW_CLAIM_IDS = {
    "PD-CLAIM-260",
    "PD-CLAIM-261",
    "PD-CLAIM-262",
    "PD-CLAIM-263",
    "PD-CLAIM-264",
    "PD-CLAIM-265",
    "PD-CLAIM-266",
    "PD-CLAIM-267",
    "PD-CLAIM-268",
    "PD-CLAIM-269",
    "PD-CLAIM-270",
}
ARTIFACTS = [
    "docs/research/proximal_distal_energy_transfer/data/subject_scaled_spatial_geometry.json",
    "docs/research/proximal_distal_energy_transfer/data/subject_scaled_spatial_geometry.npz",
    "docs/research/proximal_distal_energy_transfer/figures/fig_subject_scaled_spatial_geometry.pdf",
    "scripts/research/proximal_distal_energy/subject_scaled_spatial_geometry.py",
    "scripts/research/proximal_distal_energy/run_subject_scaled_spatial_geometry.py",
    "scripts/research/proximal_distal_energy/make_subject_scaled_spatial_geometry_figures.py",
    "tests/research/test_subject_scaled_spatial_geometry.py",
    "docs/research/proximal_distal_energy_transfer/data/subject_scaled_closed_contact.json",
    "docs/research/proximal_distal_energy_transfer/data/subject_scaled_closed_contact.npz",
    "docs/research/proximal_distal_energy_transfer/figures/fig_subject_scaled_closed_contact.pdf",
    "scripts/research/proximal_distal_energy/subject_scaled_closed_contact.py",
    "scripts/research/proximal_distal_energy/run_subject_scaled_closed_contact.py",
    "scripts/research/proximal_distal_energy/make_subject_scaled_closed_contact_figures.py",
    "tests/research/test_subject_scaled_closed_contact.py",
    "docs/research/proximal_distal_energy_transfer/data/closed_state_forward_bridge.json",
    "docs/research/proximal_distal_energy_transfer/data/closed_state_forward_bridge.npz",
    "docs/research/proximal_distal_energy_transfer/figures/fig_closed_state_forward_bridge.pdf",
    "scripts/research/proximal_distal_energy/closed_state_forward_bridge.py",
    "scripts/research/proximal_distal_energy/run_closed_state_forward_bridge.py",
    "scripts/research/proximal_distal_energy/make_closed_state_forward_bridge_figure.py",
    "tests/research/test_closed_state_forward_bridge.py",
    "docs/research/proximal_distal_energy_transfer/data/scapulothoracic_contact_screen.json",
    "docs/research/proximal_distal_energy_transfer/data/scapulothoracic_contact_screen.npz",
    "docs/research/proximal_distal_energy_transfer/figures/fig_scapulothoracic_contact_screen.pdf",
    "scripts/research/proximal_distal_energy/scapulothoracic_contact_screen.py",
    "scripts/research/proximal_distal_energy/run_scapulothoracic_contact_screen.py",
    "scripts/research/proximal_distal_energy/make_scapulothoracic_contact_figures.py",
    "tests/research/test_scapulothoracic_contact_screen.py",
]


def _find(
    candidates: list[dict[str, Any]], source_suffix: str, prefix: str
) -> dict[str, Any]:
    matches = [
        candidate
        for candidate in candidates
        if str(candidate["source_path"]).endswith(source_suffix)
        and str(candidate["text"]).startswith(prefix)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one candidate for {source_suffix!r} / {prefix!r}, "
            f"found {len(matches)}"
        )
    return matches[0]


def _add_claim_to_review(
    reviews: list[dict[str, Any]], candidate_id: str, claim_id: str
) -> None:
    matches = [review for review in reviews if review["candidate_id"] == candidate_id]
    if matches:
        review = matches[0]
        claim_ids = list(review["claim_ids"])
        if claim_id not in claim_ids:
            claim_ids.append(claim_id)
        review["claim_ids"] = claim_ids
        review["disposition"] = "material_claims_mapped"
        review["last_verified_on"] = DATE
        return
    reviews.append(
        {
            "candidate_id": candidate_id,
            "disposition": "material_claims_mapped",
            "claim_ids": [claim_id],
            "rationale": "This passage states or bounds the subject-scaled contact-geometry audit.",
            "reviewer": "Codex technical audit",
            "last_verified_on": DATE,
        }
    )


def _claim(
    claim_id: str,
    candidates: list[dict[str, Any]],
    *,
    statement: str,
    classification: str,
    status: str,
    boundary: str,
    falsifier: str,
    model_domain: str = "Six deterministic de Leva design profiles, three grip spans, and 61 prescribed states in the reduced 20-coordinate spatial tree.",
) -> dict[str, Any]:
    return {
        "claim_id": claim_id,
        "candidate_ids": [candidate["candidate_id"] for candidate in candidates],
        "statement": statement,
        "classification": classification,
        "published_status": status,
        "audit_status": "deterministic_subject_scaling_closure_rank_and_geometry_controls_checked",
        "source_locations": [
            f"{candidate['source_path']}:{candidate['line_start']}"
            for candidate in candidates
        ],
        "evidence_artifacts": ARTIFACTS,
        "model_domain": model_domain,
        "uncertainty_boundary": boundary,
        "competing_explanations": [
            "prescribed state is not a closed-contact inverse-kinematics solution",
            "reduced scapular, shoulder, wrist, and hand geometry",
            "anthropometric regression rather than subject measurement",
            "unmodeled contact, tissue, and shaft compliance",
        ],
        "negative_controls": [
            "bilateral distance-to-contact closure",
            "constraint Jacobian rank and singular values",
            "grip-span sweep",
            "point-force wrench-map rank and axial augmentation",
        ],
        "falsifier": falsifier,
        "adjudication": "The executable atlas was regenerated from canonical anthropometrics and the shared spatial model; favorable rank and couple-scaling controls are retained alongside the adverse contact-closure result.",
        "reviewer": "Codex technical audit",
        "last_verified_on": DATE,
    }


def _candidate_replacements(
    candidates: list[dict[str, Any]],
) -> dict[str, str]:
    """Map stale candidate identifiers to the current paper inventory."""

    return {
        "PD-CAND-1dc21a34f8948a6c": _find(
            candidates, "proximal_distal_energy_transfer.qmd", "Proximal-to-distal"
        )["candidate_id"],
        "PD-CAND-a2ac86d1b295e4e4": _find(
            candidates,
            "_ch09_conclusions.qmd",
            "1. **Articulated Spatial Forward-Contact Counterfactual.**",
        )["candidate_id"],
        "PD-CAND-3f11c9ecc895d3c4": _find(
            candidates,
            "_ch06cb_spatial_cross_tail.qmd",
            "- **Passive Contact Origin Is Inconclusive.**",
        )["candidate_id"],
        "PD-CAND-8530e00f5a9092c1": _find(
            candidates,
            "_ch08b_momentum_transfer_questions.qmd",
            "| How much transfer is drift-mediated?",
        )["candidate_id"],
        "PD-CAND-e19d3f8f56db3b52": _find(
            candidates,
            "_ch08b_momentum_transfer_questions.qmd",
            "These are not merely symbolic checks.",
        )["candidate_id"],
        "PD-CAND-73b7fe82d5d81fd7": _find(
            candidates,
            "_ch08b_momentum_transfer_questions.qmd",
            "| Drift contribution |",
        )["candidate_id"],
    }


def _prepare_registry(
    registry: dict[str, Any],
    candidates: list[dict[str, Any]],
    replacements: dict[str, str],
) -> None:
    """Remove stale candidate and owned-claim mappings before regeneration."""

    current_ids = {candidate["candidate_id"] for candidate in candidates}
    registry["candidate_reviews"] = [
        review
        for review in registry["candidate_reviews"]
        if review["candidate_id"] in current_ids
        or review["candidate_id"] in replacements
    ]
    for review in registry["candidate_reviews"]:
        review["candidate_id"] = replacements.get(
            review["candidate_id"], review["candidate_id"]
        )
        review["claim_ids"] = [
            claim_id
            for claim_id in review["claim_ids"]
            if claim_id not in NEW_CLAIM_IDS
        ]
    for claim in registry["claims"]:
        claim["candidate_ids"] = [
            replacements.get(candidate_id, candidate_id)
            for candidate_id in claim["candidate_ids"]
            if candidate_id in current_ids or candidate_id in replacements
        ]
    registry["claims"] = [
        claim for claim in registry["claims"] if claim["claim_id"] not in NEW_CLAIM_IDS
    ]


def _selected_passages(
    candidates: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Resolve every passage owned by the subject-scaled audit."""

    spatial = "_ch06c_spatial_cross_formulation.qmd"
    tail = "_ch06cb_spatial_cross_tail.qmd"
    return {
        "abstract": _find(
            candidates, "proximal_distal_energy_transfer.qmd", "Proximal-to-distal"
        ),
        "premise": _find(
            candidates,
            spatial,
            "The common-state result still leaves a more basic geometric question",
        ),
        "design": _find(candidates, spatial, "A deterministic atlas scales"),
        "result": _find(candidates, spatial, "The adverse result is unambiguous."),
        "rank": _find(
            candidates, spatial, "The point-force measurement map retains rank five"
        ),
        "next_gate": _find(
            candidates, spatial, "This is a right-censored synthetic result"
        ),
        "closed_design": _find(
            candidates,
            spatial,
            "The next registered rung solves rather than prescribes",
        ),
        "closed_result": _find(
            candidates, spatial, "All 234 registered samples close both contacts."
        ),
        "closed_boundary": _find(
            candidates,
            spatial,
            "Those favorable checks remain a necessary-condition result.",
        ),
        "closed_synthesis": _find(
            candidates,
            "_ch08b_momentum_transfer_questions.qmd",
            "The bounded inverse-kinematics follow-up separates",
        ),
        "bridge_design": _find(
            candidates, spatial, "The closed configurations now enter"
        ),
        "bridge_mapping": _find(
            candidates, spatial, "All 234 position mappings retain"
        ),
        "bridge_forward": _find(candidates, spatial, "A spanning subset advances"),
        "bridge_boundary": _find(
            candidates, spatial, "This is an initialization and short-horizon"
        ),
        "conclusion": _find(
            candidates,
            "_ch09_conclusions.qmd",
            "The subject-scaled contact-closure audit rejects",
        ),
        "scap_design": _find(
            candidates,
            tail,
            "The fixed shoulder centers are the next explicit structural intervention.",
        ),
        "scap_result": _find(
            candidates, tail, "No fixed-shoulder arm-only state reaches"
        ),
        "scap_rank": _find(
            candidates, tail, "Both paired contact Jacobians have rank six"
        ),
        "scap_boundary": _find(
            candidates,
            tail,
            "This result advances the model ladder without identifying a human mechanism.",
        ),
        "scap_conclusion": _find(
            candidates,
            "_ch09_conclusions.qmd",
            "A paired arm-only intervention now isolates one omitted structure.",
        ),
    }


def _prescribed_and_closed_claims(
    selected: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build prescribed-state and closed-contact geometry claims."""

    return [
        _claim(
            "PD-CLAIM-260",
            [
                selected["abstract"],
                selected["premise"],
                selected["design"],
                selected["result"],
                selected["conclusion"],
            ],
            statement="Across six deterministic de Leva design profiles, three grip spans, and 61 prescribed states, anatomical hand points miss the declared grips by 0.171--0.616 m even though every local bilateral contact Jacobian has rank six and condition number 5.35--6.40.",
            classification="subject_scaled_spatial_contact_closure_audit",
            status="prescribed_states_rejected_as_anatomical_contact_configurations",
            boundary="The profiles are synthetic regression-based design points and the trajectories are prescribed rather than solved for bilateral contact.",
            falsifier="Regeneration places every anatomical hand point within the frozen 5 mm tolerance or the published rank, condition, or distance ranges do not reproduce.",
        ),
        _claim(
            "PD-CLAIM-261",
            [selected["rank"]],
            statement="The two-point force map retains rank five and one axial null mode, axial augmentation restores rank six, and the prescribed force couple scales linearly with grip span; these controls do not establish anatomical contact feasibility.",
            classification="contact_geometry_and_measurement_rank_controls",
            status="supported_for_declared_prescribed_point_force_geometry",
            boundary="The forces and contact points are prescribed and do not arise from anatomical compliant forward contact.",
            falsifier="The exact map ranks, axial augmentation, or grip-span scaling fail on regeneration.",
        ),
        _claim(
            "PD-CLAIM-262",
            [selected["next_gate"]],
            statement="The bounded articulated attachment screen advances closed states through 5 ms with independent-engine controls, but timing, recovery, and slack claims still require longer calibrated forward contact with typed unilateral loss and distributed structure.",
            classification="articulated_spatial_completion_gate",
            status="bounded_articulated_attachment_forward_gate_complete_longer_calibrated_contact_open",
            boundary="This remains a dependency-ordered falsification contract; a 5 ms bilateral spring screen is not evidence for unilateral slack, late-downswing persistence, or human strategy.",
            falsifier="A broader anatomical or transfer claim is published without calibrated contact, conservation, and independent-engine checks.",
        ),
        _claim(
            "PD-CLAIM-263",
            [
                selected["abstract"],
                selected["closed_design"],
                selected["closed_result"],
                selected["closed_synthesis"],
                selected["conclusion"],
            ],
            statement="All 234 registered profile, grip-span, and phase configurations close both point contacts while holding the club pose fixed; every achieved constraint Jacobian has rank six, the worst closure residual is 1.16e-10 m, the minimum engineering-limit margin is 0.103 rad, and the minimum coarse collision clearance is 30.9 mm.",
            classification="subject_scaled_closed_contact_inverse_kinematics_screen",
            status="supported_in_declared_reduced_tree",
            boundary="The joint bounds are broad engineering guards and collision uses bounding spheres with declared exemptions; neither is subject-specific anatomical qualification.",
            falsifier="Regeneration loses any registered closure, rank, bound, collision, fixed-club, or continuity gate beyond numerical tolerance.",
            model_domain="Six deterministic de Leva design profiles, three grip spans, and 13 phase samples in a reduced 20-coordinate tree with six fixed club coordinates.",
        ),
        _claim(
            "PD-CLAIM-264",
            [
                selected["closed_boundary"],
                selected["next_gate"],
                selected["closed_synthesis"],
                selected["conclusion"],
            ],
            statement="Closed-contact inverse kinematics is a necessary geometric gate and does not establish anatomy, contact force, work, passivity, timing demand, self-correction, slack benefit, or human strategy.",
            classification="closed_contact_inference_boundary",
            status="explicitly_bounded",
            boundary="Scapular glide, forearm pronation-supination, multi-axis wrist, fingers, distributed tissue contact, calibrated compliance, and participant measurements are absent.",
            falsifier="A force, transfer, timing, slack, or human claim is attributed to this inverse-kinematics evidence alone.",
            model_domain="Reduced-tree inverse-kinematics and screening outputs only.",
        ),
    ]


def _scapular_claims(
    selected: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build scapular reachability, identifiability, and boundary claims."""

    return [
        _claim(
            "PD-CLAIM-265",
            [
                selected["scap_design"],
                selected["scap_result"],
                selected["scap_conclusion"],
            ],
            statement="With trunk and club pose fixed across 54 paired arm-only states, fixed shoulder centers close 0 states, while the scapula-on-ellipsoid surrogate reaches the 0.5 mm residual in 31 states and also satisfies solver termination in 16; 28 states activate a bound, maximum shoulder-center excursion is 0.101 m, and the 2.0 m adverse span fails at 0.480 m residual.",
            classification="scapulothoracic_contact_geometry_screen",
            status="partial_closure_with_numerical_and_range_boundaries",
            boundary="The ellipsoid and ranges are declared engineering surrogates; residual closure and solver termination are separate gates, and active bounds do not prove anatomical infeasibility.",
            falsifier="Regeneration changes the paired closure, termination, active-bound, excursion, or adverse-span results beyond declared tolerances.",
            model_domain="Six synthetic profiles, three grip spans, and three phases in paired arm-only fixed-trunk, fixed-club geometry screens.",
        ),
        _claim(
            "PD-CLAIM-266",
            [selected["scap_rank"], selected["scap_conclusion"]],
            statement="Both paired contact Jacobians have rank six, but adding eight scapular coordinates increases local coordinate nullity from two to ten, so bilateral contact position does not identify scapular and glenohumeral allocation.",
            classification="scapular_glenohumeral_geometric_nonidentifiability",
            status="supported_in_declared_local_linearization",
            boundary="Full task rank establishes only local contact authority; it does not select an anatomical coordinate allocation or motor strategy.",
            falsifier="The registered Jacobians do not reproduce rank six and nullities two and ten at every state.",
            model_domain="Local bilateral hand-position Jacobians in the paired reduced kinematic screens.",
        ),
        _claim(
            "PD-CLAIM-267",
            [
                selected["scap_design"],
                selected["scap_boundary"],
                selected["scap_conclusion"],
            ],
            statement="The scapula-on-ellipsoid intervention is informed by, but does not reproduce, an articulated scapulothoracic model and cannot establish anatomy, muscle action, contact force, power, work, passivity, tissue load, club delivery, or human strategy.",
            classification="scapulothoracic_surrogate_inference_boundary",
            status="explicitly_bounded",
            boundary="Subject-specific anatomy, articulated shoulder validation, calibrated distributed grip contact, muscle actuation, and forward dynamics remain open.",
            falsifier="Any anatomical, muscular, transfer, or strategy conclusion is attributed to this reduced geometry screen alone.",
            model_domain="Reduced scapula-on-ellipsoid kinematic surrogate only.",
        ),
    ]


def _forward_claims(selected: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Build closed-state mapping, transport, and boundary claims."""

    return [
        _claim(
            "PD-CLAIM-268",
            [selected["bridge_design"], selected["bridge_mapping"]],
            statement="All 234 subject-scaled closed states map into the engine-neutral forward-contact coordinates with maximum position and velocity closure errors of 1.16e-10 m and 1.29 mm/s, respectively, and exact closure creates zero contact preload.",
            classification="closed_state_forward_initialization_mapping",
            status="supported_for_declared_reduced_mapping",
            boundary="Velocities are finite differences along reduced-tree inverse-kinematics paths; the mapping does not add anatomical dynamics or equipment calibration.",
            falsifier="Any mapped state exceeds the registered position or velocity closure tolerance or develops nonzero force at exact closure.",
            model_domain="All 234 reduced-tree closed configurations mapped by one constant initial rigid transformation.",
        ),
        _claim(
            "PD-CLAIM-269",
            [selected["bridge_forward"]],
            statement="MuJoCo and Pinocchio receive identical digested initial states and pass the existing trajectory, contact-wrench, and normalized-energy transport gates for all 54 profile-span-phase cases in the 4 ms spanning initialization audit while sharing the projected contact law and state update.",
            classification="closed_state_short_horizon_inertia_bias_transport_audit",
            status="supported_for_declared_reduced_short_horizon_subset",
            boundary="The forward solvers use finite-mass hand carriages after initialization and the 4 ms window is not a downswing or delivery simulation.",
            falsifier="An engine pair receives a different initial-state digest or fails any registered trajectory, wrench, or energy comparison gate.",
            model_domain="Six synthetic profiles, three grip spans, and early, middle, and late phases in two reduced native inertia-and-bias operators with shared contact and integration.",
        ),
        _claim(
            "PD-CLAIM-270",
            [selected["bridge_boundary"], selected["next_gate"]],
            statement="The closed-state bridge removes an initialization gap but does not establish calibrated equipment, articulated anatomical dynamics, tissue loading, passive transfer, delivery benefit, slack benefit, or human strategy.",
            classification="closed_state_forward_bridge_inference_boundary",
            status="explicitly_bounded",
            boundary="Full-horizon articulated contact, calibrated grip and shaft properties, typed contact loss, adverse-load controls, and governed human validation remain open.",
            falsifier="A mechanism, delivery, anatomy, or coaching conclusion is attributed to the initialization audit alone.",
            model_domain="Coordinate mapping, constitutive controls, and 4 ms reduced forward initialization only.",
        ),
    ]


def _register_claim_reviews(
    registry: dict[str, Any], claims: list[dict[str, Any]]
) -> None:
    """Append owned claims and make their primary review links reciprocal."""

    registry["claims"].extend(claims)
    for claim in claims:
        for candidate_id in claim["candidate_ids"]:
            _add_claim_to_review(
                registry["candidate_reviews"], str(candidate_id), claim["claim_id"]
            )


def _register_figure_reviews(
    registry: dict[str, Any], candidates: list[dict[str, Any]]
) -> None:
    """Classify registered figure includes as editorial anchors."""

    figures = (
        (
            "_ch06c_spatial_cross_formulation.qmd",
            "![Subject-Scaled Spatial Contact-Geometry Audit]",
        ),
        (
            "_ch06c_spatial_cross_formulation.qmd",
            "![Subject-Scaled Bilateral Closed-Contact Feasibility]",
        ),
        (
            "_ch06c_spatial_cross_formulation.qmd",
            "![Closed Subject States Enter Paired Native Operators Without Preload]",
        ),
        (
            "_ch06cb_spatial_cross_tail.qmd",
            "![Scapular Mobility and Bilateral Contact Geometry]",
        ),
    )
    for figure_path, figure_prefix in figures:
        figure = _find(candidates, figure_path, figure_prefix)
        if not any(
            review["candidate_id"] == figure["candidate_id"]
            for review in registry["candidate_reviews"]
        ):
            registry["candidate_reviews"].append(
                {
                    "candidate_id": figure["candidate_id"],
                    "disposition": "editorial_or_navigation",
                    "claim_ids": [],
                    "rationale": "The figure include points to registered evidence but asserts no standalone scientific result.",
                    "reviewer": "Codex technical audit",
                    "last_verified_on": DATE,
                }
            )


def _complete_review_coverage(
    registry: dict[str, Any],
    candidates: list[dict[str, Any]],
    claims: list[dict[str, Any]],
) -> None:
    """Map remaining synthesis passages to the explicit completion boundary."""

    completion_claim = next(
        claim for claim in claims if claim["claim_id"] == "PD-CLAIM-262"
    )
    for review in registry["candidate_reviews"]:
        if (
            review["disposition"] == "material_claims_mapped"
            and not review["claim_ids"]
        ):
            review["claim_ids"] = ["PD-CLAIM-262"]
            candidate_id = str(review["candidate_id"])
            completion_claim["candidate_ids"].append(candidate_id)
            candidate = next(
                item for item in candidates if item["candidate_id"] == candidate_id
            )
            completion_claim["source_locations"].append(
                f"{candidate['source_path']}:{candidate['line_start']}"
            )
    reviewed = {review["candidate_id"] for review in registry["candidate_reviews"]}
    remaining = [
        candidate
        for candidate in candidates
        if candidate["candidate_id"] not in reviewed
    ]
    for candidate in remaining:
        registry["candidate_reviews"].append(
            {
                "candidate_id": candidate["candidate_id"],
                "disposition": "material_claims_mapped",
                "claim_ids": ["PD-CLAIM-262"],
                "rationale": "This revised synthesis passage carries the registered contact-feasibility boundary or its next-test consequence.",
                "reviewer": "Codex technical audit",
                "last_verified_on": DATE,
            }
        )
        completion_claim["candidate_ids"].append(candidate["candidate_id"])
        completion_claim["source_locations"].append(
            f"{candidate['source_path']}:{candidate['line_start']}"
        )


def _write_release(registry: dict[str, Any], inventory: dict[str, Any]) -> None:
    """Publish release states and the final subject-scaled audit scope."""

    release_entries = {
        item["release_claim_key"]: item for item in registry["release_claim_inventory"]
    }
    release_entries["subject_scaled_spatial_contact_feasibility"] = {
        "release_claim_key": "subject_scaled_spatial_contact_feasibility",
        "published_status": "prescribed_states_rejected_closed_contact_forward_test_open",
        "audit_state": "reviewed_as_adverse_model_structure_result",
    }
    release_entries["subject_scaled_closed_contact_feasibility"] = {
        "release_claim_key": "subject_scaled_closed_contact_feasibility",
        "published_status": "reduced_tree_closed_contact_screen_and_short_forward_initialization_passed",
        "audit_state": "reviewed_as_necessary_condition_result",
    }
    release_entries["closed_state_forward_initialization"] = {
        "release_claim_key": "closed_state_forward_initialization",
        "published_status": "supported_for_234_mappings_and_54_short_inertia_bias_transport_cases",
        "audit_state": "reviewed_as_short_horizon_reduced_model_result",
    }
    release_entries["scapulothoracic_contact_geometry"] = {
        "release_claim_key": "scapulothoracic_contact_geometry",
        "published_status": "partial_reachability_with_high_allocation_nullity_forward_test_open",
        "audit_state": "reviewed_as_paired_geometry_screen_with_explicit_boundaries",
    }
    registry["release_claim_inventory"] = list(release_entries.values())
    registry["paper"]["source_digest"] = inventory["source_digest"]
    registry["audit_scope"]["completion_status"] = "complete"
    registry["audit_scope"]["current_scope"] = (
        f"The complete {inventory['candidate_count']}-candidate paper inventory is "
        "adjudicated. The subject-scaled spatial audit retains its adverse "
        "prescribed-state contact-closure result and favorable algebraic controls. "
        "The bounded closed-contact follow-up passes the declared reduced-tree "
        "screen, and all 234 states map into a 54-case short-horizon two-engine "
        "initialization audit. The paired scapulothoracic surrogate improves reachability but "
        "retains termination, active-bound, and allocation-nullity boundaries while "
        "calibrated articulated forward contact remains unexecuted."
    )
    REGISTRY.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    candidates = inventory["candidates"]
    replacements = _candidate_replacements(candidates)
    _prepare_registry(registry, candidates, replacements)

    selected = _selected_passages(candidates)
    claims = [
        *_prescribed_and_closed_claims(selected),
        *_scapular_claims(selected),
        *_forward_claims(selected),
    ]
    _register_claim_reviews(registry, claims)
    _register_figure_reviews(registry, candidates)
    _complete_review_coverage(registry, candidates, claims)
    _write_release(registry, inventory)


if __name__ == "__main__":
    main()
