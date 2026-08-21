"""Register native-constraint formulation-discrepancy claims (#8911)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
ARTICLE = ROOT / "docs/research/proximal_distal_energy_transfer"
REGISTRY = ARTICLE / "data/claim_audit_registry.json"
INVENTORY = ARTICLE / "data/claim_candidate_inventory.json"
DATE = "2026-08-21"
CLAIM_IDS = {"PD-CLAIM-302", "PD-CLAIM-303", "PD-CLAIM-304"}
ARTIFACTS = [
    "docs/research/proximal_distal_energy_transfer/data/articulated_native_constraint_discrepancy.json",
    "docs/research/proximal_distal_energy_transfer/data/articulated_native_constraint_discrepancy.npz",
    "docs/research/proximal_distal_energy_transfer/figures/fig_articulated_native_constraint_discrepancy.pdf",
    "scripts/research/proximal_distal_energy/articulated_native_constraint_discrepancy.py",
    "scripts/research/proximal_distal_energy/run_articulated_native_constraint_discrepancy.py",
    "scripts/research/proximal_distal_energy/make_articulated_native_constraint_discrepancy_figure.py",
    "scripts/research/proximal_distal_energy/register_articulated_native_constraint_discrepancy_claims.py",
    "scripts/research/proximal_distal_energy/spatial_full_body.py",
    "scripts/research/proximal_distal_energy/articulated_forward_integration.py",
    "tests/research/test_articulated_native_constraint_discrepancy.py",
    "tests/research/test_spatial_full_body_cross_formulation.py",
]

FORWARD_CLAIM_PREFIXES: dict[str, tuple[str, ...]] = {
    "PD-CLAIM-005": ("After the killswitch",),
    "PD-CLAIM-184": (
        "The common-state experiment",
        "> Can a negative",
        "The answer is affirmative",
        "The independence claim",
        "This result closes",
    ),
    "PD-CLAIM-185": (
        "The model contains",
        "The achieved club-side",
        "The reduction is purposeful",
        "The common record fixes",
        "![Achieved Spatial Contact Geometry",
        "@fig-spatial-forward-contact-geometry",
    ),
    "PD-CLAIM-186": (
        "Each interface uses",
        "with $k_c=1800$",
        "The executed record also audits",
        "A world-referenced trajectory",
        "where $k_d=420$",
        "No force or torque is applied",
        "The equal and opposite driver wrench",
    ),
    "PD-CLAIM-187": (
        "The common experiment executes",
        "- **MuJoCo.**",
        "Both branches use",
        "The comparison evaluates",
        "![MuJoCo and Pinocchio Inertia-and-Bias",
    ),
    "PD-CLAIM-188": (
        "@fig-spatial-forward-cross-engine",
        "- club-position RMS difference",
        "In the driver-killswitch branch",
    ),
    "PD-CLAIM-189": (
        "The baseline and intervention branches",
        "![Post-Killswitch Interaction Dynamics",
        "After the killswitch",
        "The ground-pathway proxy reaches",
    ),
    "PD-CLAIM-190": (
        "Two same-force geometry controls",
        "1. transport the achieved forces",
        "The coincident-grip couple is exactly zero",
        "![Conservation and Negative Controls",
    ),
    "PD-CLAIM-191": ("The work--energy residual includes",),
    "PD-CLAIM-192": (
        "The committed record fails closed",
        "1. either imported library lacks",
        "The experiment would therefore reject",
    ),
    "PD-CLAIM-193": (
        "The evidence is stored in",
        "The next transport step is not",
    ),
}

FORWARD_CLAIM_REVISIONS = {
    "PD-CLAIM-005": (
        "model_conditional_projected_contact_transport_counterfactual",
        "The reduced MuJoCo and Pinocchio transport branches retain a negative "
        "swing-normal contact couple for 37.5 ms under one shared projected "
        "contact law and state update.",
    ),
    "PD-CLAIM-184": (
        "bounded_inertia_bias_transport_estimand",
        "The reduced forward-contact tier tests trajectory-level transport "
        "through separately assembled native inertia-and-bias operators while "
        "the projected contact law and state update remain shared.",
    ),
    "PD-CLAIM-187": (
        "native_inertia_bias_operator_identity_and_acceptance_contract",
        "MuJoCo and Pinocchio independently evaluate continuous-time rigid-body "
        "operators and spatial-force mappings while sharing projected contact "
        "and a semi-implicit update; the acceptance regions are engineering "
        "bounds rather than confidence intervals.",
    ),
    "PD-CLAIM-188": (
        "inertia_bias_transport_trajectory_result",
        "The two shared-update trajectories remain inside the declared "
        "position, orientation, complete-wrench, and energy transport regions "
        "for continuing-driver and killswitch branches.",
    ),
    "PD-CLAIM-192": (
        "registered_inertia_bias_transport_falsification_contract",
        "Operator identity, model digest, transport gates, force and power "
        "closure, branch identity, contiguous-duration, geometry, and "
        "refinement tests jointly define fail-closed falsifiers without "
        "supporting human prescription.",
    ),
}

SYNTHESIS_CANDIDATES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "_ch06cb_spatial_cross_tail.qmd",
        "- **Passive Contact Origin Is Inconclusive.**",
        ("PD-CLAIM-304",),
    ),
    (
        "_ch07_model_ladder.qmd",
        "The next tier replaces prescribed hand loads",
        ("PD-CLAIM-184", "PD-CLAIM-304"),
    ),
    (
        "_ch07_model_ladder.qmd",
        "In the exact same-state driver-killswitch branch",
        ("PD-CLAIM-005", "PD-CLAIM-188"),
    ),
    (
        "_ch07_model_ladder.qmd",
        "This executes projected forward spatial contact",
        ("PD-CLAIM-184", "PD-CLAIM-304"),
    ),
    (
        "_ch07_model_ladder.qmd",
        "1. interaction force may remain nonzero",
        ("PD-CLAIM-184", "PD-CLAIM-188"),
    ),
    (
        "_ch07_model_ladder.qmd",
        "**Reduced Spatial Forward Contact.**",
        ("PD-CLAIM-184", "PD-CLAIM-304"),
    ),
    (
        "_ch08_discussion.qmd",
        "- **Spatial Body and Distributed-Club Dynamics.**",
        ("PD-CLAIM-005", "PD-CLAIM-184"),
    ),
    (
        "_ch09_conclusions.qmd",
        "The articulated finite-base extension",
        ("PD-CLAIM-295", "PD-CLAIM-296"),
    ),
    (
        "_ch09_conclusions.qmd",
        "The reduced spatial forward-contact extension",
        ("PD-CLAIM-184", "PD-CLAIM-304"),
    ),
)


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
        raise ValueError(f"expected one native-constraint candidate for {prefix!r}")
    return matches[0]


def _find_forward(candidates: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    matches = [
        candidate
        for candidate in candidates
        if str(candidate["source_path"]).endswith("_ch06cc_spatial_forward_contact.qmd")
        and str(candidate["text"]).startswith(prefix)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one forward-transport candidate for {prefix!r}")
    return matches[0]


def _reconcile_forward_transport_claims(
    registry: dict[str, Any], candidates: list[dict[str, Any]]
) -> None:
    """Remap and narrow the legacy shared-contact forward claim records."""

    claims = {claim["claim_id"]: claim for claim in registry["claims"]}
    reviews = {
        review["candidate_id"]: review for review in registry["candidate_reviews"]
    }
    for claim_id, prefixes in FORWARD_CLAIM_PREFIXES.items():
        claim = claims[claim_id]
        retained = [
            (candidate_id, location)
            for candidate_id, location in zip(
                claim.get("candidate_ids", []),
                claim.get("source_locations", []),
                strict=True,
            )
            if not str(location)
            .split(":", 1)[0]
            .endswith("_ch06cc_spatial_forward_contact.qmd")
        ]
        selected = [_find_forward(candidates, prefix) for prefix in prefixes]
        claim["candidate_ids"] = [
            *[candidate_id for candidate_id, _ in retained],
            *[candidate["candidate_id"] for candidate in selected],
        ]
        claim["source_locations"] = [
            *[location for _, location in retained],
            *[
                f"{candidate['source_path']}:{candidate['line_start']}"
                for candidate in selected
            ],
        ]
        if claim_id in FORWARD_CLAIM_REVISIONS:
            classification, statement = FORWARD_CLAIM_REVISIONS[claim_id]
            claim["classification"] = classification
            claim["statement"] = statement
        claim["last_verified_on"] = DATE
        for candidate in selected:
            existing_claim_ids = reviews.get(candidate["candidate_id"], {}).get(
                "claim_ids", []
            )
            reviews[candidate["candidate_id"]] = {
                "candidate_id": candidate["candidate_id"],
                "disposition": "material_claims_mapped",
                "claim_ids": list(dict.fromkeys([*existing_claim_ids, claim_id])),
                "rationale": (
                    "This passage states or bounds shared-contact, shared-"
                    "integrator inertia-and-bias transport."
                ),
                "reviewer": "Codex technical audit",
                "last_verified_on": DATE,
            }
    registry["candidate_reviews"] = list(reviews.values())


def _reconcile_synthesis_candidates(
    registry: dict[str, Any], candidates: list[dict[str, Any]]
) -> None:
    """Attach revised synthesis passages to their bounded primary claims."""

    claims = {claim["claim_id"]: claim for claim in registry["claims"]}
    reviews = {
        review["candidate_id"]: review for review in registry["candidate_reviews"]
    }
    for suffix, prefix, claim_ids in SYNTHESIS_CANDIDATES:
        matches = [
            candidate
            for candidate in candidates
            if str(candidate["source_path"]).endswith(suffix)
            and str(candidate["text"]).startswith(prefix)
        ]
        if len(matches) != 1:
            raise ValueError(f"expected one synthesis candidate for {suffix}:{prefix}")
        candidate = matches[0]
        existing_claim_ids = reviews.get(candidate["candidate_id"], {}).get(
            "claim_ids", []
        )
        reviews[candidate["candidate_id"]] = {
            "candidate_id": candidate["candidate_id"],
            "disposition": "material_claims_mapped",
            "claim_ids": list(dict.fromkeys([*existing_claim_ids, *claim_ids])),
            "rationale": (
                "This synthesis passage inherits the bounded native-operator "
                "transport and formulation-discrepancy claims."
            ),
            "reviewer": "Codex technical audit",
            "last_verified_on": DATE,
        }
        location = f"{candidate['source_path']}:{candidate['line_start']}"
        for claim_id in claim_ids:
            claim = claims[claim_id]
            claim["candidate_ids"] = list(
                dict.fromkeys(
                    [*claim.get("candidate_ids", []), candidate["candidate_id"]]
                )
            )
            claim["source_locations"] = list(
                dict.fromkeys([*claim.get("source_locations", []), location])
            )
            claim["last_verified_on"] = DATE
    registry["candidate_reviews"] = list(reviews.values())


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
        "audit_status": "native_constraint_integrator_and_formulation_discrepancy_checked",
        "source_locations": [
            f"{candidate['source_path']}:{candidate['line_start']}"
            for candidate in candidates
        ],
        "evidence_artifacts": ARTIFACTS,
        "model_domain": (
            "One synthetic subject-scaled closed state, two bilateral three-row "
            "MuJoCo connect equalities, one projected bilateral Kelvin-Voigt "
            "formulation, a four-millisecond horizon, and two time steps."
        ),
        "uncertainty_boundary": boundary,
        "competing_explanations": [
            "MuJoCo regularized equality impedance differs from direct point-force compliance",
            "one state and a four-millisecond horizon",
            "engineering rather than calibrated contact parameters",
            "two-step refinement does not establish an asymptotic order",
        ],
        "negative_controls": [
            "equality-disabled zero-force killswitch",
            "identical initial generalized state",
            "six active native constraint rows",
            "nonzero formulation discrepancy required",
            "step-halving result retained without an equivalence threshold",
        ],
        "falsifier": falsifier,
        "adjudication": (
            "The native MuJoCo constraint-and-integrator branch executes and "
            "differs from the projected formulation; no equivalence or human "
            "claim is inferred."
        ),
        "reviewer": "Codex technical audit",
        "last_verified_on": DATE,
    }


def _build_claims(
    candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    selected = {
        "design": _find(candidates, "The transport comparisons below"),
        "figure": _find(candidates, "![Native Equality Dynamics"),
        "result": _find(candidates, "Both native equalities contribute"),
        "boundary": _find(candidates, "The nonzero discrepancy is the result"),
    }
    claims = [
        _claim(
            "PD-CLAIM-302",
            [selected["design"]],
            statement=(
                "A MuJoCo-native branch uses two connect equalities, mj_step, "
                "and qfrc_constraint from the same initial state as a projected "
                "bilateral compliant-contact branch."
            ),
            classification="native_constraint_formulation_discrepancy_design",
            status="complete_for_declared_single_state_control",
            boundary=(
                "The branches do not encode equivalent constitutive laws and "
                "the contact parameters are uncalibrated engineering values."
            ),
            falsifier=(
                "The native path does not call mj_step, fewer than six native "
                "constraint rows are active, or the initial states differ."
            ),
        ),
        _claim(
            "PD-CLAIM-303",
            [selected["result"]],
            statement=(
                "The native equality and projected compliant-contact branches "
                "produce a nonzero trajectory discrepancy while the disabled-"
                "equality killswitch produces zero native constraint force."
            ),
            classification="native_constraint_formulation_discrepancy_result",
            status="supported_at_declared_state_horizon_and_steps",
            boundary=(
                "The discrepancy measures two declared formulations at one "
                "state; it is not a statistical uncertainty bound or engine-"
                "equivalence result."
            ),
            falsifier=(
                "The active native force is zero, the killswitch force is "
                "nonzero, the formulations are numerically identical, or a "
                "registered finite-value gate fails."
            ),
        ),
        _claim(
            "PD-CLAIM-304",
            [selected["boundary"]],
            statement=(
                "Native constraint-and-integrator execution does not validate "
                "either contact law, anatomical grip mechanics, engine "
                "equivalence, human transfer, or coaching strategy."
            ),
            classification="native_constraint_formulation_inference_boundary",
            status="explicitly_bounded",
            boundary=(
                "Calibrated distributed grip, articulated bilateral native "
                "contact, population uncertainty, and governed human data "
                "remain open."
            ),
            falsifier=(
                "A physical-equivalence, anatomical, human-transfer, or "
                "coaching claim is attributed to this synthetic discrepancy "
                "control alone."
            ),
        ),
    ]
    return claims, selected


def _reconcile(
    registry: dict[str, Any],
    inventory: dict[str, Any],
    claims: list[dict[str, Any]],
    selected: dict[str, dict[str, Any]],
) -> None:
    registry["claims"] = [
        claim for claim in registry["claims"] if claim["claim_id"] not in CLAIM_IDS
    ] + claims
    _reconcile_forward_transport_claims(registry, inventory["candidates"])
    _reconcile_synthesis_candidates(registry, inventory["candidates"])
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
            "rationale": (
                "This passage states or bounds the native constraint-and-"
                "integrator discrepancy control."
            ),
            "reviewer": "Codex technical audit",
            "last_verified_on": DATE,
        }
    for claim in claims:
        for candidate_id in claim["candidate_ids"]:
            reviews[candidate_id]["claim_ids"].append(claim["claim_id"])
    reviews[selected["figure"]["candidate_id"]].update(
        disposition="editorial_or_navigation",
        rationale=(
            "The figure include points to registered evidence without a "
            "standalone scientific claim."
        ),
    )
    registry["candidate_reviews"] = list(reviews.values())
    entries = {
        item["release_claim_key"]: item for item in registry["release_claim_inventory"]
    }
    entries["native_constraint_formulation_discrepancy"] = {
        "release_claim_key": "native_constraint_formulation_discrepancy",
        "published_status": "native_branch_executed_nonzero_discrepancy_retained",
        "audit_state": "reviewed_as_synthetic_formulation_discrepancy_control",
    }
    registry["release_claim_inventory"] = list(entries.values())
    registry["audit_scope"]["current_scope"] = (
        "The complete paper inventory is adjudicated. A MuJoCo-native bilateral "
        "connect and integrator control is distinguished from the shared-contact "
        "inertia-and-bias transport comparisons. Calibrated articulated contact "
        "and governed human validation remain open."
    )
    registry["paper"]["source_digest"] = inventory["source_digest"]


def main() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    claims, selected = _build_claims(inventory["candidates"])
    _reconcile(registry, inventory, claims, selected)
    REGISTRY.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
