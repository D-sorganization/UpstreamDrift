"""Register numerical-prerequisite claims for issue #9126."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.research.proximal_distal_energy.register_remaining_claim_reviews import (
    _reconcile_reciprocal_claim_reviews,
)

ROOT = Path(__file__).resolve().parents[3]
ARTICLE = ROOT / "docs/research/proximal_distal_energy_transfer"
REGISTRY = ARTICLE / "data/claim_audit_registry.json"
INVENTORY = ARTICLE / "data/claim_candidate_inventory.json"
NUMERIC_CONTRACTS = ARTICLE / "data/claim_numeric_contracts.json"
REGISTRATION = (
    "docs/research/proximal_distal_energy_transfer/data/"
    "nonlinear_controller_comparison_registration.json"
)
QUALIFICATION = (
    "docs/research/proximal_distal_energy_transfer/data/"
    "nonlinear_controller_solver_qualification.json"
)
TRANSPORT = (
    "docs/research/proximal_distal_energy_transfer/data/"
    "nonlinear_controller_plant_transport.json"
)
DATE = "2026-08-26"
CLAIM_IDS = {"PD-CLAIM-324", "PD-CLAIM-325", "PD-CLAIM-326"}
RELEASE_KEY = "nonlinear_controller_numerical_qualification"
PUBLISHED_STATUS = "supported_as_registered_numerical_prerequisite_without_evaluation"
ARTIFACTS = [
    REGISTRATION,
    QUALIFICATION,
    TRANSPORT,
    "docs/research/proximal_distal_energy_transfer/"
    "NONLINEAR_CONTROLLER_COMPARISON_REGISTRATION.md",
    "docs/research/proximal_distal_energy_transfer/"
    "NONLINEAR_CONTROLLER_SOLVER_QUALIFICATION.md",
    "docs/research/proximal_distal_energy_transfer/"
    "NONLINEAR_CONTROLLER_PLANT_TRANSPORT.md",
    "scripts/research/proximal_distal_energy/nonlinear_controller_registration.py",
    "scripts/research/proximal_distal_energy/nonlinear_controller_numerics.py",
    "scripts/research/proximal_distal_energy/nonlinear_controller_kernels.py",
    "scripts/research/proximal_distal_energy/nonlinear_controller_qualification.py",
    "scripts/research/proximal_distal_energy/nonlinear_controller_plant_transport.py",
    "tests/research/test_nonlinear_controller_comparison_registration.py",
    "tests/research/test_nonlinear_controller_solver_qualification.py",
    "tests/research/test_nonlinear_controller_plant_transport.py",
]


def _numeric(
    literal_id: str,
    artifact: str,
    pointer: str,
    *,
    rtol: float = 0.0,
) -> dict[str, Any]:
    return {
        "literal_id": literal_id,
        "artifact": artifact,
        "json_pointer": pointer,
        "evidence_scope": "local_json_value",
        "scale": 1.0,
        "offset": 0.0,
        "atol": 0.0,
        "rtol": rtol,
    }


def _claim(
    *,
    claim_id: str,
    candidates: list[dict[str, Any]],
    statement: str,
    classification: str,
    boundary: str,
    falsifier: str,
    controls: list[str],
    numeric_evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "claim_id": claim_id,
        "candidate_ids": [candidate["candidate_id"] for candidate in candidates],
        "statement": statement,
        "classification": classification,
        "published_status": PUBLISHED_STATUS,
        "audit_status": (
            "registration_manufactured_solver_and_shared_equation_plant_"
            "transport_recomputed"
        ),
        "adjudication_outcome": "supported",
        "source_locations": [
            f"{candidate['source_path']}:{candidate['line_start']}"
            for candidate in candidates
        ],
        "evidence_artifacts": ARTIFACTS,
        "model_domain": (
            "One prospective analytical double-pendulum comparison, one "
            "manufactured damped-pendulum solver fixture, and shared-equation "
            "RK4 transport to the canonical ODE backend."
        ),
        "uncertainty_boundary": boundary,
        "competing_explanations": [
            "shared equations rather than independent physics validation",
            "manufactured numerical behavior rather than golf performance",
            "finite trial registration rather than global optimality",
            "solver convergence rather than model adequacy",
            "synthetic coordinate torques rather than human actuation",
        ],
        "negative_controls": controls,
        "falsifier": falsifier,
        "adjudication": (
            "The result is retained only as a numerical prerequisite. No "
            "controller comparison, human inference, passive-torque claim, or "
            "coaching recommendation is released."
        ),
        "reviewer": "Codex technical audit",
        "last_verified_on": DATE,
        "numeric_evidence": numeric_evidence,
    }


def _claims(candidates: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    boundary = candidates["boundary"]
    return (
        _claim(
            claim_id="PD-CLAIM-324",
            candidates=[candidates["registration"], boundary],
            statement=(
                "The prospective registration declares 9 controller families, "
                "24 evaluation trials, 8 disjoint tuning trials, 0 controller "
                "evaluations, and 0 ranking-eligible methods."
            ),
            classification="prospective_outcome_blind_controller_registration",
            boundary=(
                "The finite synthetic registration is not controller performance, "
                "global optimality, human control, anatomy, fatigue, or coaching evidence."
            ),
            falsifier=(
                "The deterministic report changes, tuning and evaluation overlap, "
                "a parent digest drifts, or any method becomes ranking-eligible."
            ),
            controls=[
                "three current parent digests",
                "disjoint outcome-blind split",
                "typed failure taxonomy",
                "single-worker checkpoint identity",
            ],
            numeric_evidence=[
                _numeric("9#1", REGISTRATION, "/controller_family_count"),
                _numeric("24#1", REGISTRATION, "/evaluation_trial_count"),
                _numeric("8#1", REGISTRATION, "/tuning_trial_count"),
                _numeric("0#1", REGISTRATION, "/controller_evaluation_count"),
                _numeric("0#2", REGISTRATION, "/ranking_eligible_method_count"),
            ],
        ),
        _claim(
            claim_id="PD-CLAIM-325",
            candidates=[candidates["solver"], boundary],
            statement=(
                "The manufactured fixture qualifies 1 solver kernel with a "
                "maximum directional-derivative discrepancy of "
                "1.0477729794899915e-11, maximum bound violation 0.0, "
                "0 double-pendulum evaluations, and 0 ranking-eligible methods."
            ),
            classification="manufactured_nonlinear_solver_mechanics_qualification",
            boundary=(
                "Manufactured solver mechanics are not golf performance, model "
                "adequacy, human control, passive torque, or coaching evidence."
            ),
            falsifier=(
                "Derivatives, bounds, accepted-cost descent, replay, cold/warm "
                "sensitivity, typed failure, or zero-evaluation gates fail."
            ),
            controls=[
                "independent directional derivative",
                "native or in-rollout bounds",
                "exact replay",
                "typed nonfinite dynamics failure",
            ],
            numeric_evidence=[
                _numeric("1#1", QUALIFICATION, "/qualified_solver_count"),
                _numeric(
                    "1.0477729794899915e-11#1",
                    QUALIFICATION,
                    "/directional_derivative_max_abs_error",
                ),
                _numeric("0.0#1", QUALIFICATION, "/solvers/0/maximum_bound_violation"),
                _numeric("0#1", QUALIFICATION, "/double_pendulum_evaluation_count"),
                _numeric("0#2", QUALIFICATION, "/ranking_eligible_method_count"),
            ],
        ),
        _claim(
            claim_id="PD-CLAIM-326",
            candidates=[candidates["transport"], boundary],
            statement=(
                "The plant-transport report contains 12 parity cases across 3 "
                "step sizes and 4 invalid-input controls, with maximum state "
                "parity error 0.0, 0 controller evaluations, and 0 "
                "ranking-eligible methods."
            ),
            classification="shared_equation_controller_plant_transport_parity",
            boundary=(
                "Shared-equation code-path parity is not independent physics "
                "validation, controller performance, human evidence, or coaching authority."
            ),
            falsifier=(
                "Any step differs from the canonical ODE backend beyond the gate, "
                "replay or input immutability fails, or an invalid input emits a trajectory."
            ),
            controls=[
                "0.5, 1, and 2 ms steps",
                "four state and torque cases",
                "deterministic replay and input immutability",
                "wrong-size and nonfinite typed failures",
            ],
            numeric_evidence=[
                _numeric("12#1", TRANSPORT, "/parity_case_count"),
                _numeric("3#1", TRANSPORT, "/step_size_count"),
                _numeric("4#1", TRANSPORT, "/invalid_input_case_count"),
                _numeric("0.0#1", TRANSPORT, "/maximum_state_parity_error"),
                _numeric("0#1", TRANSPORT, "/controller_evaluation_count"),
                _numeric("0#2", TRANSPORT, "/ranking_eligible_method_count"),
            ],
        ),
    )


def _find_candidates(inventory: dict[str, Any]) -> dict[str, dict[str, Any]]:
    def unique(fragment: str, *, suffix: str | None = None) -> dict[str, Any]:
        matches = [
            candidate
            for candidate in inventory["candidates"]
            if fragment in candidate["text"]
            and (suffix is None or candidate["source_path"].endswith(suffix))
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one candidate containing {fragment!r}")
        return matches[0]

    return {
        "registration": unique("next step freezes a matched nonlinear-controller"),
        "solver": unique("Before exposing the registered double-pendulum"),
        "transport": unique("A separate transport test then applies"),
        "boundary": unique("These artifacts qualify registration"),
        "appendix": unique(
            "`data/nonlinear_controller_comparison_registration.json`",
            suffix="_appendices.qmd",
        ),
    }


def _write_numeric_contracts(claims: tuple[dict[str, Any], ...]) -> None:
    contracts = json.loads(NUMERIC_CONTRACTS.read_text(encoding="utf-8"))
    contracts["claims"] = [
        item for item in contracts["claims"] if item["claim_id"] not in CLAIM_IDS
    ]
    contracts["claims"].extend(
        {
            "claim_id": claim["claim_id"],
            "statement_sha256": hashlib.sha256(
                claim["statement"].encode("utf-8")
            ).hexdigest(),
            "numeric_evidence": claim["numeric_evidence"],
        }
        for claim in claims
    )
    NUMERIC_CONTRACTS.write_text(
        json.dumps(contracts, indent=2) + "\n", encoding="utf-8", newline="\n"
    )


def _replace_candidate_reviews(
    registry: dict[str, Any],
    inventory: dict[str, Any],
    candidates: dict[str, dict[str, Any]],
    claims: tuple[dict[str, Any], ...],
) -> None:
    current_ids = {item["candidate_id"] for item in inventory["candidates"]}
    selected_ids = {item["candidate_id"] for item in candidates.values()}
    registry["candidate_reviews"] = [
        review
        for review in registry["candidate_reviews"]
        if review["candidate_id"] in current_ids
        and review["candidate_id"] not in selected_ids
        and not set(review["claim_ids"]).intersection(CLAIM_IDS)
    ]
    assignments: dict[str, list[str]] = {}
    for claim in claims:
        for candidate_id in claim["candidate_ids"]:
            assignments.setdefault(candidate_id, []).append(claim["claim_id"])
    for candidate_id, claim_ids in assignments.items():
        registry["candidate_reviews"].append(
            {
                "candidate_id": candidate_id,
                "disposition": "material_claims_mapped",
                "claim_ids": sorted(claim_ids),
                "rationale": (
                    "This passage states or bounds the registered numerical prerequisite."
                ),
                "reviewer": "Codex technical audit",
                "last_verified_on": DATE,
            }
        )
    registry["candidate_reviews"].append(
        {
            "candidate_id": candidates["appendix"]["candidate_id"],
            "disposition": "editorial_or_navigation",
            "claim_ids": [],
            "rationale": "This appendix candidate inventories governed artifacts.",
            "reviewer": "Codex technical audit",
            "last_verified_on": DATE,
        }
    )
    reviewed_ids = {review["candidate_id"] for review in registry["candidate_reviews"]}
    registry["candidate_reviews"].extend(
        {
            "candidate_id": candidate["candidate_id"],
            "disposition": "editorial_or_navigation",
            "claim_ids": [],
            "rationale": (
                "This deterministic census summarizes the governed claim registry."
            ),
            "reviewer": "Codex technical audit",
            "last_verified_on": DATE,
        }
        for candidate in inventory["candidates"]
        if candidate["source_path"].endswith("_claim_adjudication_summary.qmd")
        and candidate["candidate_id"] not in reviewed_ids
    )


def main() -> None:
    """Apply the #9126 claim review idempotently."""
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    candidates = _find_candidates(inventory)
    claims = _claims(candidates)
    registry["claims"] = [
        claim for claim in registry["claims"] if claim["claim_id"] not in CLAIM_IDS
    ]
    registry["claims"].extend(claims)
    _replace_candidate_reviews(registry, inventory, candidates, claims)
    registry["release_claim_inventory"] = [
        item
        for item in registry["release_claim_inventory"]
        if item["release_claim_key"] != RELEASE_KEY
    ]
    registry["release_claim_inventory"].append(
        {
            "release_claim_key": RELEASE_KEY,
            "published_status": PUBLISHED_STATUS,
            "audit_state": "reviewed_as_numerical_prerequisite_without_ranking",
        }
    )
    registry["paper"]["source_digest"] = inventory["source_digest"]
    registry["audit_scope"]["completion_status"] = "complete"
    registry["audit_scope"]["current_scope"] = (
        f"The complete {inventory['candidate_count']}-candidate paper inventory is "
        "adjudicated, including nonlinear-controller numerical prerequisites."
    )
    by_id = {item["candidate_id"]: item for item in inventory["candidates"]}
    claims_by_id = {claim["claim_id"]: claim for claim in registry["claims"]}
    _reconcile_reciprocal_claim_reviews(registry, by_id, claims_by_id)
    REGISTRY.write_text(
        json.dumps(registry, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    _write_numeric_contracts(claims)


if __name__ == "__main__":
    main()
