"""Register bounded nonlinear event-reachability claims for issue #9124."""

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
REPORT = (
    "docs/research/proximal_distal_energy_transfer/data/bounded_event_reachability.json"
)
DATE = "2026-08-26"
CLAIM_IDS = {"PD-CLAIM-319", "PD-CLAIM-320"}
RELEASE_KEY = "bounded_nonlinear_event_reachability"
PUBLISHED_STATUS = "supported_for_registered_local_bounded_model_scenario"
ARTIFACTS = [
    REPORT,
    "docs/research/proximal_distal_energy_transfer/data/bounded_event_reachability.npz",
    "docs/research/proximal_distal_energy_transfer/figures/"
    "fig_bounded_event_reachability.pdf",
    "docs/research/proximal_distal_energy_transfer/BOUNDED_EVENT_REACHABILITY.md",
    "docs/research/proximal_distal_energy_transfer/"
    "MODEL_COMPLETION_FALSIFICATION_MATRIX.md",
    "scripts/research/proximal_distal_energy/bounded_event_reachability.py",
    "scripts/research/proximal_distal_energy/bounded_event_multiple_shooting.py",
    "scripts/research/proximal_distal_energy/run_bounded_event_reachability.py",
    "scripts/research/proximal_distal_energy/make_bounded_event_reachability_figure.py",
    "tests/research/test_bounded_event_reachability.py",
    "tests/research/test_bounded_event_multiple_shooting.py",
    "tests/research/test_bounded_event_reachability_study.py",
    "tests/research/test_bounded_event_reachability_evidence.py",
]


def _numeric_entry(
    literal_id: str,
    json_pointer: str,
    *,
    scale: float = 1.0,
) -> dict[str, Any]:
    return {
        "literal_id": literal_id,
        "artifact": REPORT,
        "json_pointer": json_pointer,
        "evidence_scope": "local_json_value",
        "scale": scale,
        "offset": 0.0,
        "atol": 0.0,
        "rtol": 0.0,
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
            "exact_rk4_multiple_shooting_independent_replay_continuation_"
            "killswitch_refinement_adverse_and_multistart_controls_checked"
        ),
        "adjudication_outcome": "supported",
        "source_locations": [
            f"{candidate['source_path']}:{candidate['line_start']}"
            for candidate in candidates
        ],
        "evidence_artifacts": ARTIFACTS,
        "model_domain": (
            "One synthetic analytical planar double-pendulum trajectory, one "
            "positive delivery guard, small event-tangent targets, and final "
            "event time restricted to the nominal crossing bracket."
        ),
        "uncertainty_boundary": boundary,
        "competing_explanations": [
            "local optimizer convergence and nonunique minima",
            "shooting-mesh resolution",
            "integration timestep",
            "event-time shift and alternative crossing topology",
            "state and control scaling",
            "scenario rather than measured actuator bounds",
            "delay, fatigue, noise, contact, and structural model discrepancy",
        ],
        "negative_controls": controls,
        "falsifier": falsifier,
        "adjudication": (
            "The result is retained only as bounded local model-scenario "
            "feasibility. The failed multistart optimality gate suppresses "
            "channel, controller, and effort rankings. It does not establish "
            "global reachability, human capacity or strategy, passive torque, "
            "participant behavior, or coaching guidance."
        ),
        "reviewer": "Codex technical audit",
        "last_verified_on": DATE,
        "numeric_evidence": numeric_evidence,
    }


def _feasibility_claim(candidates: dict[str, dict[str, Any]]) -> dict[str, Any]:
    statement = (
        "For the registered same-bracket model scenario, 32 of 38 exact-RK4 "
        "replays are feasible; the 6 infeasible cases are exactly displaced "
        "targets under zero authority, every enabled-channel case reaches its "
        "target, and the maximum feasible tangent residual is 8.82244e-11."
    )
    return _claim(
        claim_id="PD-CLAIM-319",
        candidates=[
            candidates["formulation"],
            candidates["constraints"],
            candidates["matrix"],
            candidates["abstract"],
        ],
        statement=statement,
        classification="bounded_local_event_reachability_feasibility",
        boundary=(
            "The registered offsets are small, the event is confined to one "
            "nominal bracket, and the bounds are model scenarios rather than "
            "human measurements."
        ),
        falsifier=(
            "Any enabled-channel target fails independent replay, any displaced "
            "zero-authority target becomes feasible, event topology becomes "
            "ineligible, or mesh, step, or adverse-state controls fail."
        ),
        controls=[
            "six displaced zero-authority killswitch cases",
            "zero-authority nominal feasibility control",
            "three shooting meshes",
            "three integration timesteps",
            "two adverse initial states",
            "independent exact-RK4 replay",
        ],
        numeric_evidence=[
            _numeric_entry(
                "32#1", "/outcome_counts/replay_feasibility_status/feasible"
            ),
            _numeric_entry("38#1", "/outcome_counts/event_status/transverse"),
            _numeric_entry(
                "6#1", "/outcome_counts/replay_feasibility_status/infeasible"
            ),
            _numeric_entry(
                "8.82244e-11#1",
                "/continuation_trials/10/replay_tangent_residual",
            ),
        ],
    )


def _optimality_claim(candidates: dict[str, dict[str, Any]]) -> dict[str, Any]:
    statement = (
        "Both multistarts converge, but their scaled objectives of 2.05356e-8 "
        "and 2.56595e-8 differ by 24.9517%, exceeding the 5% gate; optimality "
        "evidence is inadequate and channel or controller rankings remain "
        "unavailable."
    )
    return _claim(
        claim_id="PD-CLAIM-320",
        candidates=[
            candidates["optimality"],
            candidates["boundary"],
            candidates["abstract"],
        ],
        statement=statement,
        classification="bounded_event_optimality_gate_failure",
        boundary=(
            "Independent feasibility survives; minimum effort, controller "
            "superiority, and biological allocation do not."
        ),
        falsifier=(
            "A preregistered multistart/global solve brings all retained "
            "objectives within the 5% gate and independently reproduces any "
            "proposed ranking under matched tasks and bounds."
        ),
        controls=[
            "zero-control initial guess",
            "deterministic bounded-walk initial guess",
            "independent replay of both converged solutions",
            "preregistered 5% relative objective-spread gate",
        ],
        numeric_evidence=[
            _numeric_entry(
                "2.05356e-8#1",
                "/falsification_controls/multistart/0/objective",
            ),
            _numeric_entry(
                "2.56595e-8#1",
                "/falsification_controls/multistart/1/objective",
            ),
            _numeric_entry(
                "24.9517#1",
                "/qualification/multistart_relative_objective_spread",
                scale=100.0,
            ),
            _numeric_entry(
                "5#1",
                "/qualification/multistart_spread_gate",
                scale=100.0,
            ),
        ],
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
        "abstract": unique("bounded nonlinear follow-up then solves"),
        "formulation": unique("next registered test replaces infinitesimal authority"),
        "constraints": unique("subject to exact step continuity"),
        "matrix": unique("continuation matrix crosses seven symmetric tangent targets"),
        "optimality": unique("Feasibility does not qualify an optimum"),
        "boundary": unique("result is restricted to one synthetic planar trajectory"),
        "figure": unique("fig_bounded_event_reachability.pdf"),
        "appendix": unique(
            "`data/bounded_event_reachability.{json,npz}`",
            suffix="_appendices.qmd",
        ),
    }


def _update_numeric_contracts(claims: tuple[dict[str, Any], ...]) -> None:
    contracts = json.loads(NUMERIC_CONTRACTS.read_text(encoding="utf-8"))
    contracts["claims"] = [
        contract
        for contract in contracts["claims"]
        if contract["claim_id"] not in CLAIM_IDS
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


def _append_candidate_reviews(
    registry: dict[str, Any],
    inventory: dict[str, Any],
    candidates: dict[str, dict[str, Any]],
) -> None:
    """Replace the bounded-study reviews while preserving unrelated reviews."""

    current_ids = {candidate["candidate_id"] for candidate in inventory["candidates"]}
    selected_ids = {candidate["candidate_id"] for candidate in candidates.values()}
    registry["candidate_reviews"] = [
        review
        for review in registry["candidate_reviews"]
        if review["candidate_id"] in current_ids
        and review["candidate_id"] not in selected_ids
        and not set(review["claim_ids"]).intersection(CLAIM_IDS)
    ]
    assignments = {
        candidates["abstract"]["candidate_id"]: sorted(CLAIM_IDS),
        candidates["formulation"]["candidate_id"]: ["PD-CLAIM-319"],
        candidates["constraints"]["candidate_id"]: ["PD-CLAIM-319"],
        candidates["matrix"]["candidate_id"]: ["PD-CLAIM-319"],
        candidates["optimality"]["candidate_id"]: ["PD-CLAIM-320"],
        candidates["boundary"]["candidate_id"]: ["PD-CLAIM-320"],
    }
    registry["candidate_reviews"].extend(
        {
            "candidate_id": candidate_id,
            "disposition": "material_claims_mapped",
            "claim_ids": claim_ids,
            "rationale": "This passage states or bounds the registered bounded-event result.",
            "reviewer": "Codex technical audit",
            "last_verified_on": DATE,
        }
        for candidate_id, claim_ids in assignments.items()
    )
    for name, rationale in (
        ("figure", "This candidate is the governed figure caption."),
        ("appendix", "This appendix paragraph inventories governed artifacts."),
    ):
        registry["candidate_reviews"].append(
            {
                "candidate_id": candidates[name]["candidate_id"],
                "disposition": "editorial_or_navigation",
                "claim_ids": [],
                "rationale": rationale,
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
                "This deterministic census paragraph summarizes the governed "
                "claim registry."
            ),
            "reviewer": "Codex technical audit",
            "last_verified_on": DATE,
        }
        for candidate in inventory["candidates"]
        if candidate["source_path"].endswith("_claim_adjudication_summary.qmd")
        and candidate["candidate_id"] not in reviewed_ids
    )


def _update_release_audit(registry: dict[str, Any], inventory: dict[str, Any]) -> None:
    """Register the bounded result and its failed optimality qualification."""

    registry["release_claim_inventory"] = [
        item
        for item in registry["release_claim_inventory"]
        if item["release_claim_key"] != RELEASE_KEY
    ]
    registry["release_claim_inventory"].append(
        {
            "release_claim_key": RELEASE_KEY,
            "published_status": PUBLISHED_STATUS,
            "audit_state": (
                "reviewed_as_local_bounded_feasibility_with_failed_optimality_gate"
            ),
        }
    )
    registry["paper"]["source_digest"] = inventory["source_digest"]
    registry["audit_scope"]["completion_status"] = "complete"
    registry["audit_scope"]["current_scope"] = (
        f"The complete {inventory['candidate_count']}-candidate paper inventory is "
        "adjudicated, including bounded nonlinear event-reachability evidence."
    )


def main() -> None:
    """Apply the frozen #9124 candidate review idempotently."""

    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    candidates = _find_candidates(inventory)
    claims = (_feasibility_claim(candidates), _optimality_claim(candidates))
    registry["claims"] = [
        claim for claim in registry["claims"] if claim["claim_id"] not in CLAIM_IDS
    ]
    registry["claims"].extend(claims)
    _append_candidate_reviews(registry, inventory, candidates)
    _update_release_audit(registry, inventory)
    by_id = {
        candidate["candidate_id"]: candidate for candidate in inventory["candidates"]
    }
    claims_by_id = {claim["claim_id"]: claim for claim in registry["claims"]}
    _reconcile_reciprocal_claim_reviews(registry, by_id, claims_by_id)
    REGISTRY.write_text(
        json.dumps(registry, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    _update_numeric_contracts(claims)


if __name__ == "__main__":
    main()
