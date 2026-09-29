"""Register trajectory-varying event-control claims for issue #9123."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
ARTICLE = ROOT / "docs/research/proximal_distal_energy_transfer"
REGISTRY = ARTICLE / "data/claim_audit_registry.json"
INVENTORY = ARTICLE / "data/claim_candidate_inventory.json"
NUMERIC_CONTRACTS = ARTICLE / "data/claim_numeric_contracts.json"
REPORT = (
    "docs/research/proximal_distal_energy_transfer/data/"
    "trajectory_control_authority.json"
)
DATE = "2026-08-26"
CLAIM_IDS = {"PD-CLAIM-317", "PD-CLAIM-318"}
RELEASE_KEY = "trajectory_varying_event_control_authority"
PUBLISHED_STATUS = "supported_for_declared_local_first_order_analytical_trajectory"
ARTIFACTS = [
    REPORT,
    "docs/research/proximal_distal_energy_transfer/data/"
    "trajectory_control_authority.npz",
    "docs/research/proximal_distal_energy_transfer/figures/"
    "fig_trajectory_control_authority.pdf",
    "docs/research/proximal_distal_energy_transfer/TRAJECTORY_CONTROL_AUTHORITY.md",
    "docs/research/proximal_distal_energy_transfer/"
    "MODEL_COMPLETION_FALSIFICATION_MATRIX.md",
    "scripts/research/proximal_distal_energy/trajectory_control_authority.py",
    "scripts/research/proximal_distal_energy/run_trajectory_control_authority.py",
    "scripts/research/proximal_distal_energy/"
    "make_trajectory_control_authority_figure.py",
    "tests/research/test_trajectory_control_authority.py",
    "tests/research/test_trajectory_control_authority_evidence.py",
]


def _numeric_entry(literal_id: str, json_pointer: str) -> dict[str, Any]:
    return {
        "literal_id": literal_id,
        "artifact": REPORT,
        "json_pointer": json_pointer,
        "evidence_scope": "local_json_value",
        "scale": 1.0,
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
            "exact_step_event_projection_channel_direct_pulse_refinement_unit_"
            "and_frozen_local_controls_checked"
        ),
        "adjudication_outcome": "supported",
        "source_locations": [
            f"{candidate['source_path']}:{candidate['line_start']}"
            for candidate in candidates
        ],
        "evidence_artifacts": ARTIFACTS,
        "model_domain": (
            "One synthetic open-loop analytical double-pendulum downswing, "
            "linearized along the registered exact RK4 trajectory through the "
            "first positive club-vertical crossing."
        ),
        "uncertainty_boundary": boundary,
        "competing_explanations": [
            "state and control scaling",
            "finite-difference step",
            "integration timestep",
            "event-time projection",
            "frozen operating-point approximation",
            "nonlinear saturation and actuator bounds",
            "open-loop program and structural model discrepancy",
        ],
        "negative_controls": controls,
        "falsifier": falsifier,
        "adjudication": (
            "The result is retained as local first-order analytical evidence. "
            "It does not establish bounded nonlinear reachability, controller "
            "superiority, human capacity or strategy, passive torque, robustness, "
            "or coaching guidance."
        ),
        "reviewer": "Codex technical audit",
        "last_verified_on": DATE,
        "numeric_evidence": numeric_evidence,
    }


def _authority_claim(candidates: dict[str, dict[str, Any]]) -> dict[str, Any]:
    statement = (
        "At the registered 35.0258 per second transverse event, the exact-step "
        "trajectory-varying Gramian has fixed-time rank 4 and event-tangent rank "
        "3; channel additivity closes within 5.17275e-12, six direct nonlinear "
        "pulses within 4.11321e-8, input-step refinement within 4.11814e-10, "
        "integration-step refinement within 1.26519e-7, and equivalent units "
        "within 3.33067e-16."
    )
    return _claim(
        claim_id="PD-CLAIM-317",
        candidates=[
            candidates["input_map"],
            candidates["normalization"],
            candidates["authority"],
            candidates["controls"],
        ],
        statement=statement,
        classification="local_trajectory_varying_event_conditioned_authority",
        boundary=(
            "Ranks and spectra depend on the declared scales, tolerance, model, "
            "trajectory, and event. They are not effort, strength, or strategy "
            "rankings."
        ),
        falsifier=(
            "The exact event loses transversality, zero input is nonzero, channel "
            "additivity or direct nonlinear pulses exceed their gates, or step and "
            "equivalent-unit controls do not reproduce."
        ),
        controls=[
            "zero-input Gramian killswitch",
            "shoulder-plus-wrist channel additivity",
            "six direct nonlinear torque pulses",
            "three input differentiation steps",
            "three integration timesteps",
            "equivalent torque and state units",
        ],
        numeric_evidence=[
            _numeric_entry(
                "35.0258#1", "/event_conditioned_authority/transversality_per_s"
            ),
            _numeric_entry("4#1", "/event_conditioned_authority/full_state/rank"),
            _numeric_entry("3#1", "/event_conditioned_authority/event_tangent/rank"),
            _numeric_entry(
                "5.17275e-12#1",
                "/falsification_controls/channel_additivity/maximum_abs_residual",
            ),
            _numeric_entry(
                "4.11321e-8#1",
                "/falsification_controls/direct_pulses/1/maximum_abs_residual",
            ),
            _numeric_entry(
                "4.11814e-10#1",
                "/falsification_controls/input_step_refinement/2/"
                "relative_event_gramian_residual",
            ),
            _numeric_entry(
                "1.26519e-7#1",
                "/falsification_controls/integration_step_refinement/0/"
                "relative_event_gramian_residual",
            ),
            _numeric_entry(
                "3.33067e-16#1",
                "/falsification_controls/equivalent_units/maximum_abs_residual",
            ),
        ],
    )


def _countermodel_claim(candidates: dict[str, dict[str, Any]]) -> dict[str, Any]:
    statement = (
        "Across four matched phase windows, the frozen-local and "
        "trajectory-varying Gramians differ by relative norms from 0.136042 to "
        "0.297782, so the frozen operating-point countermodel does not reproduce "
        "the registered finite-window authority."
    )
    return _claim(
        claim_id="PD-CLAIM-318",
        candidates=[candidates["countermodel"], candidates["boundary"]],
        statement=statement,
        classification="frozen_local_countermodel_divergence",
        boundary=(
            "This rejects equivalence only for four matched windows on the "
            "registered trajectory. It is not a global nonlinear reachability or "
            "controller-ranking result."
        ),
        falsifier=(
            "Matched-window recomputation makes every frozen-local difference "
            "numerically negligible, or the trajectory-varying result fails its "
            "direct-pulse and refinement gates."
        ),
        controls=[
            "four identical phase and horizon windows",
            "same state and control scales",
            "same continuous-energy input normalization",
            "trajectory-varying direct-pulse qualification",
        ],
        numeric_evidence=[
            _numeric_entry(
                "0.136042#1",
                "/frozen_local_countermodel/1/relative_gramian_difference",
            ),
            _numeric_entry(
                "0.297782#1",
                "/frozen_local_countermodel/2/relative_gramian_difference",
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
        "input_map": unique("finite-time transition map describes sensitivity"),
        "normalization": unique("continuous-energy-equivalent torque perturbation"),
        "authority": unique("two-channel Gramian has rank four"),
        "controls": unique("result passes four registered falsification controls"),
        "countermodel": unique("frozen-local countermodel evaluated"),
        "boundary": unique("These are local, scale-dependent first-order"),
        "figure": unique("fig_trajectory_control_authority.pdf"),
        "appendix": unique(
            "`data/trajectory_control_authority.{json,npz}`",
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


def main() -> None:
    """Apply the frozen #9123 candidate review idempotently."""

    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    candidates = _find_candidates(inventory)
    claims = (_authority_claim(candidates), _countermodel_claim(candidates))
    current_ids = {candidate["candidate_id"] for candidate in inventory["candidates"]}
    selected_ids = {candidate["candidate_id"] for candidate in candidates.values()}
    registry["claims"] = [
        claim for claim in registry["claims"] if claim["claim_id"] not in CLAIM_IDS
    ]
    registry["claims"].extend(claims)
    registry["candidate_reviews"] = [
        review
        for review in registry["candidate_reviews"]
        if review["candidate_id"] in current_ids
        and review["candidate_id"] not in selected_ids
        and not set(review["claim_ids"]).intersection(CLAIM_IDS)
    ]
    assignments: dict[str, list[str]] = {}
    for name, claim_id in (
        ("input_map", "PD-CLAIM-317"),
        ("normalization", "PD-CLAIM-317"),
        ("authority", "PD-CLAIM-317"),
        ("controls", "PD-CLAIM-317"),
        ("countermodel", "PD-CLAIM-318"),
        ("boundary", "PD-CLAIM-318"),
    ):
        candidate_claims = assignments.setdefault(candidates[name]["candidate_id"], [])
        if claim_id not in candidate_claims:
            candidate_claims.append(claim_id)
    registry["candidate_reviews"].extend(
        {
            "candidate_id": candidate_id,
            "disposition": "material_claims_mapped",
            "claim_ids": claim_ids,
            "rationale": (
                "This passage states or bounds the trajectory-control qualification."
            ),
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
                "reviewed_as_local_trajectory_varying_event_conditioned_authority"
            ),
        }
    )
    registry["paper"]["source_digest"] = inventory["source_digest"]
    registry["audit_scope"]["completion_status"] = "complete"
    registry["audit_scope"]["current_scope"] = (
        f"The complete {inventory['candidate_count']}-candidate paper inventory is "
        "adjudicated, including trajectory-varying event-control evidence."
    )
    REGISTRY.write_text(
        json.dumps(registry, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    _update_numeric_contracts(claims)


if __name__ == "__main__":
    main()
