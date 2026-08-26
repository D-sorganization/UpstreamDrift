"""Register finite-time and event-sensitivity claims for issue #9116."""

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
REPORT = "docs/research/proximal_distal_energy_transfer/data/phase_event_stability.json"
DATE = "2026-08-26"
CLAIM_IDS = {"PD-CLAIM-315", "PD-CLAIM-316"}
RELEASE_KEY = "phase_event_finite_time_stability"
ARTIFACTS = [
    REPORT,
    "docs/research/proximal_distal_energy_transfer/data/phase_event_stability.npz",
    "docs/research/proximal_distal_energy_transfer/figures/fig_phase_event_stability.pdf",
    "docs/research/proximal_distal_energy_transfer/PHASE_EVENT_STABILITY.md",
    "docs/research/proximal_distal_energy_transfer/MODEL_COMPLETION_FALSIFICATION_MATRIX.md",
    "scripts/research/proximal_distal_energy/phase_event_stability.py",
    "scripts/research/proximal_distal_energy/run_phase_event_stability.py",
    "scripts/research/proximal_distal_energy/make_phase_event_stability_figure.py",
    "tests/research/test_phase_event_stability.py",
    "tests/research/test_phase_event_stability_evidence.py",
]


def _numeric_entry(
    literal_id: str,
    json_pointer: str,
    *,
    scale: float = 1.0,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> dict[str, Any]:
    return {
        "literal_id": literal_id,
        "artifact": REPORT,
        "json_pointer": json_pointer,
        "evidence_scope": "local_json_value",
        "scale": scale,
        "offset": 0.0,
        "atol": atol,
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
        "published_status": (
            "supported_for_declared_local_nonperiodic_analytical_trajectory"
        ),
        "audit_status": (
            "nondimensional_transition_step_direct_event_grazing_saltation_unit_"
            "and_periodicity_controls_checked"
        ),
        "adjudication_outcome": "supported",
        "source_locations": [
            f"{candidate['source_path']}:{candidate['line_start']}"
            for candidate in candidates
        ],
        "evidence_artifacts": ARTIFACTS,
        "model_domain": (
            "One synthetic open-loop analytical double-pendulum downswing, "
            "integrated with the registered RK4 equations through the first "
            "positive club-vertical crossing."
        ),
        "uncertainty_boundary": boundary,
        "competing_explanations": [
            "state-coordinate scaling",
            "finite-difference step",
            "integration timestep",
            "event interpolation",
            "guard transversality",
            "open-loop program and initial state",
        ],
        "negative_controls": controls,
        "falsifier": falsifier,
        "adjudication": (
            "The result is retained as local finite-window analytical evidence. "
            "It does not establish asymptotic or global stability, neural timing "
            "demand, participant robustness, passive torque, human strategy, or "
            "coaching guidance."
        ),
        "reviewer": "Codex technical audit",
        "last_verified_on": DATE,
        "numeric_evidence": numeric_evidence,
    }


def _claims(candidates: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    finite_time_statement = (
        "On the registered analytical downswing refined to 0.125 ms, state-step "
        "multipliers 0.1 through 10 change the event transition by at most "
        "6.87737e-6, and three complete perturbation rollouts agree with the "
        "propagated fixed-horizon map within 3.45969e-7; at the 0.349256 s "
        "geometric event, the observed scaled gains span 0.093456 to 8.33244."
    )
    event_statement = (
        "The registered club-vertical guard is transverse at 35.0258 per second, "
        "with scaled-state event-time derivatives (-0.137655, -0.0240043, "
        "-0.299636, -0.243972) s and direct-rollout discrepancy no greater than "
        "5.98012e-5 s; a 0.95 reset corruption produces 0.05 saltation deviation, "
        "while the 1.48546 scaled periodicity residual exceeds 1e-6 and suppresses "
        "Floquet output."
    )
    finite_time = _claim(
        claim_id="PD-CLAIM-315",
        candidates=[candidates["transition"], candidates["finite_time"]],
        statement=finite_time_statement,
        classification="local_finite_time_state_transition_amplification",
        boundary=(
            "The gain spectrum is local to one scaled finite trajectory and is "
            "not an asymptotic, global, population, or coaching stability result."
        ),
        falsifier=(
            "Step refinement or direct perturbed rollouts exceed their registered "
            "residual gates, equivalent units change the scaled map, or the "
            "registered gain spectrum does not reproduce."
        ),
        controls=[
            "three central-difference step multipliers",
            "three complete symmetric perturbation scales",
            "equivalent radian/degree state coordinates",
            "equivalent second/millisecond exponent coordinates",
        ],
        numeric_evidence=[
            _numeric_entry("0.125#1", "/registration/analysis_dt_s", scale=1000.0),
            _numeric_entry("0.1#1", "/step_refinement/0/state_step_multiplier"),
            _numeric_entry("10#1", "/step_refinement/2/state_step_multiplier"),
            _numeric_entry(
                "6.87737e-6#1",
                "/step_refinement/0/event_transition_max_abs_residual_from_nominal",
            ),
            _numeric_entry(
                "3.45969e-7#1", "/direct_transition_controls/0/maximum_abs_residual"
            ),
            _numeric_entry("0.349256#1", "/reference_event/time_s"),
            _numeric_entry(
                "0.093456#1", "/finite_time_analysis/minimum_observed_amplification"
            ),
            _numeric_entry(
                "8.33244#1", "/finite_time_analysis/maximum_observed_amplification"
            ),
        ],
    )
    event = _claim(
        claim_id="PD-CLAIM-316",
        candidates=[
            candidates["implicit"],
            candidates["event"],
            candidates["periodicity"],
        ],
        statement=event_statement,
        classification="transverse_event_sensitivity_and_floquet_suppression",
        boundary=(
            "The derivative applies only to the declared transverse geometric "
            "guard; it is not neural timing demand, impact validation, or human "
            "robustness. No periodic orbit is registered."
        ),
        falsifier=(
            "Implicit and direct derivatives disagree beyond tolerance, the "
            "near-grazing or reset controls fail, or Floquet output is emitted "
            "without state and mode closure."
        ),
        controls=[
            "three direct event perturbation scales",
            "constructed near-grazing guard",
            "time-only identity-reset saltation",
            "corrupted-reset saltation killswitch",
            "scaled state-closure periodicity gate",
        ],
        numeric_evidence=[
            _numeric_entry(
                "35.0258#1", "/event_time_sensitivity/implicit/transversality_per_s"
            ),
            _numeric_entry(
                "-0.137655#1",
                "/event_time_sensitivity/implicit/derivative_s_per_scaled_state/0",
            ),
            _numeric_entry(
                "-0.0240043#1",
                "/event_time_sensitivity/implicit/derivative_s_per_scaled_state/1",
            ),
            _numeric_entry(
                "-0.299636#1",
                "/event_time_sensitivity/implicit/derivative_s_per_scaled_state/2",
            ),
            _numeric_entry(
                "-0.243972#1",
                "/event_time_sensitivity/implicit/derivative_s_per_scaled_state/3",
            ),
            _numeric_entry(
                "5.98012e-5#1",
                "/event_time_sensitivity/direct_trials/0/maximum_abs_residual_from_implicit_s",
            ),
            _numeric_entry("0.95#1", "/saltation_controls/corrupted_reset_diagonal"),
            _numeric_entry(
                "0.05#1",
                "/saltation_controls/corrupted_reset_max_abs_deviation_from_identity",
            ),
            _numeric_entry("1.48546#1", "/periodicity_gate/normalized_residual"),
            _numeric_entry("1e-6#1", "/periodicity_gate/tolerance"),
        ],
    )
    return finite_time, event


def _find_candidates(inventory: dict[str, Any]) -> dict[str, dict[str, Any]]:
    def unique(fragment: str, *, source_suffix: str | None = None) -> dict[str, Any]:
        matches = [
            candidate
            for candidate in inventory["candidates"]
            if fragment in candidate["text"]
            and (
                source_suffix is None
                or candidate["source_path"].endswith(source_suffix)
            )
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one candidate containing {fragment!r}")
        return matches[0]

    return {
        "transition": unique("propagates the discrete variational map"),
        "finite_time": unique("largest observed scaled gain is 8.33244"),
        "figure": unique("fig_phase_event_stability.pdf"),
        "implicit": unique("transverse-event derivative is obtained"),
        "event": unique("denominator is 35.0258"),
        "periodicity": unique("event state remains 1.48546"),
        "appendix": unique(
            "`data/phase_event_stability.{json,npz}`",
            source_suffix="_appendices.qmd",
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
        json.dumps(contracts, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    """Apply the frozen #9116 candidate review idempotently."""

    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    candidates = _find_candidates(inventory)
    claims = _claims(candidates)
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
    claim_assignments = {
        candidates["transition"]["candidate_id"]: ["PD-CLAIM-315"],
        candidates["finite_time"]["candidate_id"]: ["PD-CLAIM-315"],
        candidates["implicit"]["candidate_id"]: ["PD-CLAIM-316"],
        candidates["event"]["candidate_id"]: ["PD-CLAIM-316"],
        candidates["periodicity"]["candidate_id"]: ["PD-CLAIM-316"],
    }
    registry["candidate_reviews"].extend(
        {
            "candidate_id": candidate_id,
            "disposition": "material_claims_mapped",
            "claim_ids": claim_ids,
            "rationale": (
                "This passage states or bounds the finite-time or event-sensitivity qualification."
            ),
            "reviewer": "Codex technical audit",
            "last_verified_on": DATE,
        }
        for candidate_id, claim_ids in claim_assignments.items()
    )
    for candidate_name, rationale in (
        ("figure", "This candidate is the governed figure caption."),
        ("appendix", "This appendix paragraph inventories governed artifacts."),
    ):
        registry["candidate_reviews"].append(
            {
                "candidate_id": candidates[candidate_name]["candidate_id"],
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
                "This deterministic census paragraph summarizes the governed claim registry."
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
            "published_status": (
                "supported_for_declared_local_nonperiodic_analytical_trajectory"
            ),
            "audit_state": (
                "reviewed_as_local_finite_window_and_transverse_event_qualification"
            ),
        }
    )
    registry["paper"]["source_digest"] = inventory["source_digest"]
    registry["audit_scope"]["completion_status"] = "complete"
    registry["audit_scope"]["current_scope"] = (
        f"The complete {inventory['candidate_count']}-candidate paper inventory is "
        "adjudicated, including local finite-time and event-sensitivity evidence."
    )
    REGISTRY.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    _update_numeric_contracts(claims)


if __name__ == "__main__":
    main()
