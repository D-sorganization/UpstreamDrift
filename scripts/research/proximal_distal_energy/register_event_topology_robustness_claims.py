"""Register global event-topology robustness claims for issue #9125."""

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
PHASE_A_REPORT = (
    "docs/research/proximal_distal_energy_transfer/data/event_topology_robustness.json"
)
PHASE_B_REPORT = (
    "docs/research/proximal_distal_energy_transfer/data/"
    "event_topology_stress_extension.json"
)
PHASE_C_REPORT = (
    "docs/research/proximal_distal_energy_transfer/data/"
    "event_topology_channel_matrix.json"
)
DATE = "2026-08-26"
CLAIM_IDS = {"PD-CLAIM-321", "PD-CLAIM-322", "PD-CLAIM-323"}
RELEASE_KEY = "global_event_topology_robustness"
PUBLISHED_STATUS = "supported_for_registered_synthetic_topology_model_scenarios"
ARTIFACTS = [
    PHASE_A_REPORT,
    PHASE_A_REPORT.replace(".json", ".npz"),
    PHASE_B_REPORT,
    PHASE_B_REPORT.replace(".json", ".npz"),
    PHASE_C_REPORT,
    PHASE_C_REPORT.replace(".json", ".npz"),
    "docs/research/proximal_distal_energy_transfer/figures/"
    "fig_event_topology_robustness.pdf",
    "docs/research/proximal_distal_energy_transfer/EVENT_TOPOLOGY_ROBUSTNESS.md",
    "scripts/research/proximal_distal_energy/event_topology_robustness.py",
    "scripts/research/proximal_distal_energy/event_robustness_noise.py",
    "scripts/research/proximal_distal_energy/event_robustness_study.py",
    "scripts/research/proximal_distal_energy/event_robustness_summary.py",
    "scripts/research/proximal_distal_energy/event_topology_channel_controls.py",
    "scripts/research/proximal_distal_energy/run_event_topology_robustness.py",
    "scripts/research/proximal_distal_energy/run_event_topology_stress_extension.py",
    "scripts/research/proximal_distal_energy/run_event_topology_channel_matrix.py",
]


def _numeric_entry(
    literal_id: str,
    artifact: str,
    json_pointer: str,
) -> dict[str, Any]:
    return {
        "literal_id": literal_id,
        "artifact": artifact,
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
            "global_enumeration_delay_noise_pair_adequacy_channel_mask_"
            "step_and_horizon_controls_checked"
        ),
        "adjudication_outcome": "supported",
        "source_locations": [
            f"{candidate['source_path']}:{candidate['line_start']}"
            for candidate in candidates
        ],
        "evidence_artifacts": ARTIFACTS,
        "model_domain": (
            "One synthetic open-loop analytical planar double pendulum, one "
            "geometric guard, declared command, delay, horizon, channel-mask, "
            "and dimensionless perturbation scenarios."
        ),
        "uncertainty_boundary": boundary,
        "competing_explanations": [
            "global-horizon truncation",
            "integration-step classification drift",
            "changed crossing identity or direction",
            "event-surface rather than command perturbation",
            "coordinate channel masks rather than anatomical isolation",
            "synthetic scale rather than measured human variability",
            "topology preservation rather than target feasibility or task success",
        ],
        "negative_controls": controls,
        "falsifier": falsifier,
        "adjudication": (
            "The result is retained only as topology evidence for the declared "
            "synthetic model scenarios. Human variability, anatomy, fatigue, "
            "strategy ranking, work/power, and coaching remain unavailable."
        ),
        "reviewer": "Codex technical audit",
        "last_verified_on": DATE,
        "numeric_evidence": numeric_evidence,
    }


def _phase_a_claim(candidates: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return _claim(
        claim_id="PD-CLAIM-321",
        candidates=[candidates["intro"], candidates["phase_a"]],
        statement=(
            "At the registered 0.01 synthetic scale and 0.20 s delay, the "
            "Phase A cell retains 192 unique-transverse outcomes and 96 of 96 "
            "topology-preserved independent pairs."
        ),
        classification="small_synthetic_event_topology_preservation",
        boundary=(
            "This is pair-level topology preservation, not target accuracy, "
            "human robustness, or a success probability."
        ),
        falsifier=(
            "Any raw replay in the registered cell changes status, crossing "
            "count, or direction, or the pair-level adequacy gate fails."
        ),
        controls=[
            "complete direction-aware global enumeration",
            "common 0.60 s horizon",
            "zero-perturbation exact replay",
            "96 independent antithetic pairs",
        ],
        numeric_evidence=[
            _numeric_entry("0.01#1", PHASE_A_REPORT, "/scenarios/3/scale_fraction"),
            _numeric_entry(
                "0.20#1", PHASE_A_REPORT, "/scenarios/3/delay_summaries/10/delay_s"
            ),
            _numeric_entry(
                "192#1",
                PHASE_A_REPORT,
                "/scenarios/3/delay_summaries/10/topology_counts/unique_transverse",
            ),
            _numeric_entry(
                "96#1",
                PHASE_A_REPORT,
                "/scenarios/3/delay_summaries/10/preserved_pair_count",
            ),
            _numeric_entry(
                "96#2",
                PHASE_A_REPORT,
                "/scenarios/3/delay_summaries/10/independent_pair_count",
            ),
        ],
    )


def _phase_b_claim(candidates: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return _claim(
        claim_id="PD-CLAIM-322",
        candidates=[candidates["intro"], candidates["phase_b"]],
        statement=(
            "At the registered 0.50 synthetic stress and 0.20 s delay, Phase B "
            "retains 118 absent, 7 multiple, and 67 unique-transverse outcomes; "
            "2 of 96 independent pairs preserve nominal topology."
        ),
        classification="adaptive_synthetic_event_topology_failure_region",
        boundary=(
            "The fixed Phase B ladder is an artificial stress-to-failure map, "
            "not a human tolerance or biological noise calibration."
        ),
        falsifier=(
            "Raw arrays fail to reproduce the registered counts, the fixed stop "
            "rule is incomplete, or source/seed/pair adequacy drifts."
        ),
        controls=[
            "public fixed Phase B registration before execution",
            "all five stress levels executed",
            "raw absent and multiple crossings retained",
            "pair-level Wilson interval adequacy",
        ],
        numeric_evidence=[
            _numeric_entry("0.50#1", PHASE_B_REPORT, "/scenarios/4/scale_fraction"),
            _numeric_entry(
                "0.20#1", PHASE_B_REPORT, "/scenarios/4/delay_summaries/10/delay_s"
            ),
            _numeric_entry(
                "118#1",
                PHASE_B_REPORT,
                "/scenarios/4/delay_summaries/10/topology_counts/absent",
            ),
            _numeric_entry(
                "7#1",
                PHASE_B_REPORT,
                "/scenarios/4/delay_summaries/10/topology_counts/multiple",
            ),
            _numeric_entry(
                "67#1",
                PHASE_B_REPORT,
                "/scenarios/4/delay_summaries/10/topology_counts/unique_transverse",
            ),
            _numeric_entry(
                "2#1",
                PHASE_B_REPORT,
                "/scenarios/4/delay_summaries/10/preserved_pair_count",
            ),
            _numeric_entry(
                "96#1",
                PHASE_B_REPORT,
                "/scenarios/4/delay_summaries/10/independent_pair_count",
            ),
        ],
    )


def _phase_c_claim(candidates: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return _claim(
        claim_id="PD-CLAIM-323",
        candidates=[
            candidates["intro"],
            candidates["phase_c"],
            candidates["refinement"],
            candidates["separation"],
        ],
        statement=(
            "The 0.001, 0.002, and 0.004 s Phase C controls preserve topology "
            "identity; 0.60 and 0.80 s horizons agree, while the 0.40 s horizon "
            "misses a wrist-only crossing retained at 0.579326 s."
        ),
        classification="channel_mask_topology_and_numerical_support_control",
        boundary=(
            "Channel masks are generalized-coordinate countermodels, and the "
            "horizon result is truncation evidence rather than strategy ranking."
        ),
        falsifier=(
            "Any registered step changes topology identity, expanded horizons "
            "disagree, zero authority acquires command noise, or the wrist event "
            "is not retained at the reported expanded-horizon time."
        ),
        controls=[
            "both, shoulder-only, wrist-only, and zero coordinate masks",
            "command and command-noise masks applied together",
            "1, 2, and 4 ms physical integration steps",
            "0.40, 0.60, and 0.80 s global horizons",
        ],
        numeric_evidence=[
            _numeric_entry("0.001#1", PHASE_C_REPORT, "/registration/step_sizes_s/0"),
            _numeric_entry("0.002#1", PHASE_C_REPORT, "/registration/step_sizes_s/1"),
            _numeric_entry("0.004#1", PHASE_C_REPORT, "/registration/step_sizes_s/2"),
            _numeric_entry("0.60#1", PHASE_C_REPORT, "/registration/horizons_s/1"),
            _numeric_entry("0.80#1", PHASE_C_REPORT, "/registration/horizons_s/2"),
            _numeric_entry("0.40#1", PHASE_C_REPORT, "/registration/horizons_s/0"),
            _numeric_entry(
                "0.579326#1",
                PHASE_C_REPORT,
                "/horizon_controls/7/events/0/event_time_s",
            ),
        ],
    )


def _claims(candidates: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    return (
        _phase_a_claim(candidates),
        _phase_b_claim(candidates),
        _phase_c_claim(candidates),
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
        "intro": unique("local crossing bracket is removed"),
        "phase_a": unique("Phase A applies matched synthetic"),
        "phase_b": unique("separately preregistered Phase B"),
        "phase_c": unique("Phase C applies the same"),
        "refinement": unique("Topology identity agrees"),
        "separation": unique("Event topology, event time/state"),
        "figure": unique("fig_event_topology_robustness.pdf"),
        "appendix": unique(
            "`data/event_topology_robustness.{json,npz}`",
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


def _replace_candidate_reviews(
    registry: dict[str, Any],
    inventory: dict[str, Any],
    candidates: dict[str, dict[str, Any]],
    claims: tuple[dict[str, Any], ...],
) -> None:
    current_ids = {candidate["candidate_id"] for candidate in inventory["candidates"]}
    selected_ids = {candidate["candidate_id"] for candidate in candidates.values()}
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
                "rationale": "This passage states or bounds the registered topology result.",
                "reviewer": "Codex technical audit",
                "last_verified_on": DATE,
            }
        )
    for name, rationale in (
        ("figure", "This candidate is the governed topology figure caption."),
        ("appendix", "This appendix candidate inventories governed artifacts."),
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
            "rationale": "This deterministic census summarizes the governed claim registry.",
            "reviewer": "Codex technical audit",
            "last_verified_on": DATE,
        }
        for candidate in inventory["candidates"]
        if candidate["source_path"].endswith("_claim_adjudication_summary.qmd")
        and candidate["candidate_id"] not in reviewed_ids
    )


def main() -> None:
    """Apply the frozen #9125 claim review idempotently."""

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
            "audit_state": "reviewed_as_synthetic_global_topology_robustness",
        }
    )
    registry["paper"]["source_digest"] = inventory["source_digest"]
    registry["audit_scope"]["completion_status"] = "complete"
    registry["audit_scope"]["current_scope"] = (
        f"The complete {inventory['candidate_count']}-candidate paper inventory is "
        "adjudicated, including global event-topology robustness evidence."
    )
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
