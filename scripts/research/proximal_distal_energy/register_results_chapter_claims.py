"""Register the independently reconciled original-results audit slice."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ARTICLE = ROOT / "docs/research/proximal_distal_energy_transfer"
REGISTRY = ARTICLE / "data/claim_audit_registry.json"
INVENTORY = ARTICLE / "data/claim_candidate_inventory.json"
CHAPTER = "docs/research/proximal_distal_energy_transfer/chapters/_ch07_results.qmd"
ARTIFACTS = [
    "docs/research/proximal_distal_energy_transfer/data/results_chapter_audit.json",
    "docs/research/proximal_distal_energy_transfer/data/e1_sweep.json",
    "docs/research/proximal_distal_energy_transfer/data/results_summary.json",
    "docs/research/proximal_distal_energy_transfer/data/representative_traces.npz",
    "docs/research/proximal_distal_energy_transfer/data/e1b_bounded_sweep.json",
    "docs/research/proximal_distal_energy_transfer/data/e1c_sensitivity.json",
    "docs/research/proximal_distal_energy_transfer/data/e1d_parameter_sensitivity.json",
    "docs/research/proximal_distal_energy_transfer/data/e1e_smooth_command_sensitivity.json",
    "scripts/research/proximal_distal_energy/audit_results_chapter.py",
    "tests/research/test_results_chapter_audit.py",
]


def _claim(
    claim_id: str,
    candidate_ids: list[str],
    statement: str,
    classification: str,
    status: str,
    boundary: str,
    falsifier: str,
) -> dict[str, object]:
    return {
        "claim_id": claim_id,
        "candidate_ids": candidate_ids,
        "statement": statement,
        "classification": classification,
        "published_status": status,
        "audit_status": "independent_evidence_reconciliation_and_scope_correction_checked",
        "source_locations": [],
        "evidence_artifacts": ARTIFACTS,
        "model_domain": "Planar two-link model and finite registered command grids.",
        "uncertainty_boundary": boundary,
        "competing_explanations": [
            "impact-selection rule",
            "finite command-grid selection",
            "unmodeled parameter interactions",
            "different actuator capacity surface",
            "model geometry rather than human technique",
        ],
        "negative_controls": [
            "passive wrist program",
            "two shoulder-torque levels",
            "alternative impact criteria",
            "one-at-a-time parameter cases",
            "finite command-rise filters",
            "pointwise energy-balance closure",
        ],
        "falsifier": falsifier,
        "adjudication": (
            "Every displayed selection, percentage, work ledger, exclusion count, "
            "and robustness statement was reconciled to committed evidence. The "
            "blanket 12 percent claim was corrected to 12.4 and 7.8 percent."
        ),
        "reviewer": "Codex technical audit",
        "last_verified_on": "2026-08-13",
    }


def main() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    candidates = [
        candidate
        for candidate in inventory["candidates"]
        if candidate["source_path"] == CHAPTER
    ]
    ids = [candidate["candidate_id"] for candidate in candidates]
    by_id = {candidate["candidate_id"]: candidate for candidate in candidates}
    groups = {
        "PD-CLAIM-207": ids[:7],
        "PD-CLAIM-208": ids[7:10],
        "PD-CLAIM-209": ids[10:12],
        "PD-CLAIM-210": ids[12:15],
        "PD-CLAIM-211": ids[15:21],
        "PD-CLAIM-212": ids[21:23],
        "PD-CLAIM-213": ids[23:29],
        "PD-CLAIM-214": ids[29:],
    }
    claims = [
        _claim(
            "PD-CLAIM-207",
            groups["PD-CLAIM-207"],
            "The committed evidence contains 92 attempted programs, 63 accepted registered deliveries, 29 reason-coded exclusions, and the four reported 60 N m representatives.",
            "evidence_provenance_and_selection",
            "supported_by_deterministic_reconciliation",
            "Acceptance is a registered planar delivery-zone rule, not observed impact quality.",
            "Attempt counts, status counts, representative rows, or displayed values fail exact reconciliation.",
        ),
        _claim(
            "PD-CLAIM-208",
            groups["PD-CLAIM-208"],
            "Early drive is below passive at both levels; best late drive is 12.4 and 7.8 percent above passive, and the tested restrain program adds about 0.5 and 0.4 m/s over best drive.",
            "finite_grid_timing_order_and_effect_size",
            "supported_for_two_tested_torque_levels",
            "Effect size is not constant, and a grid-selected maximum is not a global optimum.",
            "Recalculation reverses the ordering or the paper reports one effect size for both levels.",
        ),
        _claim(
            "PD-CLAIM-209",
            groups["PD-CLAIM-209"],
            "One declared linear concentric torque-velocity bound preserves the tested ordering with compressed margins.",
            "bounded_actuator_sensitivity",
            "supported_for_one_uncalibrated_bound",
            "Constant eccentric restraint is a modeling choice, not a physiological law.",
            "The bound changes the registered ordering or is promoted as calibrated human capacity.",
        ),
        _claim(
            "PD-CLAIM-210",
            groups["PD-CLAIM-210"],
            "All selected planar traces show an arm-then-club peak sequence, with configuration differences reported only in the model coordinate convention.",
            "selected_trace_kinematic_sequence",
            "supported_for_four_selected_traces",
            "The sequence and montage do not classify a player or establish population timing.",
            "Peak order fails in a selected trace or model geometry is labeled as human technique.",
        ),
        _claim(
            "PD-CLAIM-211",
            groups["PD-CLAIM-211"],
            "The selected-trace ledger reproduces early and late segment energy and interface-work values, and club energy-rate closure is below 5e-12 W.",
            "segment_energy_and_interface_power_accounting",
            "supported_for_selected_planar_traces",
            "Joint-force power is a mechanical channel, not a named muscle, intent, or independently validated human pathway.",
            "Ledger values or pointwise balance fail reconciliation, or association is presented as human causation.",
        ),
        _claim(
            "PD-CLAIM-212",
            groups["PD-CLAIM-212"],
            "The pointwise drift-control split shows early restraint and late drift dominance at the commanded state; it is not a forward zero-command trajectory.",
            "pointwise_drift_control_signature",
            "supported_as_instantaneous_model_decomposition",
            "The tangent-field signature does not by itself establish future persistence or passive human action.",
            "Terms do not sum to modeled acceleration or are described as a zero-torque rollout.",
        ),
        _claim(
            "PD-CLAIM-213",
            groups["PD-CLAIM-213"],
            "Ordering persists across five impact families, 13 one-at-a-time parameter cases, registered delivery bounds, and 0 to 50 ms command filters.",
            "registered_local_robustness",
            "supported_but_not_global_or_population_robustness",
            "The studies omit joint parameter distributions, calibrated uncertainty, unmodeled impact physics, and global command optimization.",
            "Any registered family reverses the ordering or local results are generalized as a universal onset.",
        ),
        _claim(
            "PD-CLAIM-214",
            groups["PD-CLAIM-214"],
            "The chapter summary is restricted to the finite planar model and reports level-specific effect sizes and channel-specific energy language.",
            "results_scope_summary",
            "supported_after_scope_and_effect_size_correction",
            "No biological validation, universal strategy, or coaching prescription follows.",
            "The summary restores the blanket 12 percent claim or promotes model-local ordering to human advice.",
        ),
    ]
    claim_by_candidate = {
        candidate_id: claim_id
        for claim_id, candidate_ids in groups.items()
        for candidate_id in candidate_ids
    }
    registry["candidate_reviews"] = [
        review
        for review in registry["candidate_reviews"]
        if review["candidate_id"] not in claim_by_candidate
        and not set(review["claim_ids"]).intersection(groups)
    ]
    registry["candidate_reviews"].extend(
        {
            "candidate_id": candidate_id,
            "disposition": "material_claims_mapped",
            "claim_ids": [claim_id],
            "rationale": "This passage participates in the independently reconciled results audit.",
            "reviewer": "Codex technical audit",
            "last_verified_on": "2026-08-13",
        }
        for candidate_id, claim_id in claim_by_candidate.items()
    )
    registry["claims"] = [
        claim
        for claim in registry["claims"]
        if claim["claim_id"] not in groups and claim["claim_id"] != "PD-CLAIM-001"
    ] + claims
    for claim in claims:
        claim["source_locations"] = [
            f"{by_id[candidate_id]['source_path']}:{by_id[candidate_id]['line_start']}"
            for candidate_id in claim["candidate_ids"]
        ]
    registry["paper"]["source_digest"] = inventory["source_digest"]
    registry["audit_scope"]["current_scope"] = (
        "The complete candidate audit remains in progress. The original results "
        "chapter is now fully reconciled and adjudicated, including a corrected "
        "level-specific late-drive effect size and narrower causal language."
    )
    REGISTRY.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
