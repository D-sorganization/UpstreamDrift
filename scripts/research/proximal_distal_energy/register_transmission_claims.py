"""Register transmission robustness claims and perturbation stability."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ARTICLE = ROOT / "docs/research/proximal_distal_energy_transfer"
REGISTRY = ARTICLE / "data/claim_audit_registry.json"
INVENTORY = ARTICLE / "data/claim_candidate_inventory.json"
CHAPTER = "docs/research/proximal_distal_energy_transfer/chapters/_ch07c_transmission_robustness.qmd"
ARTIFACTS = [
    "docs/research/proximal_distal_energy_transfer/data/transmission_robustness_study.json",
    "docs/research/proximal_distal_energy_transfer/data/transmission_robustness_study.npz",
    "docs/research/proximal_distal_energy_transfer/data/transmission_stability_audit.json",
    "scripts/research/proximal_distal_energy/transmission_robustness.py",
    "scripts/research/proximal_distal_energy/run_transmission_robustness_study.py",
    "scripts/research/proximal_distal_energy/audit_transmission_stability.py",
    "tests/research/test_transmission_robustness.py",
    "tests/research/test_transmission_robustness_evidence.py",
    "tests/research/test_transmission_stability_audit.py",
]


def _claim(
    claim_id, candidate_ids, statement, classification, status, boundary, falsifier
):
    return {
        "claim_id": claim_id,
        "candidate_ids": candidate_ids,
        "statement": statement,
        "classification": classification,
        "published_status": status,
        "audit_status": "evidence_reconciled_with_jackknife_and_local_linear_audit",
        "source_locations": [],
        "evidence_artifacts": ARTIFACTS,
        "model_domain": "Synthetic moving-base two-hand compliant-club transmission model.",
        "uncertainty_boundary": boundary,
        "competing_explanations": [
            "small deterministic program set",
            "objective choice",
            "unit scaling",
            "local nonlinearity",
            "unmodeled sensing and biology",
        ],
        "negative_controls": [
            "common random numbers",
            "training/held-out split",
            "leave-one-held-out-case Pareto recomputation",
            "singular-value thresholds",
            "held-out linear prediction error",
            "pathway closure",
        ],
        "falsifier": falsifier,
        "adjudication": "All passages were reconciled to regenerated hash-bound authority, Pareto jackknives, rank thresholds, and out-of-local-envelope prediction error.",
        "reviewer": "Codex technical audit",
        "last_verified_on": "2026-08-13",
    }


def main() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    candidates = [c for c in inventory["candidates"] if c["source_path"] == CHAPTER]
    ids = [c["candidate_id"] for c in candidates]
    by_id = {c["candidate_id"]: c for c in candidates}
    groups = {
        "PD-CLAIM-230": ids[:3],
        "PD-CLAIM-231": ids[3:8],
        "PD-CLAIM-232": ids[8:14],
        "PD-CLAIM-233": ids[14:20],
        "PD-CLAIM-234": ids[20:],
    }
    claims = [
        _claim(
            "PD-CLAIM-230",
            groups["PD-CLAIM-230"],
            "The chapter asks robustness and pathway questions without equating nominal speed, torque sign, or model response with human stability.",
            "scope_and_adversarial_boundary",
            "supported",
            "The study is synthetic and four-program.",
            "Any bounded result is promoted to universal human strategy.",
        ),
        _claim(
            "PD-CLAIM-231",
            groups["PD-CLAIM-231"],
            "A reference-explicit work ledger separates constraint, direct, elastic, gravity, dissipation, and numerical terms; no scalar transfer efficiency identifies them uniquely.",
            "pathway_accounting",
            "supported_with_3.77_percent_numerical_limit",
            "Whole-system contact work cancels under ideal constraints.",
            "Pathways are conflated or residuals exceed the declared bound.",
        ),
        _claim(
            "PD-CLAIM-232",
            groups["PD-CLAIM-232"],
            "Four paired programs expose speed-face-load-effort tradeoffs; all are held-out nondominated and remain so under every leave-one-case-out recomputation.",
            "paired_multiobjective_robustness",
            "supported_for_registered_programs",
            "The tested set is not a complete control frontier.",
            "A universal optimum is claimed or jackknife membership changes.",
        ),
        _claim(
            "PD-CLAIM-233",
            groups["PD-CLAIM-233"],
            "The local map has algebraic rank three/nullity five, but practical rank is scale-dependent and held-out linear prediction error is material.",
            "local_task_map",
            "local_descriptive_only",
            "It is neither a global manifold nor evidence of neural synergy.",
            "The local partition is promoted to globally predictive or biological dimension.",
        ),
        _claim(
            "PD-CLAIM-234",
            groups["PD-CLAIM-234"],
            "Biological impedance, repeatability, coaching, ball flight, and self-stabilization remain open participant-held-out experimental questions.",
            "bounded_research_program",
            "human_validation_open",
            "No governed human outcome is analyzed.",
            "Model proxies are relabeled biological validation or prescription.",
        ),
    ]
    mapping = {
        candidate: claim for claim, members in groups.items() for candidate in members
    }
    registry["candidate_reviews"] = [
        r
        for r in registry["candidate_reviews"]
        if r["candidate_id"] not in mapping
        and not set(r["claim_ids"]).intersection(groups)
    ]
    registry["candidate_reviews"].extend(
        {
            "candidate_id": c,
            "disposition": "material_claims_mapped",
            "claim_ids": [claim],
            "rationale": "This passage participates in the transmission robustness audit.",
            "reviewer": "Codex technical audit",
            "last_verified_on": "2026-08-13",
        }
        for c, claim in mapping.items()
    )
    registry["claims"] = [
        c for c in registry["claims"] if c["claim_id"] not in groups
    ] + claims
    for claim in claims:
        claim["source_locations"] = [
            f"{by_id[c]['source_path']}:{by_id[c]['line_start']}"
            for c in claim["candidate_ids"]
        ]
    registry["paper"]["source_digest"] = inventory["source_digest"]
    registry["audit_scope"]["current_scope"] = (
        "The complete audit remains in progress. The transmission robustness chapter is fully adjudicated with jackknife and local-linear sensitivity."
    )
    REGISTRY.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
