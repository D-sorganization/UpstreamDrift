"""Register uncertainty, identifiability, and control claims with stability."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ARTICLE = ROOT / "docs/research/proximal_distal_energy_transfer"
REGISTRY = ARTICLE / "data/claim_audit_registry.json"
INVENTORY = ARTICLE / "data/claim_candidate_inventory.json"
CHAPTER = "docs/research/proximal_distal_energy_transfer/chapters/_ch06d_uncertainty_control.qmd"
ARTIFACTS = [
    "docs/research/proximal_distal_energy_transfer/data/uncertainty_control_study.json",
    "docs/research/proximal_distal_energy_transfer/data/uncertainty_control_study.npz",
    "docs/research/proximal_distal_energy_transfer/data/uncertainty_control_stability_audit.json",
    "scripts/research/proximal_distal_energy/uncertainty_control.py",
    "scripts/research/proximal_distal_energy/run_uncertainty_control_study.py",
    "scripts/research/proximal_distal_energy/audit_uncertainty_stability.py",
    "tests/research/test_uncertainty_control.py",
    "tests/research/test_uncertainty_control_evidence.py",
    "tests/research/test_uncertainty_stability_audit.py",
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
        "audit_status": "evidence_reconciliation_leave_one_out_and_threshold_sensitivity_checked",
        "source_locations": [],
        "evidence_artifacts": ARTIFACTS,
        "model_domain": "Synthetic ten-coordinate moving-base two-hand flexible-club model.",
        "uncertainty_boundary": boundary,
        "competing_explanations": [
            "small deterministic design",
            "nonmonotonic response",
            "parameter interactions",
            "threshold choice",
            "held-out case composition",
            "unmodeled biological and impact dynamics",
        ],
        "negative_controls": [
            "independent training and held-out seeds",
            "leave-one-global-sample-out PRCC",
            "leave-one-held-out-case Pareto recomputation",
            "four singular-value thresholds",
            "wrench rank-nullity audit",
            "constraint and contact-power closure",
        ],
        "falsifier": falsifier,
        "adjudication": (
            "Every chapter passage was checked against committed arrays, source "
            "contracts, leave-one-out rankings, Pareto membership perturbations, "
            "threshold sensitivity, and explicit nonhuman boundaries."
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
        "PD-CLAIM-215": ids[:3],
        "PD-CLAIM-216": ids[3:8],
        "PD-CLAIM-217": ids[8:11],
        "PD-CLAIM-218": ids[11:16],
        "PD-CLAIM-219": ids[16:19],
        "PD-CLAIM-220": ids[19:24],
        "PD-CLAIM-221": ids[24:26],
        "PD-CLAIM-222": ids[26:],
    }
    claims = [
        _claim(
            "PD-CLAIM-215",
            groups["PD-CLAIM-215"],
            "The chapter asks bounded screening, identifiability, and held-out command questions without treating the design as a human population.",
            "question_and_evidential_boundary",
            "supported_as_registered_scope",
            "Parameters are engineering envelopes and commands are not muscles.",
            "Any result is promoted to population prevalence, physiology, or coaching.",
        ),
        _claim(
            "PD-CLAIM-216",
            groups["PD-CLAIM-216"],
            "A deterministic 24-point Latin hypercube varies 12 declared inputs jointly, with separate six-case training and held-out designs.",
            "registered_coupled_design",
            "supported_with_small_sample_boundary",
            "The ranges omit fitted correlations and are sparse for interaction estimation.",
            "Seeds, ranges, sample counts, coupled model, or preselection provenance fail reconciliation.",
        ),
        _claim(
            "PD-CLAIM-217",
            groups["PD-CLAIM-217"],
            "The applied command has delay, lag, rate, asymmetric velocity, and impedance limits; outputs are mechanical proxies rather than physiological variables.",
            "actuator_and_outcome_contract",
            "supported_as_synthetic_surrogate",
            "Effort and face/path are squared-torque and planar orientation proxies.",
            "A state is relabeled neural activation, muscle force, metabolic cost, or 3-D clubface.",
        ),
        _claim(
            "PD-CLAIM-218",
            groups["PD-CLAIM-218"],
            "PRCC intervals and leaders reconcile, but leave-one-out analysis shows the hand-force leader is unstable and several nonleading coefficient signs flip.",
            "small_sample_global_sensitivity_screen",
            "conditional_screening_not_parameter_importance",
            "Five leaders persist across omissions; limb mass leads hand force in only 12 of 24 omissions.",
            "The paper claims causal, variance, monotonic, population, or uniformly stable parameter importance.",
        ),
        _claim(
            "PD-CLAIM-219",
            groups["PD-CLAIM-219"],
            "The planar net wrench leaves one hand-force null mode, while six summaries yield rank six at 1 to 10 percent singular thresholds and rank five at 20 percent.",
            "structural_and_practical_identifiability",
            "supported_with_threshold_sensitivity",
            "Rank does not identify a particular estimable parameter subset or replace profile likelihood.",
            "Individual forces become unique from net wrench alone or practical rank is claimed threshold invariant.",
        ),
        _claim(
            "PD-CLAIM-220",
            groups["PD-CLAIM-220"],
            "Seven training and eight held-out programs are nondominated, but held-out membership changes under leave-one-case-out recomputation.",
            "held_out_multiobjective_tradeoff",
            "supports_no_universal_optimum_with_small_sample_instability",
            "Six held-out cases cannot estimate a population Pareto set.",
            "One program is promoted as universal or all memberships are claimed jackknife-stable.",
        ),
        _claim(
            "PD-CLAIM-221",
            groups["PD-CLAIM-221"],
            "Early restraint improves held-out lower-tail speed versus late drive while worsening planar face/path error, with numerical residuals below reported effects.",
            "conditional_opposing_command_tradeoff",
            "supported_for_declared_six_case_holdout",
            "This is neither human preactivation nor a general strategy.",
            "The ordering reverses in a larger registration or closure error approaches the effect.",
        ),
        _claim(
            "PD-CLAIM-222",
            groups["PD-CLAIM-222"],
            "The phase supports non-identifiability and objective dependence while requiring bilateral measurement and larger registered replication.",
            "bounded_conclusions_and_falsifiers",
            "supported_with_human_validation_open",
            "No governed human outcome is analyzed.",
            "Club motion alone resolves bilateral force or the study is called human validation.",
        ),
    ]
    mapping = {
        candidate: claim
        for claim, candidates_for_claim in groups.items()
        for candidate in candidates_for_claim
    }
    registry["candidate_reviews"] = [
        review
        for review in registry["candidate_reviews"]
        if review["candidate_id"] not in mapping
        and not set(review["claim_ids"]).intersection(groups)
    ]
    registry["candidate_reviews"].extend(
        {
            "candidate_id": candidate,
            "disposition": "material_claims_mapped",
            "claim_ids": [claim],
            "rationale": "This passage participates in the uncertainty and stability audit.",
            "reviewer": "Codex technical audit",
            "last_verified_on": "2026-08-13",
        }
        for candidate, claim in mapping.items()
    )
    registry["claims"] = [
        claim for claim in registry["claims"] if claim["claim_id"] not in groups
    ] + claims
    for claim in claims:
        claim["source_locations"] = [
            f"{by_id[candidate]['source_path']}:{by_id[candidate]['line_start']}"
            for candidate in claim["candidate_ids"]
        ]
    registry["paper"]["source_digest"] = inventory["source_digest"]
    registry["audit_scope"]["current_scope"] = (
        "The complete audit remains in progress. The uncertainty, "
        "identifiability, and control chapter is fully adjudicated with "
        "leave-one-out and threshold sensitivity."
    )
    REGISTRY.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
