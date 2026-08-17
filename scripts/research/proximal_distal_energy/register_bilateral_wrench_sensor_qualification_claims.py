"""Register trajectory-level bilateral point-force qualification claims."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ARTICLE = ROOT / "docs/research/proximal_distal_energy_transfer"
REGISTRY = ARTICLE / "data/claim_audit_registry.json"
INVENTORY = ARTICLE / "data/claim_candidate_inventory.json"
CHAPTER = (
    "docs/research/proximal_distal_energy_transfer/chapters/_ch05_two_hand_wrench.qmd"
)
DATE = "2026-08-14"
CLAIM_IDS = {"PD-CLAIM-257", "PD-CLAIM-258", "PD-CLAIM-259"}
ARTIFACTS = [
    "docs/research/proximal_distal_energy_transfer/data/bilateral_wrench_sensor_qualification.json",
    "docs/research/proximal_distal_energy_transfer/figures/fig_bilateral_wrench_sensor_qualification.pdf",
    "scripts/research/proximal_distal_energy/bilateral_wrench_sensor_qualification.py",
    "scripts/research/proximal_distal_energy/run_bilateral_wrench_sensor_qualification.py",
    "tests/research/test_bilateral_wrench_sensor_qualification.py",
    "tests/research/test_bilateral_wrench_sensor_qualification_evidence.py",
]


def _claim(
    claim_id: str,
    candidates: list[dict[str, object]],
    *,
    statement: str,
    classification: str,
    status: str,
    boundary: str,
    falsifier: str,
    controls: list[str],
) -> dict[str, object]:
    return {
        "claim_id": claim_id,
        "candidate_ids": [str(candidate["candidate_id"]) for candidate in candidates],
        "statement": statement,
        "classification": classification,
        "published_status": status,
        "audit_status": "deterministic_trajectory_noise_cross_talk_contact_migration_and_null_mode_controls_checked",
        "source_locations": [
            f"{candidate['source_path']}:{candidate['line_start']}"
            for candidate in candidates
        ],
        "evidence_artifacts": ARTIFACTS,
        "model_domain": "A 301-sample deterministic synthetic trajectory of two separated three-axis point forces, evaluated over 32 seeded trials with declared normalized channel scales.",
        "uncertainty_boundary": boundary,
        "competing_explanations": [
            "unmodeled free hand moments",
            "distributed or moving contact",
            "grip and shaft compliance",
            "sensor bias, drift, and synchronization error",
            "device-specific calibration uncertainty",
        ],
        "negative_controls": controls,
        "falsifier": falsifier,
        "adjudication": "The committed evidence was regenerated from the executable estimator; exact manufactured controls and seeded perturbation trials reproduce the published metrics while retaining device and human gates.",
        "reviewer": "Codex technical audit",
        "last_verified_on": DATE,
    }


def main() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    current_ids = {
        str(candidate["candidate_id"]) for candidate in inventory["candidates"]
    }
    registry["candidate_reviews"] = [
        review
        for review in registry["candidate_reviews"]
        if review["candidate_id"] in current_ids
        and not set(review["claim_ids"]).intersection(CLAIM_IDS)
    ]
    registry["claims"] = [
        claim for claim in registry["claims"] if claim["claim_id"] not in CLAIM_IDS
    ]

    reviewed = {review["candidate_id"] for review in registry["candidate_reviews"]}
    new_candidates = [
        candidate
        for candidate in inventory["candidates"]
        if candidate["source_path"] == CHAPTER
        and candidate["candidate_id"] not in reviewed
    ]
    if len(new_candidates) != 5:
        raise ValueError(
            f"expected 5 new chapter candidates, found {len(new_candidates)}"
        )

    narrative = [
        candidate
        for candidate in new_candidates
        if not str(candidate["text"]).startswith("![")
    ]
    figure_candidates = [
        candidate
        for candidate in new_candidates
        if str(candidate["text"]).startswith("![")
    ]
    if len(narrative) != 4 or len(figure_candidates) != 1:
        raise ValueError("unexpected sensor-qualification candidate structure")

    groups = {
        "PD-CLAIM-257": narrative[:2],
        "PD-CLAIM-258": narrative[2:3],
        "PD-CLAIM-259": narrative[3:],
    }
    claims = [
        _claim(
            "PD-CLAIM-257",
            groups["PD-CLAIM-257"],
            statement="In the declared synthetic point-force trajectory, net-wrench-only inversion closes the resultant wrench to numerical precision while retaining 11.86 N allocation RMSE and 29.05 N axial-mode RMSE; the independently measured axial scalar removes that structural ambiguity in the ideal case.",
            classification="synthetic_trajectory_measurement_identifiability",
            status="supported_for_declared_synthetic_point_force_cases",
            boundary="The trajectory, normalization scales, perturbations, and sensor channel are synthetic and do not qualify a physical instrument.",
            falsifier="The net-only estimator uniquely recovers the manufactured axial allocation, or the augmented ideal estimator fails numerical closure.",
            controls=[
                "manufactured time-varying axial null mode",
                "ideal augmented axial measurement",
                "normalized net-wrench closure",
                "seeded thirty-two-trial ensemble",
            ],
        ),
        _claim(
            "PD-CLAIM-258",
            groups["PD-CLAIM-258"],
            statement="For the declared synthetic point-force estimator, calibrated cross-talk and tracked contact centers materially reduce allocation error relative to uncorrected cross-talk and fixed nominal contacts.",
            classification="synthetic_sensor_error_control",
            status="supported_for_registered_perturbation_levels",
            boundary="The numerical improvements are conditional on exact study matrices, scales, migration trajectory, and estimator assumptions; they are not general sensor tolerances.",
            falsifier="Regeneration does not reproduce the registered ordering, or the improvement disappears under the exact calibrated and tracked controls.",
            controls=[
                "noise-only baseline",
                "uncorrected versus exactly calibrated cross-talk",
                "cross-talk calibration residual",
                "fixed versus tracked contact centers",
            ],
        ),
        _claim(
            "PD-CLAIM-259",
            groups["PD-CLAIM-259"],
            statement="The synthetic qualification does not validate full bilateral six-axis wrenches, distributed contact, anatomy, human performance, or a strategy; those conclusions require traceable device and participant-held-out evidence.",
            classification="measurement_and_human_scope_boundary",
            status="full_device_and_human_validation_untested",
            boundary="Only separated point forces and a synthetic axial measurement are qualified.",
            falsifier="A governed, traceably calibrated bilateral six-axis participant dataset satisfies the frozen protocol and supports a broader registered claim.",
            controls=[
                "explicit point-force-only model boundary",
                "device calibration gate",
                "distributed-contact gate",
                "participant-held-out data gate",
            ],
        ),
    ]
    registry["claims"].extend(claims)
    for claim_id, candidates in groups.items():
        registry["candidate_reviews"].extend(
            {
                "candidate_id": candidate["candidate_id"],
                "disposition": "material_claims_mapped",
                "claim_ids": [claim_id],
                "rationale": "This passage states or bounds the trajectory-level synthetic sensor qualification.",
                "reviewer": "Codex technical audit",
                "last_verified_on": DATE,
            }
            for candidate in candidates
        )
    registry["candidate_reviews"].append(
        {
            "candidate_id": figure_candidates[0]["candidate_id"],
            "disposition": "editorial_or_navigation",
            "claim_ids": [],
            "rationale": "The figure include points to evidence but asserts no standalone scientific result.",
            "reviewer": "Codex technical audit",
            "last_verified_on": DATE,
        }
    )

    release_entries = {
        item["release_claim_key"]: item for item in registry["release_claim_inventory"]
    }
    release_entries["synthetic_bilateral_point_force_sensor_qualification"] = {
        "release_claim_key": "synthetic_bilateral_point_force_sensor_qualification",
        "published_status": "qualified_for_declared_synthetic_cases",
        "audit_state": "reviewed_as_model_conditional",
    }
    release_entries["physical_bilateral_six_axis_device_validation"] = {
        "release_claim_key": "physical_bilateral_six_axis_device_validation",
        "published_status": "untested",
        "audit_state": "reviewed_as_untested",
    }
    registry["release_claim_inventory"] = list(release_entries.values())
    registry["paper"]["source_digest"] = inventory["source_digest"]
    registry["audit_scope"]["completion_status"] = "complete"
    registry["audit_scope"]["current_scope"] = (
        "The complete 975-candidate paper inventory is adjudicated. Three new claims qualify a synthetic trajectory-level bilateral point-force estimator while retaining full-device, distributed-contact, anatomical, and human-data gates."
    )
    REGISTRY.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
