"""Register coordinate-explicit force-source claims for epic #9059."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
ARTICLE = ROOT / "docs/research/proximal_distal_energy_transfer"
REGISTRY = ARTICLE / "data/claim_audit_registry.json"
INVENTORY = ARTICLE / "data/claim_candidate_inventory.json"
CHAPTER = (
    "docs/research/proximal_distal_energy_transfer/chapters/"
    "_ch03ba_coordinate_force_sources.qmd"
)
DATE = "2026-08-25"
CLAIM_IDS = {"PD-CLAIM-305", "PD-CLAIM-306", "PD-CLAIM-307"}

MECHANICS_LINES = {5, 13, 19, 25, 38, 46, 56, 77, 79, 95, 102}
MAPPING_LINES = {119, 126, 138, 145, 152, 158, 172, 179}
STUDY_LINES = {189, 198, 204, 206, 213, 215, 223, 231, 237, 244}
EXTERNAL_LINES = {113: "PD-CLAIM-062"}

ARTIFACTS = [
    "docs/research/proximal_distal_energy_transfer/data/force_source_optimization.json",
    "docs/research/proximal_distal_energy_transfer/figures/fig_force_source_optimization.pdf",
    "scripts/research/proximal_distal_energy/force_source_optimization.py",
    "scripts/research/proximal_distal_energy/run_force_source_optimization.py",
    "src/shared/python/biomechanics/force_source_attribution.py",
    "tests/research/test_force_source_optimization.py",
    "tests/unit/biomechanics/test_force_source_attribution.py",
]


def _chapter_candidates(candidates: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    selected = {
        int(candidate["line_start"]): candidate
        for candidate in candidates
        if candidate["source_path"] == CHAPTER
    }
    expected = MECHANICS_LINES | MAPPING_LINES | STUDY_LINES | set(EXTERNAL_LINES)
    if set(selected) != expected:
        raise ValueError(
            "Coordinate-force chapter candidate inventory changed; explicit review required"
        )
    return selected


def _claim(
    claim_id: str,
    members: list[str],
    *,
    source_line: int,
    statement: str,
    classification: str,
    boundary: str,
    falsifier: str,
    numeric_evidence: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    claim: dict[str, Any] = {
        "claim_id": claim_id,
        "candidate_ids": members,
        "statement": statement,
        "classification": classification,
        "published_status": "supported_with_coordinate_and_model_qualification",
        "audit_status": "equations_executable_evidence_boundaries_and_falsifiers_checked",
        "adjudication_outcome": "supported",
        "source_locations": [f"{CHAPTER}:{source_line}"],
        "evidence_artifacts": ARTIFACTS,
        "model_domain": (
            "Exact planar double-pendulum generalized coordinates with a frozen "
            "bounded torque-program grid and an impact-truncated endpoint mapping."
        ),
        "uncertainty_boundary": boundary,
        "competing_explanations": [
            "applied generalized work",
            "gravity",
            "coordinate choice",
            "unrepresented endpoint couple",
        ],
        "negative_controls": [
            "single-rate Coriolis zero",
            "velocity-bias reconstruction residual",
            "rank-deficient endpoint mapping residual",
            "signed-versus-absolute impulse comparison",
        ],
        "falsifier": falsifier,
        "adjudication": (
            "The chapter is retained as coordinate-explicit model evidence and "
            "does not identify biological force, coaching technique, or a human optimum."
        ),
        "reviewer": "Codex technical audit",
        "last_verified_on": DATE,
    }
    if numeric_evidence is not None:
        claim["numeric_evidence"] = numeric_evidence
    return claim


def _claim_groups(
    by_line: dict[int, dict[str, Any]],
) -> tuple[dict[str, list[int]], dict[str, tuple[str, ...]]]:
    groups = {
        "PD-CLAIM-305": sorted(MECHANICS_LINES),
        "PD-CLAIM-306": sorted(MAPPING_LINES),
        "PD-CLAIM-307": sorted(STUDY_LINES),
    }
    assignments: dict[str, tuple[str, ...]] = {
        str(by_line[line]["candidate_id"]): (claim_id,)
        for claim_id, lines in groups.items()
        for line in lines
    }
    assignments.update(
        {
            str(by_line[line]["candidate_id"]): (claim_id,)
            for line, claim_id in EXTERNAL_LINES.items()
        }
    )
    return groups, assignments


def _build_claims(
    candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, tuple[str, ...]]]:
    by_line = _chapter_candidates(candidates)
    groups, assignments = _claim_groups(by_line)
    claims = [
        _claim(
            "PD-CLAIM-305",
            [by_line[line]["candidate_id"] for line in groups["PD-CLAIM-305"]],
            source_line=5,
            statement=(
                "Christoffel first-kind monomials separate cross-speed Coriolis "
                "terms from squared-speed centripetal/centrifugal terms in the "
                "declared absolute-arm/relative-club coordinates, retain an "
                "independent velocity-bias residual, and report equal-and-opposite drives."
            ),
            classification="coordinate_explicit_velocity_bias_identity",
            boundary=(
                "The partition changes under coordinate transformation and does not "
                "introduce additional physical agents or identify muscle action."
            ),
            falsifier=(
                "The monomial sum fails to reconstruct the provider velocity bias, "
                "or a cross-speed term remains when either participating rate is zero."
            ),
        ),
        _claim(
            "PD-CLAIM-306",
            [by_line[line]["candidate_id"] for line in groups["PD-CLAIM-306"]],
            source_line=119,
            statement=(
                "Virtual-work least squares reports a minimum-norm force-only endpoint "
                "equivalent together with rank and unreconstructed generalized residual; "
                "signed/absolute tangent impulse, power, and work remain distinct observables."
            ),
            classification="endpoint_mapping_and_impulse_work_contract",
            boundary=(
                "A rank-deficient force-only equivalent is not a complete grip wrench, "
                "and generalized component work is not automatically subsystem transfer."
            ),
            falsifier=(
                "The mapped force and retained residual do not reconstruct the generalized "
                "drive, or undefined zero-speed tangent intervals contribute to integration."
            ),
        ),
        _claim(
            "PD-CLAIM-307",
            [by_line[line]["candidate_id"] for line in groups["PD-CLAIM-307"]],
            source_line=189,
            statement=(
                "The frozen 135-program grid yields 91 impact-qualified programs and "
                "separate maxima for absolute Coriolis tangent impulse and clubhead speed; "
                "the maximum-impulse program has negative net Coriolis generalized work."
            ),
            classification="bounded_force_source_optimization_result",
            boundary=(
                "This is a finite model-scenario grid, not continuous optimal control, "
                "a causal human estimate, or a technique recommendation."
            ),
            falsifier=(
                "A clean replay changes grid membership or either selected optimum, fails "
                "impact qualification, or violates the published mapping and ledger diagnostics."
            ),
            numeric_evidence=[
                {
                    "literal_id": "135#1",
                    "artifact": ARTIFACTS[0],
                    "json_pointer": "/summary/candidate_count",
                    "evidence_scope": "local_json_value",
                    "scale": 1.0,
                    "offset": 0.0,
                    "atol": 0.0,
                    "rtol": 0.0,
                },
                {
                    "literal_id": "91#1",
                    "artifact": ARTIFACTS[0],
                    "json_pointer": "/summary/qualified_count",
                    "evidence_scope": "local_json_value",
                    "scale": 1.0,
                    "offset": 0.0,
                    "atol": 0.0,
                    "rtol": 0.0,
                },
            ],
        ),
    ]
    return claims, assignments


def _reconcile(
    registry: dict[str, Any],
    inventory: dict[str, Any],
    claims: list[dict[str, Any]],
    assignments: dict[str, tuple[str, ...]],
) -> None:
    selected_ids = set(assignments)
    touched_claim_ids = CLAIM_IDS | {
        claim for ids in assignments.values() for claim in ids
    }
    registry["claims"] = [
        claim for claim in registry["claims"] if claim["claim_id"] not in CLAIM_IDS
    ]
    registry["candidate_reviews"] = [
        review
        for review in registry["candidate_reviews"]
        if review["candidate_id"] not in selected_ids
    ]

    existing = {claim["claim_id"]: claim for claim in registry["claims"]}
    for candidate_id, claim_ids in assignments.items():
        for claim_id in claim_ids:
            if claim_id in existing:
                members = existing[claim_id]["candidate_ids"]
                if candidate_id not in members:
                    members.append(candidate_id)
                    members.sort()
        registry["candidate_reviews"].append(
            {
                "candidate_id": candidate_id,
                "disposition": "material_claims_mapped",
                "claim_ids": sorted(set(claim_ids)),
                "rationale": (
                    "This coordinate-force passage states or bounds an explicitly "
                    "registered mechanics, mapping, literature, or model-grid claim."
                ),
                "reviewer": "Codex technical audit",
                "last_verified_on": DATE,
            }
        )

    registry["claims"].extend(claims)
    registry["release_claim_inventory"] = [
        item
        for item in registry["release_claim_inventory"]
        if item["release_claim_key"] != "coordinate_force_source_attribution"
    ]
    registry["release_claim_inventory"].append(
        {
            "release_claim_key": "coordinate_force_source_attribution",
            "published_status": "supported_at_declared_planar_model_and_coordinate_tier",
            "audit_state": "reviewed_as_bounded_coordinate_explicit_model_result",
        }
    )
    registry["paper"]["source_digest"] = inventory["source_digest"]
    registry["audit_scope"]["completion_status"] = "complete"
    registry["audit_scope"]["current_scope"] = (
        f"The complete {inventory['candidate_count']}-candidate paper inventory is "
        "adjudicated, including the coordinate-force decomposition and bounded "
        "Coriolis-impulse optimization chapter."
    )
    if not touched_claim_ids <= {claim["claim_id"] for claim in registry["claims"]}:
        raise ValueError("Coordinate-force reconciliation references an unknown claim")


def main() -> None:
    """Apply the frozen chapter review idempotently."""
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    claims, assignments = _build_claims(inventory["candidates"])
    _reconcile(registry, inventory, claims, assignments)
    REGISTRY.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
