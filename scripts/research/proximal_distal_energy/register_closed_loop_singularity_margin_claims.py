"""Register exact feasible closed-loop singular-margin claims for #9113."""

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
    "closed_loop_singularity_margin.json"
)
DATE = "2026-08-26"
CLAIM_IDS = {"PD-CLAIM-313", "PD-CLAIM-314"}
ARTIFACTS = [
    REPORT,
    "docs/research/proximal_distal_energy_transfer/"
    "CONSTRAINT_INTERNAL_FORCE_DIAGNOSTICS.md",
    "docs/research/proximal_distal_energy_transfer/"
    "MODEL_COMPLETION_FALSIFICATION_MATRIX.md",
    "scripts/research/proximal_distal_energy/closed_loop_singularity_margin.py",
    "scripts/research/proximal_distal_energy/run_closed_loop_singularity_margin.py",
    "tests/research/test_closed_loop_singularity_margin.py",
    "tests/research/test_closed_loop_singularity_margin_evidence.py",
]


def _numeric_entry(
    literal_id: str,
    json_pointer: str,
    *,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> dict[str, Any]:
    return {
        "literal_id": literal_id,
        "artifact": REPORT,
        "json_pointer": json_pointer,
        "evidence_scope": "local_json_value",
        "scale": 1.0,
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
        "published_status": "supported_for_declared_exact_planar_kinematic_map",
        "audit_status": (
            "exact_closure_branch_phase_geometry_scale_unit_tolerance_and_"
            "manufactured_adverse_controls_checked"
        ),
        "adjudication_outcome": "supported",
        "source_locations": [
            f"{candidate['source_path']}:{candidate['line_start']}"
            for candidate in candidates
        ],
        "evidence_artifacts": ARTIFACTS,
        "model_domain": (
            "Same-origin, fixed-length planar two-arm/grip triangle with exact "
            "position closure and a scaled local velocity-constraint Jacobian."
        ),
        "uncertainty_boundary": boundary,
        "competing_explanations": [
            "distance to a triangle-degeneracy boundary",
            "relative SVD tolerance",
            "generalized-coordinate scale",
            "equivalent length-unit representation",
            "phase-grid resolution",
        ],
        "negative_controls": controls,
        "falsifier": falsifier,
        "adjudication": (
            "The result is retained as exact analytical planar kinematic evidence. "
            "It does not establish anatomy, dynamics, contact force, muscle action, "
            "passive torque, human occurrence, or coaching guidance."
        ),
        "reviewer": "Codex technical audit",
        "last_verified_on": DATE,
        "numeric_evidence": numeric_evidence,
    }


def _nominal_claim(candidates: dict[str, dict[str, Any]]) -> dict[str, Any]:
    nominal_statement = (
        "For the exact 0.75 m/0.78 m/0.25 m planar triangle, all 362 registered "
        "branch-phase samples close within 1.66533e-16 m and retain rank 4/nullity "
        "1 under 1 rad and 0.75 m coordinate scales; the minimum scaled singular "
        "value is 0.202095 m, condition number is 6.45326, and phase-spectrum "
        "spread is 2.22045e-15 m."
    )
    return _claim(
        claim_id="PD-CLAIM-313",
        candidates=[candidates["constructor"], candidates["nominal"]],
        statement=nominal_statement,
        classification="exact_feasible_closed_loop_regular_orbit",
        boundary=(
            "The result is local planar kinematics under declared coordinate "
            "scales; it neither qualifies anatomy nor identifies reactions."
        ),
        falsifier=(
            "Any registered sample violates closure, changes rank/nullity across "
            "global phase or assembly branch, or loses spectrum invariance."
        ),
        controls=[
            "both exact assembly branches",
            "181 phases per branch",
            "three phase resolutions",
            "three feasible geometries",
            "three positive translation-coordinate scales",
        ],
        numeric_evidence=[
            _numeric_entry("0.75#1", "/nominal_geometry_m/lead_arm_length_m"),
            _numeric_entry("0.78#1", "/nominal_geometry_m/trail_arm_length_m"),
            _numeric_entry("0.25#1", "/nominal_geometry_m/grip_separation_m"),
            _numeric_entry("362#1", "/nominal_orbit/sample_count"),
            _numeric_entry(
                "1.66533e-16#1", "/nominal_orbit/maximum_closure_residual_m"
            ),
            _numeric_entry("4#1", "/nominal_orbit/minimum_rank"),
            _numeric_entry("1#1", "/nominal_orbit/minimum_nullity"),
            _numeric_entry(
                "1#2", "/coordinate_scale_contract/angular_coordinate_scale_rad"
            ),
            _numeric_entry(
                "0.75#2",
                "/coordinate_scale_contract/translation_coordinate_scale_m",
            ),
            _numeric_entry(
                "0.202095#1",
                "/nominal_orbit/minimum_smallest_scaled_singular_value_m",
            ),
            _numeric_entry(
                "6.45326#1", "/nominal_orbit/maximum_scaled_condition_number"
            ),
            _numeric_entry(
                "2.22045e-15#1",
                "/nominal_orbit/maximum_scaled_singular_value_spread_m",
            ),
        ],
    )


def _boundary_claim(candidates: dict[str, dict[str, Any]]) -> dict[str, Any]:
    boundary_statement = (
        "The exact 0.03 m and 1.53 m triangle boundaries have rank 3/nullity 2; "
        "at a 1e-8 m lower-bound offset, numerical rank remains 4 through relative "
        "tolerance 1e-6 and becomes 3 at 1e-4, while equivalent centimetre units "
        "preserve rank and condition number."
    )
    return _claim(
        claim_id="PD-CLAIM-314",
        candidates=[candidates["boundary"], candidates["controls"]],
        statement=boundary_statement,
        classification="exact_triangle_degeneracy_and_numerical_margin_boundary",
        boundary=(
            "The tolerance-dependent rank transition is numerical and is not a "
            "physical singularity threshold, anatomical limit, or human strategy."
        ),
        falsifier=(
            "Either exact boundary fails to add one velocity null mode, equivalent "
            "units change rank/condition, or the offset/tolerance matrix does not "
            "reproduce."
        ),
        controls=[
            "both exact triangle boundaries",
            "five distance offsets crossed with five SVD tolerances",
            "equivalent-centimetre transformation",
            "two impossible geometries",
            "manufactured row-dependency killswitch",
        ],
        numeric_evidence=[
            _numeric_entry(
                "0.03#1",
                "/exact_triangle_degeneracies/lower_geometry_m/grip_separation_m",
            ),
            _numeric_entry(
                "1.53#1",
                "/exact_triangle_degeneracies/upper_geometry_m/grip_separation_m",
            ),
            _numeric_entry("3#1", "/exact_triangle_degeneracies/lower_rank_audit/rank"),
            _numeric_entry(
                "2#1", "/exact_triangle_degeneracies/lower_rank_audit/nullity"
            ),
            _numeric_entry(
                "1e-8#1", "/near_lower_boundary_sweep/2/distance_to_lower_degeneracy_m"
            ),
            _numeric_entry(
                "4#1", "/near_lower_boundary_sweep/2/tolerance_cases/3/rank"
            ),
            _numeric_entry(
                "1e-6#1",
                "/near_lower_boundary_sweep/2/tolerance_cases/3/relative_tolerance",
            ),
            _numeric_entry(
                "3#2", "/near_lower_boundary_sweep/2/tolerance_cases/4/rank"
            ),
            _numeric_entry(
                "1e-4#1",
                "/near_lower_boundary_sweep/2/tolerance_cases/4/relative_tolerance",
            ),
        ],
    )


def _claims(candidates: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    return _nominal_claim(candidates), _boundary_claim(candidates)


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
        json.dumps(contracts, indent=2) + "\n",
        encoding="utf-8",
    )


def _find_candidates(inventory: dict[str, Any]) -> dict[str, dict[str, Any]]:
    def unique(
        fragment: str,
        *,
        source_suffix: str | None = None,
    ) -> dict[str, Any]:
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
        "constructor": unique("same-origin triangle directly"),
        "nominal": unique("All 362 registered samples"),
        "boundary": unique("triangle inequality supplies two exact"),
        "controls": unique("equivalent centimetre representation"),
        "appendix": unique(
            "`data/closed_loop_singularity_margin.json`",
            source_suffix="_appendices.qmd",
        ),
    }


def _reset_owned_records(
    registry: dict[str, Any],
    inventory: dict[str, Any],
    candidates: dict[str, dict[str, Any]],
) -> None:
    current_ids = {candidate["candidate_id"] for candidate in inventory["candidates"]}
    selected_ids = {candidate["candidate_id"] for candidate in candidates.values()}
    registry["claims"] = [
        claim for claim in registry["claims"] if claim["claim_id"] not in CLAIM_IDS
    ]
    registry["candidate_reviews"] = [
        review
        for review in registry["candidate_reviews"]
        if review["candidate_id"] in current_ids
        and review["candidate_id"] not in selected_ids
        and not set(review["claim_ids"]).intersection(CLAIM_IDS)
    ]
    registry["release_claim_inventory"] = [
        item
        for item in registry["release_claim_inventory"]
        if item["release_claim_key"] != "feasible_closed_loop_singularity_margin"
    ]
    registry["release_claim_inventory"].append(
        {
            "release_claim_key": "feasible_closed_loop_singularity_margin",
            "published_status": (
                "supported_for_declared_exact_planar_kinematic_triangle"
            ),
            "audit_state": "reviewed_as_exact_planar_kinematic_qualification",
        }
    )


def _append_candidate_reviews(
    registry: dict[str, Any],
    inventory: dict[str, Any],
    candidates: dict[str, dict[str, Any]],
) -> None:
    assignments = {
        candidates["constructor"]["candidate_id"]: ["PD-CLAIM-313"],
        candidates["nominal"]["candidate_id"]: ["PD-CLAIM-313"],
        candidates["boundary"]["candidate_id"]: ["PD-CLAIM-314"],
        candidates["controls"]["candidate_id"]: ["PD-CLAIM-314"],
    }
    registry["candidate_reviews"].extend(
        {
            "candidate_id": candidate_id,
            "disposition": "material_claims_mapped",
            "claim_ids": claim_ids,
            "rationale": (
                "This passage states or bounds the exact feasible closed-loop "
                "singular-margin qualification."
            ),
            "reviewer": "Codex technical audit",
            "last_verified_on": DATE,
        }
        for candidate_id, claim_ids in assignments.items()
    )
    registry["candidate_reviews"].append(
        {
            "candidate_id": candidates["appendix"]["candidate_id"],
            "disposition": "editorial_or_navigation",
            "claim_ids": [],
            "rationale": "This appendix paragraph inventories governed artifacts.",
            "reviewer": "Codex technical audit",
            "last_verified_on": DATE,
        }
    )
    reviewed_ids = {review["candidate_id"] for review in registry["candidate_reviews"]}
    generated_summary_candidates = [
        candidate
        for candidate in inventory["candidates"]
        if candidate["source_path"].endswith("_claim_adjudication_summary.qmd")
        and candidate["candidate_id"] not in reviewed_ids
    ]
    registry["candidate_reviews"].extend(
        {
            "candidate_id": candidate["candidate_id"],
            "disposition": "editorial_or_navigation",
            "claim_ids": [],
            "rationale": (
                "This deterministic census paragraph summarizes the governed "
                "claim registry and adds no standalone scientific result."
            ),
            "reviewer": "Codex technical audit",
            "last_verified_on": DATE,
        }
        for candidate in generated_summary_candidates
    )


def main() -> None:
    """Apply the frozen #9113 candidate review idempotently."""

    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    candidates = _find_candidates(inventory)
    claims = _claims(candidates)
    _reset_owned_records(registry, inventory, candidates)
    registry["claims"].extend(claims)
    _append_candidate_reviews(registry, inventory, candidates)
    registry["paper"]["source_digest"] = inventory["source_digest"]
    registry["audit_scope"]["completion_status"] = "complete"
    registry["audit_scope"]["current_scope"] = (
        f"The complete {inventory['candidate_count']}-candidate paper inventory is "
        "adjudicated, including exact feasible closed-loop singular margins."
    )
    REGISTRY.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    _update_numeric_contracts(claims)


if __name__ == "__main__":
    main()
