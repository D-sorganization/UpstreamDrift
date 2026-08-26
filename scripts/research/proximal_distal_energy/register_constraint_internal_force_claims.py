"""Register scale-qualified constraint and internal-force claims for #9027."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
ARTICLE = ROOT / "docs/research/proximal_distal_energy_transfer"
REGISTRY = ARTICLE / "data/claim_audit_registry.json"
INVENTORY = ARTICLE / "data/claim_candidate_inventory.json"
NUMERIC_CONTRACTS = ARTICLE / "data/claim_numeric_contracts.json"
CHAPTER = (
    "docs/research/proximal_distal_energy_transfer/chapters/"
    "_ch06d_uncertainty_control.qmd"
)
APPENDIX = "docs/research/proximal_distal_energy_transfer/chapters/_appendices.qmd"
REPORT = (
    "docs/research/proximal_distal_energy_transfer/data/"
    "constraint_internal_force_diagnostics.json"
)
DATE = "2026-08-26"
CLAIM_IDS = {"PD-CLAIM-311", "PD-CLAIM-312"}
CHAPTER_LINES = {135, 143, 152, 159}

ARTIFACTS = [
    REPORT,
    "docs/research/proximal_distal_energy_transfer/"
    "CONSTRAINT_INTERNAL_FORCE_DIAGNOSTICS.md",
    "scripts/research/proximal_distal_energy/constraint_internal_force_diagnostics.py",
    "scripts/research/proximal_distal_energy/"
    "run_constraint_internal_force_diagnostics.py",
    "tests/research/test_constraint_internal_force_diagnostics.py",
    "tests/research/test_constraint_internal_force_diagnostics_evidence.py",
]


@dataclass(frozen=True)
class _ClaimSpec:
    claim_id: str
    statement: str
    classification: str
    status: str
    boundary: str
    falsifier: str
    negative_controls: list[str]
    numeric_evidence: list[dict[str, Any]]


def _numeric_entry(
    literal_id: str,
    json_pointer: str,
    *,
    atol: float = 0.0,
) -> dict[str, Any]:
    return {
        "literal_id": literal_id,
        "artifact": REPORT,
        "json_pointer": json_pointer,
        "evidence_scope": "local_json_value",
        "scale": 1.0,
        "offset": 0.0,
        "atol": atol,
        "rtol": 0.0,
    }


def _claim(spec: _ClaimSpec, candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "claim_id": spec.claim_id,
        "candidate_ids": [candidate["candidate_id"] for candidate in candidates],
        "statement": spec.statement,
        "classification": spec.classification,
        "published_status": spec.status,
        "audit_status": (
            "scaled_rank_adverse_geometry_tolerance_and_inference_boundary_checked"
        ),
        "adjudication_outcome": "supported",
        "source_locations": [
            f"{candidate['source_path']}:{candidate['line_start']}"
            for candidate in candidates
        ],
        "evidence_artifacts": ARTIFACTS,
        "model_domain": (
            "Instantaneous analytical planar closure and normalized bilateral "
            "wrench maps at declared geometries, scales, and SVD tolerances."
        ),
        "uncertainty_boundary": spec.boundary,
        "competing_explanations": [
            "generalized-coordinate scaling",
            "wrench force-moment scaling",
            "rank tolerance",
            "contact geometry",
            "constraint or contact infeasibility",
        ],
        "negative_controls": spec.negative_controls,
        "falsifier": spec.falsifier,
        "adjudication": (
            "The result is retained as scale-qualified analytical and registered "
            "synthetic evidence; it does not identify participant force, biological "
            "allocation, human strategy, or coaching guidance."
        ),
        "reviewer": "Codex technical audit",
        "last_verified_on": DATE,
        "numeric_evidence": spec.numeric_evidence,
    }


def _planar_claim(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    spec = _ClaimSpec(
        claim_id="PD-CLAIM-311",
        statement=(
            "Under declared 1 rad angular and 0.75 m translational coordinate "
            "scales, the regular planar closure map has rank 4/nullity 1 and the "
            "constructed adverse alignment has rank 3/nullity 2."
        ),
        classification="scaled_planar_constraint_singularity",
        status="supported_for_declared_analytical_maps",
        boundary=(
            "Rank and nullity are local kinematic properties; conditioning depends "
            "on the declared scales, the adverse alignment is not a qualified closed "
            "anatomical pose, and no constraint force or human strategy is identified."
        ),
        falsifier=(
            "A clean recomputation changes either rank/nullity result, positive "
            "coordinate scaling changes exact rank, or the alignment is promoted "
            "without a feasible closed-state test."
        ),
        negative_controls=[
            "three positive translation-coordinate scales",
            "three near-singular angular offsets",
            "explicit rank tolerance",
            "raw-versus-scaled rank equality",
            "nonhuman inference gate",
        ],
        numeric_evidence=[
            _numeric_entry(
                "1#1",
                "/planar_closed_loop/coordinate_scale_contract/"
                "angular_coordinate_scale_rad",
                atol=0.5000001,
            ),
            _numeric_entry(
                "0.75#1",
                "/planar_closed_loop/coordinate_scale_contract/"
                "translation_coordinate_scale_m",
                atol=0.005000001,
            ),
            _numeric_entry("4#1", "/planar_closed_loop/regular_case/rank"),
            _numeric_entry("1#2", "/planar_closed_loop/regular_case/nullity"),
            _numeric_entry("3#1", "/planar_closed_loop/exact_singular_case/rank"),
            _numeric_entry(
                "2#1",
                "/planar_closed_loop/exact_singular_case/nullity",
            ),
        ],
    )
    return _claim(spec, candidates)


def _contact_claim(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    spec = _ClaimSpec(
        claim_id="PD-CLAIM-312",
        statement=(
            "With a declared 0.10 m wrench length, the separated 0.20 m "
            "point-force map has rank 5/nullity 1 and the coincident map has "
            "rank 3/nullity 3; the near-coincident numerical rank changes across "
            "declared SVD tolerances."
        ),
        classification="normalized_contact_geometry_rank_boundary",
        status="supported_for_declared_analytical_maps",
        boundary=(
            "The map establishes sensing/allocation ambiguity only; it does not "
            "show that a human produces the null-mode force, identify complete hand "
            "wrenches, or qualify compliant or distributed contact."
        ),
        falsifier=(
            "Coincident contacts retain moment observability above rank three, the "
            "separated map loses its axial null mode, or rank is reported without "
            "the geometry, normalization, and tolerance contract."
        ),
        negative_controls=[
            "coincident-contact killswitch",
            "eight contact spans",
            "three relative SVD tolerances",
            "one axial measurement augmentation",
            "full bilateral six-axis input map",
        ],
        numeric_evidence=[
            _numeric_entry(
                "0.10#1",
                "/bilateral_point_force/normalization/"
                "moment_rows_divided_by_reference_length_m",
                atol=0.005000001,
            ),
            _numeric_entry(
                "0.20#1",
                "/bilateral_point_force/registered_span_case/span_m",
                atol=0.005000001,
            ),
            _numeric_entry(
                "5#1",
                "/bilateral_point_force/registered_span_case/rank",
            ),
            _numeric_entry(
                "1#1",
                "/bilateral_point_force/registered_span_case/nullity",
            ),
            _numeric_entry(
                "3#1",
                "/bilateral_point_force/coincident_contact_case/rank",
            ),
            _numeric_entry(
                "3#2",
                "/bilateral_point_force/coincident_contact_case/nullity",
            ),
        ],
    )
    return _claim(spec, candidates)


def _update_numeric_contracts(new_claims: tuple[dict[str, Any], ...]) -> None:
    """Replace the numeric contracts owned by this registration slice."""

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
        for claim in new_claims
    )
    NUMERIC_CONTRACTS.write_text(
        json.dumps(contracts, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Apply the frozen #9027 candidate review idempotently."""

    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    current_ids = {candidate["candidate_id"] for candidate in inventory["candidates"]}
    chapter = {
        int(candidate["line_start"]): candidate
        for candidate in inventory["candidates"]
        if candidate["source_path"] == CHAPTER
        and int(candidate["line_start"]) in CHAPTER_LINES
    }
    if set(chapter) != CHAPTER_LINES:
        raise ValueError(
            "Constraint/internal-force candidates changed; explicit review required"
        )
    appendix = [
        candidate
        for candidate in inventory["candidates"]
        if candidate["source_path"] == APPENDIX
        and "constraint_internal_force_diagnostics.json" in candidate["text"]
    ]
    if len(appendix) != 1:
        raise ValueError("Expected one constraint-diagnostic appendix candidate")

    selected_ids = {candidate["candidate_id"] for candidate in chapter.values()}
    selected_ids.add(appendix[0]["candidate_id"])
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

    assignments = {
        135: ("PD-CLAIM-311",),
        143: ("PD-CLAIM-311",),
        152: ("PD-CLAIM-312",),
        159: ("PD-CLAIM-311", "PD-CLAIM-312"),
    }
    new_claims = (
        _planar_claim([chapter[135], chapter[143], chapter[159]]),
        _contact_claim([chapter[152], chapter[159]]),
    )
    registry["claims"].extend(new_claims)
    registry["candidate_reviews"].extend(
        {
            "candidate_id": chapter[line]["candidate_id"],
            "disposition": "material_claims_mapped",
            "claim_ids": list(claim_ids),
            "rationale": (
                "This passage states or bounds the scale-qualified constraint and "
                "internal-force diagnostic."
            ),
            "reviewer": "Codex technical audit",
            "last_verified_on": DATE,
        }
        for line, claim_ids in assignments.items()
    )
    registry["candidate_reviews"].append(
        {
            "candidate_id": appendix[0]["candidate_id"],
            "disposition": "editorial_or_navigation",
            "claim_ids": [],
            "rationale": (
                "This appendix paragraph inventories governed artifacts and adds "
                "no standalone scientific result."
            ),
            "reviewer": "Codex technical audit",
            "last_verified_on": DATE,
        }
    )
    registry["paper"]["source_digest"] = inventory["source_digest"]
    registry["audit_scope"]["completion_status"] = "complete"
    registry["audit_scope"]["current_scope"] = (
        f"The complete {inventory['candidate_count']}-candidate paper inventory is "
        "adjudicated, including scale-qualified constraint singularity and "
        "contact-geometry/tolerance boundaries."
    )
    REGISTRY.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    _update_numeric_contracts(new_claims)


if __name__ == "__main__":
    main()
