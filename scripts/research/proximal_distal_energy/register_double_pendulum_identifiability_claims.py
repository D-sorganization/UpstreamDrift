"""Register bounded double-pendulum identifiability claims for issue #9104."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
ARTICLE = ROOT / "docs/research/proximal_distal_energy_transfer"
REGISTRY = ARTICLE / "data/claim_audit_registry.json"
INVENTORY = ARTICLE / "data/claim_candidate_inventory.json"
REPORT = (
    "docs/research/proximal_distal_energy_transfer/data/"
    "double_pendulum_identifiability.json"
)
MASTER = (
    "docs/research/proximal_distal_energy_transfer/proximal_distal_energy_transfer.qmd"
)
CHAPTER = "docs/research/proximal_distal_energy_transfer/chapters/_ch02_mechanics.qmd"
APPENDIX = "docs/research/proximal_distal_energy_transfer/chapters/_appendices.qmd"
DATE = "2026-08-25"
CLAIM_IDS = {"PD-CLAIM-308", "PD-CLAIM-309", "PD-CLAIM-310"}
OLD_ABSTRACT_ID = "PD-CAND-f9b891e62e1d24de"
OLD_APPENDIX_ID = "PD-CAND-6f363aa905e209c1"
CHAPTER_LINES = {217, 225, 237, 248}

ARTIFACTS = [
    REPORT,
    "docs/research/proximal_distal_energy_transfer/DOUBLE_PENDULUM_IDENTIFIABILITY.md",
    "scripts/research/proximal_distal_energy/double_pendulum_identifiability.py",
    "scripts/research/proximal_distal_energy/double_pendulum_identifiability_contract.py",
    "scripts/research/proximal_distal_energy/double_pendulum_identifiability_reporting.py",
    "scripts/research/proximal_distal_energy/double_pendulum_identifiability_validation.py",
    "scripts/research/proximal_distal_energy/double_pendulum_physical_parameters.py",
    "scripts/research/proximal_distal_energy/run_double_pendulum_identifiability.py",
    "tests/research/test_double_pendulum_identifiability.py",
    "tests/research/test_double_pendulum_identifiability_evidence.py",
]


@dataclass(frozen=True)
class _ClaimSpec:
    claim_id: str
    source_line: int
    statement: str
    classification: str
    published_status: str
    boundary: str
    falsifier: str
    negative_controls: list[str]
    numeric_evidence: list[dict[str, Any]]


def _numeric_entry(
    literal_id: str,
    json_pointer: str,
    *,
    scale: float = 1.0,
    atol: float = 0.0,
) -> dict[str, Any]:
    return {
        "literal_id": literal_id,
        "artifact": REPORT,
        "json_pointer": json_pointer,
        "evidence_scope": "local_json_value",
        "scale": scale,
        "offset": 0.0,
        "atol": atol,
        "rtol": 0.0,
    }


def _selected_candidates(
    candidates: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    abstracts = [
        candidate
        for candidate in candidates
        if candidate["source_path"] == MASTER and int(candidate["line_start"]) == 10
    ]
    if len(abstracts) != 1:
        raise ValueError("Expected exactly one current paper abstract candidate")
    chapter = {
        int(candidate["line_start"]): candidate
        for candidate in candidates
        if candidate["source_path"] == CHAPTER
        and int(candidate["line_start"]) in CHAPTER_LINES
    }
    if set(chapter) != CHAPTER_LINES:
        raise ValueError(
            "Double-pendulum identifiability candidates changed; explicit review required"
        )
    return abstracts[0], chapter


def _claim(
    spec: _ClaimSpec,
    members: list[str],
) -> dict[str, Any]:
    return {
        "claim_id": spec.claim_id,
        "candidate_ids": members,
        "statement": spec.statement,
        "classification": spec.classification,
        "published_status": spec.published_status,
        "audit_status": (
            "exact_map_dimensionless_evidence_and_inference_boundary_checked"
        ),
        "adjudication_outcome": "supported",
        "source_locations": [f"{CHAPTER}:{spec.source_line}"],
        "evidence_artifacts": ARTIFACTS,
        "model_domain": (
            "Planar fixed-hub analytical double pendulum in declared absolute-arm/"
            "relative-club coordinates with oracle inverse-dynamics kinematics."
        ),
        "uncertainty_boundary": spec.boundary,
        "competing_explanations": [
            "insufficient trajectory excitation",
            "parameter grouping in the equation of motion",
            "scale choice",
            "unmodeled dynamics and measurement error",
        ],
        "negative_controls": spec.negative_controls,
        "falsifier": spec.falsifier,
        "adjudication": (
            "The result is retained as exact or registered synthetic model evidence; "
            "it does not identify participant anatomy, human strategy, or coaching advice."
        ),
        "reviewer": "Codex technical audit",
        "last_verified_on": DATE,
        "numeric_evidence": spec.numeric_evidence,
    }


def _structural_claim(members: list[str]) -> dict[str, Any]:
    spec = _ClaimSpec(
        claim_id="PD-CLAIM-308",
        source_line=217,
        statement=(
            "The declared physical-parameter map has analytic rank 7 and nullity 4, "
            "with three exact nonunique parameter families that preserve every base "
            "coefficient."
        ),
        classification="double_pendulum_exact_physical_map_nonuniqueness",
        published_status="structurally_non_identifiable_under_declared_model",
        boundary=(
            "The exact result applies only to the declared eleven-entry reduced "
            "physical map; it does not test model adequacy or a richer sensor model."
        ),
        falsifier=(
            "The analytic rank witness vanishes inside the declared parameter domain "
            "or any registered alternative changes a base coefficient."
        ),
        negative_controls=[
            "analytic nonzero-minor witness",
            "three exact coefficient-preserving alternatives",
            "independent finite-difference Jacobian comparison",
        ],
        numeric_evidence=[
            _numeric_entry("7#1", "/physical_parameter_map/rank"),
            _numeric_entry("4#1", "/physical_parameter_map/nullity"),
        ],
    )
    return _claim(spec, members)


def _finite_record_claim(members: list[str]) -> dict[str, Any]:
    spec = _ClaimSpec(
        claim_id="PD-CLAIM-309",
        source_line=237,
        statement=(
            "Under declared coefficient and 60 N m torque scales, the registered "
            "synthetic record yields dimensionless regressor rank 7 and condition "
            "180.853; equivalent coefficient units change the matrix by at most "
            "4.44089e-16, while zero motion returns rank 0."
        ),
        classification="double_pendulum_dimensionless_finite_record_excitation",
        published_status="full_rank_for_registered_synthetic_record",
        boundary=(
            "Rank and condition are finite-record, tolerance-, coordinate-, and "
            "scale-contract results; raw dimensional conditioning is not interpreted."
        ),
        falsifier=(
            "A clean replay changes the registered rank, equivalent units change a "
            "dimensionless rank decision, or the zero-motion killswitch has nonzero rank."
        ),
        negative_controls=[
            "equivalent coefficient-unit conversion",
            "two positive scale alternatives",
            "zero-motion rank killswitch",
            "manufactured inverse-dynamics reconstruction",
        ],
        numeric_evidence=[
            _numeric_entry("60#1", "/nondimensional_scale_contract/torque_scale_nm"),
            _numeric_entry("7#1", "/finite_record_regressor/rank"),
            _numeric_entry(
                "180.853#1", "/finite_record_regressor/retained_condition_number"
            ),
            _numeric_entry(
                "4.44089e-16#1",
                "/finite_record_regressor/unit_invariance/"
                "max_abs_dimensionless_regressor_difference",
            ),
            _numeric_entry("0#1", "/zero_motion_killswitch/rank"),
        ],
    )
    return _claim(spec, members)


def _oracle_claim(members: list[str]) -> dict[str, Any]:
    spec = _ClaimSpec(
        claim_id="PD-CLAIM-310",
        source_line=248,
        statement=(
            "With exact kinematics and iid Gaussian torque noise of 1 N m, the oracle "
            "lower-bound screen gives worst relative 95% half-width 0.123266 over the "
            "full record and 498.504 over its first 10%; this is not practical or "
            "participant identifiability."
        ),
        classification="double_pendulum_identifiability_inference_boundary",
        published_status="explicitly_bounded_oracle_lower_bound_only",
        boundary=(
            "The screen omits kinematic differentiation noise, correlated errors, "
            "model discrepancy, event uncertainty, unknown noise scale, priors, "
            "repeated participants, and held-out prediction."
        ),
        falsifier=(
            "The uncertainty calculation reports pseudo-precision for a deficient "
            "record or the paper promotes the oracle bound to practical inference."
        ),
        negative_controls=[
            "four torque-noise levels",
            "four cumulative record windows",
            "rank-deficient Fisher fail-closed case",
            "explicit practical-identifiability non-promotion",
        ],
        numeric_evidence=[
            _numeric_entry(
                "1#1", "/noise_aware_lower_bound_screen/reference_window_noise_sd_nm"
            ),
            _numeric_entry(
                "95#1",
                "/noise_aware_lower_bound_screen/confidence_level",
                scale=100.0,
            ),
            _numeric_entry(
                "0.123266#1",
                "/noise_aware_lower_bound_screen/full_record_cases/2/"
                "worst_ci95_relative_half_width",
            ),
            _numeric_entry(
                "498.504#1",
                "/noise_aware_lower_bound_screen/window_cases/0/"
                "worst_ci95_relative_half_width",
            ),
            _numeric_entry(
                "10#1",
                "/noise_aware_lower_bound_screen/window_cases/0/fraction",
                scale=100.0,
            ),
        ],
    )
    return _claim(spec, members)


def _build_claims(
    candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, tuple[str, ...]], str]:
    abstract, chapter = _selected_candidates(candidates)
    abstract_id = str(abstract["candidate_id"])
    setup_id = str(chapter[217]["candidate_id"])
    structural_id = str(chapter[225]["candidate_id"])
    finite_id = str(chapter[237]["candidate_id"])
    boundary_id = str(chapter[248]["candidate_id"])
    assignments = {
        abstract_id: tuple(sorted(CLAIM_IDS)),
        setup_id: ("PD-CLAIM-308", "PD-CLAIM-309"),
        structural_id: ("PD-CLAIM-308",),
        finite_id: ("PD-CLAIM-309",),
        boundary_id: ("PD-CLAIM-310",),
    }
    claims = [
        _structural_claim([abstract_id, setup_id, structural_id]),
        _finite_record_claim([abstract_id, setup_id, finite_id]),
        _oracle_claim([abstract_id, boundary_id]),
    ]
    return claims, assignments, abstract_id


def _migrate_abstract(
    registry: dict[str, Any], inventory_ids: set[str], abstract_id: str
) -> set[str]:
    prior_claim_ids: set[str] = set()
    for review in registry["candidate_reviews"]:
        if review["candidate_id"] in {OLD_ABSTRACT_ID, abstract_id}:
            prior_claim_ids.update(review["claim_ids"])
    stale_abstracts = {
        candidate_id
        for claim in registry["claims"]
        if f"{MASTER}:10" in claim.get("source_locations", [])
        for candidate_id in claim["candidate_ids"]
        if candidate_id not in inventory_ids
    }
    if stale_abstracts not in (set(), {OLD_ABSTRACT_ID}):
        raise ValueError(f"Unexpected stale abstract candidates: {stale_abstracts}")
    for claim in registry["claims"]:
        if OLD_ABSTRACT_ID in claim["candidate_ids"]:
            claim["candidate_ids"] = sorted(
                {
                    abstract_id if item == OLD_ABSTRACT_ID else item
                    for item in claim["candidate_ids"]
                }
            )
    return prior_claim_ids


def _reconcile_release_inventory(registry: dict[str, Any]) -> None:
    release_items = {
        "double_pendulum_base_coefficient_excitation": (
            "full_rank_for_registered_synthetic_record",
            "reviewed_as_dimensionless_finite_record_excitation",
        ),
        "double_pendulum_physical_parameter_identifiability": (
            "structurally_non_identifiable_under_declared_model",
            "reviewed_as_exact_structural_nonidentifiability",
        ),
        "double_pendulum_practical_identifiability": (
            "not_established_oracle_kinematics_lower_bound_only",
            "reviewed_as_oracle_lower_bound_only",
        ),
    }
    registry["release_claim_inventory"] = [
        item
        for item in registry["release_claim_inventory"]
        if item["release_claim_key"] not in release_items
    ]
    registry["release_claim_inventory"].extend(
        {
            "release_claim_key": key,
            "published_status": values[0],
            "audit_state": values[1],
        }
        for key, values in release_items.items()
    )


def _reconcile(
    registry: dict[str, Any],
    inventory: dict[str, Any],
    claims: list[dict[str, Any]],
    assignments: dict[str, tuple[str, ...]],
    abstract_id: str,
) -> None:
    inventory_ids = {str(item["candidate_id"]) for item in inventory["candidates"]}
    appendix_candidates = [
        item
        for item in inventory["candidates"]
        if item["source_path"] == APPENDIX
        and "data/double_pendulum_identifiability.json" in str(item["text"])
    ]
    if len(appendix_candidates) != 1:
        raise ValueError("Expected one appendix artifact-inventory candidate")
    appendix_id = str(appendix_candidates[0]["candidate_id"])
    prior_abstract_claims = _migrate_abstract(registry, inventory_ids, abstract_id)
    selected_ids = set(assignments) | {
        OLD_ABSTRACT_ID,
        OLD_APPENDIX_ID,
        appendix_id,
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
    for candidate_id, assigned_claims in assignments.items():
        claim_ids = set(assigned_claims)
        if candidate_id == abstract_id:
            claim_ids.update(prior_abstract_claims)
        for claim_id in claim_ids:
            if (
                claim_id in existing
                and candidate_id not in existing[claim_id]["candidate_ids"]
            ):
                existing[claim_id]["candidate_ids"].append(candidate_id)
                existing[claim_id]["candidate_ids"].sort()
        registry["candidate_reviews"].append(
            {
                "candidate_id": candidate_id,
                "disposition": "material_claims_mapped",
                "claim_ids": sorted(claim_ids),
                "rationale": (
                    "This passage states, repeats, or bounds the exact physical-map, "
                    "dimensionless finite-record, or oracle lower-bound claim."
                ),
                "reviewer": "Codex technical audit",
                "last_verified_on": DATE,
            }
        )
    registry["candidate_reviews"].append(
        {
            "candidate_id": appendix_id,
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
    registry["claims"].extend(claims)
    _reconcile_release_inventory(registry)
    registry["paper"]["source_digest"] = inventory["source_digest"]
    registry["audit_scope"]["completion_status"] = "complete"
    registry["audit_scope"]["current_scope"] = (
        f"The complete {inventory['candidate_count']}-candidate paper inventory is "
        "adjudicated, including exact double-pendulum physical-map nonuniqueness, "
        "dimensionless finite-record excitation, and the oracle inference boundary."
    )


def main() -> None:
    """Apply the frozen #9104 candidate review idempotently."""
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    claims, assignments, abstract_id = _build_claims(inventory["candidates"])
    _reconcile(registry, inventory, claims, assignments, abstract_id)
    REGISTRY.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
