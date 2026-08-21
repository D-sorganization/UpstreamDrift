"""Register independent articulated manufactured-solution claims (#8910)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
ARTICLE = ROOT / "docs/research/proximal_distal_energy_transfer"
REGISTRY = ARTICLE / "data/claim_audit_registry.json"
INVENTORY = ARTICLE / "data/claim_candidate_inventory.json"
DATE = "2026-08-21"
CLAIM_IDS = {f"PD-CLAIM-{number}" for number in range(297, 302)}
ARTIFACTS = [
    "docs/research/proximal_distal_energy_transfer/data/articulated_manufactured_solution.json",
    "scripts/research/proximal_distal_energy/articulated_manufactured_solution.py",
    "scripts/research/proximal_distal_energy/run_articulated_manufactured_solution.py",
    "scripts/research/proximal_distal_energy/register_articulated_manufactured_solution_claims.py",
    "tests/research/test_articulated_manufactured_solution.py",
]


def _find(candidates: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    matches = [
        candidate
        for candidate in candidates
        if str(candidate["source_path"]).endswith(
            "_ch06c_spatial_cross_formulation.qmd"
        )
        and str(candidate["text"]).startswith(prefix)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one manufactured candidate beginning {prefix!r}")
    return matches[0]


def _claim(
    claim_id: str,
    candidate: dict[str, Any],
    statement: str,
    classification: str,
    status: str,
    boundary: str,
    falsifier: str,
) -> dict[str, Any]:
    return {
        "claim_id": claim_id,
        "candidate_ids": [candidate["candidate_id"]],
        "statement": statement,
        "classification": classification,
        "published_status": status,
        "audit_status": "independent_manufactured_solution_controls_executed",
        "source_locations": [f"{candidate['source_path']}:{candidate['line_start']}"],
        "evidence_artifacts": ARTIFACTS,
        "model_domain": (
            "One synthetic subject-scaled closed state, a 10 ms manufactured "
            "trajectory, three inverse-dynamics formulations, and a free-club "
            "gravity-free zero-torque conservation rollout."
        ),
        "uncertainty_boundary": boundary,
        "competing_explanations": [
            "common idealized rigid-body specification",
            "finite-difference Christoffel mass gradient",
            "short manufactured horizon",
            "world-supported pelvis branch excluded from conservation",
        ],
        "negative_controls": [
            "10 N m MuJoCo corruption killswitch",
            "three independently named inverse-dynamics paths",
            "adjacent three-level Richardson estimates",
            "measured rather than hardcoded conservation drift",
            "exact constrained identities separated from numerical evidence",
        ],
        "falsifier": falsifier,
        "adjudication": (
            "The registered numerical controls pass and remain restricted to "
            "synthetic operator and integrator verification."
        ),
        "reviewer": "Codex technical audit",
        "last_verified_on": DATE,
    }


def _build_claims(
    candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected = [
        _find(candidates, "Cross-engine agreement alone"),
        _find(candidates, "For the registered closed state"),
        _find(candidates, "Momentum is assessed"),
        _find(candidates, "The constrained control coordinates"),
        _find(candidates, "The machine-readable record is"),
    ]
    claims = [
        _claim(
            "PD-CLAIM-297",
            selected[0],
            "The manufactured torque is independently evaluated by analytical Lagrange--Christoffel, MuJoCo mj_inverse, and Pinocchio RNEA, with a corruption killswitch.",
            "articulated_manufactured_solution_design",
            "complete_for_declared_numerical_control",
            "The paths share one idealized model specification but not one dynamics operator.",
            "The 10 N m corruption does not fail the gate, an operator aliases another, or a named native API is not called.",
        ),
        _claim(
            "PD-CLAIM-298",
            selected[1],
            "All three inverse-dynamics comparisons are small but nonzero and semi-implicit Euler exhibits registered first-order convergence.",
            "articulated_manufactured_solution_result",
            "supported_at_declared_state_and_steps",
            "The result is one manufactured state and a 10 ms horizon, not a population or complete downswing.",
            "Any relative dynamics residual exceeds 0.05 or either adjacent Richardson order leaves 0.9--1.1.",
        ),
        _claim(
            "PD-CLAIM-299",
            selected[2],
            "The free-club gravity-free zero-torque rollout has measured linear-momentum, angular-momentum, and kinetic-energy drift below the registered 0.02 relative bound.",
            "articulated_conservation_result",
            "supported_for_free_floating_club_subtree",
            "The supported pelvis tree is excluded because its reactions invalidate whole-model momentum conservation.",
            "Any measured drift exceeds 0.02 or supported-body motion is silently included as a free-system invariant.",
        ),
        _claim(
            "PD-CLAIM-300",
            selected[3],
            "The coordinated constrained trajectory recovers the imposed multiplier through both native operators with nonzero cross-engine numerical residuals.",
            "articulated_constrained_manufactured_result",
            "supported_for_declared_coordinated_trajectory",
            "This is one three-component lead-hand-to-grip point constraint, not a simultaneous two-hand closure test; exact position, velocity, and virtual-power zeros are construction identities and not independent evidence.",
            "Closure is not achieved, multiplier or equilibrium error exceeds 0.05, or an exact identity is presented as an independent engine comparison.",
        ),
        _claim(
            "PD-CLAIM-301",
            selected[4],
            "Manufactured-solution controls verify numerical operators and integration order but do not validate a human swing, biological torque, or coaching strategy.",
            "articulated_manufactured_solution_inference_boundary",
            "explicitly_bounded",
            "Human mechanics and strategy require governed empirical evidence outside this synthetic control.",
            "A human, physiological, or coaching conclusion is attributed to these manufactured trajectories alone.",
        ),
    ]
    return claims, selected


def _reconcile(
    registry: dict[str, Any],
    inventory: dict[str, Any],
    claims: list[dict[str, Any]],
    selected: list[dict[str, Any]],
) -> None:
    registry["claims"] = [
        claim for claim in registry["claims"] if claim["claim_id"] not in CLAIM_IDS
    ]
    valid_ids = {candidate["candidate_id"] for candidate in inventory["candidates"]}
    reviews = {
        review["candidate_id"]: review
        for review in registry["candidate_reviews"]
        if review["candidate_id"] in valid_ids
    }
    for claim, candidate in zip(claims, selected, strict=True):
        reviews[candidate["candidate_id"]] = {
            "candidate_id": candidate["candidate_id"],
            "disposition": "material_claims_mapped",
            "claim_ids": [claim["claim_id"]],
            "rationale": "This passage states or bounds the independent manufactured-solution control.",
            "reviewer": "Codex technical audit",
            "last_verified_on": DATE,
        }
    appendix_matches = [
        candidate
        for candidate in inventory["candidates"]
        if str(candidate["source_path"]).endswith("_appendices.qmd")
        and str(candidate["text"]).startswith("- `data/e1_sweep.json`")
    ]
    if len(appendix_matches) != 1:
        raise ValueError("expected one appendix artifact-list candidate")
    appendix = appendix_matches[0]
    reviews[appendix["candidate_id"]] = {
        "candidate_id": appendix["candidate_id"],
        "disposition": "editorial_or_navigation",
        "claim_ids": [],
        "rationale": "This appendix paragraph inventories governed artifacts and adds no standalone scientific result.",
        "reviewer": "Codex technical audit",
        "last_verified_on": DATE,
    }
    registry["candidate_reviews"] = list(reviews.values())
    registry["claims"].extend(claims)
    entries = {
        item["release_claim_key"]: item for item in registry["release_claim_inventory"]
    }
    entries["articulated_manufactured_solution"] = {
        "release_claim_key": "articulated_manufactured_solution",
        "published_status": "independent_numerical_controls_qualified",
        "audit_state": "reviewed_as_synthetic_operator_and_integrator_evidence",
    }
    registry["release_claim_inventory"] = list(entries.values())
    registry["paper"]["source_digest"] = inventory["source_digest"]


def main() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    claims, selected = _build_claims(inventory["candidates"])
    _reconcile(registry, inventory, claims, selected)
    REGISTRY.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
