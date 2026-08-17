"""Register distributed-grip discretization claims and inference boundaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
ARTICLE = ROOT / "docs/research/proximal_distal_energy_transfer"
REGISTRY = ARTICLE / "data/claim_audit_registry.json"
INVENTORY = ARTICLE / "data/claim_candidate_inventory.json"
DATE = "2026-08-15"
CLAIM_IDS = {"PD-CLAIM-286", "PD-CLAIM-287", "PD-CLAIM-288"}
ARTIFACTS = [
    "docs/research/proximal_distal_energy_transfer/data/articulated_distributed_grip_atlas.json",
    "docs/research/proximal_distal_energy_transfer/data/articulated_distributed_grip_atlas.npz",
    "docs/research/proximal_distal_energy_transfer/figures/fig_articulated_distributed_grip_atlas.pdf",
    "scripts/research/proximal_distal_energy/articulated_distributed_grip.py",
    "scripts/research/proximal_distal_energy/articulated_distributed_forward.py",
    "scripts/research/proximal_distal_energy/articulated_distributed_atlas.py",
    "scripts/research/proximal_distal_energy/run_distributed_grip_atlas.py",
    "scripts/research/proximal_distal_energy/make_distributed_grip_figure.py",
    "scripts/research/proximal_distal_energy/register_distributed_grip_claims.py",
    "tests/research/test_articulated_distributed_grip.py",
    "tests/research/test_articulated_distributed_forward.py",
    "tests/research/test_articulated_distributed_atlas.py",
]


def _find(candidates: list[dict[str, Any]], suffix: str, prefix: str) -> dict[str, Any]:
    matches = [
        candidate
        for candidate in candidates
        if str(candidate["source_path"]).endswith(suffix)
        and str(candidate["text"]).startswith(prefix)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one distributed-grip candidate for {prefix!r}")
    return matches[0]


def _selected(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    chapter = "_ch06c_spatial_cross_formulation.qmd"
    return {
        "law": _find(candidates, chapter, "The next gate isolates contact"),
        "control": _find(candidates, chapter, "Thus station count does not"),
        "design": _find(candidates, chapter, "The atlas selects twelve"),
        "figure": _find(candidates, chapter, "![Distributed Grip Discretization"),
        "numerics": _find(candidates, chapter, "All registered numerical"),
        "active": _find(candidates, chapter, "The active-set result is structural"),
        "boundary": _find(candidates, chapter, "The fibers remain memoryless"),
        "summary": _find(candidates, chapter, "The bounded articulated experiments"),
        "ladder": _find(candidates, "_ch07_model_ladder.qmd", "- the three-link"),
        "next": _find(candidates, "_ch07_model_ladder.qmd", "The next decisive"),
        "slack": _find(
            candidates,
            "_ch08b_momentum_transfer_questions.qmd",
            "The next articulated atlas",
        ),
    }


def _claim(
    claim_id: str,
    candidates: list[dict[str, Any]],
    statement: str,
    classification: str,
    status: str,
    boundary: str,
    falsifier: str,
) -> dict[str, Any]:
    return {
        "claim_id": claim_id,
        "candidate_ids": [candidate["candidate_id"] for candidate in candidates],
        "statement": statement,
        "classification": classification,
        "published_status": status,
        "audit_status": "distributed_grip_horizon_and_discretization_checked",
        "source_locations": [
            f"{candidate['source_path']}:{candidate['line_start']}"
            for candidate in candidates
        ],
        "evidence_artifacts": ARTIFACTS,
        "model_domain": (
            "Twelve synthetic articulated states, one/three/five fibers per hand, "
            "two initial velocity signs, two time steps, two native engines, and "
            "nested 4/10/25/50 millisecond observations from 288 trajectories."
        ),
        "uncertainty_boundary": boundary,
        "competing_explanations": [
            "fiber geometry rather than physical pressure distribution",
            "state-registered free lengths rather than tissue preload",
            "rigid-club or unchanged-support dynamics",
            "unmatched load and delivery state",
        ],
        "negative_controls": [
            "one-fiber point-law reduction",
            "equal total stiffness and damping across station counts",
            "coincident and reversed moment arms",
            "initial-velocity reversal",
            "time-step, station-count, and native-engine comparisons",
        ],
        "falsifier": falsifier,
        "adjudication": (
            "Discretization and active-set results are reported without assigning "
            "measured pressure, benefit, human intent, or coaching meaning."
        ),
        "reviewer": "Codex technical audit",
        "last_verified_on": DATE,
    }


def _claims(selected: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        _claim(
            "PD-CLAIM-286",
            [selected["law"], selected["control"], selected["design"]],
            "A registered articulated atlas compares one-, three-, and five-fiber tension grips with equal total stiffness/damping, state-registered free lengths, and nested 4--50 ms horizons.",
            "distributed_grip_discretization_design",
            "complete_for_declared_synthetic_matrix",
            "The fibers are frictionless engineering lines, not measured pressure, fingers, or tissue.",
            "Station count changes total stiffness, horizons are reinitialized, or the one-fiber reduction fails.",
        ),
        _claim(
            "PD-CLAIM-287",
            [selected["numerics"], selected["active"]],
            "All 288 trajectories pass registered power, passivity, work-energy, geometry, refinement, and MuJoCo--Pinocchio gates through 50 ms; multi-fiber active sets are partly open but do not transition.",
            "distributed_grip_discretization_result",
            "supported_through_declared_fifty_millisecond_synthetic_horizon",
            "No-transition observations are right-censored, and delivery states are not matched for load or work.",
            "Any reproduced cell exceeds a gate, a geometric control fails, or a reported active transition is absent from committed evidence.",
        ),
        _claim(
            "PD-CLAIM-288",
            [selected["boundary"]],
            "The distributed-fiber atlas establishes discretization sensitivity, not physical grip-pressure benefit, slack benefit, equipment response, timing economy, human transfer, or strategy.",
            "distributed_grip_inference_boundary",
            "explicitly_bounded",
            "Shaft bending/torsion, ground/free-moment work, friction, tissue, matched delivery, and governed human measurements remain open.",
            "A physical, biological, equipment, timing, or coaching claim is attributed to this synthetic atlas alone.",
        ),
    ]


def _review_primary(
    reviews: dict[str, dict[str, Any]],
    selected: dict[str, Any],
    claims: list[dict[str, Any]],
) -> None:
    for name in ("law", "control", "design", "numerics", "active", "boundary"):
        candidate = selected[name]
        reviews[candidate["candidate_id"]] = {
            "candidate_id": candidate["candidate_id"],
            "disposition": "material_claims_mapped",
            "claim_ids": [],
            "rationale": "This passage states or bounds the distributed-grip gate.",
            "reviewer": "Codex technical audit",
            "last_verified_on": DATE,
        }
    for claim in claims:
        for candidate_id in claim["candidate_ids"]:
            reviews[candidate_id]["claim_ids"].append(claim["claim_id"])
    figure = selected["figure"]
    reviews[figure["candidate_id"]] = {
        "candidate_id": figure["candidate_id"],
        "disposition": "editorial_or_navigation",
        "claim_ids": [],
        "rationale": "The figure include points to governed evidence without a standalone claim.",
        "reviewer": "Codex technical audit",
        "last_verified_on": DATE,
    }


def _attach_repeated(
    registry: dict[str, Any],
    reviews: dict[str, dict[str, Any]],
    selected: dict[str, Any],
) -> None:
    mapping = {
        "summary": ("PD-CLAIM-128", "PD-CLAIM-262", "PD-CLAIM-288"),
        "ladder": ("PD-CLAIM-128", "PD-CLAIM-288"),
        "next": ("PD-CLAIM-128", "PD-CLAIM-288"),
        "slack": ("PD-CLAIM-253", "PD-CLAIM-287", "PD-CLAIM-288"),
    }
    claims = {claim["claim_id"]: claim for claim in registry["claims"]}
    for name, claim_ids in mapping.items():
        candidate = selected[name]
        candidate_id = candidate["candidate_id"]
        location = f"{candidate['source_path']}:{candidate['line_start']}"
        for claim_id in claim_ids:
            claim = claims[claim_id]
            claim["candidate_ids"] = list(
                dict.fromkeys([*claim.get("candidate_ids", []), candidate_id])
            )
            claim["source_locations"] = list(
                dict.fromkeys([*claim.get("source_locations", []), location])
            )
        reviews[candidate_id] = {
            "candidate_id": candidate_id,
            "disposition": "material_claims_mapped",
            "claim_ids": list(claim_ids),
            "rationale": "This repeated ladder or boundary passage inherits the primary claim limits.",
            "reviewer": "Codex technical audit",
            "last_verified_on": DATE,
        }
    claims["PD-CLAIM-128"]["statement"] = (
        "The discrepancy matrix records explicit branch capabilities rather than "
        "cumulative triangular inheritance; eleven bounded findings are supported "
        "somewhere in executed tiers, while calibrated anatomical, shaft, ground, "
        "and independently measured human transport remain untested."
    )
    claims["PD-CLAIM-253"]["statement"] = (
        "Typed scalar and articulated point-contact audits separate slack classes, "
        "and a 288-trajectory distributed-fiber atlas exposes station-count and "
        "active-set sensitivity; none establishes physical class identity or benefit."
    )
    claims["PD-CLAIM-262"]["statement"] = (
        "Articulated point-attachment screens pass through 5 ms and the distributed "
        "fiber discretization passes through 50 ms, but timing, recovery, equipment, "
        "and benefit claims require calibrated shaft, ground, tissue, and human data."
    )


def main() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    valid_ids = {candidate["candidate_id"] for candidate in inventory["candidates"]}
    registry["claims"] = [
        claim for claim in registry["claims"] if claim["claim_id"] not in CLAIM_IDS
    ]
    for claim in registry["claims"]:
        claim["candidate_ids"] = [
            candidate_id
            for candidate_id in claim.get("candidate_ids", [])
            if candidate_id in valid_ids
        ]
    selected = _selected(inventory["candidates"])
    claims = _claims(selected)
    registry["claims"].extend(claims)
    reviews = {
        review["candidate_id"]: review
        for review in registry["candidate_reviews"]
        if review["candidate_id"] in valid_ids
    }
    _review_primary(reviews, selected, claims)
    _attach_repeated(registry, reviews, selected)
    registry["candidate_reviews"] = list(reviews.values())
    release = {
        item["release_claim_key"]: item for item in registry["release_claim_inventory"]
    }
    release["distributed_grip_discretization"] = {
        "release_claim_key": "distributed_grip_discretization",
        "published_status": "fifty_millisecond_distributed_fiber_gate_qualified",
        "audit_state": "reviewed_as_synthetic_discretization_result",
    }
    registry["release_claim_inventory"] = list(release.values())
    registry["audit_scope"]["current_scope"] = (
        "The complete paper inventory is adjudicated. Articulated point-attachment "
        "and distributed-fiber tiers pass their registered gates through 5 and 50 ms. "
        "Calibrated shaft, ground, tissue, and governed human validation remain open."
    )
    registry["paper"]["source_digest"] = inventory["source_digest"]
    REGISTRY.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
