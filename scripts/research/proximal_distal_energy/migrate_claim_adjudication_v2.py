"""Apply the explicitly reviewed v2 outcome census to the frozen claim snapshot.

This migration deliberately does not infer scientific outcomes from
``published_status`` or ``audit_status``.  It is locked to the exact paper
digest and claim count reviewed for issue #8724.  A later paper revision or
claim split must receive a new explicit outcome review instead of inheriting a
default.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .claim_audit import SCHEMA_VERSION, build_candidate_inventory

PRE_ADJUDICATION_SOURCE_DIGEST = (
    "ad5a9e7338f9f8ef9517464f8c8cf9f70a2dc033de50312de760417438a1d6d5"
)
PRIOR_REVIEWED_SOURCE_DIGEST = (
    "034692d69e1447ea0552c2d5a8f36c05299294c1f23635cd74e6690ad304815b"
)
REVIEWED_SOURCE_DIGEST = (
    "7407e8f00842ecdf95769d65ac7d2fe3f8d495cb0d11d405640e7582e6b8560a"
)
REVIEWED_CLAIM_COUNT = 303
REVIEWER_PROJECTION_CANDIDATE_IDS = frozenset(
    {
        "PD-CAND-9345c1e6be2ef186",
        "PD-CAND-aeecc7c4cec6b96f",
        "PD-CAND-ab4689630944a0fe",
        "PD-CAND-aa6efbe9274b5d53",
        "PD-CAND-87d26eacc282b21c",
        "PD-CAND-d2f2b5b07b466265",
        "PD-CAND-bcfcc2b3a9631de1",
        "PD-CAND-bb48a75174a745ce",
        "PD-CAND-fdcf5c685afbb41a",
        "PD-CAND-feff7c3d6f4ddc55",
        "PD-CAND-9519d7e6dfb308ef",
        "PD-CAND-be8a26a0593eab4f",
        "PD-CAND-165a6caf21ef85e2",
        "PD-CAND-f1e25f4b524e7a06",
        "PD-CAND-cd8822807d808531",
        "PD-CAND-39f42d06f3e621a5",
        "PD-CAND-0acfb3375ef769f8",
        "PD-CAND-9906005bff75ba72",
        "PD-CAND-b06a3cbe5b2d0e01",
        "PD-CAND-f5697c26aed70275",
        "PD-CAND-c6f7607002d58a93",
        "PD-CAND-b5b2526e23b77d70",
    }
)
PRIOR_REVIEWER_PROJECTION_CANDIDATE_IDS = REVIEWER_PROJECTION_CANDIDATE_IDS - {
    "PD-CAND-b06a3cbe5b2d0e01",
    "PD-CAND-f5697c26aed70275",
    "PD-CAND-c6f7607002d58a93",
} | {
    "PD-CAND-1e8415652d3743aa",
    "PD-CAND-585da748b9832879",
    "PD-CAND-f0b8fbb7ca966f01",
}

# These sets are the exhaustive finding-level review authority for the locked
# 303-claim snapshot. They are intentionally explicit: no claim can inherit an
# outcome merely because it is absent from an exception list.
SUPPORTED_CLAIM_IDS = frozenset(
    {
        "PD-CLAIM-002",
        "PD-CLAIM-003",
        "PD-CLAIM-004",
        "PD-CLAIM-005",
        "PD-CLAIM-006",
        "PD-CLAIM-007",
        "PD-CLAIM-008",
        "PD-CLAIM-009",
        "PD-CLAIM-010",
        "PD-CLAIM-011",
        "PD-CLAIM-012",
        "PD-CLAIM-013",
        "PD-CLAIM-014",
        "PD-CLAIM-015",
        "PD-CLAIM-016",
        "PD-CLAIM-017",
        "PD-CLAIM-018",
        "PD-CLAIM-019",
        "PD-CLAIM-020",
        "PD-CLAIM-021",
        "PD-CLAIM-022",
        "PD-CLAIM-023",
        "PD-CLAIM-024",
        "PD-CLAIM-025",
        "PD-CLAIM-026",
        "PD-CLAIM-027",
        "PD-CLAIM-028",
        "PD-CLAIM-029",
        "PD-CLAIM-030",
        "PD-CLAIM-031",
        "PD-CLAIM-032",
        "PD-CLAIM-033",
        "PD-CLAIM-034",
        "PD-CLAIM-035",
        "PD-CLAIM-036",
        "PD-CLAIM-037",
        "PD-CLAIM-038",
        "PD-CLAIM-039",
        "PD-CLAIM-040",
        "PD-CLAIM-041",
        "PD-CLAIM-042",
        "PD-CLAIM-043",
        "PD-CLAIM-044",
        "PD-CLAIM-045",
        "PD-CLAIM-046",
        "PD-CLAIM-047",
        "PD-CLAIM-048",
        "PD-CLAIM-049",
        "PD-CLAIM-050",
        "PD-CLAIM-051",
        "PD-CLAIM-052",
        "PD-CLAIM-053",
        "PD-CLAIM-054",
        "PD-CLAIM-055",
        "PD-CLAIM-056",
        "PD-CLAIM-057",
        "PD-CLAIM-058",
        "PD-CLAIM-060",
        "PD-CLAIM-061",
        "PD-CLAIM-062",
        "PD-CLAIM-063",
        "PD-CLAIM-064",
        "PD-CLAIM-065",
        "PD-CLAIM-066",
        "PD-CLAIM-067",
        "PD-CLAIM-068",
        "PD-CLAIM-069",
        "PD-CLAIM-071",
        "PD-CLAIM-072",
        "PD-CLAIM-073",
        "PD-CLAIM-074",
        "PD-CLAIM-075",
        "PD-CLAIM-076",
        "PD-CLAIM-077",
        "PD-CLAIM-078",
        "PD-CLAIM-080",
        "PD-CLAIM-081",
        "PD-CLAIM-082",
        "PD-CLAIM-083",
        "PD-CLAIM-084",
        "PD-CLAIM-085",
        "PD-CLAIM-086",
        "PD-CLAIM-087",
        "PD-CLAIM-088",
        "PD-CLAIM-089",
        "PD-CLAIM-090",
        "PD-CLAIM-091",
        "PD-CLAIM-092",
        "PD-CLAIM-095",
        "PD-CLAIM-096",
        "PD-CLAIM-097",
        "PD-CLAIM-098",
        "PD-CLAIM-099",
        "PD-CLAIM-100",
        "PD-CLAIM-101",
        "PD-CLAIM-102",
        "PD-CLAIM-103",
        "PD-CLAIM-104",
        "PD-CLAIM-105",
        "PD-CLAIM-106",
        "PD-CLAIM-107",
        "PD-CLAIM-108",
        "PD-CLAIM-109",
        "PD-CLAIM-110",
        "PD-CLAIM-111",
        "PD-CLAIM-112",
        "PD-CLAIM-113",
        "PD-CLAIM-114",
        "PD-CLAIM-115",
        "PD-CLAIM-116",
        "PD-CLAIM-118",
        "PD-CLAIM-119",
        "PD-CLAIM-120",
        "PD-CLAIM-121",
        "PD-CLAIM-122",
        "PD-CLAIM-123",
        "PD-CLAIM-124",
        "PD-CLAIM-125",
        "PD-CLAIM-126",
        "PD-CLAIM-127",
        "PD-CLAIM-128",
        "PD-CLAIM-129",
        "PD-CLAIM-131",
        "PD-CLAIM-132",
        "PD-CLAIM-133",
        "PD-CLAIM-134",
        "PD-CLAIM-135",
        "PD-CLAIM-136",
        "PD-CLAIM-137",
        "PD-CLAIM-138",
        "PD-CLAIM-139",
        "PD-CLAIM-140",
        "PD-CLAIM-141",
        "PD-CLAIM-142",
        "PD-CLAIM-146",
        "PD-CLAIM-147",
        "PD-CLAIM-148",
        "PD-CLAIM-149",
        "PD-CLAIM-150",
        "PD-CLAIM-151",
        "PD-CLAIM-152",
        "PD-CLAIM-153",
        "PD-CLAIM-154",
        "PD-CLAIM-155",
        "PD-CLAIM-156",
        "PD-CLAIM-157",
        "PD-CLAIM-158",
        "PD-CLAIM-159",
        "PD-CLAIM-160",
        "PD-CLAIM-161",
        "PD-CLAIM-162",
        "PD-CLAIM-163",
        "PD-CLAIM-164",
        "PD-CLAIM-165",
        "PD-CLAIM-166",
        "PD-CLAIM-167",
        "PD-CLAIM-168",
        "PD-CLAIM-169",
        "PD-CLAIM-170",
        "PD-CLAIM-171",
        "PD-CLAIM-172",
        "PD-CLAIM-173",
        "PD-CLAIM-174",
        "PD-CLAIM-175",
        "PD-CLAIM-176",
        "PD-CLAIM-177",
        "PD-CLAIM-178",
        "PD-CLAIM-179",
        "PD-CLAIM-180",
        "PD-CLAIM-181",
        "PD-CLAIM-182",
        "PD-CLAIM-184",
        "PD-CLAIM-185",
        "PD-CLAIM-186",
        "PD-CLAIM-187",
        "PD-CLAIM-188",
        "PD-CLAIM-189",
        "PD-CLAIM-190",
        "PD-CLAIM-191",
        "PD-CLAIM-192",
        "PD-CLAIM-193",
        "PD-CLAIM-194",
        "PD-CLAIM-195",
        "PD-CLAIM-196",
        "PD-CLAIM-197",
        "PD-CLAIM-201",
        "PD-CLAIM-202",
        "PD-CLAIM-203",
        "PD-CLAIM-204",
        "PD-CLAIM-205",
        "PD-CLAIM-207",
        "PD-CLAIM-208",
        "PD-CLAIM-209",
        "PD-CLAIM-210",
        "PD-CLAIM-211",
        "PD-CLAIM-212",
        "PD-CLAIM-213",
        "PD-CLAIM-214",
        "PD-CLAIM-215",
        "PD-CLAIM-216",
        "PD-CLAIM-217",
        "PD-CLAIM-218",
        "PD-CLAIM-219",
        "PD-CLAIM-220",
        "PD-CLAIM-221",
        "PD-CLAIM-222",
        "PD-CLAIM-223",
        "PD-CLAIM-224",
        "PD-CLAIM-225",
        "PD-CLAIM-226",
        "PD-CLAIM-227",
        "PD-CLAIM-228",
        "PD-CLAIM-229",
        "PD-CLAIM-230",
        "PD-CLAIM-231",
        "PD-CLAIM-232",
        "PD-CLAIM-233",
        "PD-CLAIM-235",
        "PD-CLAIM-236",
        "PD-CLAIM-237",
        "PD-CLAIM-238",
        "PD-CLAIM-239",
        "PD-CLAIM-240",
        "PD-CLAIM-241",
        "PD-CLAIM-242",
        "PD-CLAIM-243",
        "PD-CLAIM-244",
        "PD-CLAIM-247",
        "PD-CLAIM-248",
        "PD-CLAIM-249",
        "PD-CLAIM-250",
        "PD-CLAIM-251",
        "PD-CLAIM-252",
        "PD-CLAIM-253",
        "PD-CLAIM-254",
        "PD-CLAIM-255",
        "PD-CLAIM-256",
        "PD-CLAIM-257",
        "PD-CLAIM-258",
        "PD-CLAIM-260",
        "PD-CLAIM-261",
        "PD-CLAIM-262",
        "PD-CLAIM-263",
        "PD-CLAIM-264",
        "PD-CLAIM-265",
        "PD-CLAIM-266",
        "PD-CLAIM-267",
        "PD-CLAIM-268",
        "PD-CLAIM-269",
        "PD-CLAIM-270",
        "PD-CLAIM-271",
        "PD-CLAIM-272",
        "PD-CLAIM-273",
        "PD-CLAIM-274",
        "PD-CLAIM-275",
        "PD-CLAIM-276",
        "PD-CLAIM-277",
        "PD-CLAIM-278",
        "PD-CLAIM-279",
        "PD-CLAIM-280",
        "PD-CLAIM-281",
        "PD-CLAIM-282",
        "PD-CLAIM-283",
        "PD-CLAIM-284",
        "PD-CLAIM-285",
        "PD-CLAIM-286",
        "PD-CLAIM-287",
        "PD-CLAIM-288",
        "PD-CLAIM-289",
        "PD-CLAIM-290",
        "PD-CLAIM-291",
        "PD-CLAIM-292",
        "PD-CLAIM-293",
        "PD-CLAIM-294",
        "PD-CLAIM-295",
        "PD-CLAIM-297",
        "PD-CLAIM-298",
        "PD-CLAIM-299",
        "PD-CLAIM-300",
        "PD-CLAIM-301",
        "PD-CLAIM-302",
        "PD-CLAIM-303",
        "PD-CLAIM-304",
    }
)
UNTESTED_CLAIM_IDS = frozenset(
    {
        "PD-CLAIM-059",
        "PD-CLAIM-070",
        "PD-CLAIM-079",
        "PD-CLAIM-094",
        "PD-CLAIM-117",
        "PD-CLAIM-130",
        "PD-CLAIM-143",
        "PD-CLAIM-144",
        "PD-CLAIM-145",
        "PD-CLAIM-198",
        "PD-CLAIM-199",
        "PD-CLAIM-206",
        "PD-CLAIM-234",
        "PD-CLAIM-245",
        "PD-CLAIM-259",
    }
)
INCONCLUSIVE_CLAIM_IDS = frozenset(
    {
        "PD-CLAIM-093",
        "PD-CLAIM-183",
        "PD-CLAIM-200",
        "PD-CLAIM-246",
        "PD-CLAIM-296",
    }
)
CONTRADICTED_CLAIM_IDS: frozenset[str] = frozenset()


def _paths(root: Path) -> tuple[Path, Path]:
    data = root / "docs/research/proximal_distal_energy_transfer/data"
    return data / "claim_audit_registry.json", data / "claim_candidate_inventory.json"


def _reconcile_reviewer_projection(
    registry: dict[str, Any], inventory: dict[str, Any]
) -> None:
    candidate_ids = {item["candidate_id"] for item in inventory["candidates"]}
    reviews = {item["candidate_id"]: item for item in registry["candidate_reviews"]}
    for candidate_id in PRIOR_REVIEWER_PROJECTION_CANDIDATE_IDS:
        reviews.pop(candidate_id, None)
    unknown_reviews = set(reviews) - candidate_ids
    if unknown_reviews:
        raise ValueError(
            f"Reviewed candidates disappeared from the paper: {sorted(unknown_reviews)}"
        )
    missing_reviews = candidate_ids - set(reviews)
    if missing_reviews != REVIEWER_PROJECTION_CANDIDATE_IDS:
        raise ValueError(
            "Reviewer projection candidates differ from the explicitly reviewed "
            f"set: {sorted(missing_reviews)}"
        )
    for candidate_id in sorted(REVIEWER_PROJECTION_CANDIDATE_IDS):
        reviews[candidate_id] = {
            "candidate_id": candidate_id,
            "disposition": "editorial_or_navigation",
            "claim_ids": [],
            "rationale": (
                "Generated reviewer projection of existing claim outcomes and "
                "qualification axes; it introduces no new scientific estimand."
            ),
            "reviewer": "Codex technical audit",
            "last_verified_on": "2026-08-23",
        }
    registry["candidate_reviews"] = list(reviews.values())
    registry["paper"]["source_digest"] = REVIEWED_SOURCE_DIGEST
    registry["audit_scope"]["current_scope"] = (
        "The complete 1100-candidate paper inventory is adjudicated. Repeated "
        "methods, summary, limitation, provenance, and model-tier passages inherit "
        "the primary claim boundaries; generated reviewer tables and editorial "
        "anchors are explicitly classified as nonclaims."
    )


def _outcome(claim_id: str) -> str:
    if claim_id in SUPPORTED_CLAIM_IDS:
        return "supported"
    if claim_id in UNTESTED_CLAIM_IDS:
        return "untested"
    if claim_id in INCONCLUSIVE_CLAIM_IDS:
        return "inconclusive"
    if claim_id in CONTRADICTED_CLAIM_IDS:
        return "contradicted"
    raise ValueError(f"{claim_id}: no explicit reviewed adjudication outcome")


def migrate(root: Path) -> dict[str, int]:
    """Migrate the frozen registry and inventory or fail before writing."""
    registry_path, inventory_path = _paths(root.resolve())
    registry: dict[str, Any] = json.loads(registry_path.read_text(encoding="utf-8"))
    digest = registry.get("paper", {}).get("source_digest")
    claims = registry.get("claims")
    if digest not in {
        PRE_ADJUDICATION_SOURCE_DIGEST,
        PRIOR_REVIEWED_SOURCE_DIGEST,
        REVIEWED_SOURCE_DIGEST,
    }:
        raise ValueError(
            "Paper digest differs from the explicitly reviewed v2 snapshot"
        )
    if not isinstance(claims, list) or len(claims) != REVIEWED_CLAIM_COUNT:
        raise ValueError("Claim count differs from the explicitly reviewed v2 snapshot")

    paper_path = root.resolve() / registry["paper"]["source"]
    inventory = build_candidate_inventory(paper_path, repository_root=root.resolve())
    if inventory["source_digest"] != REVIEWED_SOURCE_DIGEST:
        raise ValueError("Reviewer projection differs from the reviewed paper snapshot")
    if digest in {PRE_ADJUDICATION_SOURCE_DIGEST, PRIOR_REVIEWED_SOURCE_DIGEST}:
        _reconcile_reviewer_projection(registry, inventory)
    else:
        candidate_ids = {item["candidate_id"] for item in inventory["candidates"]}
        reviewed_ids = {item["candidate_id"] for item in registry["candidate_reviews"]}
        if reviewed_ids != candidate_ids:
            raise ValueError("Reconciled reviewer projection coverage is incomplete")

    claim_ids = {claim.get("claim_id") for claim in claims}
    reviewed_claim_ids = (
        SUPPORTED_CLAIM_IDS
        | UNTESTED_CLAIM_IDS
        | INCONCLUSIVE_CLAIM_IDS
        | CONTRADICTED_CLAIM_IDS
    )
    unreviewed = sorted(claim_ids - reviewed_claim_ids)
    if unreviewed:
        raise ValueError(
            f"Registry contains claims without explicit reviewed outcomes: {unreviewed}"
        )
    missing = sorted(reviewed_claim_ids - claim_ids)
    if missing:
        raise ValueError(
            f"Reviewed outcome IDs are missing from the registry: {missing}"
        )
    overlaps = (
        (SUPPORTED_CLAIM_IDS & UNTESTED_CLAIM_IDS)
        | (SUPPORTED_CLAIM_IDS & INCONCLUSIVE_CLAIM_IDS)
        | (SUPPORTED_CLAIM_IDS & CONTRADICTED_CLAIM_IDS)
        | (UNTESTED_CLAIM_IDS & INCONCLUSIVE_CLAIM_IDS)
        | (UNTESTED_CLAIM_IDS & CONTRADICTED_CLAIM_IDS)
        | (INCONCLUSIVE_CLAIM_IDS & CONTRADICTED_CLAIM_IDS)
    )
    if overlaps:
        raise ValueError(f"Claim IDs have conflicting outcomes: {sorted(overlaps)}")

    counts = dict.fromkeys(("supported", "contradicted", "inconclusive", "untested"), 0)
    migrated_claims: list[dict[str, Any]] = []
    for claim in claims:
        claim_id = claim["claim_id"]
        outcome = _outcome(claim_id)
        counts[outcome] += 1
        migrated: dict[str, Any] = {}
        for key, value in claim.items():
            migrated[key] = value
            if key == "audit_status":
                migrated["adjudication_outcome"] = outcome
        migrated_claims.append(migrated)

    registry["schema_version"] = SCHEMA_VERSION
    registry["claims"] = migrated_claims
    registry["audit_scope"]["normalized_adjudication_status"] = "complete"
    registry["audit_scope"]["normalized_adjudication_policy"] = (
        "Finding-level supported, contradicted, inconclusive, or untested outcome; "
        "never inferred from published_status or audit_status."
    )
    inventory["schema_version"] = SCHEMA_VERSION
    registry_path.write_text(
        json.dumps(registry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    inventory_path.write_text(
        json.dumps(inventory, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return counts


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    counts = migrate(root)
    print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
