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

from .claim_audit import SCHEMA_VERSION

REVIEWED_SOURCE_DIGEST = (
    "46dc909be65d129414533e41800f4f68a118c2086faf7def710dcfe4c38a03ac"
)
REVIEWED_CLAIM_COUNT = 295

# These sets encode the finding-level review, not a text-pattern mapping.
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


def _outcome(claim_id: str) -> str:
    if claim_id in UNTESTED_CLAIM_IDS:
        return "untested"
    if claim_id in INCONCLUSIVE_CLAIM_IDS:
        return "inconclusive"
    if claim_id in CONTRADICTED_CLAIM_IDS:
        return "contradicted"
    return "supported"


def migrate(root: Path) -> dict[str, int]:
    """Migrate the frozen registry and inventory or fail before writing."""
    registry_path, inventory_path = _paths(root.resolve())
    registry: dict[str, Any] = json.loads(registry_path.read_text(encoding="utf-8"))
    inventory: dict[str, Any] = json.loads(inventory_path.read_text(encoding="utf-8"))
    digest = registry.get("paper", {}).get("source_digest")
    claims = registry.get("claims")
    if digest != REVIEWED_SOURCE_DIGEST:
        raise ValueError(
            "Paper digest differs from the explicitly reviewed v2 snapshot"
        )
    if not isinstance(claims, list) or len(claims) != REVIEWED_CLAIM_COUNT:
        raise ValueError("Claim count differs from the explicitly reviewed v2 snapshot")

    claim_ids = {claim.get("claim_id") for claim in claims}
    reviewed_exceptions = (
        UNTESTED_CLAIM_IDS | INCONCLUSIVE_CLAIM_IDS | CONTRADICTED_CLAIM_IDS
    )
    missing = sorted(reviewed_exceptions - claim_ids)
    if missing:
        raise ValueError(
            f"Reviewed outcome IDs are missing from the registry: {missing}"
        )
    overlaps = (
        (UNTESTED_CLAIM_IDS & INCONCLUSIVE_CLAIM_IDS)
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
