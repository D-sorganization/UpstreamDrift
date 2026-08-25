"""Register reviewed numeric-evidence pointer maps in the claim authority.

The contract file is canonical input.  This command never infers pointers from
claim prose: it only applies reviewed mappings whose statement digest and
literal inventory still match the registry.  Any changed, added, or removed
numeric claim therefore fails closed until its numeric contract is reviewed.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.research.proximal_distal_energy.claim_numeric_audit import (
    extract_numeric_literals,
)


ROOT = Path(__file__).resolve().parents[3]
ARTICLE = ROOT / "docs/research/proximal_distal_energy_transfer"
REGISTRY = ARTICLE / "data/claim_audit_registry.json"
CONTRACTS = ARTICLE / "data/claim_numeric_contracts.json"


def statement_digest(statement: str) -> str:
    """Return the stable digest binding a pointer map to exact claim prose."""
    return hashlib.sha256(statement.encode("utf-8")).hexdigest()


def _claims_by_id(records: object, *, context: str) -> dict[str, dict[str, Any]]:
    if not isinstance(records, list):
        raise ValueError(f"{context} claims must be a list")
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"{context} claim must be an object")
        claim_id = record.get("claim_id")
        if not isinstance(claim_id, str) or not claim_id:
            raise ValueError(f"{context} claim_id must be non-empty")
        if claim_id in result:
            raise ValueError(f"{context} duplicate claim_id {claim_id}")
        result[claim_id] = record
    return result


def apply_numeric_contracts(
    registry: dict[str, Any], contracts: dict[str, Any]
) -> dict[str, Any]:
    """Return a registry with complete reviewed numeric mappings applied."""
    if contracts.get("schema_version") != "claim-numeric-contract-v1":
        raise ValueError("unsupported numeric contract schema_version")
    contract_records = contracts.get("claims")
    registry_by_id = _claims_by_id(registry.get("claims"), context="registry")
    contracts_by_id = _claims_by_id(contract_records, context="contract")
    numeric_ids = {
        claim_id
        for claim_id, claim in registry_by_id.items()
        if extract_numeric_literals(str(claim.get("statement", "")))
    }
    contract_ids = set(contracts_by_id)
    if numeric_ids != contract_ids:
        raise ValueError(
            "numeric contract claim coverage mismatch; "
            f"missing={sorted(numeric_ids - contract_ids)}, "
            f"extra={sorted(contract_ids - numeric_ids)}"
        )

    updated = copy.deepcopy(registry)
    updated_by_id = _claims_by_id(updated["claims"], context="registry")
    for claim_id in sorted(numeric_ids):
        claim = updated_by_id[claim_id]
        statement = claim.get("statement")
        if not isinstance(statement, str):
            raise ValueError(f"{claim_id}: statement must be a string")
        contract = contracts_by_id[claim_id]
        expected_digest = statement_digest(statement)
        if contract.get("statement_sha256") != expected_digest:
            raise ValueError(f"{claim_id}: stale numeric contract statement digest")
        entries = contract.get("numeric_evidence")
        if not isinstance(entries, list):
            raise ValueError(f"{claim_id}: numeric_evidence must be a list")
        expected_literals = {
            str(item["literal_id"]) for item in extract_numeric_literals(statement)
        }
        actual_literals = {
            str(item.get("literal_id")) for item in entries if isinstance(item, dict)
        }
        if expected_literals != actual_literals or len(entries) != len(actual_literals):
            raise ValueError(f"{claim_id}: stale numeric contract literal inventory")
        artifacts = claim.get("evidence_artifacts")
        if not isinstance(artifacts, list):
            raise ValueError(f"{claim_id}: evidence_artifacts must be a list")
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError(
                    f"{claim_id}: numeric evidence entry must be an object"
                )
            artifact = entry.get("artifact")
            if not isinstance(artifact, str) or not artifact:
                raise ValueError(f"{claim_id}: numeric artifact must be non-empty")
            if artifact not in artifacts:
                artifacts.append(artifact)
        claim["numeric_evidence"] = copy.deepcopy(entries)
        comparisons = contract.get("numeric_comparisons", [])
        if not isinstance(comparisons, list):
            raise ValueError(f"{claim_id}: numeric_comparisons must be a list")
        if comparisons:
            for comparison in comparisons:
                if not isinstance(comparison, dict):
                    raise ValueError(
                        f"{claim_id}: numeric comparison must be an object"
                    )
                artifact = comparison.get("artifact")
                if not isinstance(artifact, str) or not artifact:
                    raise ValueError(
                        f"{claim_id}: comparison artifact must be non-empty"
                    )
                if artifact not in artifacts:
                    artifacts.append(artifact)
            claim["numeric_comparisons"] = copy.deepcopy(comparisons)
        else:
            claim.pop("numeric_comparisons", None)
    return updated


def _serialized(value: object) -> str:
    return json.dumps(value, indent=2) + "\n"


def register(*, check: bool = False) -> dict[str, int | str]:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    contracts = json.loads(CONTRACTS.read_text(encoding="utf-8"))
    updated = apply_numeric_contracts(registry, contracts)
    rendered = _serialized(updated)
    if check:
        if registry != updated:
            raise ValueError("claim registry numeric evidence is stale; run register")
    else:
        REGISTRY.write_text(rendered, encoding="utf-8")
    return {
        "claim_count": len(updated["claims"]),
        "numeric_contract_count": len(contracts["claims"]),
        "mode": "check" if check else "register",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", nargs="?", choices=("register", "check"), default="register"
    )
    args = parser.parse_args()
    print(json.dumps(register(check=args.mode == "check"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
