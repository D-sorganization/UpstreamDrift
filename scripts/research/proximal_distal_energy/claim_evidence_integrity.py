"""Build and validate deterministic integrity records for claim support."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

ARTICLE_REL = Path("docs/research/proximal_distal_energy_transfer")
REGISTRY_REL = ARTICLE_REL / "data/claim_audit_registry.json"
MANIFEST_REL = ARTICLE_REL / "data/claim_evidence_manifest.json"
SCHEMA_VERSION = "proximal-distal-claim-evidence-integrity-v1"


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _local_path(reference: str, root: Path) -> tuple[str, Path]:
    path_text = reference.partition("#")[0]
    relative = Path(path_text)
    if relative.is_absolute():
        raise ValueError(
            f"Local evidence path must be repository-relative: {reference}"
        )
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Local evidence path escapes repository root: {reference}")
    if not resolved.is_file():
        raise ValueError(f"Missing local evidence artifact: {reference}")
    return relative.as_posix(), resolved


def build_claim_evidence_manifest(
    root: str | Path, registry_path: str | Path | None = None
) -> dict[str, Any]:
    """Hash-pin local support and inventory external claim-support URLs."""
    root_path = Path(root).resolve()
    registry_file = Path(registry_path or root_path / REGISTRY_REL).resolve()
    registry = json.loads(registry_file.read_text(encoding="utf-8"))
    local_claims: dict[str, set[str]] = {}
    external_claims: dict[str, set[str]] = {}
    claims: dict[str, dict[str, list[str]]] = {}
    reference_count = 0

    for claim in registry["claims"]:
        claim_id = claim["claim_id"]
        local_refs: list[str] = []
        external_refs: list[str] = []
        for reference in claim["evidence_artifacts"]:
            reference_count += 1
            if reference.startswith(("https://", "http://")):
                external_refs.append(reference)
                external_claims.setdefault(reference, set()).add(claim_id)
                continue
            relative, _ = _local_path(reference, root_path)
            local_refs.append(reference)
            local_claims.setdefault(relative, set()).add(claim_id)
        claims[claim_id] = {
            "local_references": local_refs,
            "external_references": external_refs,
        }

    local_artifacts = {}
    for relative in sorted(local_claims):
        path = root_path / relative
        local_artifacts[relative] = {
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
            "referenced_by": sorted(local_claims[relative]),
        }
    external_urls = {}
    for url in sorted(external_claims):
        parsed = urlsplit(url)
        external_urls[url] = {
            "scheme": parsed.scheme,
            "host": parsed.hostname,
            "referenced_by": sorted(external_claims[url]),
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "registry": {
            "path": registry_file.relative_to(root_path).as_posix(),
            "sha256": _sha256(registry_file),
        },
        "scope": {
            "local_artifact_semantics": "content_integrity_not_independent_validation",
            "external_url_semantics": "inventory_only_not_scientific_validation",
            "network_access_required": False,
        },
        "summary": {
            "claim_count": len(claims),
            "evidence_reference_count": reference_count,
            "local_artifact_count": len(local_artifacts),
            "external_url_count": len(external_urls),
        },
        "claims": claims,
        "local_artifacts": local_artifacts,
        "external_urls": external_urls,
    }


def validate_claim_evidence_manifest(
    root: str | Path, manifest: dict[str, Any]
) -> dict[str, Any]:
    """Fail closed when claim coverage, file content, or URL inventory drifts."""
    expected = build_claim_evidence_manifest(root)
    if manifest != expected:
        mismatches = [key for key in expected if manifest.get(key) != expected.get(key)]
        raise ValueError(
            "claim evidence manifest validation failed: mismatched sections="
            + ",".join(mismatches)
        )
    return {
        "valid": True,
        "mismatches": [],
        **expected["summary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("write", "validate"))
    args = parser.parse_args()
    root = _repository_root()
    path = root / MANIFEST_REL
    if args.action == "write":
        path.write_text(
            json.dumps(build_claim_evidence_manifest(root), indent=2) + "\n",
            encoding="utf-8",
        )
        print(path)
        return
    manifest = json.loads(path.read_text(encoding="utf-8"))
    print(json.dumps(validate_claim_evidence_manifest(root, manifest), indent=2))


if __name__ == "__main__":
    main()
