"""Write or validate the proximal-distal open-resource release bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .claim_evidence_integrity import (
    MANIFEST_REL as CLAIM_EVIDENCE_MANIFEST_REL,
    build_claim_evidence_manifest,
    validate_claim_evidence_manifest,
)
from .external_source_review import (
    REVIEW_REL as EXTERNAL_SOURCE_REVIEW_REL,
    validate_external_source_review,
)
from .release_bundle import (
    ARTICLE_REL,
    build_release_manifest,
    checksum_lines,
    validate_release_manifest,
)
from .release_claim_review import validate_review

ROOT = Path(__file__).resolve().parents[3]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("write", "validate", "list-presets"))
    return parser


def main() -> None:
    """Execute the release qualification action."""
    action = _parser().parse_args().action
    manifest_path = ROOT / ARTICLE_REL / "release_manifest.json"
    if action == "write":
        validate_review()
        claim_manifest_path = ROOT / CLAIM_EVIDENCE_MANIFEST_REL
        claim_manifest = build_claim_evidence_manifest(ROOT)
        external_review = json.loads(
            (ROOT / EXTERNAL_SOURCE_REVIEW_REL).read_text(encoding="utf-8")
        )
        validate_external_source_review(ROOT, external_review, claim_manifest)
        manifest = build_release_manifest(ROOT)
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        checksum_path = ROOT / ARTICLE_REL / "CHECKSUMS.sha256"
        checksum_path.write_text(
            "\n".join(checksum_lines(manifest)) + "\n", encoding="utf-8"
        )
        claim_manifest = build_claim_evidence_manifest(ROOT)
        claim_manifest_path.write_text(
            json.dumps(claim_manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        print(manifest_path)
        print(checksum_path)
        print(claim_manifest_path)
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if action == "validate":
        validate_review()
        result = validate_release_manifest(ROOT, manifest)
        claim_manifest_path = ROOT / CLAIM_EVIDENCE_MANIFEST_REL
        claim_manifest = json.loads(claim_manifest_path.read_text(encoding="utf-8"))
        result["claim_evidence"] = validate_claim_evidence_manifest(
            ROOT, claim_manifest
        )
        external_review = json.loads(
            (ROOT / EXTERNAL_SOURCE_REVIEW_REL).read_text(encoding="utf-8")
        )
        result["external_sources"] = validate_external_source_review(
            ROOT, external_review, claim_manifest
        )
        print(json.dumps(result, indent=2))
        return
    for name, preset in manifest["presets"].items():
        print(f"{name}: {preset['command']}")


if __name__ == "__main__":
    main()
