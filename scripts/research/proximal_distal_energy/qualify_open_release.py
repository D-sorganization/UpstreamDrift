"""Write or validate the proximal-distal open-resource release bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from .biomechanics_evidence_bridge import (
    BRIDGE_REL as BIOMECHANICS_EVIDENCE_BRIDGE_REL,
    validate_biomechanics_evidence_bridge,
)
from .biomechanics_source_register import (
    SOURCE_REGISTER_REL as BIOMECHANICS_SOURCE_REGISTER_REL,
    validate_biomechanics_source_register,
)
from .biomechanics_evidence_surfaces import (
    validate_biomechanics_evidence_surfaces,
)
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
from .publication_quality import (
    EXPECTED_AUTHOR,
    EXPECTED_TITLE,
    SOURCE_REPOSITORY,
    inspect_publication_pdf,
    validate_publication_quality,
)
from .publication_quality_contract import PublicationProfile

ROOT = Path(__file__).resolve().parents[3]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("write", "validate", "list-presets"))
    parser.add_argument(
        "--source-revision",
        help="Exact 40-character source revision; defaults to the checked-out HEAD.",
    )
    parser.add_argument(
        "--publication-profile",
        choices=("computational", "archival"),
        default="computational",
    )
    parser.add_argument("--publication-report", type=Path)
    return parser


def _source_revision(explicit: str | None) -> str:
    if explicit is not None:
        return explicit
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _publication_report(
    manifest_payload: str,
    *,
    source_revision: str,
    profile: PublicationProfile,
) -> dict[str, object]:
    report = inspect_publication_pdf(
        ROOT / ARTICLE_REL / "proximal_distal_energy_transfer.pdf",
        expected_title=EXPECTED_TITLE,
        expected_author=EXPECTED_AUTHOR,
        source_repository=SOURCE_REPOSITORY,
        source_revision=source_revision,
        release_manifest_sha256=hashlib.sha256(
            manifest_payload.encode("utf-8")
        ).hexdigest(),
    )
    validate_publication_quality(report, profile=profile)
    return report


def _write_publication_report(path: Path | None, report: dict[str, object]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    """Execute the release qualification action."""
    args = _parser().parse_args()
    action = args.action
    manifest_path = ROOT / ARTICLE_REL / "release_manifest.json"
    if action == "write":
        validate_review()
        claim_manifest_path = ROOT / CLAIM_EVIDENCE_MANIFEST_REL
        claim_manifest = build_claim_evidence_manifest(ROOT)
        external_review = json.loads(
            (ROOT / EXTERNAL_SOURCE_REVIEW_REL).read_text(encoding="utf-8")
        )
        validate_external_source_review(ROOT, external_review, claim_manifest)
        biomechanics_bridge = json.loads(
            (ROOT / BIOMECHANICS_EVIDENCE_BRIDGE_REL).read_text(encoding="utf-8")
        )
        validate_biomechanics_evidence_bridge(
            ROOT, biomechanics_bridge, external_review
        )
        biomechanics_sources = json.loads(
            (ROOT / BIOMECHANICS_SOURCE_REGISTER_REL).read_text(encoding="utf-8")
        )
        validate_biomechanics_source_register(ROOT, biomechanics_sources)
        validate_biomechanics_evidence_surfaces(ROOT)
        manifest = build_release_manifest(ROOT)
        manifest_payload = json.dumps(manifest, indent=2) + "\n"
        publication_report = _publication_report(
            manifest_payload,
            source_revision=_source_revision(args.source_revision),
            profile=args.publication_profile,
        )
        manifest_path.write_text(manifest_payload, encoding="utf-8")
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
        _write_publication_report(args.publication_report, publication_report)
        return
    manifest_payload = manifest_path.read_text(encoding="utf-8")
    manifest = json.loads(manifest_payload)
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
        biomechanics_bridge = json.loads(
            (ROOT / BIOMECHANICS_EVIDENCE_BRIDGE_REL).read_text(encoding="utf-8")
        )
        result["biomechanics_evidence_bridge"] = validate_biomechanics_evidence_bridge(
            ROOT, biomechanics_bridge, external_review
        )
        biomechanics_sources = json.loads(
            (ROOT / BIOMECHANICS_SOURCE_REGISTER_REL).read_text(encoding="utf-8")
        )
        result["biomechanics_source_register"] = validate_biomechanics_source_register(
            ROOT, biomechanics_sources
        )
        result["biomechanics_evidence_surfaces"] = (
            validate_biomechanics_evidence_surfaces(ROOT)
        )
        publication_report = _publication_report(
            manifest_payload,
            source_revision=_source_revision(args.source_revision),
            profile=args.publication_profile,
        )
        result["publication_quality"] = publication_report
        _write_publication_report(args.publication_report, publication_report)
        print(json.dumps(result, indent=2))
        return
    for name, preset in manifest["presets"].items():
        print(f"{name}: {preset['command']}")


if __name__ == "__main__":
    main()
