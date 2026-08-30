"""Publication-quality contracts for the proximal-distal monograph PDF."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

try:
    import fitz
except ImportError:  # pragma: no cover - exercised by the dependency-light CI lane
    fitz = None

from scripts.research.proximal_distal_energy.publication_quality import (
    inspect_publication_pdf,
    validate_publication_quality,
)

ROOT = Path(__file__).resolve().parents[2]
ARTICLE = ROOT / "docs/research/proximal_distal_energy_transfer"
PDF = ARTICLE / "proximal_distal_energy_transfer.pdf"
MANIFEST = ARTICLE / "release_manifest.json"
SOURCE_REPOSITORY = "https://github.com/D-sorganization/UpstreamDrift"
SOURCE_REVISION = "a" * 40
pytestmark = pytest.mark.unit
requires_fitz = pytest.mark.skipif(
    fitz is None,
    reason="PyMuPDF is installed only by the publication-quality profile",
)


def _write_pdf(path: Path, *, title: str = "Release Paper") -> None:
    assert fitz is not None
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Computational evidence")
    page.insert_link(
        {
            "kind": fitz.LINK_URI,
            "from": fitz.Rect(70, 55, 210, 80),
            "uri": "https://example.org/evidence",
        }
    )
    document.set_toc([[1, "Evidence", 1]])
    document.set_metadata({"title": title, "author": "Dieter Olson"})
    document.save(path)
    document.close()


def _inspect(path: Path, *, title: str = "Release Paper") -> dict[str, object]:
    return inspect_publication_pdf(
        path,
        expected_title=title,
        expected_author="Dieter Olson",
        source_repository=SOURCE_REPOSITORY,
        source_revision=SOURCE_REVISION,
        release_manifest_sha256="b" * 64,
        render_zoom=0.25,
    )


@requires_fitz
def test_computational_profile_requires_identity_metadata_and_every_page_render(
    tmp_path: Path,
) -> None:
    path = tmp_path / "paper.pdf"
    _write_pdf(path)

    report = _inspect(path)

    assert report["schema_version"] == "proximal-distal-publication-quality-v1"
    assert report["source"] == {
        "repository": SOURCE_REPOSITORY,
        "revision": SOURCE_REVISION,
        "release_manifest_sha256": "b" * 64,
    }
    publication = report["publication"]
    assert publication["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert publication["metadata"]["title"] == "Release Paper"
    assert publication["metadata"]["author"] == "Dieter Olson"
    assert report["navigation"]["outline_entries"] == 1
    assert report["navigation"]["uri_links"] == 1
    assert report["rendering"] == {"pages_rendered": 1, "errors": []}
    assert report["readiness"]["computational_release"] is True
    assert validate_publication_quality(report, profile="computational")["valid"]


@requires_fitz
def test_archive_profile_discloses_untagged_and_nonoptimized_pdf(
    tmp_path: Path,
) -> None:
    path = tmp_path / "paper.pdf"
    _write_pdf(path)

    report = _inspect(path)

    assert report["accessibility"]["tagged"] is False
    assert report["publication"]["fast_web_access"] is False
    assert report["readiness"]["archival_publication"] is False
    finding_codes = {finding["code"] for finding in report["findings"]}
    assert {"pdf-not-tagged", "pdf-not-fast-web-access"} <= finding_codes
    with pytest.raises(ValueError, match="archival publication"):
        validate_publication_quality(report, profile="archival")


@pytest.mark.parametrize(
    ("revision", "manifest_sha"),
    [("main", "b" * 64), ("a" * 40, "not-a-digest")],
)
@requires_fitz
def test_source_identity_rejects_unpinned_values(
    tmp_path: Path, revision: str, manifest_sha: str
) -> None:
    path = tmp_path / "paper.pdf"
    _write_pdf(path)

    with pytest.raises(ValueError, match="full lowercase hexadecimal"):
        inspect_publication_pdf(
            path,
            expected_title="Release Paper",
            expected_author="Dieter Olson",
            source_repository=SOURCE_REPOSITORY,
            source_revision=revision,
            release_manifest_sha256=manifest_sha,
        )


@requires_fitz
def test_metadata_mismatch_fails_the_computational_profile(tmp_path: Path) -> None:
    path = tmp_path / "paper.pdf"
    _write_pdf(path, title="Wrong Paper")

    report = _inspect(path)

    assert report["readiness"]["computational_release"] is False
    assert "metadata-title-mismatch" in {
        finding["code"] for finding in report["findings"]
    }
    with pytest.raises(ValueError, match="computational release"):
        validate_publication_quality(report, profile="computational")


def test_validator_rejects_a_forged_ready_flag() -> None:
    with pytest.raises(ValueError, match="schema version"):
        validate_publication_quality(
            {"readiness": {"computational_release": True}},
            profile="computational",
        )


@requires_fitz
def test_validator_rejects_findings_not_derived_from_the_pdf(tmp_path: Path) -> None:
    path = tmp_path / "paper.pdf"
    _write_pdf(path)
    report = deepcopy(_inspect(path))
    report["findings"].append(
        {
            "code": "metadata-title-mismatch",
            "level": "blocker",
            "message": "Invented blocker.",
        }
    )

    with pytest.raises(ValueError, match="findings are inconsistent"):
        validate_publication_quality(report, profile="computational")


@requires_fitz
def test_validator_rejects_malformed_finding_metadata(tmp_path: Path) -> None:
    path = tmp_path / "paper.pdf"
    _write_pdf(path)
    report = deepcopy(_inspect(path))
    report["findings"][0]["level"] = "blocker"

    with pytest.raises(ValueError, match="malformed finding"):
        validate_publication_quality(report, profile="computational")


def test_canonical_pdf_byte_identity_is_dependency_free() -> None:
    digest = hashlib.sha256(PDF.read_bytes()).hexdigest()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    artifact = manifest["artifacts"][
        "docs/research/proximal_distal_energy_transfer/"
        "proximal_distal_energy_transfer.pdf"
    ]

    assert digest == (
        "72334a25bffb2d0c1cf4eafe8de9290f91dc30656352eb82fc66877d295f8064"
    )
    assert PDF.stat().st_size == 2_012_265
    assert artifact == {"sha256": digest, "bytes": 2_012_265}


@requires_fitz
def test_opening_evidence_callout_is_not_split_across_pages() -> None:
    assert fitz is not None
    opening = "Scope and Evidence Categories"
    closing = "prior literature is not independent empirical confirmation."
    with fitz.open(PDF) as document:
        page_text = [page.get_text() for page in document]

    opening_pages = {index for index, text in enumerate(page_text) if opening in text}
    closing_pages = {index for index, text in enumerate(page_text) if closing in text}
    assert opening_pages == closing_pages, (
        "The opening evidence-category callout must render on one page; "
        f"opening={sorted(opening_pages)}, closing={sorted(closing_pages)}"
    )


@requires_fitz
def test_canonical_pdf_passes_the_computational_profile() -> None:
    report = inspect_publication_pdf(
        PDF,
        expected_title="Proximal-to-Distal Energy Transfer in the Golf Swing",
        expected_author="Dieter Olson",
        source_repository=SOURCE_REPOSITORY,
        source_revision=SOURCE_REVISION,
        release_manifest_sha256=hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        render_zoom=0.2,
    )

    assert report["publication"]["sha256"] == (
        "72334a25bffb2d0c1cf4eafe8de9290f91dc30656352eb82fc66877d295f8064"
    )
    assert report["publication"]["bytes"] == 2_012_265
    assert report["publication"]["pages"] == 256
    assert report["publication"]["fast_web_access"] is True
    assert report["navigation"] == {
        "outline_entries": 255,
        "uri_links": 197,
        "internal_links": 0,
        "invalid_uri_links": [],
        "invalid_internal_links": [],
    }
    assert report["rendering"]["pages_rendered"] == 256
    assert report["rendering"]["errors"] == []
    assert report["accessibility"] == {
        "tagged": False,
        "pages_with_extractable_text": 256,
        "font_inventory": {
            "resources": 143,
            "types": {"Type0": 17, "Type1": 2, "Type3": 124},
            "type3_resources": 124,
            "unembedded_resources": 2,
        },
    }
    assert validate_publication_quality(report, profile="computational")["valid"]
    assert report["readiness"]["archival_publication"] is False
