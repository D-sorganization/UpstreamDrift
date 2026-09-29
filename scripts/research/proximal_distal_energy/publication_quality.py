"""Inspect and validate the publication quality of the canonical monograph PDF."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .publication_quality_contract import (
    ARCHIVAL_GAPS,
    COMPUTATIONAL_BLOCKERS,
    require_hex,
    require_repository,
    validate_publication_quality,
)

ROOT = Path(__file__).resolve().parents[3]
ARTICLE = ROOT / "docs/research/proximal_distal_energy_transfer"
DEFAULT_PDF = ARTICLE / "proximal_distal_energy_transfer.pdf"
DEFAULT_MANIFEST = ARTICLE / "release_manifest.json"
EXPECTED_TITLE = "Proximal-to-Distal Energy Transfer in the Golf Swing"
EXPECTED_AUTHOR = "Dieter Olson"
SOURCE_REPOSITORY = "https://github.com/D-sorganization/UpstreamDrift"


def _load_fitz():
    try:
        import fitz
    except ImportError as error:
        raise RuntimeError(
            "PyMuPDF is required for PDF publication inspection; install `pymupdf`."
        ) from error
    return fitz


def _tagged(document: Any) -> bool:
    catalog = document.pdf_catalog()
    mark_type, mark_value = document.xref_get_key(catalog, "MarkInfo")
    tree_type, tree_value = document.xref_get_key(catalog, "StructTreeRoot")
    marked = mark_type != "null" and "/Marked true" in mark_value
    structure = tree_type != "null" and tree_value != "null"
    return marked and structure


def _font_inventory(document: Any) -> dict[str, Any]:
    fonts: dict[int, tuple[Any, ...]] = {}
    for page in document:
        for record in page.get_fonts(full=True):
            fonts.setdefault(int(record[0]), record)
    type_counts: dict[str, int] = {}
    type3 = 0
    unembedded = 0
    for xref, record in fonts.items():
        font_type = str(record[2])
        type_counts[font_type] = type_counts.get(font_type, 0) + 1
        if font_type == "Type3":
            type3 += 1
            continue
        try:
            embedded_bytes = document.extract_font(xref)[3]
        except (RuntimeError, ValueError):
            embedded_bytes = b""
        if not embedded_bytes:
            unembedded += 1
    return {
        "resources": len(fonts),
        "types": dict(sorted(type_counts.items())),
        "type3_resources": type3,
        "unembedded_resources": unembedded,
    }


def _valid_uri(uri: str) -> bool:
    parsed = urlparse(uri)
    if parsed.scheme in {"http", "https"}:
        return bool(parsed.netloc)
    if parsed.scheme == "mailto":
        return bool(parsed.path and "@" in parsed.path)
    return False


def _inspect_navigation(document: Any, fitz: Any) -> dict[str, Any]:
    uri_links = 0
    internal_links = 0
    invalid_uris: list[dict[str, Any]] = []
    invalid_internal: list[dict[str, Any]] = []
    for page_index, page in enumerate(document):
        for link in page.get_links():
            kind = link.get("kind")
            if kind == fitz.LINK_URI:
                uri_links += 1
                uri = str(link.get("uri", ""))
                if not _valid_uri(uri):
                    invalid_uris.append({"page": page_index + 1, "uri": uri})
            elif kind == fitz.LINK_GOTO:
                internal_links += 1
                destination = link.get("page")
                if not isinstance(destination, int) or not (
                    0 <= destination < document.page_count
                ):
                    invalid_internal.append(
                        {"page": page_index + 1, "destination": destination}
                    )
    return {
        "outline_entries": len(document.get_toc()),
        "uri_links": uri_links,
        "internal_links": internal_links,
        "invalid_uri_links": invalid_uris,
        "invalid_internal_links": invalid_internal,
    }


def _render_pages(document: Any, fitz: Any, zoom: float) -> tuple[dict[str, Any], int]:
    errors: list[dict[str, Any]] = []
    rendered = 0
    text_pages = 0
    matrix = fitz.Matrix(zoom, zoom)
    for page_index, page in enumerate(document):
        if page.get_text("text").strip():
            text_pages += 1
        try:
            pixmap = page.get_pixmap(matrix=matrix, colorspace=fitz.csGRAY, alpha=False)
            if pixmap.width <= 0 or pixmap.height <= 0 or not pixmap.samples:
                raise RuntimeError("renderer returned an empty bitmap")
            rendered += 1
        except (RuntimeError, ValueError) as error:
            errors.append({"page": page_index + 1, "error": str(error)[:240]})
    return {"pages_rendered": rendered, "errors": errors}, text_pages


def _findings(
    *,
    metadata: dict[str, str],
    expected_metadata: dict[str, str],
    navigation: dict[str, Any],
    rendering: dict[str, Any],
    text_pages: int,
    tagged: bool,
    fast_web_access: bool,
    fonts: dict[str, Any],
) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []

    def add(code: str, message: str, level: str) -> None:
        findings.append({"code": code, "level": level, "message": message})

    if metadata.get("title") != expected_metadata["title"]:
        add(
            "metadata-title-mismatch",
            "PDF title does not match the source title.",
            "blocker",
        )
    if metadata.get("author") != expected_metadata["author"]:
        add(
            "metadata-author-mismatch",
            "PDF author does not match the source author.",
            "blocker",
        )
    if navigation["outline_entries"] == 0:
        add("missing-outline", "PDF has no navigable document outline.", "blocker")
    if navigation["invalid_uri_links"]:
        add("invalid-uri-link", "PDF contains a malformed external URI.", "blocker")
    if navigation["invalid_internal_links"]:
        add(
            "invalid-internal-link",
            "PDF contains an invalid internal destination.",
            "blocker",
        )
    if rendering["errors"]:
        add("page-render-failed", "At least one PDF page failed to render.", "blocker")
    if text_pages == 0:
        add("no-extractable-text", "PDF contains no extractable text.", "blocker")
    if not tagged:
        add("pdf-not-tagged", "PDF lacks a tagged structure tree.", "archival-gap")
    if not fast_web_access:
        add(
            "pdf-not-fast-web-access",
            "PDF is not linearized for web access.",
            "archival-gap",
        )
    if fonts["type3_resources"]:
        add(
            "type3-font-resource", "PDF contains Type 3 font resources.", "archival-gap"
        )
    if fonts["unembedded_resources"]:
        add(
            "unembedded-font-resource",
            "PDF contains unembedded font resources.",
            "archival-gap",
        )
    return findings


def inspect_publication_pdf(
    path: Path,
    *,
    expected_title: str,
    expected_author: str,
    source_repository: str,
    source_revision: str,
    release_manifest_sha256: str,
    render_zoom: float = 0.25,
) -> dict[str, Any]:
    """Return a complete immutable inspection report without mutating the PDF."""
    pdf_path = path.resolve()
    if not pdf_path.is_file():
        raise FileNotFoundError(pdf_path)
    if not expected_title.strip() or not expected_author.strip():
        raise ValueError("expected title and author must be non-empty")
    if not math.isfinite(render_zoom) or not 0 < render_zoom <= 4:
        raise ValueError("render_zoom must be finite and in (0, 4]")
    source = {
        "repository": require_repository(source_repository),
        "revision": require_hex(source_revision, length=40, label="source_revision"),
        "release_manifest_sha256": require_hex(
            release_manifest_sha256,
            length=64,
            label="release_manifest_sha256",
        ),
    }
    fitz = _load_fitz()
    with fitz.open(pdf_path) as document:
        if not document.is_pdf or document.page_count < 1:
            raise ValueError("publication PDF must contain at least one page")
        metadata = {
            key: str(document.metadata.get(key, ""))
            for key in (
                "title",
                "author",
                "subject",
                "keywords",
                "creator",
                "producer",
                "format",
            )
        }
        navigation = _inspect_navigation(document, fitz)
        rendering, text_pages = _render_pages(document, fitz, render_zoom)
        fonts = _font_inventory(document)
        tagged = _tagged(document)
        fast_web_access = bool(document.is_fast_webaccess)
        pages = document.page_count
    findings = _findings(
        metadata=metadata,
        expected_metadata={"title": expected_title, "author": expected_author},
        navigation=navigation,
        rendering=rendering,
        text_pages=text_pages,
        tagged=tagged,
        fast_web_access=fast_web_access,
        fonts=fonts,
    )
    codes = {finding["code"] for finding in findings}
    computational_ready = not bool(codes & COMPUTATIONAL_BLOCKERS)
    archival_ready = computational_ready and not bool(codes & ARCHIVAL_GAPS)
    return {
        "schema_version": "proximal-distal-publication-quality-v1",
        "source": source,
        "publication": {
            "path": pdf_path.name,
            "sha256": hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
            "bytes": pdf_path.stat().st_size,
            "pages": pages,
            "metadata": metadata,
            "expected_metadata": {
                "title": expected_title,
                "author": expected_author,
            },
            "fast_web_access": fast_web_access,
        },
        "navigation": navigation,
        "rendering": rendering,
        "accessibility": {
            "tagged": tagged,
            "pages_with_extractable_text": text_pages,
            "font_inventory": fonts,
        },
        "findings": findings,
        "readiness": {
            "computational_release": computational_ready,
            "archival_publication": archival_ready,
        },
    }


def main() -> None:
    """Inspect the canonical PDF and optionally write its quality report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument(
        "--profile", choices=("computational", "archival"), default="computational"
    )
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = inspect_publication_pdf(
        args.pdf,
        expected_title=EXPECTED_TITLE,
        expected_author=EXPECTED_AUTHOR,
        source_repository=SOURCE_REPOSITORY,
        source_revision=args.source_revision,
        release_manifest_sha256=hashlib.sha256(
            DEFAULT_MANIFEST.read_bytes()
        ).hexdigest(),
    )
    validate_publication_quality(report, profile=args.profile)
    payload = json.dumps(report, indent=2) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
