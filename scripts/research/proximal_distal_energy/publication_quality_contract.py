"""Design-by-contract validation for publication-quality inspection reports."""

from __future__ import annotations

import re
from typing import Any, Literal
from urllib.parse import urlparse

PublicationProfile = Literal["computational", "archival"]

COMPUTATIONAL_BLOCKERS = frozenset(
    {
        "invalid-internal-link",
        "invalid-uri-link",
        "metadata-author-mismatch",
        "metadata-title-mismatch",
        "missing-outline",
        "no-extractable-text",
        "page-render-failed",
    }
)
ARCHIVAL_GAPS = frozenset(
    {
        "pdf-not-fast-web-access",
        "pdf-not-tagged",
        "type3-font-resource",
        "unembedded-font-resource",
    }
)
_FINDING_LEVELS = {
    **dict.fromkeys(COMPUTATIONAL_BLOCKERS, "blocker"),
    **dict.fromkeys(ARCHIVAL_GAPS, "archival-gap"),
}
_HEX_PATTERN = re.compile(r"^[0-9a-f]+$")


def require_hex(value: str, *, length: int, label: str) -> str:
    """Return a pinned hexadecimal value or reject it before publication."""
    if len(value) != length or _HEX_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a full lowercase hexadecimal value")
    return value


def require_repository(value: str) -> str:
    """Return a normalized absolute HTTPS repository URL."""
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError("source_repository must be an absolute HTTPS repository URL")
    return value.rstrip("/")


def _derived_findings(
    publication: dict[str, Any],
    rendering: dict[str, Any],
    navigation: dict[str, Any],
    accessibility: dict[str, Any],
) -> set[str]:
    pages = publication.get("pages")
    if type(pages) is not int or pages < 1:
        raise ValueError("publication quality report has an invalid page count")
    metadata = publication.get("metadata")
    expected = publication.get("expected_metadata")
    if not isinstance(metadata, dict) or not isinstance(expected, dict):
        raise ValueError("publication quality report is missing metadata authority")
    derived: set[str] = set()
    if metadata.get("title") != expected.get("title"):
        derived.add("metadata-title-mismatch")
    if metadata.get("author") != expected.get("author"):
        derived.add("metadata-author-mismatch")
    if navigation.get("outline_entries") == 0:
        derived.add("missing-outline")
    if navigation.get("invalid_uri_links"):
        derived.add("invalid-uri-link")
    if navigation.get("invalid_internal_links"):
        derived.add("invalid-internal-link")
    if rendering.get("pages_rendered") != pages or rendering.get("errors"):
        derived.add("page-render-failed")
    if accessibility.get("pages_with_extractable_text") == 0:
        derived.add("no-extractable-text")
    fonts = accessibility.get("font_inventory")
    if not isinstance(fonts, dict):
        raise ValueError("publication quality report is missing font inventory")
    if accessibility.get("tagged") is not True:
        derived.add("pdf-not-tagged")
    if publication.get("fast_web_access") is not True:
        derived.add("pdf-not-fast-web-access")
    if fonts.get("type3_resources"):
        derived.add("type3-font-resource")
    if fonts.get("unembedded_resources"):
        derived.add("unembedded-font-resource")
    return derived


def _finding_codes(findings: list[Any]) -> set[str]:
    codes: set[str] = set()
    for finding in findings:
        if not isinstance(finding, dict):
            raise ValueError("publication quality report has a malformed finding")
        code = finding.get("code")
        level = finding.get("level")
        message = finding.get("message")
        if (
            not isinstance(code, str)
            or code not in _FINDING_LEVELS
            or level != _FINDING_LEVELS[code]
            or not isinstance(message, str)
            or not message.strip()
            or code in codes
        ):
            raise ValueError("publication quality report has a malformed finding")
        codes.add(code)
    return codes


def validate_publication_quality(
    report: dict[str, Any],
    *,
    profile: PublicationProfile = "computational",
) -> dict[str, Any]:
    """Fail closed when the requested publication-readiness profile is not met."""
    if profile not in {"computational", "archival"}:
        raise ValueError("profile must be 'computational' or 'archival'")
    if report.get("schema_version") != "proximal-distal-publication-quality-v1":
        raise ValueError("publication quality report has an invalid schema version")
    source = report.get("source")
    if not isinstance(source, dict):
        raise ValueError("publication quality report is missing source identity")
    require_repository(str(source.get("repository", "")))
    require_hex(str(source.get("revision", "")), length=40, label="source_revision")
    require_hex(
        str(source.get("release_manifest_sha256", "")),
        length=64,
        label="release_manifest_sha256",
    )
    publication = report.get("publication")
    rendering = report.get("rendering")
    navigation = report.get("navigation")
    accessibility = report.get("accessibility")
    findings = report.get("findings")
    if (
        not isinstance(publication, dict)
        or not isinstance(rendering, dict)
        or not isinstance(navigation, dict)
        or not isinstance(accessibility, dict)
        or not isinstance(findings, list)
    ):
        raise ValueError("publication quality report is structurally incomplete")
    require_hex(
        str(publication.get("sha256", "")), length=64, label="publication sha256"
    )
    codes = _finding_codes(findings)
    derived = _derived_findings(publication, rendering, navigation, accessibility)
    if derived != codes:
        raise ValueError("publication quality report findings are inconsistent")
    readiness = report.get("readiness")
    if not isinstance(readiness, dict):
        raise ValueError("publication quality report is missing readiness")
    computational = not bool(derived & COMPUTATIONAL_BLOCKERS)
    archival = computational and not bool(derived & ARCHIVAL_GAPS)
    if readiness.get("computational_release") is not computational or (
        readiness.get("archival_publication") is not archival
    ):
        raise ValueError("publication quality report readiness is inconsistent")
    key = (
        "computational_release"
        if profile == "computational"
        else "archival_publication"
    )
    if readiness.get(key) is not True:
        label = (
            "computational release"
            if profile == "computational"
            else "archival publication"
        )
        raise ValueError(f"publication quality validation failed for {label}")
    return {"valid": True, "profile": profile, "findings": findings}
