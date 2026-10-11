#!/usr/bin/env python3
"""Restore illustrated shot-pattern LaTeX figure assets for CI typesetting."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

RELEASE_TAG = "shot-pattern-phase2-research-2026-10-10"
ARCHIVE_NAME = "illustrated-report-source.zip"
ARCHIVE_SHA256 = "6567a9dcee60d6e18e4b1fe39ca34b3631eac09a578ce3eacd64914c93c7c1d2"
DEFAULT_REPO = "D-sorganization/UpstreamDrift"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _report_root() -> Path:
    return _repo_root() / "docs/research/shot_pattern_analysis"


MANIFEST_RELATIVE = "phase2/presentation_v2/figure_assets.json"
SELECTION_RELATIVE = "phase2/validation/illustrated_article_v1/figure_selection.json"
#: Only rendered figure assets may ever be restored; never TeX or other sources.
ALLOWED_SUFFIXES = frozenset({".pdf", ".png", ".svg"})
MAX_MEMBER_BYTES = 64 * 1024 * 1024
_SHA256_LEN = 64


class RestoreError(ValueError):
    """Raised when a manifest, selection or archive member violates the contract."""


def _safe_relative(report_root: Path, rel: object) -> Path:
    """Return the validated absolute target for ``rel`` inside ``report_root``.

    Precondition: ``rel`` is a POSIX-style relative path of an allowlisted
    figure type. Rejects absolute paths, drive letters, backslashes, NUL, any
    ``..`` segment, symlinked path components and escapes from ``report_root``.
    """
    if not isinstance(rel, str) or not rel:
        raise RestoreError(f"path must be a non-empty string: {rel!r}")
    if "\\" in rel or "\x00" in rel or rel.startswith("/") or ":" in rel:
        raise RestoreError(f"unsafe path: {rel!r}")
    parts = PurePosixPath(rel).parts
    if not parts or any(part in ("..", ".", "") for part in parts):
        raise RestoreError(f"unsafe path segment: {rel!r}")
    if PurePosixPath(rel).suffix.lower() not in ALLOWED_SUFFIXES:
        raise RestoreError(f"not an allowlisted figure type: {rel!r}")
    root = report_root.resolve()
    cursor = root
    for part in parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise RestoreError(f"symlinked path component: {rel!r}")
    target = cursor.resolve()
    if not target.is_relative_to(root):
        raise RestoreError(f"path escapes report root: {rel!r}")
    return target


def _load_entries(
    report_root: Path, document: Path, key: str
) -> list[tuple[str, Path, str]]:
    try:
        data = json.loads(document.read_text(encoding="utf-8"))
        raw = data[key]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise RestoreError(f"unreadable {document.name}: {exc}") from exc
    if not isinstance(raw, list) or not raw:
        raise RestoreError(f"{document.name}: '{key}' must be a non-empty list")
    entries: list[tuple[str, Path, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            raise RestoreError(f"{document.name}: malformed entry {item!r}")
        digest = item.get("sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != _SHA256_LEN
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise RestoreError(f"{document.name}: bad sha256 for {item.get('path')!r}")
        rel = item.get("path")
        entries.append((str(rel), _safe_relative(report_root, rel), digest))
    return entries


def _verify_manifest(report_root: Path, manifest: Path) -> None:
    """Fail closed unless every manifest asset exists with its recorded hash."""
    for rel, target, digest in _load_entries(report_root, manifest, "assets"):
        if not target.is_file():
            raise RestoreError(f"manifest asset missing: {rel}")
        if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise RestoreError(f"manifest asset hash mismatch: {rel}")


def _verify_archive_digest(archive: Path) -> bool:
    return hashlib.sha256(archive.read_bytes()).hexdigest() == ARCHIVE_SHA256


def _extract_allowlisted(
    archive: Path, entries: list[tuple[str, Path, str]]
) -> dict[Path, bytes]:
    """Read and verify allowlisted members; nothing is written by this function."""
    verified: dict[Path, bytes] = {}
    with zipfile.ZipFile(archive) as bundle:
        names = {info.filename: info for info in bundle.infolist()}
        for rel, target, digest in entries:
            info = names.get(rel)
            if info is None:
                raise RestoreError(f"archive lacks allowlisted member: {rel}")
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode) or info.is_dir():
                raise RestoreError(f"archive member is not a regular file: {rel}")
            if info.file_size > MAX_MEMBER_BYTES:
                raise RestoreError(f"archive member too large: {rel}")
            data = bundle.read(info)
            if hashlib.sha256(data).hexdigest() != digest:
                raise RestoreError(f"archive member hash mismatch: {rel}")
            verified[target] = data
    return verified


def _download_archive(dest: Path) -> None:
    repo = os.environ.get("GITHUB_REPOSITORY", DEFAULT_REPO)
    url = f"https://github.com/{repo}/releases/download/{RELEASE_TAG}/{ARCHIVE_NAME}"
    if not url.startswith("https://github.com/"):
        msg = f"Insecure download URL: {url}"
        raise ValueError(msg)
    headers: dict[str, str] = {}
    token = os.environ.get("GITHUB_TOKEN", "")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    # nosemgrep: python.lang.security.audit.dynamic-urllib-use-detected.dynamic-urllib-use-detected
    with urllib.request.urlopen(request) as response, dest.open("wb") as handle:  # nosec B310 # noqa: S310
        handle.write(response.read())


def main() -> int:
    report_root = _report_root()
    manifest = report_root / MANIFEST_RELATIVE
    try:
        if manifest.is_file():
            _verify_manifest(report_root, manifest)
            print("Figure assets match the tracked manifest; nothing to restore.")
            return 0
        entries = _load_entries(
            report_root, report_root / SELECTION_RELATIVE, "figures"
        )
        missing = [e for e in entries if not e[1].is_file()]
        if not missing:
            print("All illustrated figure assets present; nothing to restore.")
            return 0
        print(
            f"Missing {len(missing)} illustrated figure asset(s); "
            f"restoring from release {RELEASE_TAG}."
        )
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / ARCHIVE_NAME
            _download_archive(archive)
            if not _verify_archive_digest(archive):
                raise RestoreError("archive SHA256 mismatch")
            verified = _extract_allowlisted(archive, missing)
    except (RestoreError, zipfile.BadZipFile) as exc:
        print(f"Restore refused: {exc}", file=sys.stderr)
        return 1
    for target, data in verified.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    print("Illustrated figure assets restored.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
