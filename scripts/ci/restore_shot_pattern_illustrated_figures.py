#!/usr/bin/env python3
"""Restore illustrated shot-pattern LaTeX figure assets for CI typesetting.

Security contract (#12304): the legacy release archive is untrusted input.

* When the tracked ``figure_assets.json`` manifest exists the script only
  verifies hashes and never downloads or unpacks anything.
* Otherwise only figures registered in ``figure_selection.json`` are read from
  the archive -- by exact member name, as unique regular files, with a size cap
  -- and every byte is SHA-256 verified before anything is written.
* Every destination is a strict allowlisted ``[A-Za-z0-9_.-]`` POSIX path that
  must resolve to itself under the report root (no traversal, drive letters,
  NTFS streams, case aliases or symlinked components), so canonical TeX and
  other non-figure sources can never be overwritten.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
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
CURRENT_MANIFEST = "phase2/presentation_v2/figure_assets.json"
CURRENT_PDF_NAMES = (
    "summary.pdf",
    "driver_baseline_density.pdf",
    "driver_baseline_flight_paths.pdf",
    "seven_iron_baseline_density.pdf",
    "seven_iron_baseline_flight_paths.pdf",
    "pitching_wedge_baseline_density.pdf",
    "pitching_wedge_baseline_flight_paths.pdf",
    "driver_shaft_rotation_sd_density.pdf",
    "driver_curve_strength_density.pdf",
    "seven_iron_shaft_rotation_sd_density.pdf",
    "seven_iron_curve_strength_density.pdf",
    "pitching_wedge_shaft_rotation_sd_density.pdf",
    "pitching_wedge_curve_strength_density.pdf",
    "driver_across_lean_flight_paths.pdf",
)
LEGACY_PREFIXES = (
    "phase2/final_figures_v1/details/",
    "phase2/supplement_report_v2_layout_repair/figures/comparisons/",
)
LEGACY_SUFFIXES = (".pdf", ".png")
# The pinned archive is ~3.4 MB; these caps are >10x headroom, not budgets.
MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
MAX_MEMBER_BYTES = 32 * 1024 * 1024
DOWNLOAD_TIMEOUT_SECONDS = 120
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
# One path component: no separators, drive colons, NTFS streams, whitespace,
# control characters, or leading dot/dash.  Trailing dots are rejected
# separately because Windows silently strips them.
_COMPONENT = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*\Z")
_REPO_PART = re.compile(r"[A-Za-z0-9_.-]+\Z")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _report_root() -> Path:
    return _repo_root() / "docs/research/shot_pattern_analysis"


def _check_allowlisted(relative: str, *, current: bool) -> None:
    if current:
        expected = {f"phase2/presentation_v2/{name}" for name in CURRENT_PDF_NAMES}
        if relative not in expected:
            raise ValueError(f"unexpected current figure path: {relative!r}")
    elif not relative.startswith(LEGACY_PREFIXES) or (
        PurePosixPath(relative).suffix not in LEGACY_SUFFIXES
    ):
        raise ValueError(f"unexpected legacy figure path: {relative!r}")


def _safe_target(relative: str, *, current: bool) -> Path:
    """Return the on-disk path for an allowlisted figure, or raise ValueError.

    Postcondition: the returned path lies strictly under the resolved report
    root and resolves to itself, so no component is a symlink or junction and
    no case/short-name alias can redirect the write onto another file.
    """
    if not isinstance(relative, str):
        raise ValueError("figure path must be a string")
    parts = relative.split("/")
    if not all(_COMPONENT.fullmatch(p) and not p.endswith(".") for p in parts):
        raise ValueError(f"unsafe figure path: {relative!r}")
    _check_allowlisted(relative, current=current)
    report_root = _report_root().resolve()
    target = report_root.joinpath(*parts)
    if target.resolve() != target:
        raise ValueError(f"figure path escapes report root via a link: {relative!r}")
    return target


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _current_assets(manifest: Path) -> dict[Path, str]:
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("files_sha256"), dict):
        raise ValueError("current figure manifest lacks files_sha256")
    files = data["files_sha256"]
    expected = {f"phase2/presentation_v2/{name}" for name in CURRENT_PDF_NAMES}
    if set(files) != expected:
        raise ValueError("current figure manifest must list all 14 reviewed PDFs")
    assets = {}
    for relative, digest in files.items():
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise ValueError(f"invalid figure SHA256: {relative}")
        assets[_safe_target(relative, current=True)] = digest
    return assets


def _legacy_assets() -> dict[str, str]:
    selection = (
        _report_root()
        / "phase2/validation/illustrated_article_v1/figure_selection.json"
    )
    data = json.loads(selection.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("figures"), list):
        raise ValueError("legacy figure selection is malformed")
    assets: dict[str, str] = {}
    folded: set[str] = set()
    for figure in data["figures"]:
        if not isinstance(figure, dict):
            raise ValueError("legacy figure selection is malformed")
        relative, digest = figure.get("path"), figure.get("sha256")
        if not isinstance(relative, str):
            raise ValueError("legacy figure path must be a string")
        _safe_target(relative, current=False)
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise ValueError(f"invalid legacy figure SHA256: {relative}")
        # Case-insensitive filesystems (Windows, macOS) would fold these.
        if relative.casefold() in folded:
            raise ValueError(f"duplicate legacy figure path: {relative}")
        folded.add(relative.casefold())
        assets[relative] = digest
    if not assets:
        raise ValueError("legacy figure selection is empty")
    return assets


def _download_url() -> str:
    repo = os.environ.get("GITHUB_REPOSITORY") or DEFAULT_REPO
    parts = repo.split("/")
    if len(parts) != 2 or any(
        part in (".", "..") or not _REPO_PART.fullmatch(part) for part in parts
    ):
        raise ValueError(f"malformed GITHUB_REPOSITORY: {repo!r}")
    return f"https://github.com/{repo}/releases/download/{RELEASE_TAG}/{ARCHIVE_NAME}"


def _download_archive(dest: Path) -> None:
    """Stream the pinned release asset to *dest*, capped at MAX_ARCHIVE_BYTES."""
    request = urllib.request.Request(_download_url())
    token = os.environ.get("GITHUB_TOKEN", "")
    if token:
        # Unredirected: the token must not follow the redirect to the CDN host.
        request.add_unredirected_header("Authorization", f"Bearer {token}")
    received = 0
    with (
        urllib.request.urlopen(  # nosemgrep: python.lang.security.audit.dynamic-urllib-use-detected.dynamic-urllib-use-detected
            request,
            timeout=DOWNLOAD_TIMEOUT_SECONDS,
        ) as response,  # nosec B310 # noqa: S310
        dest.open("wb") as handle,
    ):
        while chunk := response.read(1 << 20):
            received += len(chunk)
            if received > MAX_ARCHIVE_BYTES:
                raise ValueError(f"legacy archive exceeds {MAX_ARCHIVE_BYTES} bytes")
            handle.write(chunk)


def _verify_current(manifest: Path) -> None:
    for path, digest in _current_assets(manifest).items():
        if not path.is_file() or _sha256_file(path) != digest:
            raise ValueError(f"current figure missing or SHA256 mismatch: {path}")


def _read_member(bundle: zipfile.ZipFile, relative: str) -> bytes:
    """Read exactly one regular, size-capped member named *relative*.

    Any backslash or case alias of *relative* makes the archive ambiguous
    (zipfile and NTFS fold such names together on Windows), so it fails closed.
    """
    key = relative.casefold()
    matches = [
        info
        for info in bundle.infolist()
        if key in (info.filename.casefold(), info.orig_filename.casefold())
        or info.orig_filename.replace("\\", "/").casefold() == key
    ]
    if len(matches) != 1 or matches[0].orig_filename != relative:
        raise ValueError(f"legacy archive lacks a unique exact member: {relative}")
    info = matches[0]
    if info.is_dir() or stat.S_ISLNK(info.external_attr >> 16):
        raise ValueError(f"legacy archive member is not a regular file: {relative}")
    if info.file_size > MAX_MEMBER_BYTES:
        raise ValueError(f"legacy archive member too large: {relative}")
    with bundle.open(info) as stream:
        content = stream.read(MAX_MEMBER_BYTES + 1)
    if len(content) > MAX_MEMBER_BYTES:
        raise ValueError(f"legacy archive member too large: {relative}")
    return content


def _write_figure(relative: str, content: bytes) -> None:
    """Atomically replace one allowlisted figure without following links."""
    target = _safe_target(relative, current=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    target = _safe_target(relative, current=False)  # re-check after mkdir
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, suffix=".restore")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
        tmp.chmod(0o644)  # mkstemp creates 0600; match a normal checkout file
        os.replace(tmp, target)  # replaces a link itself, never its referent
    finally:
        tmp.unlink(missing_ok=True)


def _restore_legacy(assets: dict[str, str]) -> None:
    missing = {
        relative: digest
        for relative, digest in assets.items()
        if not (target := _safe_target(relative, current=False)).is_file()
        or _sha256_file(target) != digest
    }
    if not missing:
        print("All legacy illustrated figure assets verified.")
        return
    print(f"Restoring {len(missing)} allowlisted legacy figure asset(s).")
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / ARCHIVE_NAME
        _download_archive(archive)
        if _sha256_file(archive) != ARCHIVE_SHA256:
            raise ValueError("legacy archive SHA256 mismatch")
        with zipfile.ZipFile(archive) as bundle:
            contents: dict[str, bytes] = {}
            for relative, digest in missing.items():
                content = _read_member(bundle, relative)
                if hashlib.sha256(content).hexdigest() != digest:
                    raise ValueError(f"legacy figure SHA256 mismatch: {relative}")
                contents[relative] = content
        for relative, content in contents.items():
            _write_figure(relative, content)
    for relative, digest in assets.items():
        target = _safe_target(relative, current=False)
        if not target.is_file() or _sha256_file(target) != digest:
            raise ValueError(f"legacy figure verification failed: {relative}")
    print("Allowlisted legacy figure assets restored and verified.")


def main() -> int:
    try:
        manifest = _report_root() / CURRENT_MANIFEST
        if manifest.exists():
            _verify_current(manifest)
            print("All current illustrated figure assets verified; no restore needed.")
            return 0
        _restore_legacy(_legacy_assets())
        return 0
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        print(f"Illustrated figure verification failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
