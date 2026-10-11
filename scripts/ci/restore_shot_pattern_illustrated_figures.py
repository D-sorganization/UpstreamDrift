#!/usr/bin/env python3
"""Restore illustrated shot-pattern LaTeX figure assets for CI typesetting."""

from __future__ import annotations

import hashlib
import json
import os
import re
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
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _report_root() -> Path:
    return _repo_root() / "docs/research/shot_pattern_analysis"


def _safe_target(relative: str, *, current: bool) -> Path:
    if not isinstance(relative, str) or "\\" in relative:
        raise ValueError("figure path must be a POSIX relative path")
    parts = relative.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"unsafe figure path: {relative}")
    path = PurePosixPath(relative)
    if path.is_absolute():
        raise ValueError(f"unsafe figure path: {relative}")
    if current:
        expected = {f"phase2/presentation_v2/{name}" for name in CURRENT_PDF_NAMES}
        if relative not in expected:
            raise ValueError(f"unexpected current figure path: {relative}")
    elif not relative.startswith(LEGACY_PREFIXES) or path.suffix not in (
        ".pdf",
        ".png",
    ):
        raise ValueError(f"unexpected legacy figure path: {relative}")
    report_root = _report_root().resolve()
    target = _report_root() / relative
    if not target.resolve().is_relative_to(report_root):
        raise ValueError(f"figure path escapes report root: {relative}")
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
    for figure in data["figures"]:
        if not isinstance(figure, dict):
            raise ValueError("legacy figure selection is malformed")
        relative, digest = figure.get("path"), figure.get("sha256")
        if not isinstance(relative, str):
            raise ValueError("legacy figure path must be a string")
        _safe_target(relative, current=False)
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise ValueError(f"invalid legacy figure SHA256: {relative}")
        if relative in assets:
            raise ValueError(f"duplicate legacy figure path: {relative}")
        assets[relative] = digest
    if not assets:
        raise ValueError("legacy figure selection is empty")
    return assets


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


def _verify_current(manifest: Path) -> None:
    for path, digest in _current_assets(manifest).items():
        if not path.is_file() or _sha256_file(path) != digest:
            raise ValueError(f"current figure missing or SHA256 mismatch: {path}")


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
            names = bundle.namelist()
            for relative, digest in missing.items():
                if names.count(relative) != 1:
                    raise ValueError(f"legacy archive lacks unique figure: {relative}")
                content = bundle.read(relative)
                if hashlib.sha256(content).hexdigest() != digest:
                    raise ValueError(f"legacy figure SHA256 mismatch: {relative}")
                contents[relative] = content
            for relative, content in contents.items():
                target = _safe_target(relative, current=False)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
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
