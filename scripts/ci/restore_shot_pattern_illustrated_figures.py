#!/usr/bin/env python3
"""Restore illustrated shot-pattern LaTeX figure assets for CI typesetting."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

RELEASE_TAG = "shot-pattern-phase2-research-2026-10-10"
ARCHIVE_NAME = "illustrated-report-source.zip"
ARCHIVE_SHA256 = "6567a9dcee60d6e18e4b1fe39ca34b3631eac09a578ce3eacd64914c93c7c1d2"
DEFAULT_REPO = "D-sorganization/UpstreamDrift"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _report_root() -> Path:
    return _repo_root() / "docs/research/shot_pattern_analysis"


def _required_paths() -> list[Path]:
    selection = (
        _report_root()
        / "phase2/validation/illustrated_article_v1/figure_selection.json"
    )
    data = json.loads(selection.read_text(encoding="utf-8"))
    return [(_report_root() / figure["path"]).resolve() for figure in data["figures"]]


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
    with urllib.request.urlopen(request) as response, dest.open("wb") as handle:  # noqa: S310
        handle.write(response.read())


def main() -> int:
    missing = [path for path in _required_paths() if not path.is_file()]
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
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        if digest != ARCHIVE_SHA256:
            print(f"Archive SHA256 mismatch: {digest}", file=sys.stderr)
            return 1
        report_root = _report_root()
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.namelist():
                if member.endswith("/"):
                    continue
                target = report_root / member
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(bundle.read(member))
    still_missing = [path for path in _required_paths() if not path.is_file()]
    if still_missing:
        print("Restore finished but assets still missing:", file=sys.stderr)
        for path in still_missing:
            print(f"  {path.relative_to(_report_root())}", file=sys.stderr)
        return 1
    print("Illustrated figure assets restored.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
