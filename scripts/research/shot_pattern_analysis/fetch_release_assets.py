#!/usr/bin/env python3
"""Restore shot-pattern binary artefacts from the research release.

Study data archives and review photographs are published as release assets so
they stay out of git history. ``RELEASE_ASSETS.json`` records each file's
repository path, asset name, size and SHA-256. This script downloads only the
missing or changed files and verifies every byte before writing it.

Usage::

    python3 -m scripts.research.shot_pattern_analysis.fetch_release_assets
    python3 -m scripts.research.shot_pattern_analysis.fetch_release_assets \
        --only data_archives/final_primary_v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

SCHEMA = "shot-pattern-release-assets/1"
DEFAULT_REPO = "D-sorganization/UpstreamDrift"
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INDEX = (
    REPO_ROOT / "docs/research/shot_pattern_analysis/phase2/RELEASE_ASSETS.json"
)
_PREFIX = "docs/research/shot_pattern_analysis/"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ASSET = re.compile(r"[A-Za-z0-9._-]+\Z")

Downloader = Callable[[str, str], bytes]


@dataclass(frozen=True)
class ReleaseAsset:
    """One repository file published as a release asset."""

    path: str
    asset: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class AssetIndex:
    """Validated release index: the tag plus every published file."""

    release_tag: str
    entries: tuple[ReleaseAsset, ...]

    def __iter__(self):
        return iter(self.entries)


def _safe_path(path: object) -> str:
    if not isinstance(path, str) or "\\" in path or not path.startswith(_PREFIX):
        raise ValueError(f"asset path must be a POSIX path under {_PREFIX}: {path!r}")
    pure = PurePosixPath(path)
    if pure.is_absolute() or any(part in ("", ".", "..") for part in path.split("/")):
        raise ValueError(f"unsafe asset path: {path!r}")
    return path


def load_index(index_path: Path) -> AssetIndex:
    """Parse and validate the release index (precondition for every fetch)."""
    data = json.loads(Path(index_path).read_text(encoding="utf-8"))
    if data.get("schema") != SCHEMA:
        raise ValueError(f"unexpected release index schema: {data.get('schema')!r}")
    tag = data.get("release_tag")
    if not isinstance(tag, str) or not tag:
        raise ValueError("release index needs a release_tag")
    files = data.get("files")
    if not isinstance(files, dict):
        raise ValueError("release index needs a files mapping")
    entries = []
    for path, meta in files.items():
        entry = ReleaseAsset(
            path=_safe_path(path),
            asset=str(meta.get("asset", "")),
            bytes=meta.get("bytes"),
            sha256=str(meta.get("sha256", "")),
        )
        if not _ASSET.match(entry.asset):
            raise ValueError(f"invalid asset name for {path}: {entry.asset!r}")
        if not isinstance(entry.bytes, int) or isinstance(entry.bytes, bool):
            raise ValueError(f"asset size must be an integer for {path}")
        if entry.bytes < 0 or not _SHA256.match(entry.sha256):
            raise ValueError(f"invalid size or sha256 for {path}")
        entries.append(entry)
    return AssetIndex(tag, tuple(entries))


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_matches(path: Path, entry: ReleaseAsset) -> bool:
    if not path.is_file() or path.stat().st_size != entry.bytes:
        return False
    return _digest(path.read_bytes()) == entry.sha256


def github_download(repo: str) -> Downloader:
    """Return a downloader for public release assets of ``repo``."""

    def download(tag: str, asset: str) -> bytes:
        url = f"https://github.com/{repo}/releases/download/{tag}/{asset}"
        if not url.startswith("https://github.com/"):
            raise ValueError("release downloads must use https://github.com")
        with urllib.request.urlopen(url, timeout=300) as response:  # noqa: S310  # nosec B310 - fixed https host checked above
            return response.read()

    return download


def _write_atomic(target: Path, data: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
        os.replace(temporary, target)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def fetch(
    index_path: Path,
    repo_root: Path,
    *,
    download: Downloader,
    only: str | None = None,
) -> list[Path]:
    """Download missing or changed assets; return the paths written.

    Postcondition: every returned file matches its indexed size and SHA-256.
    A download that fails verification raises ValueError and writes nothing
    for that file.
    """
    index = load_index(index_path)
    written: list[Path] = []
    for entry in index:
        if only is not None and only not in entry.path:
            continue
        target = Path(repo_root) / entry.path
        if _file_matches(target, entry):
            continue
        data = download(index.release_tag, entry.asset)
        if len(data) != entry.bytes or _digest(data) != entry.sha256:
            raise ValueError(f"downloaded {entry.asset} does not match its sha256")
        _write_atomic(target, data)
        written.append(target)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--only", help="restore only paths containing this text")
    args = parser.parse_args(argv)
    written = fetch(
        args.index, REPO_ROOT, download=github_download(args.repo), only=args.only
    )
    print(f"restored {len(written)} file(s)")  # noqa: T201 - CLI output
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
