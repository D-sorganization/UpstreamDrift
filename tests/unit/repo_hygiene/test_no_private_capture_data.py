"""Repo-hygiene guard: private motion-capture data must not be tracked.

Native capture files and club workbooks are private. They live in the
private capture data repository and resolve at runtime through
``src.motion_capture.capture_registry`` (``CAPTURE_DATA_DIR``). Only the
policy-approved anonymized C3D copies stay in this public tree.

Two rules, both evaluated against ``git ls-files``:

1. No tracked ``*.gpcap`` file (the native capture format) is allowed.
2. No tracked file may have a SHA-256 listed in
   ``private_capture_sha256.txt``. The denylist holds digests only, so it
   discloses nothing about the removed files beyond their hashes.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess  # nosec B404 - git invocation, no shell, fixed args
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DENYLIST = Path(__file__).with_name("private_capture_sha256.txt")
_FORBIDDEN_SUFFIXES = frozenset({".gpcap"})
_SHA256_HEX_LENGTH = 64
_CHUNK_BYTES = 1 << 20


def _tracked_files() -> list[Path]:
    git = shutil.which("git")
    if git is None or not (_REPO_ROOT / ".git").exists():
        pytest.skip("git checkout not available")
    result = subprocess.run(  # nosec B603 - fixed argv, no shell
        [git, "ls-files", "-z"],
        cwd=_REPO_ROOT,
        capture_output=True,
        check=True,
    )
    names = result.stdout.decode("utf-8", errors="surrogateescape").split("\0")
    return [_REPO_ROOT / name for name in names if name]


def _load_denylist() -> frozenset[str]:
    digests = set()
    for raw in _DENYLIST.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        digests.add(line.lower())
    return frozenset(digests)


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_BYTES):
            hasher.update(chunk)
    return hasher.hexdigest()


def test_denylist_is_well_formed() -> None:
    digests = _load_denylist()
    assert digests, "private capture denylist must not be empty"
    malformed = [
        d
        for d in digests
        if len(d) != _SHA256_HEX_LENGTH or set(d) - set("0123456789abcdef")
    ]
    assert not malformed, f"malformed SHA-256 entries: {malformed}"


def test_no_tracked_native_capture_files() -> None:
    offenders = [
        path.relative_to(_REPO_ROOT).as_posix()
        for path in _tracked_files()
        if path.suffix.lower() in _FORBIDDEN_SUFFIXES
    ]
    assert not offenders, (
        "Native capture files are private and must not be tracked; move them "
        f"to the private capture data repository: {offenders}"
    )


def test_no_tracked_file_matches_private_capture_hash() -> None:
    denylist = _load_denylist()
    offenders = [
        path.relative_to(_REPO_ROOT).as_posix()
        for path in _tracked_files()
        if path.is_file() and not path.is_symlink() and _sha256(path) in denylist
    ]
    assert not offenders, (
        "Tracked files match the private capture denylist; remove them and "
        f"resolve the data through CAPTURE_DATA_DIR instead: {offenders}"
    )
