"""Repo-hygiene guard: private motion-capture data must not be tracked.

Native capture files, their sidecars and club workbooks are private. They live
in the private capture data repository and resolve at runtime through
``src.motion_capture.capture_registry`` (``CAPTURE_DATA_DIR``). Only the
policy-approved anonymized C3D copies stay in this public tree.

Two rules, both evaluated against the git index (``git ls-files -s``), so no
file is read or hashed:

1. No tracked native capture file or sidecar: ``*.gpcap``, ``Capture.info``
   style session sidecars, ``PlayerInfo.bin``, or anything under a
   ``* Capture Files`` session directory.
2. No tracked blob may appear in ``private_capture_blob_ids.txt``, which lists
   git blob IDs only and so discloses nothing about the removed files.
"""

from __future__ import annotations

import re
import shutil
import subprocess  # nosec B404 - git invocation, no shell, fixed args
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DENYLIST = Path(__file__).with_name("private_capture_blob_ids.txt")
_BLOB_ID_PATTERN = re.compile(r"^[0-9a-f]{40}$")
# Native capture sessions: the capture file, its per-capture sidecar, the
# player sidecar, and any file inside a "<vendor> Capture Files" session tree.
_PRIVATE_PATH_PATTERN = re.compile(
    r"(\.gpcap$)|((^|/)([^/]* )?Capture\.info$)|((^|/)PlayerInfo\.bin$)"
    r"|((^|/)[^/]* Capture Files/)",
    re.IGNORECASE,
)


def _index_entries() -> list[tuple[str, str]]:
    """Return ``(blob_id, path)`` for every tracked file in the git index."""
    git = shutil.which("git")
    if git is None or not (_REPO_ROOT / ".git").exists():
        pytest.skip("git checkout not available")
    result = subprocess.run(  # nosec B603 - fixed argv, no shell
        [git, "ls-files", "-s", "-z"],
        cwd=_REPO_ROOT,
        capture_output=True,
        check=True,
    )
    entries = []
    for record in result.stdout.decode("utf-8", errors="surrogateescape").split("\0"):
        if not record:
            continue
        meta, path = record.split("\t", 1)
        entries.append((meta.split()[1], path))
    return entries


def _load_denylist() -> frozenset[str]:
    ids = set()
    for raw in _DENYLIST.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            ids.add(line.lower())
    return frozenset(ids)


def test_denylist_is_well_formed() -> None:
    ids = _load_denylist()
    assert ids, "private capture denylist must not be empty"
    malformed = sorted(i for i in ids if not _BLOB_ID_PATTERN.match(i))
    assert not malformed, f"malformed blob IDs: {malformed}"


@pytest.mark.parametrize(
    ("path", "private"),
    [
        ("a/b/Session Capture Files/x/2019-01-01/12 Capture.gpcap", True),
        ("a/12 Capture.info", True),
        ("a/Capture.info", True),
        ("a/PlayerInfo.bin", True),
        ("a/Vendor Capture Files/notes.txt", True),
        ("data/C3D_TA_Driver.c3d", False),
        ("a/Mocap C3D Files/CaptureExport Tour average.c3d", False),
        ("a/model.info.json", False),
        ("a/.MATLABDriveTag", False),
    ],
)
def test_private_path_pattern(path: str, private: bool) -> None:
    assert bool(_PRIVATE_PATH_PATTERN.search(path)) is private


def test_no_tracked_native_capture_files() -> None:
    offenders = [p for _, p in _index_entries() if _PRIVATE_PATH_PATTERN.search(p)]
    assert not offenders, (
        "Native capture files and sidecars are private and must not be tracked; "
        f"move them to the private capture data repository: {offenders}"
    )


def test_no_tracked_file_matches_private_capture_blob() -> None:
    denylist = _load_denylist()
    offenders = [p for blob, p in _index_entries() if blob in denylist]
    assert not offenders, (
        "Tracked files match the private capture denylist; remove them and "
        f"resolve the data through CAPTURE_DATA_DIR instead: {offenders}"
    )
