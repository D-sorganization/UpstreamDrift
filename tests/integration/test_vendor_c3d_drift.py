"""Vendor drift sentinel for the canonical C3D reader.

The vendored copy at
``vendor/ud-tools/src/shared/python/sidekick/lab/bio/c3d_reader.py``
is duplicated content sourced from the ``D-sorganization/Tools`` repo per the
cross-repo dependency contract. Until that contract migrates the canonical
import path (issue #4484), the two copies must remain byte-identical to
prevent silent drift.

The vendored tree ships as a git submodule and is not always materialised
on disk (CI checkout, sparse fetches, fresh clones with
``--recurse-submodules`` off). The test skips when the vendored file is
absent rather than failing.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest


def _repo_root() -> Path:
    """Return the repository root.

    Walk up from this file until we find a directory containing ``CLAUDE.md``
    or ``pyproject.toml``. This keeps the test resilient to worktree layouts.
    """
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / "pyproject.toml").is_file() or (parent / "CLAUDE.md").is_file():
            return parent
    raise RuntimeError("Could not locate repository root from test file location")


def _sha256(path: Path) -> str:
    """Return the hex SHA-256 digest of a file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


CANONICAL_REL = Path("src/shared/python/sidekick/lab/bio/c3d_reader.py")
VENDOR_REL = Path("vendor/ud-tools/src/shared/python/sidekick/lab/bio/c3d_reader.py")


@pytest.mark.integration
def test_vendor_c3d_reader_matches_canonical() -> None:
    """Canonical and vendored ``c3d_reader.py`` must hash-match.

    Drift here means an out-of-tree change has shipped via the vendor
    submodule that the canonical reader has not adopted (or vice versa).
    Either re-vendor or update canonical so they agree before merging.
    """
    root = _repo_root()
    canonical = root / CANONICAL_REL
    vendored = root / VENDOR_REL
    if not canonical.is_file():
        pytest.fail(f"Canonical C3D reader missing at {canonical}")
    if not vendored.is_file():
        pytest.skip(
            "Vendor submodule not materialised; "
            f"missing {VENDOR_REL}. Run `git submodule update --init "
            "vendor/ud-tools` to populate."
        )
    canonical_hash = _sha256(canonical)
    vendor_hash = _sha256(vendored)
    assert canonical_hash == vendor_hash, (
        "Vendor C3D reader has drifted from canonical. "
        f"canonical={canonical_hash}, vendor={vendor_hash}. "
        "Re-vendor or align changes (see issue #4484)."
    )
