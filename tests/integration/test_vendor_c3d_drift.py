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
