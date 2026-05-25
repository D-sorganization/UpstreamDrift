#!/usr/bin/env python3
"""Ensure VERSION, pyproject.toml, and SPEC.md declare the same release version.

Usage::

    python3 scripts/ci/check_version_consistency.py

Exits non-zero (with a diagnostic on stderr) when any of these three surfaces
disagree:

* ``VERSION`` -- single-line file at the repository root.
* ``pyproject.toml`` -- ``[project] version`` field.
* ``SPEC.md`` -- ``| **Current Version** | <value> |`` row in the identity table.

This is a minimal sibling to ``scripts/check_version_consistency.py`` (which
audits the broader set of release metadata surfaces, including Cargo and
package.json). The intent here is to guard against the specific drift between
the human-readable spec, the packaging metadata, and the plain-text VERSION
file that downstream tooling (e.g. ``src/launchers/about_dialog.py``) reads.

Pure stdlib so it can run before any project dependencies are installed.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

_SPEC_CURRENT_VERSION_RE = re.compile(
    r"^\|\s*\*\*Current Version\*\*\s*\|\s*([^|]+?)\s*\|", re.MULTILINE
)


def _read_version_file(path: Path) -> str:
    """Return the trimmed contents of the ``VERSION`` file."""
    if not path.is_file():
        raise FileNotFoundError(f"VERSION file is missing: {path}")
    version = path.read_text(encoding="utf-8").strip()
    if not version:
        raise ValueError(f"VERSION file is empty: {path}")
    return version


def _read_pyproject_version(path: Path) -> str:
    """Return ``[project] version`` from a ``pyproject.toml`` file."""
    if not path.is_file():
        raise FileNotFoundError(f"pyproject.toml is missing: {path}")
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    project = data.get("project")
    if not isinstance(project, dict):
        raise ValueError(f"{path} is missing a [project] table")
    version = project.get("version")
    if not isinstance(version, str) or not version:
        raise ValueError(f"{path} is missing a non-empty project.version")
    return version


def _read_spec_current_version(path: Path) -> str:
    """Return the ``Current Version`` cell from ``SPEC.md``."""
    if not path.is_file():
        raise FileNotFoundError(f"SPEC.md is missing: {path}")
    text = path.read_text(encoding="utf-8")
    match = _SPEC_CURRENT_VERSION_RE.search(text)
    if match is None:
        raise ValueError(
            f"Could not locate a '**Current Version**' row in {path}; "
            "expected `| **Current Version** | X.Y.Z |`"
        )
    return match.group(1).strip()


def main() -> int:
    """Compare the three surfaces and exit non-zero on disagreement."""
    surfaces = {
        "VERSION": _read_version_file(REPO_ROOT / "VERSION"),
        "pyproject.toml [project.version]": _read_pyproject_version(
            REPO_ROOT / "pyproject.toml"
        ),
        "SPEC.md Current Version": _read_spec_current_version(REPO_ROOT / "SPEC.md"),
    }

    distinct = set(surfaces.values())
    if len(distinct) == 1:
        version = next(iter(distinct))
        print(f"Version consistency check passed: {version}")
        return 0

    print("Version consistency check FAILED:", file=sys.stderr)
    for name, value in surfaces.items():
        print(f"  - {name}: {value!r}", file=sys.stderr)
    print(
        "Update the disagreeing surfaces so all three match, "
        "or see scripts/check_version_consistency.py for the full audit.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
