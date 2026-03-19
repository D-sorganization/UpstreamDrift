#!/usr/bin/env python3
"""Verify that Dockerfile.heavy_test and heavy-tests-opt-in.yml install the same
core Python packages for the heavy integration test suite.

Checks that both files are supersets of a canonical list of heavy dependencies
defined in this script. This catches regressions where a dep is added to one
entry point but forgotten in the other.

Exit 0 if parity is maintained; exit 1 with a diff report if they diverge.

Usage::

    python3 scripts/check_heavy_dep_parity.py

Design by Contract
------------------
Pre:  Dockerfile.heavy_test and heavy-tests-opt-in.yml exist at repo root.
Post: Exits 0 iff all CANONICAL_HEAVY_DEPS are present in both files.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parents[1]
DOCKERFILE = REPO_ROOT / "Dockerfile.heavy_test"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "heavy-tests-opt-in.yml"

# ---------------------------------------------------------------------------
# Canonical heavy dependencies — must be present in BOTH files.
# When you add a new heavy dep, add it here AND to both files.
# ---------------------------------------------------------------------------
CANONICAL_HEAVY_DEPS: frozenset[str] = frozenset(
    {
        "pinocchio",
        "pink",
        "mujoco",
        "mediapipe",
        "meshcat",
        "pyvista",
        "vtk",
        "scipy",
        "sympy",
        "trimesh",
        "c3d",
        "ezdxf",
    }
)


def _extract_pip_packages(text: str) -> set[str]:
    """Extract lowercase pip package names from any pip install commands in text.

    Handles multi-line pip install blocks (backslash-continued lines in Dockerfile
    and YAML multi-line strings). Ignores shell keywords, flags (--...), and
    conditional/optional install patterns (|| echo ...).
    """
    packages: set[str] = set()
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        # Detect start of a pip install command (skip comments)
        if "pip install" in stripped and not stripped.lstrip().startswith("#"):
            # Collect this and continuation lines
            block_lines = [stripped]
            while stripped.rstrip().endswith("\\") and i + 1 < len(lines):
                i += 1
                stripped = lines[i].strip()
                block_lines.append(stripped)
            _parse_pip_block(block_lines, packages)
        i += 1
    return packages


def _parse_pip_block(block_lines: list[str], packages: set[str]) -> None:
    """Parse package names from a collected pip install block."""
    import re

    # Join and strip continuation markers
    joined = " ".join(b.rstrip("\\").strip() for b in block_lines)
    # Remove the pip install command prefix and everything up to it
    parts = re.split(r"pip install\s+", joined)
    for part in parts[1:]:  # skip the leading part before pip install
        # Trim conditional suffix: || echo "..." or ; true etc.
        part = re.split(r"\|\||\band\b|;", part)[0]
        tokens = part.split()
        for token in tokens:
            # Skip flags and shell constructs
            if token.startswith("-"):
                continue
            if token in {
                "pip",
                "install",
                "upgrade",
                "if",
                "fi",
                "then",
                "else",
                "echo",
                "true",
                "false",
                "or",
                "&&",
                "||",
                "e",
                ".",
            }:
                continue
            # Skip glob/path patterns
            if any(c in token for c in ["/", "[", "{"]):
                continue
            # Normalize: lowercase, strip version specifiers
            import re as _re

            pkg = _re.sub(r"[>=<!;\\\"'()@].*", "", token).strip().lower()
            if pkg and pkg[0].isalpha():
                packages.add(pkg)


def main() -> int:
    assert DOCKERFILE.exists(), f"Dockerfile not found: {DOCKERFILE}"
    assert WORKFLOW.exists(), f"Workflow not found: {WORKFLOW}"

    dockerfile_text = DOCKERFILE.read_text(encoding="utf-8")
    workflow_text = WORKFLOW.read_text(encoding="utf-8")

    dockerfile_pkgs = _extract_pip_packages(dockerfile_text)
    workflow_pkgs = _extract_pip_packages(workflow_text)

    # Check that each canonical dep is present in both files
    missing_from_dockerfile = CANONICAL_HEAVY_DEPS - dockerfile_pkgs
    missing_from_workflow = CANONICAL_HEAVY_DEPS - workflow_pkgs

    if not missing_from_dockerfile and not missing_from_workflow:
        print("✓ Heavy test dependency parity OK")
        print(f"  Canonical deps verified: {len(CANONICAL_HEAVY_DEPS)}")
        print(f"  Dockerfile pip packages found: {len(dockerfile_pkgs)}")
        print(f"  Workflow pip packages found:   {len(workflow_pkgs)}")
        return 0

    print("✗ Heavy test dependency parity VIOLATION")
    print()
    if missing_from_dockerfile:
        print("Canonical deps missing from Dockerfile.heavy_test:")
        for pkg in sorted(missing_from_dockerfile):
            print(f"  - {pkg}")
        print("  Fix: add to a RUN pip install block in Dockerfile.heavy_test")
        print()
    if missing_from_workflow:
        print("Canonical deps missing from heavy-tests-opt-in.yml:")
        for pkg in sorted(missing_from_workflow):
            print(f"  - {pkg}")
        print(
            "  Fix: add to the pip install step in "
            ".github/workflows/heavy-tests-opt-in.yml"
        )
        print()
    print(
        "To add a new canonical dep: add it to CANONICAL_HEAVY_DEPS in "
        "scripts/check_heavy_dep_parity.py AND to both files above."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
