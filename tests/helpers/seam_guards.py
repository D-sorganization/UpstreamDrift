"""Unified seam guard verification and vendor requirement helpers.

See UpstreamDrift issue #9501.
Seam guards protect the Tools <-> UpstreamDrift convergence boundary.
When vendor/ud-tools is unpopulated, seam checks must FAIL (not silently skip)
unless explicitly opted out via SEAM_TESTS_ALLOW_SKIP=1.

Workaround for linked worktrees where submodule update fails:
    git -C <path-to-tools-clone> archive <pin> | tar -x -C vendor/ud-tools
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.shared.python.config.tools_vendor_authority import (
    expected_tools_gitlink_sha,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_VENDOR_ROOT = _REPO_ROOT / "vendor" / "ud-tools"


def seam_tests_allow_skip() -> bool:
    """Return True only if explicitly opted out via SEAM_TESTS_ALLOW_SKIP=1."""
    return os.environ.get("SEAM_TESTS_ALLOW_SKIP", "").strip() == "1"


def missing_vendor_instructions(target_path: Path | None = None) -> str:
    """Generate clear, actionable instructions naming the git archive workaround."""
    pin = expected_tools_gitlink_sha(_REPO_ROOT) or "<pin>"
    loc = target_path if target_path is not None else _VENDOR_ROOT
    return (
        f"The vendored Tools tree is missing at {loc}.\n"
        "Seam tests fail closed by default to prevent silent regression (issue #9501).\n"
        "To populate vendor/ud-tools, run:\n"
        "    git submodule update --init vendor/ud-tools\n"
        "Or if in a linked worktree where submodule initialization fails, run:\n"
        f"    git -C ../Tools archive {pin} | tar -x -C vendor/ud-tools\n"
        "To explicitly opt out and skip locally, set SEAM_TESTS_ALLOW_SKIP=1."
    )


def require_vendor_path(path: Path) -> Path:
    """Require that a vendor file or directory exists; fail closed or skip if opted out.

    Raises:
        AssertionError: If path is missing and SEAM_TESTS_ALLOW_SKIP is not '1' (or in CI).
        pytest.skip.Exception: If path is missing and SEAM_TESTS_ALLOW_SKIP is '1' outside CI.
    """
    if path.exists():
        if path.is_dir():
            # If it's a directory, make sure it's not an empty placeholder
            try:
                if any(path.iterdir()):
                    return path
            except OSError:
                pass
        else:
            return path

    msg = missing_vendor_instructions(path)
    if os.environ.get("CI") or not seam_tests_allow_skip():
        raise AssertionError(msg)
    pytest.skip(msg)
