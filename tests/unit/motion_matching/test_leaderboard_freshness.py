"""Unit tests for the leaderboard freshness sidecar (issue #4237).

The motion-matching-leaderboard CI workflow gates PRs on the
`motion_matching/results/.leaderboard_meta.json` sidecar. The MATLAB
runner can't be invoked in CI (no Simulink), so these tests verify the
parts of the contract we *can* check from Python:

  * the sidecar is committed at the canonical path
  * its schema matches what the workflow's parser expects
  * its `git_head` resolves to a real commit in the repository
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
META_PATH = (
    REPO_ROOT
    / "src"
    / "engines"
    / "Simscape_Multibody_Models"
    / "3D_Golf_Model"
    / "matlab"
    / "motion_matching"
    / "results"
    / ".leaderboard_meta.json"
)


@pytest.mark.unit
def test_leaderboard_meta_is_committed():
    assert META_PATH.is_file(), (
        f"Expected leaderboard freshness sidecar at {META_PATH}. The CI "
        "workflow .github/workflows/motion-matching-leaderboard.yml requires it."
    )


@pytest.mark.unit
def test_leaderboard_meta_schema():
    """Schema mirrors what the workflow's python -c parser reads."""
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    assert isinstance(meta, dict)
    # Required keys
    assert "schema_version" in meta
    assert meta["schema_version"] == 1
    assert "git_head" in meta
    assert "regenerated_at" in meta
    # git_head must be a non-empty string that looks like a SHA-1.
    sha = meta["git_head"]
    assert isinstance(sha, str)
    assert sha.strip(), "git_head must not be empty"
    assert len(sha) >= 7
    assert all(
        c in "0123456789abcdef" for c in sha.lower()
    ), f"git_head {sha!r} should be hex"


def _is_shallow_repo() -> bool:
    """Return True when the working tree is a shallow git clone."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--is-shallow-repository"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except FileNotFoundError:
        return False
    return result.returncode == 0 and result.stdout.strip().lower() == "true"
