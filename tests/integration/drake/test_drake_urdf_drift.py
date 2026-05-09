"""Local pytest mirror of the drake URDF drift CI gate (#4129).

The CI gate (`.github/workflows/ci-engine-models.yml`) runs
``python3 scripts/build_humanoid_models.py --engine drake --check`` on
every PR that touches the shared YAML, the URDF generator, the canonical
URDF, or the build script. This module runs the *same* check via pytest
so contributors catch drift locally before pushing.

A failure here means the on-disk URDF at
``src/engines/physics_engines/drake/models/generated/golfer.urdf`` does
not match a fresh regeneration from the shared YAML. Fix:

    python3 scripts/build_humanoid_models.py --engine drake
    git add src/engines/physics_engines/drake/models/generated/golfer.urdf
    git commit -m "regen drake URDF"

Hand-edits to generated engine model files are forbidden by
``CROSS_ENGINE_PARITY_SPEC.md`` §6.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD_SCRIPT = REPO_ROOT / "scripts" / "build_humanoid_models.py"
CANONICAL_URDF = (
    REPO_ROOT
    / "src"
    / "engines"
    / "physics_engines"
    / "drake"
    / "models"
    / "generated"
    / "golfer.urdf"
)


@pytest.mark.unit
def test_build_script_supports_check_flag() -> None:
    """``--check`` is a documented CLI flag (#4129 AC: build-script support)."""
    result = subprocess.run(
        [sys.executable, str(BUILD_SCRIPT), "--help"],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO_ROOT,
    )
    assert "--check" in result.stdout, (
        "scripts/build_humanoid_models.py must expose --check; see #4129."
    )


@pytest.mark.unit
def test_canonical_urdf_committed() -> None:
    """The on-disk URDF the CI gate diffs against must be tracked."""
    assert CANONICAL_URDF.exists(), (
        f"Canonical URDF missing: {CANONICAL_URDF}. "
        "Run `python3 scripts/build_humanoid_models.py --engine drake` "
        "and commit the result."
    )


@pytest.mark.unit
def test_drake_urdf_no_drift() -> None:
    """Mirror of the CI gate: regenerate and assert byte-equality.

    Skips cleanly when the URDF generator dependencies (``pyyaml`` and
    friends) are not importable on this machine — CI installs them
    explicitly. Locally, contributors who do not have them are not the
    audience that can introduce drift in the first place.
    """
    pytest.importorskip("yaml", reason="PyYAML required to regenerate URDF")

    # Importability of the generator itself is part of the contract; if
    # this import fails the script will also fail in CI, so surface it
    # as a real test failure rather than a skip.
    from src.engines.physics_engines.drake.python.motion_matching.humanoid_urdf import (  # noqa: E501
        build_humanoid_urdf,  # noqa: F401
    )

    result = subprocess.run(
        [sys.executable, str(BUILD_SCRIPT), "--engine", "drake", "--check"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )

    if result.returncode != 0:
        pytest.fail(
            "drake URDF drift detected (#4129).\n"
            f"--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}\n"
            "Fix locally:\n"
            "  python3 scripts/build_humanoid_models.py --engine drake\n"
            "  git add "
            "src/engines/physics_engines/drake/models/generated/golfer.urdf\n"
            "  git commit -m 'regen drake URDF'\n"
        )
