"""Evidence manifest module probes must not execute package imports (#15015)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from src.shared.python.motion_matching.evidence_scripts import (
    EvidenceScriptDisposition,
    module_source_exists,
    validate_evidence_run_dir,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN102_DIR = (
    REPO_ROOT
    / "docs"
    / "development"
    / "simscape_tour_matching"
    / "native_evidence"
    / "two_window_fit_9967_102"
)


@pytest.mark.parametrize(
    "module_name",
    [
        "src.engines.physics_engines.pinocchio.python.native_node_chart",
        "src.shared.python.motion_matching.native_candidate",
    ],
)
def test_missing_native_modules_are_absent_by_source_path(module_name: str) -> None:
    assert module_source_exists(module_name, repo_root=REPO_ROOT) is False


def test_run102_archived_validation_does_not_import_pinocchio_stack() -> None:
    """Archived run 102 stays valid when find_spec would traverse optional deps."""
    result = validate_evidence_run_dir(RUN102_DIR, repo_root=REPO_ROOT)
    assert result.valid is True
    assert result.disposition == EvidenceScriptDisposition.ARCHIVED


def test_run102_script_exits_before_native_imports() -> None:
    """Direct execution fails closed with the archived manifest message."""
    script = RUN102_DIR / "two_window_fit_102.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "archived on this branch" in proc.stderr
