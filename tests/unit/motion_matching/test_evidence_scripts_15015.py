"""Evidence run scripts must be runnable on main or explicitly archived (#15015)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.shared.python.motion_matching.evidence_scripts import (
    EvidenceScriptDisposition,
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


def test_run102_evidence_script_is_archived_not_silent() -> None:
    """Run 102 optimizer script cannot import on main; manifest records the restore ref."""
    result = validate_evidence_run_dir(RUN102_DIR, repo_root=REPO_ROOT)
    assert result.valid is True
    assert result.disposition == EvidenceScriptDisposition.ARCHIVED
    assert result.script_path.name == "two_window_fit_102.py"
    assert result.script_path.is_file()
    assert "7bbca06c74b901e06db184b88e196224ae166e01" in result.detail
