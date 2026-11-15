"""Collation must run the size-aware devlog archiver before proposing (#15166).

``collate-changes.yml`` grows ``DEVELOPMENT_LOG.md`` every run. Without the
headroom archiver the fleet policy gate rejects merges once the log passes
``MAX_BYTES``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

WORKFLOW = (
    Path(__file__).resolve().parents[3]
    / ".github"
    / "workflows"
    / "collate-changes.yml"
)

pytestmark = pytest.mark.unit


def _steps() -> list[dict]:
    data = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    return data["jobs"]["collate"]["steps"]


def _step_names() -> list[str]:
    return [str(step.get("name")) for step in _steps()]


def test_collate_workflow_archives_devlog_before_proposing_pr() -> None:
    names = _step_names()
    assert "Collate fragments" in names
    assert "Propose collation as a pull request" in names
    archive_index = names.index("Archive development log with headroom")
    assert archive_index > names.index("Collate fragments")
    assert archive_index < names.index("Propose collation as a pull request")


def test_archive_step_invokes_headroom_entry_point() -> None:
    step = next(
        s for s in _steps() if s.get("name") == "Archive development log with headroom"
    )
    run = step["run"]
    assert "scripts/ci/archive_devlog_with_headroom.py" in run
