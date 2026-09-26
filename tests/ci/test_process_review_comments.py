"""Retirement guard for the review-comment-to-issue converter (RM#1755).

The converter workflow and its processor were retired fleet-wide. This file
keeps its historical path so the deleted-test gate stays meaningful, and pins
the retirement so the converter cannot be reintroduced without a decision.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
RETIRED_PATHS = (
    ROOT / ".github" / "workflows" / "Comment-to-Issue-Converter.yml",
    ROOT / "scripts" / "ci" / "process_review_comments.py",
)


@pytest.mark.parametrize("path", RETIRED_PATHS, ids=lambda p: p.name)
def test_retired_converter_file_is_absent(path: Path) -> None:
    """RM#1755: the converter's workflow and processor must stay deleted."""
    assert not path.exists(), f"retired converter file reintroduced: {path}"


def test_no_workflow_references_the_retired_processor() -> None:
    """No remaining workflow may invoke the retired processor script."""
    offenders = [
        workflow.name
        for workflow in sorted((ROOT / ".github" / "workflows").glob("*.y*ml"))
        if "process_review_comments" in workflow.read_text(encoding="utf-8")
    ]
    assert offenders == [], f"workflows still call the retired processor: {offenders}"


def test_workflow_inventory_has_no_converter_row() -> None:
    """The workflow inventory must not list the retired converter."""
    inventory = (ROOT / ".github" / "WORKFLOWS.md").read_text(encoding="utf-8")
    assert "Comment-to-Issue-Converter.yml" not in inventory
