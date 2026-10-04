"""Merge-queue runs must scope the test lanes like a pull request (RM#1900).

On a ``merge_group`` event ``github.base_ref`` is empty and there is no
``github.event.before``. Without explicit handling, ``tests`` fell through to
the whole-``src`` coverage lane (11.8 % against a 75 % floor) and
``unit-test-gate`` had no ``origin/main`` for the Tools child-copy guard, so
every queued PR failed regardless of its content.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci-standard.yml"


def _job(name: str) -> dict[str, Any]:
    yaml = pytest.importorskip("yaml")
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    job: dict[str, Any] = workflow["jobs"][name]
    return job


def _step(job: dict[str, Any], name: str) -> dict[str, Any]:
    for step in job["steps"]:
        if step.get("name") == name:
            return step
    raise AssertionError(f"step {name!r} not found")


@pytest.mark.unit
def test_core_tests_diff_against_merge_group_base() -> None:
    run = _step(_job("tests"), "Run Core Test Suite")["run"]
    assert 'github.event_name }}" = "merge_group"' in run
    assert 'diff_base="${{ github.event.merge_group.base_sha }}"' in run
