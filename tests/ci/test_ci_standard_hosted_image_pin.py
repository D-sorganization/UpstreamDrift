"""ci-standard hosted runs stay on ubuntu-24.04 until the 26.04 migration (#14398, #14397).

GitHub is moving ``ubuntu-latest`` to ubuntu-26.04, where four exact-hash fixture
tests drift and merge groups are ejected depending on which image a job lands on.
``pick-runner`` therefore routes hosted jobs to the pinned ``ubuntu-24.04`` label.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

WORKFLOW = Path(__file__).resolve().parents[2] / ".github/workflows/ci-standard.yml"


def _pick_runner_script() -> str:
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    return workflow["jobs"]["pick-runner"]["steps"][0]["run"]


def test_pick_runner_routes_hosted_jobs_to_pinned_ubuntu_2404() -> None:
    hosted = re.findall(r'runner="(ubuntu-[^"]+)"', _pick_runner_script())

    assert hosted, "pick-runner must assign a hosted runner label"
    assert set(hosted) == {"ubuntu-24.04"}, (
        "hosted pick-runner targets must stay pinned to ubuntu-24.04 until #14397 "
        f"migrates the fixture hashes to ubuntu-26.04; found {hosted}"
    )
