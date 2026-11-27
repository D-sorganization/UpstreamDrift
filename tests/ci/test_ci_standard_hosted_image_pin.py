"""ci-standard hosted runs use ubuntu-latest (#14398, #14397).

#14397 canonicalized float hashing across runner CPU/glibc images, allowing
pick-runner to target ubuntu-latest without exact-hash fixture drift.
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


def test_pick_runner_routes_hosted_jobs_to_ubuntu_latest() -> None:
    hosted = re.findall(r'runner="(ubuntu-[^"]+)"', _pick_runner_script())

    assert hosted, "pick-runner must assign a hosted runner label"
    assert set(hosted) == {"ubuntu-latest"}, (
        f"hosted pick-runner targets should use ubuntu-latest (#14397); found {hosted}"
    )
