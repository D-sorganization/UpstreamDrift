"""ci-standard hosted runs use the pinned ubuntu-26.04 label (#14398, #14397).

The exact-hash fixture drift first seen on ubuntu-26.04 came from the runner CPU,
not the image, and unit-test-gate now pins its numeric kernels. ``pick-runner``
names the image explicitly instead of ``ubuntu-latest`` so the next image rollout
cannot silently change the merge-queue environment.
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


def test_pick_runner_routes_hosted_jobs_to_pinned_ubuntu_2604() -> None:
    hosted = re.findall(r'runner="(ubuntu-[^"]+)"', _pick_runner_script())

    assert hosted, "pick-runner must assign a hosted runner label"
    assert set(hosted) == {"ubuntu-26.04"}, (
        "hosted pick-runner targets must name ubuntu-26.04 explicitly (#14397), "
        f"not ubuntu-latest or the retired 24.04 pin; found {hosted}"
    )
