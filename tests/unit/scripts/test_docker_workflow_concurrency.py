"""Docker CI workflows must not share one GitHub Actions concurrency group (#15086)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOWS = (
    "docker-size-gates.yml",
    "docker-security-scan.yml",
    "docker-smoke.yml",
)

pytestmark = pytest.mark.unit


def _concurrency_group(workflow_name: str) -> str:
    data = yaml.safe_load(
        (REPO_ROOT / ".github" / "workflows" / workflow_name).read_text(
            encoding="utf-8"
        )
    )
    concurrency = data["concurrency"]
    assert concurrency.get("cancel-in-progress") is False
    return str(concurrency["group"])


def test_each_docker_workflow_has_a_distinct_concurrency_group() -> None:
    groups = [_concurrency_group(name) for name in WORKFLOWS]
    assert len(groups) == len(set(groups)), (
        "Sharing one concurrency group cancels a third workflow when Dockerfile "
        "paths trigger docker-size-gates, docker-security-scan, and docker-smoke "
        "together (#15086)."
    )
    assert "docker-image-build" not in "".join(groups)
