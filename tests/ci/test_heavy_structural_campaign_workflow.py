"""Hosted structural-campaign workflow resource contracts."""

from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.unit


def test_structural_campaign_retains_partial_artifacts_with_safe_timeout() -> None:
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/heavy-tests-opt-in.yml").read_text(encoding="utf-8")
    )
    job = workflow["jobs"]["structural-runtime-audit"]

    assert job["timeout-minutes"] == 90
    upload = next(
        step
        for step in job["steps"]
        if step.get("name") == "Upload Structural Campaign Checkpoints"
    )
    assert "always()" in upload["if"]
    assert upload["with"]["if-no-files-found"] == "error"
