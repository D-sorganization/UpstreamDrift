"""The nightly realtime soak skips cleanly when no soak runner exists (#12286).

``realtime-soak.yml`` targets the ``d-sorg-fleet-soak`` label, which no org
runner carried on 2026-10-11, so every 02:00 UTC run queued until GitHub
expired it. The scheduled job now runs only when the repository variable
``SOAK_RUNNER_AVAILABLE`` is ``'true'``; a manual dispatch always runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.unit

WORKFLOW = (
    Path(__file__).resolve().parents[2] / ".github" / "workflows" / "realtime-soak.yml"
)


def _load() -> dict[str, Any]:
    yaml = pytest.importorskip("yaml")
    data = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    assert isinstance(data, dict), "realtime-soak.yml did not parse to a mapping"
    return data


def test_soak_job_still_targets_the_soak_runner() -> None:
    assert _load()["jobs"]["soak"]["runs-on"] == "d-sorg-fleet-soak"


def test_scheduled_soak_needs_runner_variable() -> None:
    condition = _load()["jobs"]["soak"]["if"]
    assert "vars.SOAK_RUNNER_AVAILABLE == 'true'" in condition
    assert "github.event_name == 'workflow_dispatch'" in condition
