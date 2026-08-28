"""CLI contract tests for the serial rigid-refinement extension."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.research.proximal_distal_energy import (
    articulated_rigid_refinement_launcher as launcher,
)


def test_launcher_reports_atomic_case_statuses(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"design": {}}), encoding="utf-8")

    class Checkpoint:
        def __init__(self, status: str, resumed: bool) -> None:
            self.status = status
            self.resumed = resumed

    import scripts.research.proximal_distal_energy.articulated_forward_smoke_evaluator as evaluator

    monkeypatch.setattr(
        evaluator,
        "run_registered_rigid_smoke",
        lambda **_kwargs: (
            Checkpoint("completed", False),
            Checkpoint("unavailable", True),
        ),
    )

    result = launcher.launch_rigid_refinement(
        plan_path=plan,
        execution_revision="a" * 40,
        checkpoint_dir=tmp_path / "checkpoints",
    )

    assert result["case_count"] == 2
    assert result["status_counts"] == {"completed": 1, "unavailable": 1}
    assert result["resumed_count"] == 1
