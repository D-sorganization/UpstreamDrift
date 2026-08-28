"""CLI contract for the serial structural-factorial launcher."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.research.proximal_distal_energy import (
    articulated_structural_factorial_launcher as launcher,
)
from scripts.research.proximal_distal_energy.articulated_structural_factorial_runner import (
    build_launch_manifest,
)

pytestmark = pytest.mark.scientific

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/research/proximal_distal_energy_transfer/data"


def test_launcher_reports_statuses_without_owning_execution_logic(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan_path = tmp_path / "plan.json"
    launch_path = tmp_path / "launch.json"
    plan_path.write_text(json.dumps({"design": {}}), encoding="utf-8")
    launch_path.write_text(json.dumps({"schema_version": "fixture"}), encoding="utf-8")

    class Checkpoint:
        def __init__(self, status: str, resumed: bool) -> None:
            self.status = status
            self.resumed = resumed

    import scripts.research.proximal_distal_energy.articulated_structural_factorial_evaluator as evaluator
    import scripts.research.proximal_distal_energy.articulated_structural_factorial_runner as runner

    monkeypatch.setattr(evaluator, "evaluate_structural_case", lambda *_args: {})
    monkeypatch.setattr(
        runner,
        "run_serial_cases",
        lambda **_kwargs: (
            Checkpoint("completed", False),
            Checkpoint("unavailable", True),
        ),
    )

    result = launcher.launch_structural_factorial(
        plan_path=plan_path,
        launch_path=launch_path,
        checkpoint_dir=tmp_path / "checkpoints",
    )

    assert result["case_count"] == 2
    assert result["status_counts"] == {"completed": 1, "unavailable": 1}
    assert result["resumed_count"] == 1


def test_committed_launch_exactly_binds_the_immutable_runner() -> None:
    plan = json.loads(
        (DATA / "articulated_structural_factorial_plan.json").read_text(
            encoding="utf-8"
        )
    )
    committed = json.loads(
        (DATA / "articulated_structural_factorial_launch.json").read_text(
            encoding="utf-8"
        )
    )

    assert committed == build_launch_manifest(
        plan=plan,
        execution_revision="d6bffd0c6ae6e7d90fdc18778799c0bd4b592dae",
    )
