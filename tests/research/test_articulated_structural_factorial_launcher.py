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
    audit_path = tmp_path / "runtime-audit.json"
    audit_path.write_text(json.dumps({"schema_version": "fixture"}), encoding="utf-8")

    class Checkpoint:
        def __init__(self, status: str, resumed: bool) -> None:
            self.status = status
            self.resumed = resumed

    import scripts.research.proximal_distal_energy.articulated_structural_factorial_evaluator as evaluator
    import scripts.research.proximal_distal_energy.articulated_structural_factorial_runner as runner

    received: dict[str, object] = {}

    def run_slice(**kwargs: object) -> tuple[Checkpoint, ...]:
        received.update(kwargs)
        return (Checkpoint("completed", False), Checkpoint("unavailable", True))

    monkeypatch.setattr(evaluator, "evaluate_structural_case", lambda *_args: {})
    monkeypatch.setattr(launcher, "validate_runtime_audit", lambda **_kwargs: "a" * 64)
    monkeypatch.setattr(runner, "run_serial_cases", run_slice)

    result = launcher.launch_structural_factorial(
        plan_path=plan_path,
        launch_path=launch_path,
        runtime_audit_path=audit_path,
        checkpoint_dir=tmp_path / "checkpoints",
        case_start=8,
        case_stop=16,
    )

    assert result["case_count"] == 2
    assert result["status_counts"] == {"completed": 1, "unavailable": 1}
    assert result["resumed_count"] == 1
    assert result["runtime_identity_sha256"] == "a" * 64
    assert result["case_start"] == 8
    assert result["case_stop"] == 16
    assert received["case_start"] == 8
    assert received["case_stop"] == 16
    session_path = tmp_path / "checkpoints" / "execution-session.json"
    assert result["execution_session_path"] == str(session_path.resolve())
    session = json.loads(session_path.read_text(encoding="utf-8"))
    assert session["schema_version"] == "articulated-structural-factorial-session/1.0.0"
    assert session["runtime_identity_sha256"] == "a" * 64
    assert session["plan_file_sha256"]
    assert session["launch_file_sha256"]
    assert session["runtime_audit_file_sha256"]


def test_launcher_rejects_an_unqualified_runtime_before_importing_evaluator(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan_path = tmp_path / "plan.json"
    launch_path = tmp_path / "launch.json"
    audit_path = tmp_path / "runtime-audit.json"
    plan_path.write_text(json.dumps({"design": {}}), encoding="utf-8")
    launch_path.write_text(json.dumps({"schema_version": "fixture"}), encoding="utf-8")
    audit_path.write_text(
        json.dumps({"qualified_for_execution": False}), encoding="utf-8"
    )

    def reject(**_kwargs: object) -> str:
        raise ValueError("runtime audit is not qualified for execution")

    monkeypatch.setattr(launcher, "validate_runtime_audit", reject)

    with pytest.raises(ValueError, match="not qualified"):
        launcher.launch_structural_factorial(
            plan_path=plan_path,
            launch_path=launch_path,
            runtime_audit_path=audit_path,
            checkpoint_dir=tmp_path / "checkpoints",
        )


def test_launcher_rejects_unbound_preexisting_checkpoint_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan_path = tmp_path / "plan.json"
    launch_path = tmp_path / "launch.json"
    audit_path = tmp_path / "runtime-audit.json"
    plan_path.write_text(json.dumps({"design": {}}), encoding="utf-8")
    launch_path.write_text(
        json.dumps({"execution_revision": "b" * 40}), encoding="utf-8"
    )
    audit_path.write_text(json.dumps({}), encoding="utf-8")
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "case-unbound.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(launcher, "validate_runtime_audit", lambda **_kwargs: "a" * 64)

    with pytest.raises(ValueError, match="lacks an execution session"):
        launcher.launch_structural_factorial(
            plan_path=plan_path,
            launch_path=launch_path,
            runtime_audit_path=audit_path,
            checkpoint_dir=checkpoint_dir,
        )


def test_launcher_rejects_execution_session_identity_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan_path = tmp_path / "plan.json"
    launch_path = tmp_path / "launch.json"
    audit_path = tmp_path / "runtime-audit.json"
    plan_path.write_text(json.dumps({"design": {}}), encoding="utf-8")
    launch_path.write_text(
        json.dumps({"execution_revision": "b" * 40}), encoding="utf-8"
    )
    audit_path.write_text(json.dumps({}), encoding="utf-8")
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "execution-session.json").write_text(
        json.dumps({"schema_version": "wrong"}), encoding="utf-8"
    )
    monkeypatch.setattr(launcher, "validate_runtime_audit", lambda **_kwargs: "a" * 64)

    with pytest.raises(ValueError, match="session identity"):
        launcher.launch_structural_factorial(
            plan_path=plan_path,
            launch_path=launch_path,
            runtime_audit_path=audit_path,
            checkpoint_dir=checkpoint_dir,
        )


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
        execution_revision="2e5145fdefdd837bdf7401f7f1d5dfdda1cebf4f",
    )
