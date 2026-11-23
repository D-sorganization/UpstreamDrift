"""Recorded total-effort CLI contract for optimization epic #15493."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.audit_optimization_effort import audit_bundle, main
from src.shared.python.optimization.solution_bundle import save_solution_bundle
from tests.unit.optimization.test_solution_bundle import JOINTS, _bundle

pytestmark = pytest.mark.unit


@pytest.fixture
def recorded(tmp_path: Path) -> tuple[Path, dict]:
    bundle = _bundle()
    bundle.arrays["time"][:] = [0, 0.05, 0.1, 0.15, 0.2, 0.25]
    bundle.arrays["tau_ff_ref"][:] = 0
    bundle.arrays["tau_cmd"][:] = 0
    bundle.arrays["tau_applied"][:] = [2, -3, 0]
    save_solution_bundle(bundle, tmp_path / "bundle")
    limits = {
        joint: {"lower_nm": -4, "upper_nm": 4, "slew_nm_s": 100}
        for joint in reversed(JOINTS)
    }
    return tmp_path / "bundle", limits


def test_reports_applied_torque_in_bundle_joint_order(recorded: tuple) -> None:
    path, limits = recorded
    result = audit_bundle(path, limits)
    assert result["sampled_effort_feasible"] is True
    assert result["forward_replay_verified"] is False
    assert result["solution_id"] == "sol-0001"
    assert result["sample_count"] == 6
    assert result["duration_s"] == pytest.approx(0.25)
    assert [row["joint"] for row in result["joints"]] == list(JOINTS)
    assert [row["torque_margin_nm"] for row in result["joints"]] == [2, 1, 4]
    assert result["violating_joints"] == []
    json.dumps(result, allow_nan=False)


def test_feedback_violation_is_not_hidden_by_zero_feedforward(recorded: tuple) -> None:
    path, limits = recorded
    limits[JOINTS[0]]["upper_nm"] = 1
    result = audit_bundle(path, limits)
    assert result["sampled_effort_feasible"] is False
    assert result["violating_joints"] == [JOINTS[0]]
    assert result["joints"][0]["torque_margin_nm"] == -1


def test_slew_violation_propagates_from_canonical_audit(recorded: tuple) -> None:
    path, limits = recorded
    from src.shared.python.optimization.solution_bundle import load_solution_bundle

    bundle = load_solution_bundle(path)
    bundle.arrays["tau_applied"][1:, 0] = 3
    changed_path = path.parent / "changed"
    save_solution_bundle(bundle, changed_path)
    limits[JOINTS[0]]["slew_nm_s"] = 10
    result = audit_bundle(changed_path, limits)
    assert result["sampled_effort_feasible"] is False
    assert result["joints"][0]["slew_margin_nm_s"] == pytest.approx(-10)


@pytest.mark.parametrize("mutation", ["missing", "extra", "field", "nan", "bool"])
def test_rejects_incomplete_or_invalid_limits(recorded: tuple, mutation: str) -> None:
    path, limits = recorded
    if mutation == "missing":
        limits.pop(JOINTS[0])
    elif mutation == "extra":
        limits["unknown_joint"] = limits[JOINTS[0]]
    elif mutation == "field":
        limits[JOINTS[0]]["upper_nmm"] = 4
    elif mutation == "nan":
        limits[JOINTS[0]]["upper_nm"] = float("nan")
    else:
        limits[JOINTS[0]]["upper_nm"] = True
    with pytest.raises(ValueError):
        audit_bundle(path, limits)


def test_cli_strict_json_and_exit_codes(
    recorded: tuple, capsys: pytest.CaptureFixture
) -> None:
    path, limits = recorded
    config = path.parent / "limits.json"
    config.write_text(json.dumps(limits), encoding="utf-8")
    assert main([str(path), "--limits", str(config)]) == 0
    assert json.loads(capsys.readouterr().out)["sampled_effort_feasible"] is True
    limits[JOINTS[0]]["upper_nm"] = 1
    config.write_text(json.dumps(limits), encoding="utf-8")
    assert main([str(path), "--limits", str(config)]) == 1
    assert json.loads(capsys.readouterr().out)["violating_joints"] == [JOINTS[0]]


def test_cli_corrupt_bundle_fails_without_success_report(
    recorded: tuple, capsys: pytest.CaptureFixture
) -> None:
    path, limits = recorded
    config = path.parent / "limits.json"
    config.write_text(json.dumps(limits), encoding="utf-8")
    (path / "arrays.npz").write_bytes(b"corrupt")
    with pytest.raises(SystemExit) as exc:
        main([str(path), "--limits", str(config)])
    assert exc.value.code == 2
    assert capsys.readouterr().out == ""
