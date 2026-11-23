"""Sampled kinematic guardrails for optimization (Refs #15493)."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scripts.audit_optimization_kinematics import audit_bundle, main
from src.shared.python.optimization.solution_bundle import (
    SolutionBundle,
    save_solution_bundle,
)
from tests.unit.optimization.test_solution_bundle import _bundle

pytestmark = pytest.mark.unit


def inputs() -> tuple[SolutionBundle, dict[str, Any]]:
    bundle = _bundle()
    arrays = {key: value.copy() for key, value in bundle.arrays.items()}
    for key in ("q_rollout", "qd_rollout", "q_ref", "qd_ref"):
        arrays[key][:] = 0.0
    limits = {
        "joint_order": list(bundle.joint_order),
        "position_lower_rad": [-1.0] * 3,
        "position_upper_rad": [2.0] * 3,
        "speed_limit_rad_s": [3.0] * 3,
        "position_error_limit_rad": [0.5] * 3,
        "velocity_error_limit_rad_s": [1.0] * 3,
    }
    return replace(bundle, arrays=arrays), limits


def test_reports_independent_signed_worst_sample_margins() -> None:
    bundle, limits = inputs()
    bundle.arrays["q_rollout"][0] = [-1.25, 2.5, 0.5]
    bundle.arrays["qd_rollout"][1] = [-4.0, 2.0, 0.0]
    report = audit_bundle(bundle, limits)
    assert report["sampled_feasible"] is False
    assert report["position_margin_rad"] == pytest.approx([-0.25, -0.5, 1.0])
    assert report["speed_margin_rad_s"] == pytest.approx([-1.0, 1.0, 3.0])
    assert report["position_tracking_margin_rad"] == pytest.approx([-0.75, -2, 0])
    assert report["velocity_tracking_margin_rad_s"] == pytest.approx([-3, -1, 1])
    assert report["joint_order"] == list(bundle.joint_order)
    assert report["solution_id"] == bundle.solution_id
    json.dumps(report, allow_nan=False)


def test_zero_margin_is_feasible_and_reference_cannot_hide_actual_rom() -> None:
    bundle, limits = inputs()
    bundle.arrays["q_rollout"][:] = 2.0
    bundle.arrays["q_ref"][:] = 2.5
    bundle.arrays["qd_rollout"][:] = -3.0
    bundle.arrays["qd_ref"][:] = -2.0
    report = audit_bundle(bundle, limits)
    assert report["sampled_feasible"] is True
    for key in report:
        if "margin" in key:
            assert report[key] == [0.0] * 3
    bundle.arrays["q_rollout"][2, 0] = 2.1
    bundle.arrays["q_ref"][2, 0] = 2.1
    assert audit_bundle(bundle, limits)["sampled_feasible"] is False


@pytest.mark.parametrize("key", ["q_ref", "qd_ref", "q_rollout", "qd_rollout"])
@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_rejects_nonfinite_trajectory(key: str, bad: float) -> None:
    bundle, limits = inputs()
    bundle.arrays[key][0, 0] = bad
    with pytest.raises(ValueError, match="finite"):
        audit_bundle(bundle, limits)


@pytest.mark.parametrize("key", ["q_ref", "qd_ref", "q_rollout", "qd_rollout"])
def test_rejects_misaligned_trajectory(key: str) -> None:
    bundle, limits = inputs()
    bundle.arrays[key] = np.zeros((2, 3))
    with pytest.raises(ValueError, match="shape"):
        audit_bundle(bundle, limits)


@pytest.mark.parametrize(
    "bad", [[], ["wrong"], ["lead_wrist_flex", "thorax_rz", "pelvis_rz"]]
)
def test_requires_exact_joint_order(bad: list[str]) -> None:
    bundle, limits = inputs()
    limits["joint_order"] = bad
    with pytest.raises(ValueError, match="joint_order"):
        audit_bundle(bundle, limits)


@pytest.mark.parametrize("bad", [-1, np.inf, True, "1", [1]])
def test_rejects_invalid_tracking_limits(bad: Any) -> None:
    bundle, limits = inputs()
    limits["position_error_limit_rad"] = [bad] * 3
    with pytest.raises(ValueError):
        audit_bundle(bundle, limits)


def test_rejects_missing_and_inverted_limits() -> None:
    bundle, limits = inputs()
    with pytest.raises(ValueError, match="missing"):
        audit_bundle(bundle, {})
    limits["position_lower_rad"] = [3.0] * 3
    with pytest.raises(ValueError, match="lower"):
        audit_bundle(bundle, limits)


@pytest.mark.parametrize("key", ["speed_limit_rad_s", "velocity_error_limit_rad_s"])
def test_rejects_negative_speed_and_velocity_error_limits(key: str) -> None:
    bundle, limits = inputs()
    limits[key][1] = -0.1
    with pytest.raises(ValueError, match="nonnegative"):
        audit_bundle(bundle, limits)


def test_cli_rejects_malformed_limits_without_success_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bundle, _ = inputs()
    save_solution_bundle(bundle, tmp_path / "bundle")
    path = tmp_path / "limits.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        main([str(tmp_path / "bundle"), str(path)])
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "limits must be an object" in captured.err


def test_overflow_cannot_produce_a_feasible_report() -> None:
    bundle, limits = inputs()
    bundle.arrays["q_rollout"][:] = 1e308
    bundle.arrays["q_ref"][:] = -1e308
    with pytest.raises(ValueError, match="finite"):
        audit_bundle(bundle, limits)


@pytest.mark.parametrize("times", [[], [0], [0, 0], [1, 0], [0, np.inf]])
def test_rejects_invalid_time_support(times: list[float]) -> None:
    bundle, limits = inputs()
    bundle.arrays["time"] = np.asarray(times)
    with pytest.raises(ValueError):
        audit_bundle(bundle, limits)


def test_cli_loads_existing_codec_and_emits_strict_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bundle, limits = inputs()
    save_solution_bundle(bundle, tmp_path / "bundle")
    limits_path = tmp_path / "limits.json"
    limits_path.write_text(json.dumps(limits), encoding="utf-8")
    assert main([str(tmp_path / "bundle"), str(limits_path)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["sampled_feasible"] is True
    bundle.arrays["qd_rollout"][0, 0] = 4.0
    save_solution_bundle(bundle, tmp_path / "bad")
    assert main([str(tmp_path / "bad"), str(limits_path)]) == 1
    assert json.loads(capsys.readouterr().out)["sampled_feasible"] is False
