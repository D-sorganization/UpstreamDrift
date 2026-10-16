"""Characterization tests for the ``evaluate`` call surface (issue #12845)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.unit.test_golf_ml_matching_diagnostics import (
    _club_frame,
    _load_module,
    _torque_frame,
)

_TUNING = {
    "impact_time": None,
    "impact_window_s": 0.1,
    "effort_weight": 1.0e-8,
    "smoothness_weight": 1.0e-10,
}


def _csvs(tmp_path: Path) -> dict[str, Path]:
    paths = {
        "target_csv": tmp_path / "target.csv",
        "sim_csv": tmp_path / "sim.csv",
        "torque_csv": tmp_path / "torques.csv",
    }
    _club_frame().to_csv(paths["target_csv"], index=False)
    _club_frame(offset=0.01).to_csv(paths["sim_csv"], index=False)
    _torque_frame().to_csv(paths["torque_csv"], index=False)
    return paths


@pytest.mark.unit
def test_evaluate_report_values_are_pinned(tmp_path: Path) -> None:
    module = _load_module()
    report = module.evaluate(
        **_csvs(tmp_path),
        output_dir=tmp_path / "o",
        scenario="sc",
        run_label="rl",
        **_TUNING,
    )
    assert report["objective"]["value"] == pytest.approx(
        0.00015630801065644631, rel=1e-12
    )
    assert report["objective"]["terms"]["effort_source"] == "l2_torque_effort_proxy"
    assert report["objective"]["weights"]["effort"] == 1.0e-8
    assert report["objective"]["weights"]["smoothness"] == 1.0e-10
    assert report["matching"]["impact_window_s"] == 0.1
    assert report["matching"]["impact_window"]["position"]["samples"] == 2
    assert report["mechanical_work"] == {
        "available": False,
        "reason": "joint velocity data was not provided",
    }
    assert report["body_state_csv"] is None
    assert report["joint_velocity_csv"] is None
    assert sorted(report) == [
        "body_state_csv",
        "effort",
        "joint_velocity_csv",
        "matching",
        "mechanical_work",
        "objective",
        "plots",
        "run_label",
        "scenario",
        "sim_csv",
        "target_csv",
        "torque_csv",
    ]
    metrics = tmp_path / "o" / "rl_matching_metrics.json"
    saved = json.loads(metrics.read_text(encoding="utf-8"))
    assert saved["objective"]["value"] == report["objective"]["value"]


@pytest.mark.unit
def test_evaluate_records_optional_velocity_sources(tmp_path: Path) -> None:
    module = _load_module()
    csvs = _csvs(tmp_path)
    joint = tmp_path / "joint.csv"
    body = tmp_path / "body.csv"
    _torque_frame().to_csv(joint, index=False)
    _torque_frame().to_csv(body, index=False)
    report = module.evaluate(
        **csvs,
        output_dir=tmp_path / "o",
        scenario="sc",
        run_label="rl",
        body_state_csv=body,
        joint_velocity_csv=joint,
        **_TUNING,
    )
    assert report["joint_velocity_csv"] == str(joint)
    assert report["body_state_csv"] == str(body)
    assert report["mechanical_work"]["velocity_csv"] == str(joint)


@pytest.mark.unit
def test_evaluate_accepts_leading_positional_arguments(tmp_path: Path) -> None:
    module = _load_module()
    csvs = _csvs(tmp_path)
    report = module.evaluate(
        csvs["target_csv"],
        csvs["sim_csv"],
        csvs["torque_csv"],
        tmp_path / "o",
        "sc",
        "rl",
        **_TUNING,
    )
    assert report["run_label"] == "rl"


@pytest.mark.unit
def test_evaluate_rejects_missing_and_unknown_options(tmp_path: Path) -> None:
    module = _load_module()
    csvs = _csvs(tmp_path)
    base = {**csvs, "output_dir": tmp_path / "o", "scenario": "s", "run_label": "r"}
    with pytest.raises(TypeError):
        module.evaluate(**base, impact_window_s=0.1)
    with pytest.raises(TypeError):
        module.evaluate(**base, **_TUNING, bogus=1)
