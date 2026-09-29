from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "engines"
    / "Simscape_Multibody_Models"
    / "3D_Golf_Model"
    / "MachineLearning"
    / "evaluate_matching_workflow.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "golf_ml_mechanical_work", SCRIPT_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compute_mechanical_work_integrates_positive_negative_and_net_work() -> None:
    module = _load_module()
    torque_frame = pd.DataFrame(
        {
            "time": [0.0, 1.0, 2.0],
            "ShoulderTorqueX": [2.0, 2.0, 2.0],
            "ElbowTorqueZ": [-1.0, -1.0, -1.0],
        }
    )
    velocity_frame = pd.DataFrame(
        {
            "time": [0.0, 1.0, 2.0],
            "ShoulderVelocityX": [3.0, 3.0, 3.0],
            "ElbowAngularVelocityZ": [4.0, 4.0, 4.0],
        }
    )

    work = module.compute_mechanical_work(torque_frame, velocity_frame)

    assert work["available"] is True
    assert work["paired_columns"] == {
        "ShoulderTorqueX": "ShoulderVelocityX",
        "ElbowTorqueZ": "ElbowAngularVelocityZ",
    }
    assert work["positive_mechanical_work"] == pytest.approx(12.0)
    assert work["negative_mechanical_work_abs"] == pytest.approx(8.0)
    assert work["net_mechanical_work"] == pytest.approx(4.0)
    assert work["per_joint_ranking"][0]["torque_column"] == "ShoulderTorqueX"
    assert work["per_joint_ranking"][0]["positive_mechanical_work"] == pytest.approx(
        12.0
    )


def test_torque_only_evaluate_reports_missing_qdot_fallback(tmp_path: Path) -> None:
    module = _load_module()
    torque_csv = tmp_path / "torques.csv"
    pd.DataFrame(
        {
            "time": [0.0, 1.0],
            "LSLogs_ActuatorTorqueX": [1.0, 1.0],
        }
    ).to_csv(torque_csv, index=False)

    report = module.evaluate(
        target_csv=None,
        sim_csv=None,
        torque_csv=torque_csv,
        output_dir=tmp_path / "reports",
        scenario="downswing",
        run_label="torque_only",
        impact_time=None,
        impact_window_s=0.02,
        effort_weight=1.0e-8,
        smoothness_weight=1.0e-10,
    )

    assert report["effort"]["available"] is True
    assert report["mechanical_work"]["available"] is False
    assert "joint velocity" in report["mechanical_work"]["reason"]
