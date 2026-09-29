from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

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
        "golf_ml_matching_diagnostics", SCRIPT_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _club_frame(offset: float = 0.0) -> pd.DataFrame:
    time = np.linspace(0.0, 1.0, 31)
    return pd.DataFrame(
        {
            "time": time,
            "clubface_x": np.sin(time) + offset,
            "clubface_y": np.cos(time) - offset,
            "clubface_z": time**2 + offset,
            "clubface_vx": np.cos(time),
            "clubface_vy": -np.sin(time),
            "clubface_vz": 2.0 * time,
            "clubface_ax": -np.sin(time),
            "clubface_ay": -np.cos(time),
            "clubface_az": np.full_like(time, 2.0),
        }
    )


def _torque_frame() -> pd.DataFrame:
    time = np.linspace(0.0, 1.0, 31)
    return pd.DataFrame(
        {
            "time": time,
            "LSLogs_ActuatorTorqueX": np.sin(2.0 * np.pi * time),
            "RSLogs_ActuatorTorqueY": 0.5 * np.cos(2.0 * np.pi * time),
        }
    )


def test_matching_diagnostics_writes_report_files(tmp_path: Path) -> None:
    module = _load_module()
    target_csv = tmp_path / "target.csv"
    sim_csv = tmp_path / "sim.csv"
    torque_csv = tmp_path / "torques.csv"
    _club_frame().to_csv(target_csv, index=False)
    _club_frame(offset=0.01).to_csv(sim_csv, index=False)
    _torque_frame().to_csv(torque_csv, index=False)

    report = module.evaluate(
        target_csv=target_csv,
        sim_csv=sim_csv,
        torque_csv=torque_csv,
        output_dir=tmp_path / "reports",
        scenario="downswing",
        run_label="synthetic",
        impact_time=None,
        impact_window_s=0.1,
        effort_weight=1.0e-8,
        smoothness_weight=1.0e-10,
    )

    metrics_path = tmp_path / "reports" / "synthetic_matching_metrics.json"
    summary_path = tmp_path / "reports" / "synthetic_matching_summary.md"
    assert metrics_path.exists()
    assert summary_path.exists()
    saved = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert saved["matching"]["groups"]["position"]["normalized_vector_rmse"] > 0.0
    assert saved["effort"]["available"] is True
    assert report["objective"]["terms"]["tracking"] is not None


def test_matching_diagnostics_supports_torque_only_report(
    tmp_path: Path,
) -> None:
    module = _load_module()
    torque_csv = tmp_path / "torques.csv"
    _torque_frame().to_csv(torque_csv, index=False)

    report = module.evaluate(
        target_csv=None,
        sim_csv=None,
        torque_csv=torque_csv,
        output_dir=tmp_path / "reports",
        scenario="full-swing",
        run_label="torque_only",
        impact_time=None,
        impact_window_s=0.02,
        effort_weight=1.0e-8,
        smoothness_weight=1.0e-10,
    )

    assert report["matching"] is None
    assert report["effort"]["available"] is True
    assert (tmp_path / "reports" / "torque_only_matching_summary.md").exists()
