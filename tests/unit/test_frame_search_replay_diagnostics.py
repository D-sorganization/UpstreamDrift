"""Unit tests for the frame-search replay diagnostics module (#3979)."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "engines"
    / "Simscape_Multibody_Models"
    / "3D_Golf_Model"
    / "MachineLearning"
    / "frame_search_replay_diagnostics.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "frame_search_replay_diagnostics", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["frame_search_replay_diagnostics"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def replay_module():
    return _load_module()


def _make_target_frame() -> pd.DataFrame:
    time = np.linspace(-0.05, 0.05, 51)
    return pd.DataFrame(
        {
            "time": time,
            "clubface_x": 0.5 * np.sin(20.0 * time),
            "clubface_y": 0.5 * np.cos(20.0 * time),
            "clubface_z": 0.1 + time,
            "clubface_vx": 0.5 * 20.0 * np.cos(20.0 * time),
            "clubface_vy": -0.5 * 20.0 * np.sin(20.0 * time),
            "clubface_vz": np.ones_like(time),
        }
    )


def _make_simulated_frame(offset: float = 0.01) -> pd.DataFrame:
    target = _make_target_frame()
    return pd.DataFrame(
        {
            "time": target["time"],
            "ClubLogs_CHGlobalPosition_1": target["clubface_x"] + offset,
            "ClubLogs_CHGlobalPosition_2": target["clubface_y"] + offset,
            "ClubLogs_CHGlobalPosition_3": target["clubface_z"] + offset,
            "ClubLogs_CHGlobalVelocity_1": target["clubface_vx"],
            "ClubLogs_CHGlobalVelocity_2": target["clubface_vy"],
            "ClubLogs_CHGlobalVelocity_3": target["clubface_vz"],
        }
    )


def _make_torque_frame() -> pd.DataFrame:
    time = np.linspace(-0.05, 0.05, 51)
    return pd.DataFrame(
        {
            "time": time,
            "LSLogs_ActuatorTorqueX": np.sin(2.0 * np.pi * time * 10.0),
            "RSLogs_ActuatorTorqueY": 0.5 * np.cos(2.0 * np.pi * time * 10.0),
        }
    )


def _write_pipeline(tmp_path: Path) -> dict[str, Path]:
    target_csv = tmp_path / "target.csv"
    sim_csv = tmp_path / "simulated.csv"
    torque_csv = tmp_path / "torques.csv"
    poly_mat = tmp_path / "poly.mat"
    poly_mat.write_bytes(b"")  # placeholder
    _make_target_frame().to_csv(target_csv, index=False)
    _make_simulated_frame().to_csv(sim_csv, index=False)
    _make_torque_frame().to_csv(torque_csv, index=False)
    return {
        "target": target_csv,
        "simulated": sim_csv,
        "torque": torque_csv,
        "poly": poly_mat,
    }


def test_replay_pipeline_runs_with_mocked_replay(replay_module, tmp_path):
    paths = _write_pipeline(tmp_path)
    out_dir = tmp_path / "out"

    fake_script = tmp_path / "replay_matching_workflow.py"
    fake_script.write_text("# placeholder", encoding="utf-8")

    inputs = replay_module.ReplayInputs(
        polynomial_mat=paths["poly"],
        target_csv=paths["target"],
        simulated_club_csv=paths["simulated"],
        torque_csv=paths["torque"],
        output_dir=out_dir,
        replay_script=fake_script,
    )

    completed = types.SimpleNamespace(returncode=0, stdout="", stderr="")
    with patch.object(replay_module.subprocess, "run", return_value=completed) as m:
        diagnostics = replay_module.run_replay_diagnostics(inputs)

    m.assert_called_once()
    assert diagnostics.replay_executed is True
    # Position offsets of 0.01 should appear in residuals.
    assert diagnostics.position_error_rms["x"] == pytest.approx(0.01, abs=1e-9)
    assert diagnostics.impact_position_error["x"] == pytest.approx(0.01, abs=1e-9)
    assert diagnostics.torque_effort["effort_l1"] > 0.0
    summary = out_dir / "frame_search_replay_summary.json"
    assert summary.exists()
    record = json.loads(summary.read_text())
    assert record["replay_executed"] is True


def test_metric_emission_when_metrics_module_present(replay_module, tmp_path):
    paths = _write_pipeline(tmp_path)
    out_dir = tmp_path / "out"

    captured: dict = {}
    fake_metrics = types.ModuleType("ud.metrics")

    def record_metric(name, payload):
        captured["name"] = name
        captured["payload"] = payload

    fake_metrics.record_metric = record_metric  # type: ignore[attr-defined]

    inputs = replay_module.ReplayInputs(
        polynomial_mat=paths["poly"],
        target_csv=paths["target"],
        simulated_club_csv=paths["simulated"],
        torque_csv=paths["torque"],
        output_dir=out_dir,
    )

    with patch.dict(
        sys.modules, {"ud": types.ModuleType("ud"), "ud.metrics": fake_metrics}
    ):
        diagnostics = replay_module.run_replay_diagnostics(inputs)

    assert captured.get("name") == "frame_search_replay"
    assert "position_error_rms" in captured["payload"]
    assert diagnostics.metrics_emitted_via.startswith("module:")


def test_handles_missing_simulation_csv_gracefully(replay_module, tmp_path):
    paths = _write_pipeline(tmp_path)
    out_dir = tmp_path / "out"
    missing_sim = tmp_path / "does_not_exist.csv"

    inputs = replay_module.ReplayInputs(
        polynomial_mat=paths["poly"],
        target_csv=paths["target"],
        simulated_club_csv=missing_sim,
        torque_csv=paths["torque"],
        output_dir=out_dir,
    )

    diagnostics = replay_module.run_replay_diagnostics(inputs)

    assert diagnostics.replay_executed is False  # no script present
    assert diagnostics.position_error_rms == {}
    assert diagnostics.impact_position_error == {}
    # Torque effort still computed even though sim CSV is missing.
    assert diagnostics.torque_effort["effort_l1"] > 0.0
    fallback = out_dir / "frame_search_replay_metrics.json"
    summary = out_dir / "frame_search_replay_summary.json"
    assert summary.exists()
    # metrics_emitted_via should be JSON fallback when no metrics module exposed.
    assert diagnostics.metrics_emitted_via.startswith(
        "json:"
    ) or diagnostics.metrics_emitted_via.startswith("module:")
    if diagnostics.metrics_emitted_via.startswith("json:"):
        assert fallback.exists()
