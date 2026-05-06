"""Unit tests for the closed-loop MATLAB replay diagnostics harness (#3970)."""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

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
    / "replay_matching_workflow.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "replay_matching_workflow", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def harness():
    return _load_module()


@pytest.fixture()
def fake_target_csv(tmp_path: Path) -> Path:
    time = np.linspace(0.0, 1.0, 21)
    frame = pd.DataFrame(
        {
            "time": time,
            "clubface_x": np.sin(time),
            "clubface_y": np.cos(time),
            "clubface_z": time**2,
            "clubface_vx": np.cos(time),
            "clubface_vy": -np.sin(time),
            "clubface_vz": 2.0 * time,
            "clubface_ax": -np.sin(time),
            "clubface_ay": -np.cos(time),
            "clubface_az": np.full_like(time, 2.0),
        }
    )
    path = tmp_path / "target_club.csv"
    frame.to_csv(path, index=False)
    return path


@pytest.fixture()
def fake_sim_csv(tmp_path: Path) -> Path:
    time = np.linspace(0.0, 1.0, 21)
    frame = pd.DataFrame(
        {
            "time": time,
            "clubface_x": np.sin(time) + 0.01,
            "clubface_y": np.cos(time) - 0.01,
            "clubface_z": time**2 + 0.005,
            "clubface_vx": np.cos(time),
            "clubface_vy": -np.sin(time),
            "clubface_vz": 2.0 * time,
            "clubface_ax": -np.sin(time),
            "clubface_ay": -np.cos(time),
            "clubface_az": np.full_like(time, 2.0),
        }
    )
    path = tmp_path / "existing_sim.csv"
    frame.to_csv(path, index=False)
    return path


@pytest.fixture()
def fake_polynomial_mat(tmp_path: Path) -> Path:
    path = tmp_path / "polynomial_inputs.mat"
    path.write_bytes(b"\x00MAT")
    return path


def test_skips_simulation_when_matlab_unavailable(
    harness, fake_polynomial_mat, fake_target_csv, fake_sim_csv, tmp_path, monkeypatch
):
    monkeypatch.setattr(harness, "_matlab_engine_available", lambda: False)
    result = harness.replay(
        polynomial_mat=fake_polynomial_mat,
        target_csv=fake_target_csv,
        scenario="full-swing",
        output_root=tmp_path / "reports",
        existing_sim_csv=fake_sim_csv,
        timestamp=datetime(2026, 5, 5, 12, 0, 0, tzinfo=timezone.utc),
    )
    assert result.matlab_used is False
    assert result.skipped_reason is not None
    assert result.sim_csv is not None and result.sim_csv.exists()
    assert result.metrics_path is not None and result.metrics_path.exists()
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["matlab_used"] is False
    assert manifest["scenario"] == "full-swing"


def test_full_pipeline_with_mocked_engine(
    harness, fake_polynomial_mat, fake_target_csv, fake_sim_csv, tmp_path, monkeypatch
):
    fake_engine = MagicMock()
    eng_instance = MagicMock()
    fake_engine.start_matlab.return_value = eng_instance

    def fake_run_replay(
        polynomial, scenario, start_state, sim_csv_path, joint_csv_path, nargout
    ):
        # MATLAB driver would write the simulated CSV. Emulate it here.
        Path(sim_csv_path).write_bytes(fake_sim_csv.read_bytes())
        assert nargout == 0

    eng_instance.run_replay.side_effect = fake_run_replay

    monkeypatch.setattr(harness, "_matlab_engine_available", lambda: True)

    # Provide a stub matlab driver path that exists.
    driver_path = tmp_path / "driver" / "run_replay.m"
    driver_path.parent.mkdir(parents=True)
    driver_path.write_text("% stub")

    result = harness.replay(
        polynomial_mat=fake_polynomial_mat,
        target_csv=fake_target_csv,
        scenario="downswing",
        output_root=tmp_path / "reports",
        matlab_driver=driver_path,
        matlab_engine_module=fake_engine,
        timestamp=datetime(2026, 5, 5, 12, 0, 0, tzinfo=timezone.utc),
    )

    fake_engine.start_matlab.assert_called_once()
    eng_instance.addpath.assert_any_call(str(driver_path.parent), nargout=0)
    eng_instance.run_replay.assert_called_once()
    eng_instance.quit.assert_called_once()

    assert result.matlab_used is True
    assert result.sim_csv is not None and result.sim_csv.exists()
    assert result.metrics_path is not None and result.metrics_path.exists()
    assert (result.run_dir / "run_manifest.json").exists()


def test_emits_canonical_metrics_per_schema(
    harness, fake_polynomial_mat, fake_target_csv, fake_sim_csv, tmp_path, monkeypatch
):
    # Tolerate the canonical Metrics schema not yet being on main (#4046).
    pytest.importorskip(
        "src.shared.python.motion_matching.metrics",
        reason="canonical Metrics schema (#4046) not yet on main",
    )
    monkeypatch.setattr(harness, "_matlab_engine_available", lambda: False)
    result = harness.replay(
        polynomial_mat=fake_polynomial_mat,
        target_csv=fake_target_csv,
        scenario="full-swing",
        output_root=tmp_path / "reports",
        existing_sim_csv=fake_sim_csv,
    )
    assert result.metrics_path is not None
    payload = json.loads(result.metrics_path.read_text())
    # Schema-level expectations for canonical Metrics output.
    assert "scenario" in payload
    assert "matching" in payload
    assert "objective" in payload


def test_failed_sim_propagates_clear_error(
    harness, fake_polynomial_mat, fake_target_csv, tmp_path, monkeypatch
):
    fake_engine = MagicMock()
    eng_instance = MagicMock()
    fake_engine.start_matlab.return_value = eng_instance
    eng_instance.run_replay.side_effect = RuntimeError("Simscape blew up")

    monkeypatch.setattr(harness, "_matlab_engine_available", lambda: True)

    driver_path = tmp_path / "driver" / "run_replay.m"
    driver_path.parent.mkdir(parents=True)
    driver_path.write_text("% stub")

    with pytest.raises(harness.ReplayError) as excinfo:
        harness.replay(
            polynomial_mat=fake_polynomial_mat,
            target_csv=fake_target_csv,
            scenario="downswing",
            output_root=tmp_path / "reports",
            matlab_driver=driver_path,
            matlab_engine_module=fake_engine,
        )
    assert "downswing" in str(excinfo.value)
    assert "Simscape blew up" in str(excinfo.value)
    eng_instance.quit.assert_called_once()


def test_run_dir_layout(
    harness, fake_polynomial_mat, fake_target_csv, fake_sim_csv, tmp_path, monkeypatch
):
    monkeypatch.setattr(harness, "_matlab_engine_available", lambda: False)
    when = datetime(2026, 5, 5, 12, 34, 56, tzinfo=timezone.utc)
    result = harness.replay(
        polynomial_mat=fake_polynomial_mat,
        target_csv=fake_target_csv,
        scenario="full-swing",
        output_root=tmp_path / "reports",
        existing_sim_csv=fake_sim_csv,
        run_label="unit",
        timestamp=when,
    )

    expected_dir = tmp_path / "reports" / "full-swing" / "20260505T123456Z"
    assert result.run_dir == expected_dir
    assert (expected_dir / "run_manifest.json").exists()
    assert (expected_dir / "simulated_club_motion.csv").exists()
    assert (expected_dir / "unit_matching_metrics.json").exists()
    assert (expected_dir / "unit_matching_summary.md").exists()


def test_invalid_scenario_rejected(harness, fake_polynomial_mat, fake_target_csv):
    with pytest.raises(ValueError, match="scenario"):
        harness.build_inputs(
            polynomial_mat=fake_polynomial_mat,
            target_csv=fake_target_csv,
            scenario="not-a-scenario",
        )


def test_missing_polynomial_mat_rejected(tmp_path, harness, fake_target_csv):
    with pytest.raises(FileNotFoundError):
        harness.build_inputs(
            polynomial_mat=tmp_path / "missing.mat",
            target_csv=fake_target_csv,
            scenario="full-swing",
        )
