from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

from src.tools.drift_control.analyzer import DriftControlAnalyzer

pytestmark = pytest.mark.unit


def test_drift_control_analyzer_loads_npz_and_computes_force_ratio(tmp_path) -> None:
    trajectory_path = tmp_path / "expert.npz"
    np.savez(
        trajectory_path,
        time=np.array([0.0, 0.1]),
        qfrc_bias=np.array([[3.0, 4.0], [0.0, 6.0]]),
        qfrc_actuator=np.array([[0.0, 5.0], [8.0, 0.0]]),
    )

    analyzer = DriftControlAnalyzer()
    trajectory = analyzer.load_expert_trajectory(trajectory_path)
    ratio = analyzer.compute_ratio(trajectory)

    assert trajectory.sample_count == 2
    assert np.allclose(ratio, np.array([1.0, 0.75]))
    assert analyzer.summarize_ratio(ratio) == {
        "sample_count": 2,
        "minimum": 0.75,
        "maximum": 1.0,
        "mean": 0.875,
    }


def test_drift_control_analyzer_rejects_shape_mismatch(tmp_path) -> None:
    trajectory_path = tmp_path / "bad.npz"
    np.savez(
        trajectory_path,
        drift_generalized_force=np.ones((2, 2)),
        control_generalized_force=np.ones((3, 2)),
    )

    with pytest.raises(ValueError, match="same shape"):
        DriftControlAnalyzer().load_expert_trajectory(trajectory_path)


def test_drift_control_cli_outputs_json_summary(tmp_path) -> None:
    trajectory_path = tmp_path / "expert.npz"
    np.savez(
        trajectory_path,
        drift_generalized_force=np.array([[3.0, 4.0]]),
        control_generalized_force=np.array([[0.0, 5.0]]),
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.tools.drift_control",
            str(trajectory_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["ratio"]["mean"] == 1.0
    assert payload["trajectory"]["sample_count"] == 1
