from __future__ import annotations

import importlib.util
import json
from pathlib import Path

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
    / "prepare_frame_by_frame_search.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "prepare_frame_by_frame_search", SCRIPT_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _column_manifest(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "input_columns": {
                    "time": ["time"],
                    "joint_positions": ["HipLogs_HipPositionX"],
                    "joint_velocities": ["HipLogs_HipVelocityX"],
                    "applied_controls": [
                        "LSLogs_ActuatorTorqueX",
                        "RSLogs_ActuatorTorqueY",
                    ],
                },
                "target_columns": {
                    "club": ["ClubLogs_CHGlobalPosition_1"],
                },
            }
        ),
        encoding="utf-8",
    )


def _target_csv(path: Path) -> None:
    time = np.array([0.0, 0.01, 0.02], dtype=float)
    pd.DataFrame(
        {
            "time": time,
            "clubface_x": [0.0, 0.1, 0.2],
            "clubface_y": [0.0, 0.0, 0.0],
            "clubface_z": [1.0, 1.1, 1.2],
        }
    ).to_csv(path, index=False)


def test_build_search_manifest_writes_deterministic_contract(tmp_path: Path) -> None:
    module = _load_module()
    manifest_path = tmp_path / "columns.json"
    target_path = tmp_path / "desired.csv"
    output_path = tmp_path / "search.json"
    _column_manifest(manifest_path)
    _target_csv(target_path)

    manifest = module.build_search_manifest(
        desired_target_csv=target_path,
        column_manifest=manifest_path,
        output_json=output_path,
        torque_output_csv=tmp_path / "torques.csv",
        polynomial_output_mat=tmp_path / "torques.mat",
        candidate_step=2.5,
        candidate_levels=[-1.0, 0.0, 1.0],
        requested_control_columns=["LSLogs_ActuatorTorqueX"],
    )

    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert manifest == saved
    assert saved["workflow"] == "frame_by_frame_torque_search"
    assert saved["search"]["candidate_step"] == 2.5
    assert saved["search"]["candidate_strategy"] == "coordinate"
    assert saved["validation"]["target_rows"] == 3
    assert saved["validation"]["candidates_per_frame"] == 3
    assert saved["columns"]["target_columns"] == [
        "clubface_x",
        "clubface_y",
        "clubface_z",
    ]
    assert saved["outputs"]["polynomial_summary_json"].endswith(".summary.json")


def test_build_search_manifest_rejects_non_monotonic_time(tmp_path: Path) -> None:
    module = _load_module()
    manifest_path = tmp_path / "columns.json"
    target_path = tmp_path / "desired.csv"
    _column_manifest(manifest_path)
    pd.DataFrame(
        {
            "time": [0.0, 0.02, 0.01],
            "clubface_x": [0.0, 0.1, 0.2],
        }
    ).to_csv(target_path, index=False)

    with pytest.raises(ValueError, match="strictly increasing"):
        module.build_search_manifest(
            desired_target_csv=target_path,
            column_manifest=manifest_path,
            output_json=tmp_path / "search.json",
        )


def test_build_search_manifest_caps_candidate_expansion(tmp_path: Path) -> None:
    module = _load_module()
    manifest_path = tmp_path / "columns.json"
    target_path = tmp_path / "desired.csv"
    _column_manifest(manifest_path)
    _target_csv(target_path)

    with pytest.raises(ValueError, match="above max"):
        module.build_search_manifest(
            desired_target_csv=target_path,
            column_manifest=manifest_path,
            output_json=tmp_path / "search.json",
            candidate_strategy="cartesian",
            candidate_levels=[-1.0, 0.0, 1.0],
            max_candidates_per_frame=4,
        )
