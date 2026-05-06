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
    / "validate_club_calibration.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "golf_ml_calibration_validation", SCRIPT_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _measured_frame() -> pd.DataFrame:
    time = np.linspace(0.0, 1.0, 21)
    return pd.DataFrame(
        {
            "time": time,
            "clubface_x": time,
            "clubface_y": time**2,
            "clubface_z": np.sin(time),
            "clubface_vx": np.ones_like(time),
            "clubface_vy": 2.0 * time,
            "clubface_vz": np.cos(time),
            "clubface_ax": np.zeros_like(time),
            "clubface_ay": np.full_like(time, 2.0),
            "clubface_az": -np.sin(time),
        }
    )


def _model_frame_from_source(
    source: pd.DataFrame,
    scale: float = 1.0,
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> pd.DataFrame:
    output = pd.DataFrame({"time": source["time"]})
    position_source = source[["clubface_x", "clubface_y", "clubface_z"]].to_numpy(
        dtype=float
    )
    velocity_source = source[["clubface_vx", "clubface_vy", "clubface_vz"]].to_numpy(
        dtype=float
    )
    accel_source = source[["clubface_ax", "clubface_ay", "clubface_az"]].to_numpy(
        dtype=float
    )
    for idx, column in enumerate(
        [
            "ClubLogs_CHGlobalPosition_1",
            "ClubLogs_CHGlobalPosition_2",
            "ClubLogs_CHGlobalPosition_3",
        ]
    ):
        output[column] = scale * position_source[:, idx] + translation[idx]
    for idx, column in enumerate(
        [
            "ClubLogs_CHGlobalVelocity_1",
            "ClubLogs_CHGlobalVelocity_2",
            "ClubLogs_CHGlobalVelocity_3",
        ]
    ):
        output[column] = scale * velocity_source[:, idx]
    for idx, column in enumerate(
        [
            "ClubLogs_CHGlobalAcceleration_1",
            "ClubLogs_CHGlobalAcceleration_2",
            "ClubLogs_CHGlobalAcceleration_3",
        ]
    ):
        output[column] = scale * accel_source[:, idx]
    return output


def test_calibration_validation_reports_known_translation_scale_improvement(
    tmp_path: Path,
) -> None:
    module = _load_module()
    measured = _measured_frame()
    simulated = _model_frame_from_source(
        measured, scale=1.5, translation=(0.2, -0.1, 0.3)
    )
    transform_path = tmp_path / "transform.json"
    transform_path.write_text(
        json.dumps(
            {
                "transform": {
                    "scale": 1.5,
                    "matrix": (np.eye(3) * 1.5).tolist(),
                    "translation": [0.2, -0.1, 0.3],
                }
            }
        ),
        encoding="utf-8",
    )
    measured_csv = tmp_path / "measured.csv"
    calibrated_csv = tmp_path / "calibrated.csv"
    sim_csv = tmp_path / "sim.csv"
    measured.to_csv(measured_csv, index=False)
    simulated.to_csv(calibrated_csv, index=False)
    simulated.to_csv(sim_csv, index=False)

    report = module.validate(
        measured_target_csv=measured_csv,
        calibrated_target_csv=calibrated_csv,
        sim_csv=sim_csv,
        output_dir=tmp_path / "reports",
        run_label="known_scale",
        transform_json=transform_path,
        impact_window_s=0.2,
        write_plots=False,
    )

    before = report["residuals"]["before"]["groups"]["position"]["vector_rmse"]
    after = report["residuals"]["after"]["groups"]["position"]["vector_rmse"]
    assert before > 0.1
    assert after == 0.0
    assert report["transform"]["matrix_shape"] == [3, 3]
    assert report["transform"]["determinant"] > 0.0
    assert report["transform"]["translation"] == [0.2, -0.1, 0.3]
    assert (tmp_path / "reports" / "known_scale_calibration_validation.json").exists()
    assert (tmp_path / "reports" / "known_scale_calibration_validation.md").exists()


def test_calibration_validation_emits_warning_flags(tmp_path: Path) -> None:
    module = _load_module()
    time = np.linspace(0.0, 1.0, 7)
    measured = pd.DataFrame(
        {
            "time": time,
            "clubface_x": time,
            "clubface_y": time,
            "clubface_z": time,
        }
    )
    calibrated = pd.DataFrame(
        {
            "time": time,
            "ClubLogs_CHGlobalPosition_1": time + 5.0,
            "ClubLogs_CHGlobalPosition_2": time,
            "ClubLogs_CHGlobalPosition_3": time,
        }
    )
    simulated = pd.DataFrame(
        {
            "time": time,
            "ClubLogs_CHGlobalPosition_1": time,
            "ClubLogs_CHGlobalPosition_2": time,
            "ClubLogs_CHGlobalPosition_3": time,
        }
    )
    transform_path = tmp_path / "mirror.json"
    transform_path.write_text(
        json.dumps(
            {
                "matrix": [[10.0, 0.0, 0.0], [0.0, -10.0, 0.0], [0.0, 0.0, 10.0]],
                "translation": [0.0, 0.0, 0.0],
            }
        ),
        encoding="utf-8",
    )
    measured_csv = tmp_path / "measured.csv"
    calibrated_csv = tmp_path / "calibrated.csv"
    sim_csv = tmp_path / "sim.csv"
    measured.to_csv(measured_csv, index=False)
    calibrated.to_csv(calibrated_csv, index=False)
    simulated.to_csv(sim_csv, index=False)

    report = module.validate(
        measured_target_csv=measured_csv,
        calibrated_target_csv=calibrated_csv,
        sim_csv=sim_csv,
        output_dir=tmp_path / "reports",
        run_label="warnings",
        transform_json=transform_path,
        impact_window_s=0.0,
        write_plots=False,
        poor_impact_threshold=0.01,
        anisotropy_threshold=2.0,
        extreme_scale_max=5.0,
    )

    warning_codes = {warning["code"] for warning in report["warnings"]}
    assert "mirror_flip" in warning_codes
    assert "extreme_scale" in warning_codes
    assert "poor_impact_window_fit" in warning_codes
    assert "residual_anisotropy" in warning_codes
    assert "too_few_finite_samples" in warning_codes
