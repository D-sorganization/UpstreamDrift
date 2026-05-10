"""Calibrate measured club coordinates into the Simscape club-log frame."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TARGET = SCRIPT_DIR / "data" / "processed" / "TW_ProV1_club_target.csv"
DEFAULT_OUTPUT = (
    SCRIPT_DIR / "data" / "processed" / "TW_ProV1_club_target_calibrated.csv"
)

SOURCE_POSITION = ["clubface_x", "clubface_y", "clubface_z"]
SOURCE_VELOCITY = ["clubface_vx", "clubface_vy", "clubface_vz"]
SOURCE_ACCELERATION = ["clubface_ax", "clubface_ay", "clubface_az"]
MODEL_POSITION = [
    "ClubLogs_CHGlobalPosition_1",
    "ClubLogs_CHGlobalPosition_2",
    "ClubLogs_CHGlobalPosition_3",
]
MODEL_VELOCITY = [
    "ClubLogs_CHGlobalVelocity_1",
    "ClubLogs_CHGlobalVelocity_2",
    "ClubLogs_CHGlobalVelocity_3",
]
MODEL_ACCELERATION = [
    "ClubLogs_CHGlobalAcceleration_1",
    "ClubLogs_CHGlobalAcceleration_2",
    "ClubLogs_CHGlobalAcceleration_3",
]


def _normalize_time(values: np.ndarray) -> np.ndarray:
    if len(values) == 1:
        return np.zeros_like(values, dtype=float)
    start = float(values[0])
    span = float(values[-1] - values[0])
    if abs(span) < 1e-12:
        return np.linspace(0.0, 1.0, len(values))
    return (values - start) / span


def _time_vector(frame: pd.DataFrame) -> np.ndarray:
    if "time" in frame.columns:
        return frame["time"].to_numpy(dtype=float)
    return np.arange(len(frame), dtype=float)


def _interpolate_columns(
    frame: pd.DataFrame,
    columns: list[str],
    query_time: np.ndarray,
) -> np.ndarray:
    source_time = _normalize_time(_time_vector(frame))
    output = np.zeros((len(query_time), len(columns)), dtype=float)
    for idx, column in enumerate(columns):
        output[:, idx] = np.interp(
            query_time, source_time, frame[column].to_numpy(dtype=float)
        )
    return output


def _fit_similarity(
    source: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, float, np.ndarray]:
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = source_centered.T @ target_centered / len(source)
    u, singular_values, vt = np.linalg.svd(covariance)
    sign = np.sign(np.linalg.det(u @ vt))
    correction = np.diag([1.0, 1.0, sign])
    rotation = u @ correction @ vt
    variance = np.mean(np.sum(source_centered**2, axis=1))
    scale = float(np.sum(singular_values * np.diag(correction)) / variance)
    translation = target_mean - scale * (source_mean @ rotation)
    return rotation, scale, translation


def _fit_affine(
    source: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    design = np.column_stack([source, np.ones(len(source))])
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    matrix = coefficients[:3, :]
    offset = coefficients[3, :]
    return matrix, offset


def calibrate(
    target_csv: Path,
    sim_csv: Path,
    output_csv: Path,
    output_json: Path,
    method: str,
) -> None:
    measured = pd.read_csv(target_csv)
    simulated = pd.read_csv(sim_csv)
    missing_measured = [
        column for column in SOURCE_POSITION if column not in measured.columns
    ]
    missing_sim = [
        column for column in MODEL_POSITION if column not in simulated.columns
    ]
    if missing_measured:
        raise ValueError(f"Measured target is missing columns: {missing_measured}")
    if missing_sim:
        raise ValueError(f"Simulated club CSV is missing columns: {missing_sim}")

    query_time = _normalize_time(_time_vector(measured))
    source_position = measured[SOURCE_POSITION].to_numpy(dtype=float)
    target_position = _interpolate_columns(simulated, MODEL_POSITION, query_time)

    if method == "similarity":
        rotation, scale, translation = _fit_similarity(source_position, target_position)
        transform_matrix = scale * rotation
        calibrated_position = source_position @ transform_matrix + translation
        transform_payload = {
            "method": method,
            "scale": scale,
            "rotation": rotation.tolist(),
            "translation": translation.tolist(),
            "matrix": transform_matrix.tolist(),
        }
    else:
        matrix, translation = _fit_affine(source_position, target_position)
        transform_matrix = matrix
        calibrated_position = source_position @ transform_matrix + translation
        transform_payload = {
            "method": method,
            "matrix": transform_matrix.tolist(),
            "translation": translation.tolist(),
        }

    output = pd.DataFrame(
        {
            "time": (
                measured["time"]
                if "time" in measured.columns
                else np.arange(len(measured))
            )
        }
    )
    for idx, column in enumerate(MODEL_POSITION):
        output[column] = calibrated_position[:, idx]

    for source_columns, model_columns in (
        (SOURCE_VELOCITY, MODEL_VELOCITY),
        (SOURCE_ACCELERATION, MODEL_ACCELERATION),
    ):
        if all(column in measured.columns for column in source_columns):
            transformed = (
                measured[source_columns].to_numpy(dtype=float) @ transform_matrix
            )
            for idx, column in enumerate(model_columns):
                output[column] = transformed[:, idx]

    fit_error = calibrated_position - target_position
    rmse = np.sqrt(np.mean(fit_error**2, axis=0))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_csv, index=False)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(
            {
                "target_csv": str(target_csv),
                "sim_csv": str(sim_csv),
                "output_csv": str(output_csv),
                "position_rmse": rmse.tolist(),
                "position_rmse_mean": float(np.mean(rmse)),
                "transform": transform_payload,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-csv", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--sim-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=SCRIPT_DIR / "data" / "processed" / "club_target_calibration.json",
    )
    parser.add_argument(
        "--method", choices=["similarity", "affine"], default="similarity"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calibrate(
        target_csv=args.target_csv,
        sim_csv=args.sim_csv,
        output_csv=args.output_csv,
        output_json=args.output_json,
        method=args.method,
    )


if __name__ == "__main__":
    main()
