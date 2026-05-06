"""Fit MATLAB 3D golf model torque polynomial inputs from a torque timeseries."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import savemat

LOGGER = logging.getLogger(__name__)
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SCRIPT_DIR / "data" / "processed" / "ml_torque_polynomial_inputs.mat"

COEFFICIENT_LETTERS = "ABCDEFG"

TORQUE_TO_POLYNOMIAL_BASE = {
    "LScapLogs_ActuatorTorqueX": "LScapInputX",
    "LScapLogs_ActuatorTorqueY": "LScapInputY",
    "RScapLogs_ActuatorTorqueX": "RScapInputX",
    "RScapLogs_ActuatorTorqueY": "RScapInputY",
    "LSLogs_ActuatorTorqueX": "LSInputX",
    "LSLogs_ActuatorTorqueY": "LSInputY",
    "LSLogs_ActuatorTorqueZ": "LSInputZ",
    "RSLogs_ActuatorTorqueX": "RSInputX",
    "RSLogs_ActuatorTorqueY": "RSInputY",
    "RSLogs_ActuatorTorqueZ": "RSInputZ",
    "SpineLogs_ActuatorTorqueX": "SpineInputX",
    "SpineLogs_ActuatorTorqueY": "SpineInputY",
    "HipLogs_TranslationForceXInput": "TranslationInputX",
    "HipLogs_TranslationForceYInput": "TranslationInputY",
    "HipLogs_TranslationForceZInput": "TranslationInputZ",
    "HipLogs_HipTorqueXInput": "HipInputX",
    "HipLogs_HipTorqueYInput": "HipInputY",
    "HipLogs_HipTorqueZInput": "HipInputZ",
    "LScapTorqueXInput": "LScapInputX",
    "LScapTorqueYInput": "LScapInputY",
    "RScapTorqueXInput": "RScapInputX",
    "RScapTorqueYInput": "RScapInputY",
    "LSTorqueXInput": "LSInputX",
    "LSTorqueYInput": "LSInputY",
    "LSTorqueZInput": "LSInputZ",
    "RSTorqueXInput": "RSInputX",
    "RSTorqueYInput": "RSInputY",
    "RSTorqueZInput": "RSInputZ",
    "HipTorqueXInput": "HipInputX",
    "HipTorqueYInput": "HipInputY",
    "HipTorqueZInput": "HipInputZ",
}


def _fit_hex_polynomial(time: np.ndarray, values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(time) & np.isfinite(values)
    if finite.sum() < 2:
        raise ValueError("Need at least two finite samples to fit a polynomial")
    degree = min(6, int(finite.sum()) - 1)
    coeffs = np.polyfit(time[finite], values[finite], degree)
    if degree < 6:
        coeffs = np.pad(coeffs, (6 - degree, 0), mode="constant")
    return coeffs.astype(float)


def _mapped_torque_columns(columns: list[str]) -> dict[str, str]:
    mapped = {}
    for column in columns:
        if column in TORQUE_TO_POLYNOMIAL_BASE:
            mapped[column] = TORQUE_TO_POLYNOMIAL_BASE[column]
        elif column.endswith("Input") and column[:-1] in TORQUE_TO_POLYNOMIAL_BASE:
            mapped[column] = TORQUE_TO_POLYNOMIAL_BASE[column[:-1]]
    return mapped


def export_polynomial_inputs(
    torque_csv: Path,
    output_mat: Path,
    output_json: Path | None,
    time_column: str,
) -> None:
    frame = pd.read_csv(torque_csv)
    if time_column not in frame.columns:
        raise ValueError(f"Torque CSV is missing required time column: {time_column}")

    time = frame[time_column].to_numpy(dtype=float)
    mapped_columns = _mapped_torque_columns(list(frame.columns))
    if not mapped_columns:
        raise ValueError("No torque columns matched known MATLAB polynomial inputs")

    mat_payload: dict[str, np.ndarray] = {}
    fit_summaries: dict[str, dict[str, object]] = {}
    summary: dict[str, object] = {
        "source": str(torque_csv),
        "output_mat": str(output_mat),
        "time_column": time_column,
        "time_min": float(np.nanmin(time)),
        "time_max": float(np.nanmax(time)),
        "fits": fit_summaries,
    }

    for torque_column, matlab_base in mapped_columns.items():
        values = frame[torque_column].to_numpy(dtype=float)
        coeffs = _fit_hex_polynomial(time, values)
        for letter, value in zip(COEFFICIENT_LETTERS, coeffs, strict=True):
            mat_payload[f"{matlab_base}{letter}"] = np.asarray([[value]], dtype=float)
        fit_summaries[torque_column] = {
            "matlab_base": matlab_base,
            "coefficients_A_to_G": coeffs.tolist(),
            "rmse": float(np.sqrt(np.mean((np.polyval(coeffs, time) - values) ** 2))),
        }

    output_mat.parent.mkdir(parents=True, exist_ok=True)
    savemat(output_mat, mat_payload, do_compression=True)

    if output_json is None:
        output_json = output_mat.with_suffix(".summary.json")
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOGGER.info("%s", json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torque-csv", type=Path, required=True)
    parser.add_argument("--output-mat", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--time-column", default="time")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    export_polynomial_inputs(
        torque_csv=args.torque_csv,
        output_mat=args.output_mat,
        output_json=args.output_json,
        time_column=args.time_column,
    )


if __name__ == "__main__":
    main()
