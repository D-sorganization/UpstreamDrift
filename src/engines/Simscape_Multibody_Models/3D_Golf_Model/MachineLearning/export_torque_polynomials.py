"""Fit MATLAB 3D golf model torque polynomial inputs from a torque timeseries."""

from __future__ import annotations

import argparse
import importlib.util as _importlib_util
import json
import logging
import sys as _sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import savemat

_SMOOTHING_PATH = Path(__file__).resolve().parent / "torque_smoothing.py"
if "_torque_smoothing_loader" not in globals():
    _spec = _importlib_util.spec_from_file_location(
        "torque_smoothing_module", _SMOOTHING_PATH
    )
    if _spec is None or _spec.loader is None:
        raise ImportError(f"Could not load torque_smoothing from {_SMOOTHING_PATH}")
    _torque_smoothing_loader = _importlib_util.module_from_spec(_spec)
    _sys.modules.setdefault("torque_smoothing_module", _torque_smoothing_loader)
    _spec.loader.exec_module(_torque_smoothing_loader)

VALID_METHODS = _torque_smoothing_loader.VALID_METHODS
SmoothingConfig = _torque_smoothing_loader.SmoothingConfig
polynomial_residual_diagnostic = _torque_smoothing_loader.polynomial_residual_diagnostic
smooth_torque = _torque_smoothing_loader.smooth_torque

from src.shared.python.motion_matching.control_names import (
    COEFFICIENT_LETTERS as _CANONICAL_COEFFICIENT_LETTERS,
)
from src.shared.python.motion_matching.control_names import (
    TORQUE_TO_POLYNOMIAL_BASE,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_RESIDUAL_THRESHOLD = 0.5  # N*m absolute residual warning threshold
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SCRIPT_DIR / "data" / "processed" / "ml_torque_polynomial_inputs.mat"

# Preserve the legacy string form ("ABCDEFG") expected by ``zip`` consumers
# below; the canonical tuple lives in ``motion_matching.control_names``.
COEFFICIENT_LETTERS = "".join(_CANONICAL_COEFFICIENT_LETTERS)


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
    smoothing: SmoothingConfig | None = None,
    residual_threshold: float = DEFAULT_RESIDUAL_THRESHOLD,
    smoothed_csv: Path | None = None,
) -> dict[str, object]:
    """Fit MATLAB sixth-order polynomial inputs from a torque CSV.

    Parameters
    ----------
    torque_csv:
        Source CSV containing piecewise-constant torque rows.
    output_mat:
        Destination MAT file.
    output_json:
        Optional JSON summary path. Defaults to ``output_mat`` with a
        ``.summary.json`` suffix.
    time_column:
        Name of the time column in ``torque_csv``.
    smoothing:
        Optional smoothing configuration applied to torque columns prior to
        polynomial fitting. ``None`` disables smoothing (raw rows fit).
    residual_threshold:
        Absolute residual (N*m) above which a polynomial fit is flagged in
        the diagnostics summary.
    smoothed_csv:
        Optional path to write the smoothed-torque CSV alongside the MAT.
    """
    frame = pd.read_csv(torque_csv)
    if time_column not in frame.columns:
        raise ValueError(f"Torque CSV is missing required time column: {time_column}")

    time = frame[time_column].to_numpy(dtype=float)
    mapped_columns = _mapped_torque_columns(list(frame.columns))
    if not mapped_columns:
        raise ValueError("No torque columns matched known MATLAB polynomial inputs")

    mat_payload: dict[str, np.ndarray] = {}
    fit_summaries: dict[str, dict[str, object]] = {}
    smoothing_method = smoothing.method if smoothing is not None else "none"
    summary: dict[str, object] = {
        "source": str(torque_csv),
        "output_mat": str(output_mat),
        "time_column": time_column,
        "time_min": float(np.nanmin(time)),
        "time_max": float(np.nanmax(time)),
        "smoothing": {
            "method": smoothing_method,
            "config": (
                {
                    "window": smoothing.window,
                    "polyorder": smoothing.polyorder,
                    "cutoff_hz": smoothing.cutoff_hz,
                    "butter_order": smoothing.butter_order,
                    "spline_s": smoothing.spline_s,
                }
                if smoothing is not None
                else None
            ),
            "residual_threshold": float(residual_threshold),
        },
        "fits": fit_summaries,
        "flagged_columns": [],
    }
    flagged: list[str] = summary["flagged_columns"]  # type: ignore[assignment]
    smoothed_frame = pd.DataFrame({time_column: time}) if smoothed_csv else None

    for torque_column, matlab_base in mapped_columns.items():
        raw = frame[torque_column].to_numpy(dtype=float)
        if smoothing is not None:
            grid = smooth_torque(time, raw, smoothing)
        else:
            grid = raw
        if smoothed_frame is not None:
            smoothed_frame[torque_column] = grid

        coeffs = _fit_hex_polynomial(time, grid)
        for letter, value in zip(COEFFICIENT_LETTERS, coeffs, strict=True):
            mat_payload[f"{matlab_base}{letter}"] = np.asarray([[value]], dtype=float)

        diagnostic = polynomial_residual_diagnostic(
            time, grid, coeffs, residual_threshold
        )
        if diagnostic["exceeds_threshold"]:
            flagged.append(torque_column)
        fit_summaries[torque_column] = {
            "matlab_base": matlab_base,
            "coefficients_A_to_G": coeffs.tolist(),
            "rmse": diagnostic["rmse"],
            "max_abs_residual": diagnostic["max_abs_residual"],
            "exceeds_residual_threshold": diagnostic["exceeds_threshold"],
            "raw_peak": float(np.max(np.abs(raw))),
            "smoothed_peak": float(np.max(np.abs(grid))),
        }

    output_mat.parent.mkdir(parents=True, exist_ok=True)
    savemat(output_mat, mat_payload, do_compression=True)

    if smoothed_frame is not None and smoothed_csv is not None:
        smoothed_csv.parent.mkdir(parents=True, exist_ok=True)
        smoothed_frame.to_csv(smoothed_csv, index=False)
        summary["smoothed_csv"] = str(smoothed_csv)

    if output_json is None:
        output_json = output_mat.with_suffix(".summary.json")
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOGGER.info("%s", json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torque-csv", type=Path, required=True)
    parser.add_argument("--output-mat", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--time-column", default="time")
    parser.add_argument(
        "--smoothing-method",
        choices=("none", *VALID_METHODS),
        default="none",
        help="Smoothing applied prior to polynomial fitting.",
    )
    parser.add_argument("--smoothing-window", type=int, default=5)
    parser.add_argument("--smoothing-polyorder", type=int, default=3)
    parser.add_argument("--smoothing-cutoff-hz", type=float, default=25.0)
    parser.add_argument("--smoothing-butter-order", type=int, default=4)
    parser.add_argument("--smoothing-spline-s", type=float, default=None)
    parser.add_argument(
        "--residual-threshold",
        type=float,
        default=DEFAULT_RESIDUAL_THRESHOLD,
        help="Max absolute polynomial residual before a column is flagged.",
    )
    parser.add_argument("--smoothed-csv", type=Path)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    smoothing: SmoothingConfig | None
    if args.smoothing_method == "none":
        smoothing = None
    else:
        smoothing = SmoothingConfig(
            method=args.smoothing_method,
            window=args.smoothing_window,
            polyorder=args.smoothing_polyorder,
            cutoff_hz=args.smoothing_cutoff_hz,
            butter_order=args.smoothing_butter_order,
            spline_s=args.smoothing_spline_s,
        )
    export_polynomial_inputs(
        torque_csv=args.torque_csv,
        output_mat=args.output_mat,
        output_json=args.output_json,
        time_column=args.time_column,
        smoothing=smoothing,
        residual_threshold=args.residual_threshold,
        smoothed_csv=args.smoothed_csv,
    )


if __name__ == "__main__":
    main()
