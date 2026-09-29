"""Prepare measured club trajectory workbooks for surrogate-control targets."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_WORKBOOK = (
    SCRIPT_DIR.parent
    / "matlab"
    / "src"
    / "apps"
    / "golf_gui"
    / "Motion Capture Plotter"
    / "Wiffle_ProV1_club_3D_data.xlsx"
)
DEFAULT_OUTPUT = SCRIPT_DIR / "data" / "processed" / "TW_ProV1_club_target.csv"
LOGGER = logging.getLogger(__name__)


def _derivative(values: np.ndarray, time: np.ndarray) -> np.ndarray:
    if len(values) > 1 and np.all(np.isfinite(time)) and len(np.unique(time)) > 1:
        return np.gradient(values, time, axis=0, edge_order=1)
    return np.full_like(values, np.nan, dtype=np.float64)


def prepare_target(workbook: Path, sheet: str, output: Path) -> None:
    raw = pd.read_excel(workbook, sheet_name=sheet, header=None)
    header_row = raw.index[raw.iloc[:, 0].astype(str).str.strip().eq("Sample #")]
    if len(header_row) != 1:
        raise ValueError(
            f"Could not identify a unique 'Sample #' header row in {sheet}"
        )

    start = int(header_row[0]) + 1
    data = raw.iloc[start:, :26].copy()
    data.columns = [
        "sample",
        "time",
        "midhand_x",
        "midhand_y",
        "midhand_z",
        "midhand_xx",
        "midhand_xy",
        "midhand_xz",
        "midhand_yx",
        "midhand_yy",
        "midhand_yz",
        "midhand_zx",
        "midhand_zy",
        "midhand_zz",
        "clubface_x",
        "clubface_y",
        "clubface_z",
        "clubface_xx",
        "clubface_xy",
        "clubface_xz",
        "clubface_yx",
        "clubface_yy",
        "clubface_yz",
        "clubface_zx",
        "clubface_zy",
        "clubface_zz",
    ]
    numeric = data.apply(pd.to_numeric, errors="coerce").dropna(
        subset=["sample", "time"]
    )
    numeric = numeric.sort_values("time").reset_index(drop=True)

    time = numeric["time"].to_numpy(dtype=np.float64)
    position = numeric[["clubface_x", "clubface_y", "clubface_z"]].to_numpy(
        dtype=np.float64
    )
    velocity = _derivative(position, time)
    acceleration = _derivative(velocity, time)

    output_frame = pd.DataFrame(
        {
            "sample": numeric["sample"].to_numpy(dtype=np.int64),
            "time": time,
            "clubface_x": position[:, 0],
            "clubface_y": position[:, 1],
            "clubface_z": position[:, 2],
            "clubface_vx": velocity[:, 0],
            "clubface_vy": velocity[:, 1],
            "clubface_vz": velocity[:, 2],
            "clubface_ax": acceleration[:, 0],
            "clubface_ay": acceleration[:, 1],
            "clubface_az": acceleration[:, 2],
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output_frame.to_csv(output, index=False)

    summary = {
        "workbook": str(workbook),
        "sheet": sheet,
        "output": str(output),
        "rows": int(len(output_frame)),
        "time_min": float(output_frame["time"].min()),
        "time_max": float(output_frame["time"].max()),
        "columns": list(output_frame.columns),
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    LOGGER.info("%s", json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    parser.add_argument("--sheet", default="TW_ProV1")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    prepare_target(workbook=args.workbook, sheet=args.sheet, output=args.output)


if __name__ == "__main__":
    main()
