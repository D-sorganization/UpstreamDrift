"""Slice a prepared measured club trajectory into full-swing or downswing targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "data" / "processed" / "TW_ProV1_club_target.csv"


def _speed(frame: pd.DataFrame) -> np.ndarray:
    velocity_columns = ["clubface_vx", "clubface_vy", "clubface_vz"]
    if all(column in frame.columns for column in velocity_columns):
        values = frame[velocity_columns].to_numpy(dtype=float)
        return np.linalg.norm(values, axis=1)
    position_columns = ["clubface_x", "clubface_y", "clubface_z"]
    if "time" not in frame.columns or not all(
        column in frame.columns for column in position_columns
    ):
        raise ValueError(
            "Cannot infer downswing start without clubface velocity "
            "or position/time columns"
        )
    position = frame[position_columns].to_numpy(dtype=float)
    time = frame["time"].to_numpy(dtype=float)
    velocity = np.gradient(position, time, axis=0)
    return np.linalg.norm(velocity, axis=1)


def infer_top_of_backswing_index(frame: pd.DataFrame) -> int:
    """Infer top of backswing as the last low-speed sample before peak speed."""

    speeds = _speed(frame)
    peak_index = int(np.nanargmax(speeds))
    if peak_index <= 1:
        return 0
    pre_peak = speeds[:peak_index]
    threshold = np.nanpercentile(pre_peak, 15)
    candidates = np.flatnonzero(pre_peak <= threshold)
    if len(candidates) == 0:
        return int(np.nanargmin(pre_peak))
    return int(candidates[-1])


def slice_target(
    input_csv: Path,
    output_csv: Path,
    scenario: str,
    start_sample: int | None,
    start_time: float | None,
    reset_time: bool,
) -> None:
    frame = pd.read_csv(input_csv)
    if frame.empty:
        raise ValueError(f"Target CSV is empty: {input_csv}")

    start_index = 0
    start_reason = "first row"
    if scenario == "downswing":
        if start_sample is not None:
            if "sample" in frame.columns:
                sample_matches = frame.index[frame["sample"] == start_sample]
            else:
                sample_matches = pd.Index([])
            if not sample_matches.empty:
                start_index = int(sample_matches[0])
                start_reason = f"sample {start_sample}"
            else:
                start_index = max(0, min(int(start_sample), len(frame) - 1))
                start_reason = f"row index {start_sample}"
        elif start_time is not None:
            if "time" not in frame.columns:
                raise ValueError("--start-time requires a time column")
            start_index = int(
                np.argmin(np.abs(frame["time"].to_numpy(dtype=float) - start_time))
            )
            start_reason = f"nearest time {start_time}"
        else:
            start_index = infer_top_of_backswing_index(frame)
            start_reason = "inferred last low-speed sample before peak speed"

    sliced = frame.iloc[start_index:].copy().reset_index(drop=True)
    if reset_time and "time" in sliced.columns:
        sliced["time"] = sliced["time"] - float(sliced["time"].iloc[0])

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    sliced.to_csv(output_csv, index=False)
    output_csv.with_suffix(".summary.json").write_text(
        json.dumps(
            {
                "input_csv": str(input_csv),
                "output_csv": str(output_csv),
                "scenario": scenario,
                "start_index": start_index,
                "start_reason": start_reason,
                "rows": int(len(sliced)),
                "reset_time": reset_time,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument(
        "--scenario", choices=["full-swing", "downswing"], default="full-swing"
    )
    parser.add_argument("--start-sample", type=int)
    parser.add_argument("--start-time", type=float)
    parser.add_argument("--reset-time", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    slice_target(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        scenario=args.scenario,
        start_sample=args.start_sample,
        start_time=args.start_time,
        reset_time=args.reset_time,
    )


if __name__ == "__main__":
    main()
