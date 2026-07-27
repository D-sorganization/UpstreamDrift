"""Compare a desired club trajectory against a MATLAB/Simscape club-log CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_GROUPS = {
    "position": [
        "ClubLogs_CHGlobalPosition_1",
        "ClubLogs_CHGlobalPosition_2",
        "ClubLogs_CHGlobalPosition_3",
    ],
    "velocity": [
        "ClubLogs_CHGlobalVelocity_1",
        "ClubLogs_CHGlobalVelocity_2",
        "ClubLogs_CHGlobalVelocity_3",
    ],
    "acceleration": [
        "ClubLogs_CHGlobalAcceleration_1",
        "ClubLogs_CHGlobalAcceleration_2",
        "ClubLogs_CHGlobalAcceleration_3",
    ],
}

SOURCE_TO_MODEL = {
    "clubface_x": "ClubLogs_CHGlobalPosition_1",
    "clubface_y": "ClubLogs_CHGlobalPosition_2",
    "clubface_z": "ClubLogs_CHGlobalPosition_3",
    "clubface_vx": "ClubLogs_CHGlobalVelocity_1",
    "clubface_vy": "ClubLogs_CHGlobalVelocity_2",
    "clubface_vz": "ClubLogs_CHGlobalVelocity_3",
    "clubface_ax": "ClubLogs_CHGlobalAcceleration_1",
    "clubface_ay": "ClubLogs_CHGlobalAcceleration_2",
    "clubface_az": "ClubLogs_CHGlobalAcceleration_3",
}


def _normalize_time(frame: pd.DataFrame) -> np.ndarray:
    time = (
        frame["time"].to_numpy(dtype=float)
        if "time" in frame.columns
        else np.arange(len(frame), dtype=float)
    )
    if len(time) == 1:
        return np.zeros_like(time, dtype=float)
    span = float(time[-1] - time[0])
    if abs(span) < 1e-12:
        return np.linspace(0.0, 1.0, len(time))
    return (time - float(time[0])) / span


def _canonical_target(frame: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame(
        {"time": frame["time"] if "time" in frame.columns else np.arange(len(frame))}
    )
    for model_column in {
        column for columns in MODEL_GROUPS.values() for column in columns
    }:
        if model_column in frame.columns:
            output[model_column] = frame[model_column]
    for source_column, model_column in SOURCE_TO_MODEL.items():
        if source_column in frame.columns and model_column not in output.columns:
            output[model_column] = frame[source_column]
    return output


def _interpolate(
    frame: pd.DataFrame, columns: list[str], query_time: np.ndarray
) -> np.ndarray:
    source_time = _normalize_time(frame)
    values = np.zeros((len(query_time), len(columns)), dtype=float)
    for idx, column in enumerate(columns):
        values[:, idx] = np.interp(
            query_time, source_time, frame[column].to_numpy(dtype=float)
        )
    return values


def _write_plot(
    target: pd.DataFrame, simulated: pd.DataFrame, output_png: Path
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    query_time = _normalize_time(target)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for axis_index, (group, columns) in enumerate(MODEL_GROUPS.items()):
        available = [
            column
            for column in columns
            if column in target.columns and column in simulated.columns
        ]
        if not available:
            continue
        sim_values = _interpolate(simulated, available, query_time)
        for idx, column in enumerate(available):
            axes[axis_index].plot(
                query_time,
                target[column].to_numpy(dtype=float),
                label=f"target {column[-1]}",
            )
            axes[axis_index].plot(
                query_time,
                sim_values[:, idx],
                linestyle="--",
                label=f"sim {column[-1]}",
            )
        axes[axis_index].set_ylabel(group)
        axes[axis_index].legend(loc="best", fontsize="small")
    axes[-1].set_xlabel("normalized time")
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=150)
    plt.close(fig)


def compare(
    target_csv: Path, sim_csv: Path, output_json: Path, output_png: Path | None
) -> None:
    target = _canonical_target(pd.read_csv(target_csv))
    simulated = pd.read_csv(sim_csv)
    query_time = _normalize_time(target)
    group_metrics: dict[str, dict[str, object]] = {}
    metrics: dict[str, object] = {
        "target_csv": str(target_csv),
        "sim_csv": str(sim_csv),
        "rows_target": int(len(target)),
        "rows_sim": int(len(simulated)),
        "groups": group_metrics,
    }

    for group, columns in MODEL_GROUPS.items():
        available = [
            column
            for column in columns
            if column in target.columns and column in simulated.columns
        ]
        if not available:
            continue
        target_values = target[available].to_numpy(dtype=float)
        sim_values = _interpolate(simulated, available, query_time)
        error = sim_values - target_values
        # ⚡ Bolt: np.einsum is ~2x faster than np.mean(..., axis=0)
        rmse = np.sqrt(np.einsum("ij,ij->j", error, error) / error.shape[0])
        group_metrics[group] = {
            "columns": available,
            "rmse": rmse.tolist(),
            "rmse_mean": float(np.mean(rmse)),
            "max_abs_error": np.max(np.abs(error), axis=0).tolist(),
        }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if output_png is not None:
        _write_plot(target, simulated, output_png)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-csv", type=Path, required=True)
    parser.add_argument("--sim-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-png", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compare(
        target_csv=args.target_csv,
        sim_csv=args.sim_csv,
        output_json=args.output_json,
        output_png=args.output_png,
    )


if __name__ == "__main__":
    main()
