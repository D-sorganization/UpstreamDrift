"""Prepare a manifest for sequential frame-by-frame torque search.

The MATLAB runner uses this JSON manifest to keep the long-running Simscape
search reproducible. This helper validates the target trajectory and column
manifest up front so overnight jobs fail before MATLAB starts evaluating
candidate torques.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_COLUMN_MANIFEST = SCRIPT_DIR / "column_manifest.json"
DEFAULT_OUTPUT = SCRIPT_DIR / "data" / "processed" / "frame_by_frame_search.json"
DEFAULT_TORQUE_CSV = (
    SCRIPT_DIR / "data" / "processed" / "frame_by_frame_torque_sequence.csv"
)
DEFAULT_POLYNOMIAL_MAT = (
    SCRIPT_DIR / "data" / "processed" / "frame_by_frame_torque_polynomials.mat"
)

CLUB_TARGET_ALIASES = {
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


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _flatten_columns(section: dict[str, Any]) -> list[str]:
    columns: list[str] = []
    for value in section.values():
        if isinstance(value, list):
            columns.extend(str(item) for item in value)
    return columns


def _manifest_controls(manifest: dict[str, Any]) -> list[str]:
    inputs = manifest.get("input_columns")
    if not isinstance(inputs, dict):
        raise ValueError("Column manifest is missing input_columns")
    controls = inputs.get("applied_controls")
    if not isinstance(controls, list) or not controls:
        raise ValueError("Column manifest is missing input_columns.applied_controls")
    return [str(column) for column in controls]


def _target_columns(frame: pd.DataFrame, requested: list[str] | None) -> list[str]:
    if requested:
        missing = [column for column in requested if column not in frame.columns]
        if missing:
            raise ValueError(f"Desired target CSV is missing columns: {missing}")
        return requested

    canonical_columns = set(CLUB_TARGET_ALIASES.values())
    targets = [column for column in frame.columns if column in canonical_columns]
    for alias, canonical in CLUB_TARGET_ALIASES.items():
        if alias in frame.columns and canonical not in targets:
            targets.append(alias)
    if not targets:
        raise ValueError(
            "Desired target CSV has no recognizable club position, velocity, "
            "or acceleration target columns"
        )
    return targets


def _validate_time(frame: pd.DataFrame) -> dict[str, float]:
    if "time" not in frame.columns:
        raise ValueError("Desired target CSV must include a time column")
    if frame.empty:
        raise ValueError("Desired target CSV is empty")

    time = frame["time"].to_numpy(dtype=float)
    if not np.all(np.isfinite(time)):
        raise ValueError("Desired target time column contains non-finite values")
    if len(time) > 1 and not np.all(np.diff(time) > 0.0):
        raise ValueError("Desired target time column must be strictly increasing")

    if len(time) > 1:
        diffs = np.diff(time)
        median_step = float(np.median(diffs))
        min_step = float(np.min(diffs))
        max_step = float(np.max(diffs))
    else:
        median_step = min_step = max_step = 0.0

    return {
        "time_start": float(time[0]),
        "time_end": float(time[-1]),
        "median_step_seconds": median_step,
        "min_step_seconds": min_step,
        "max_step_seconds": max_step,
    }


def _candidate_count(
    control_count: int, candidate_levels: list[float], strategy: str
) -> int:
    if strategy == "cartesian":
        return int(len(candidate_levels) ** control_count)
    if strategy != "coordinate":
        raise ValueError("candidate_strategy must be 'coordinate' or 'cartesian'")
    non_zero = sum(1 for level in candidate_levels if abs(level) > 0.0)
    has_zero = any(abs(level) == 0.0 for level in candidate_levels)
    return int((1 if has_zero else 0) + control_count * non_zero)


def build_search_manifest(  # noqa: C901
    desired_target_csv: Path,
    column_manifest: Path = DEFAULT_COLUMN_MANIFEST,
    output_json: Path = DEFAULT_OUTPUT,
    model_name: str = "GolfSwing3D_Kinetic",
    starting_state_file: Path | None = None,
    torque_output_csv: Path = DEFAULT_TORQUE_CSV,
    polynomial_output_mat: Path = DEFAULT_POLYNOMIAL_MAT,
    horizon_frames: int = 1,
    candidate_step: float = 5.0,
    candidate_levels: list[float] | None = None,
    candidate_strategy: str = "coordinate",
    max_candidates_per_frame: int = 2048,
    requested_target_columns: list[str] | None = None,
    requested_control_columns: list[str] | None = None,
    use_parallel: str = "auto",
    position_weight: float = 1.0,
    velocity_weight: float = 0.25,
    acceleration_weight: float = 0.25,
    effort_weight: float = 1.0e-6,
    smoothness_weight: float = 1.0e-4,
    smoothing_window_frames: int = 7,
    polynomial_degree: int = 6,
    run_dir: Path | None = None,
    checkpoint_interval_frames: int = 10,
) -> dict[str, Any]:
    if horizon_frames < 1:
        raise ValueError("horizon_frames must be >= 1")
    if candidate_step <= 0.0:
        raise ValueError("candidate_step must be positive")
    if max_candidates_per_frame < 1:
        raise ValueError("max_candidates_per_frame must be >= 1")
    if smoothing_window_frames < 1:
        raise ValueError("smoothing_window_frames must be >= 1")
    if polynomial_degree < 1:
        raise ValueError("polynomial_degree must be >= 1")
    if use_parallel not in {"auto", "always", "never"}:
        raise ValueError("use_parallel must be one of: auto, always, never")
    if checkpoint_interval_frames < 1:
        raise ValueError("checkpoint_interval_frames must be >= 1")

    levels = candidate_levels if candidate_levels is not None else [-1.0, 0.0, 1.0]
    if not levels:
        raise ValueError("candidate_levels must contain at least one level")
    if not all(np.isfinite(level) for level in levels):
        raise ValueError("candidate_levels must be finite")

    source_manifest = _load_json(column_manifest)
    manifest_controls = _manifest_controls(source_manifest)
    controls = requested_control_columns or manifest_controls
    missing_controls = [
        column for column in controls if column not in manifest_controls
    ]
    if missing_controls:
        raise ValueError(
            f"Requested controls are not in the manifest: {missing_controls}"
        )

    desired = pd.read_csv(desired_target_csv)
    time_summary = _validate_time(desired)
    targets = _target_columns(desired, requested_target_columns)
    candidate_total = _candidate_count(len(controls), levels, candidate_strategy)
    if candidate_total > max_candidates_per_frame:
        raise ValueError(
            "Candidate plan expands to "
            f"{candidate_total} candidates per frame, above max "
            f"{max_candidates_per_frame}. Reduce controls/levels or raise the cap."
        )

    if run_dir is None:
        run_dir = output_json.parent / f"{output_json.stem}_run"
    run_dir = Path(run_dir)
    outputs = {
        "torque_csv": str(torque_output_csv),
        "polynomial_mat": str(polynomial_output_mat),
        "polynomial_summary_json": str(
            polynomial_output_mat.with_suffix(".summary.json")
        ),
        "run_dir": str(run_dir),
        "progress_csv": str(run_dir / "progress.csv"),
        "checkpoint_mat": str(run_dir / "checkpoint.mat"),
        "summary_json": str(run_dir / "summary.json"),
        "manifest_copy_json": str(run_dir / "manifest.json"),
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "workflow": "frame_by_frame_torque_search",
        "description": (
            "Sequential short-horizon torque candidate search for the 3D Golf "
            "Simscape model. MATLAB owns simulation stepping through explicit hooks."
        ),
        "inputs": {
            "desired_target_csv": str(desired_target_csv),
            "column_manifest": str(column_manifest),
            "starting_state_file": (
                str(starting_state_file) if starting_state_file is not None else ""
            ),
        },
        "simulation": {
            "model_name": model_name,
            "step_hook": "evaluateFrameByFrameTorqueCandidate",
            "state_hook": "extractFrameByFrameState",
        },
        "columns": {
            "time": "time",
            "target_columns": targets,
            "control_columns": controls,
            "available_input_columns": _flatten_columns(
                source_manifest["input_columns"]
            ),
        },
        "search": {
            "horizon_frames": int(horizon_frames),
            "candidate_step": float(candidate_step),
            "candidate_levels": [float(level) for level in levels],
            "candidate_strategy": candidate_strategy,
            "max_candidates_per_frame": int(max_candidates_per_frame),
            "use_parallel": use_parallel,
            "weights": {
                "position": float(position_weight),
                "velocity": float(velocity_weight),
                "acceleration": float(acceleration_weight),
                "effort": float(effort_weight),
                "smoothness": float(smoothness_weight),
            },
        },
        "postprocess": {
            "smoothing_window_frames": int(smoothing_window_frames),
            "polynomial_degree": int(polynomial_degree),
        },
        "checkpoint": {
            "interval_frames": int(checkpoint_interval_frames),
            "stale_lock_multiplier": 2.0,
        },
        "outputs": outputs,
        "validation": {
            "target_rows": int(len(desired)),
            "target_columns_found": int(len(targets)),
            "control_columns_found": int(len(controls)),
            "candidates_per_frame": candidate_total,
            "estimated_candidate_evaluations": int(candidate_total * len(desired)),
            **time_summary,
        },
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, indent=2).encode("utf-8")
    output_json.write_bytes(payload)
    run_dir.mkdir(parents=True, exist_ok=True)
    Path(outputs["manifest_copy_json"]).write_bytes(payload)
    LOGGER.info("Wrote frame-by-frame search manifest to %s", output_json)
    # The manifest dict returned mirrors the on-disk JSON exactly; callers
    # that need the SHA-256 should call frame_search_artifacts.manifest_sha256
    # on the written file (the MATLAB runner does the same on its side).
    return manifest


def _csv_list(value: str | None) -> list[str] | None:
    if value is None or value == "":
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--desired-target-csv", type=Path, required=True)
    parser.add_argument("--column-manifest", type=Path, default=DEFAULT_COLUMN_MANIFEST)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-name", default="GolfSwing3D_Kinetic")
    parser.add_argument("--starting-state-file", type=Path)
    parser.add_argument("--torque-output-csv", type=Path, default=DEFAULT_TORQUE_CSV)
    parser.add_argument(
        "--polynomial-output-mat", type=Path, default=DEFAULT_POLYNOMIAL_MAT
    )
    parser.add_argument("--horizon-frames", type=int, default=1)
    parser.add_argument("--candidate-step", type=float, default=5.0)
    parser.add_argument("--candidate-levels", default="-1,0,1")
    parser.add_argument(
        "--candidate-strategy",
        choices=["coordinate", "cartesian"],
        default="coordinate",
    )
    parser.add_argument("--max-candidates-per-frame", type=int, default=2048)
    parser.add_argument("--target-columns")
    parser.add_argument("--control-columns")
    parser.add_argument(
        "--use-parallel", choices=["auto", "always", "never"], default="auto"
    )
    parser.add_argument("--position-weight", type=float, default=1.0)
    parser.add_argument("--velocity-weight", type=float, default=0.25)
    parser.add_argument("--acceleration-weight", type=float, default=0.25)
    parser.add_argument("--effort-weight", type=float, default=1.0e-6)
    parser.add_argument("--smoothness-weight", type=float, default=1.0e-4)
    parser.add_argument("--smoothing-window-frames", type=int, default=7)
    parser.add_argument("--polynomial-degree", type=int, default=6)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-interval-frames", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    levels = [float(item) for item in _csv_list(args.candidate_levels) or []]
    build_search_manifest(
        desired_target_csv=args.desired_target_csv,
        column_manifest=args.column_manifest,
        output_json=args.output_json,
        model_name=args.model_name,
        starting_state_file=args.starting_state_file,
        torque_output_csv=args.torque_output_csv,
        polynomial_output_mat=args.polynomial_output_mat,
        horizon_frames=args.horizon_frames,
        candidate_step=args.candidate_step,
        candidate_levels=levels,
        candidate_strategy=args.candidate_strategy,
        max_candidates_per_frame=args.max_candidates_per_frame,
        requested_target_columns=_csv_list(args.target_columns),
        requested_control_columns=_csv_list(args.control_columns),
        use_parallel=args.use_parallel,
        position_weight=args.position_weight,
        velocity_weight=args.velocity_weight,
        acceleration_weight=args.acceleration_weight,
        effort_weight=args.effort_weight,
        smoothness_weight=args.smoothness_weight,
        smoothing_window_frames=args.smoothing_window_frames,
        polynomial_degree=args.polynomial_degree,
        run_dir=args.run_dir,
        checkpoint_interval_frames=args.checkpoint_interval_frames,
    )


if __name__ == "__main__":
    main()
