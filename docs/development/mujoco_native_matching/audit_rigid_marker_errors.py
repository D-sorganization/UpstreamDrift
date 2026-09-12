"""Counterfactual marker decomposition; never modifies or accepts a trajectory."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from src.shared.python.body_part_viz.fitters._kabsch import kabsch_rotation
from src.shared.python.motion_matching.marker_replay_report import observed_rms


def decompose(
    prediction: np.ndarray, target: np.ndarray, valid: np.ndarray
) -> dict[str, Any]:
    """Fit translation and then proper rigid rotation on observed equal-weight points."""
    prediction, target, valid = (
        np.asarray(prediction),
        np.asarray(target),
        np.asarray(valid),
    )
    if (
        prediction.ndim != 2
        or prediction.shape[1] != 3
        or prediction.shape != target.shape
        or valid.shape != prediction.shape[:1]
        or valid.dtype != np.bool_
        or valid.sum() < 3
        or not np.isfinite(prediction).all()
        or not np.isfinite(target[valid]).all()
    ):
        raise ValueError(
            "Finite corresponding points and at least three observed markers required"
        )
    p, q = prediction[valid], target[valid]
    p_mean, q_mean = p.mean(axis=0), q.mean(axis=0)
    p_centered, q_centered = p - p_mean, q - q_mean
    if min(np.linalg.matrix_rank(p_centered), np.linalg.matrix_rank(q_centered)) < 2:
        raise ValueError("Noncollinear observations required for a rigid rotation")
    shift = q_mean - p_mean
    rotation = kabsch_rotation(p_centered, q_centered)
    translation = q_mean - rotation @ p_mean
    residuals = [p - q, p + shift - q, p @ rotation.T + translation - q]
    errors = [np.linalg.norm(value, axis=1) for value in residuals]
    rms = [observed_rms(value, np.ones(len(p), dtype=bool)) for value in errors]
    squared = [float(value) ** 2 for value in rms]
    return {
        "observed_count": len(p),
        "original_rms_m": rms[0],
        "translation_rms_m": rms[1],
        "rigid_rms_m": rms[2],
        "translation_correction_m": shift.tolist(),
        "translation_norm_m": float(np.linalg.norm(shift)),
        "rotation_prediction_to_target": rotation.tolist(),
        "rotation_angle_deg": float(
            np.degrees(np.linalg.norm(Rotation.from_matrix(rotation).as_rotvec()))
        ),
        "rigid_translation_about_world_origin_m": translation.tolist(),
        "transform_convention": "target approx prediction @ rotation.T + rigid_translation; no scaling or reflection",
        "mse_fraction_removed_by_translation": (squared[0] - squared[1]) / squared[0]
        if squared[0]
        else None,
        "mse_fraction_removed_by_additional_rotation": (squared[1] - squared[2])
        / squared[0]
        if squared[0]
        else None,
        "mse_fraction_remaining_after_rigid": squared[2] / squared[0]
        if squared[0]
        else None,
        "per_observed_marker_errors_m": {
            name: value.tolist()
            for name, value in zip(
                ["original", "translation", "rigid"], errors, strict=True
            )
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "target", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    receipt_path, trajectory_path = (
        args.run / "independent-replay.json",
        args.run / "independent-trajectory.npz",
    )
    receipt, payload = (
        json.loads(receipt_path.read_bytes()),
        json.loads(args.target.read_bytes()),
    )
    if (
        hashlib.sha256(args.target.read_bytes()).hexdigest()
        != receipt["input_sha256"]["target"]
    ):
        raise ValueError("Target payload identity mismatch")
    with np.load(trajectory_path, allow_pickle=False) as data:
        if str(data["candidate_sha256"]) != receipt["candidate_sha256"]:
            raise ValueError("Trajectory candidate identity mismatch")
        clock, labels = data["time_s"], data["labels"].tolist()
        source_indices = [payload["labels"].index(label) for label in labels]
        payload_clock = np.asarray(payload["time_s"])
        rows = []
        for time in (0.6, 0.7, 0.8, 0.85):
            indices = np.flatnonzero(np.isclose(clock, time, rtol=0, atol=1e-12))
            native_indices = np.flatnonzero(
                np.isclose(payload_clock, time, rtol=0, atol=1e-12)
            )
            if len(indices) != 1 or len(native_indices) != 1:
                raise ValueError(
                    "Requested time absent or ambiguous; no interpolation permitted"
                )
            index, native_index = int(indices[0]), int(native_indices[0])
            target = np.asarray(payload["points_world_m"])[native_index, source_indices]
            valid = np.asarray(payload["valid"], dtype=bool)[
                native_index, source_indices
            ]
            if not np.array_equal(valid, data["valid"][index]) or not np.array_equal(
                target[valid], data["target_m"][index, valid]
            ):
                raise ValueError(
                    "Trajectory target points or masks differ from source payload"
                )
            row = decompose(
                data["prediction_m"][index],
                data["target_m"][index],
                data["valid"][index],
            )
            rows.append(
                {
                    "requested_time_s": time,
                    "actual_time_s": float(clock[index]),
                    "observed_labels": [
                        label for label, flag in zip(labels, valid, strict=True) if flag
                    ],
                    **row,
                }
            )
    if not np.isclose(
        rows[-1]["original_rms_m"], receipt["terminal_rms_m"], atol=1e-12, rtol=0
    ):
        raise ValueError("Terminal unmodified error differs from replay receipt")
    paths = [
        trajectory_path,
        receipt_path,
        args.target,
        Path(__file__),
        Path("src/shared/python/body_part_viz/fitters/_kabsch.py"),
    ]
    report = {
        "qualification": "counterfactual per-frame global registration only; no accepted-metric changes, trajectory resets, dynamics or reachability claim",
        "candidate_sha256": receipt["candidate_sha256"],
        "model_sha256": receipt["input_sha256"]["model"],
        "capture_sha256": payload["source_sha256"],
        "target_payload_sha256": receipt["input_sha256"]["target"],
        "rms_definition": "Euclidean marker RMS over original observed mask, equal weights; no missing data filled",
        "residual_interpretation": "After-rigid residual includes articulation, geometry/attachments and measurement disagreement; it is not a pure articulation estimate",
        "rows": rows,
        "source_and_input_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
        },
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    sys.stdout.write(
        json.dumps(
            [
                {
                    k: row[k]
                    for k in [
                        "actual_time_s",
                        "original_rms_m",
                        "translation_rms_m",
                        "rigid_rms_m",
                        "translation_correction_m",
                        "rotation_angle_deg",
                    ]
                }
                for row in rows
            ]
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
