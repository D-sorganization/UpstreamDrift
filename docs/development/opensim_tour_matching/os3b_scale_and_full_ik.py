"""OS-3b: OpenSim segment scaling, marker calibration, and full 654-frame IK.

Runs where ``import opensim`` succeeds (e.g. ControlTower opensim-10003 venv).
1. Computes segment scale factors from frame 0 marker pairs and audits rigid assumption.
2. Scales the OpenSim golf model segments.
3. Calibrates 34 body-fixed marker offsets with alternating IK (retaining best iteration).
4. Solves Inverse Kinematics over all 654 frames (stride 1).
5. Unwraps rotational coordinates and computes the five shared metrics.
6. Exports scaled models, full motion (.mot and .npz), overlay animation GIF, and receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.engines.physics_engines.opensim.python.tour_matching import (  # noqa: E402
    GOLF_HUMANOID_MARKER_BODIES,
    CalibrationResult,
    MarkerPlacement,
    SegmentScaleResult,
    SharedMetrics,
    apply_segment_scaling,
    attach_marker_set,
    calibrate_marker_offsets,
    compute_shared_metrics,
    estimate_segment_scales,
    read_trc,
    unlock_coordinates,
    write_model,
    write_trc,
)
from src.engines.physics_engines.opensim.python.tour_matching.marker_set import (  # noqa: E402
    locked_coordinates,
    parse_model,
)
from src.shared.python.motion_matching.tour_capture_contract import (  # noqa: E402
    MARKER_SEGMENTS,
    TourCapture,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def subsample(capture: TourCapture, stride: int) -> TourCapture:
    rows = np.arange(0, capture.frames, stride)
    time = capture.time_s[rows] - capture.time_s[rows][0]
    return TourCapture(
        time, capture.labels, capture.points_m[rows], capture.valid[rows]
    )


def tracking_variant(osim: Path, offsets: dict, unlock: list[str]) -> object:
    tree = parse_model(osim)
    unlock_coordinates(tree, unlock)
    attach_marker_set(tree, {k: MarkerPlacement(b, o) for k, (b, o) in offsets.items()})
    return tree


def generate_overlay_gif(
    observed_m: np.ndarray,
    predicted_m: np.ndarray,
    valid_mask: np.ndarray,
    time_s: np.ndarray,
    out_path: Path,
    stride: int = 6,
) -> None:
    """Generate 3D animated GIF comparing observed vs model markers."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
    except ImportError:
        return

    frames_to_plot = np.arange(0, len(time_s), stride)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    def update(frame_idx):
        ax.clear()
        f = frames_to_plot[frame_idx]
        v = valid_mask[f]
        obs = observed_m[f, v]
        pred = predicted_m[f, v]

        ax.scatter(
            obs[:, 0],
            obs[:, 2],
            obs[:, 1],
            c="blue",
            label="Observed Mocap",
            s=20,
            alpha=0.7,
        )
        ax.scatter(
            pred[:, 0],
            pred[:, 2],
            pred[:, 1],
            c="red",
            label="OpenSim Model",
            s=20,
            alpha=0.7,
        )

        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)
        ax.set_zlim(0.0, 2.0)
        ax.set_xlabel("X (forward/target) [m]")
        ax.set_ylabel("Z (lateral) [m]")
        ax.set_zlabel("Y (up) [m]")
        ax.set_title(
            f"Tour Driver Matching - Frame {f}/{len(time_s)} (t = {time_s[f]:.3f}s)"
        )
        ax.legend(loc="upper right")
        return []

    anim = FuncAnimation(fig, update, frames=len(frames_to_plot), blit=False)
    anim.save(str(out_path), writer="pillow", fps=15)
    plt.close(fig)


def _solve_ik_and_evaluate(
    scaled_osim: Path,
    cal_result: CalibrationResult,
    full: TourCapture,
    names: list[str],
    rotational: list[bool],
    pose_fn: object,
    ik_fn: object,
    output_dir: Path,
) -> tuple[np.ndarray, np.ndarray, list[float], SharedMetrics, Path, str]:
    full_q = ik_fn(cal_result.offsets, full)
    for i in range(len(names)):
        if rotational[i]:
            full_q[:, i] = np.unwrap(full_q[:, i])

    predicted_markers = np.zeros((full.frames, len(full.labels), 3))
    for f in range(full.frames):
        poses = pose_fn(full_q[f])
        for i, label in enumerate(full.labels):
            body, offset = cal_result.offsets[label]
            r, t = poses[body]
            predicted_markers[f, i] = r @ np.asarray(offset) + t

    diffs = predicted_markers - full.points_m
    sq_err = np.sum(diffs**2, axis=-1)
    per_frame_rms = [
        (
            float(np.sqrt(np.mean(sq_err[f, full.valid[f]])))
            if np.any(full.valid[f])
            else 0.0
        )
        for f in range(full.frames)
    ]
    shared = compute_shared_metrics(full, predicted_markers)

    calibrated_model_tree = tracking_variant(scaled_osim, cal_result.offsets, [])
    final_model_path = write_model(
        calibrated_model_tree,
        output_dir / "golf_humanoid_scaled_tour_markers.osim",
    )

    np.savez_compressed(
        output_dir / "ik_full_654.npz",
        time_s=full.time_s,
        q=full_q,
        coordinate_names=np.array(names),
        predicted_markers=predicted_markers,
        per_frame_rms_m=np.array(per_frame_rms),
    )

    gif_name = f"overlay_{sha(final_model_path)[:12]}.gif"
    generate_overlay_gif(
        full.points_m,
        predicted_markers,
        full.valid,
        full.time_s,
        output_dir / gif_name,
        stride=6,
    )
    return full_q, predicted_markers, per_frame_rms, shared, final_model_path, gif_name


def _build_pose_fn(
    base_model: object, base_state: object, names: list[str], bodies: list[str]
) -> object:
    def pose_fn(q: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        for i, name in enumerate(names):
            base_model.getCoordinateSet().get(name).setValue(
                base_state, float(q[i]), False
            )
        base_model.realizePosition(base_state)
        poses = {}
        for body in bodies:
            transform = (
                base_model.getBodySet().get(body).getTransformInGround(base_state)
            )
            rot = transform.R().asMat33()
            r = np.array([[rot.get(i, j) for j in range(3)] for i in range(3)])
            poses[body] = (r, transform.p().to_numpy())
        return poses

    return pose_fn


def _run_ik_motion(
    scaled_osim: Path,
    offsets: dict,
    cap: TourCapture,
    names: list[str],
    rotational: list[bool],
) -> np.ndarray:
    import opensim  # type: ignore[import-not-found]

    with tempfile.TemporaryDirectory() as tmp:
        osim_tmp = write_model(
            tracking_variant(scaled_osim, offsets, []), Path(tmp) / "cal_model.osim"
        )
        dt = float(np.mean(np.diff(cap.time_s))) if cap.frames > 1 else (1.0 / 360.0)
        trc_tmp = write_trc(cap, Path(tmp) / "cal.trc", rate_hz=1.0 / dt)
        model = opensim.Model(str(osim_tmp))
        model.initSystem()
        tool = opensim.InverseKinematicsTool()
        tool.setModel(model)
        tool.setMarkerDataFileName(str(trc_tmp))
        tool.setStartTime(float(cap.time_s[0]))
        tool.setEndTime(float(cap.time_s[-1]))
        tool.setResultsDir(tmp)
        tool.setOutputMotionFileName(str(Path(tmp) / "ik.mot"))
        tasks = tool.getIKTaskSet()
        for label in cap.labels:
            task = opensim.IKMarkerTask()
            task.setName(label)
            task.setWeight(
                10.0 if "Club" in GOLF_HUMANOID_MARKER_BODIES.get(label, "") else 1.0
            )
            task.setApply(True)
            tasks.cloneAndAppend(task)
        tool.run()
        table = opensim.TimeSeriesTable(str(Path(tmp) / "ik.mot"))
        in_degrees = table.getTableMetaDataAsString("inDegrees") == "yes"
        labels = list(table.getColumnLabels())
        q = np.zeros((table.getNumRows(), len(names)))
        for i, name in enumerate(names):
            col = labels.index(name)
            col_data = np.array(
                [table.getRowAtIndex(r)[col] for r in range(table.getNumRows())]
            )
            q[:, i] = (
                np.radians(col_data) if (in_degrees and rotational[i]) else col_data
            )
        return q


def _prepare_calibration_and_run(
    scaled_osim: Path,
    cal_capture: TourCapture,
    names: list[str],
    rotational: list[bool],
    bodies: list[str],
    base_model: object,
    base_state: object,
    iterations: int,
) -> tuple[CalibrationResult, object, object]:
    pose_fn = _build_pose_fn(base_model, base_state, names, bodies)

    def ik_fn(offsets: dict, cap: TourCapture) -> np.ndarray:
        return _run_ik_motion(scaled_osim, offsets, cap, names, rotational)

    waist = [
        cal_capture.index(k)
        for k in ("WaistLeft", "WaistRight", "WaistLBack", "WaistRBack")
    ]
    centroid = np.nanmean(cal_capture.points_m[0, waist], axis=0)
    initial = np.zeros(len(names))
    for axis, key in enumerate(("pelvis_tx", "pelvis_ty", "pelvis_tz")):
        initial[names.index(key)] = float(centroid[axis])

    cal_result = calibrate_marker_offsets(
        cal_capture,
        GOLF_HUMANOID_MARKER_BODIES,
        pose_fn,
        ik_fn,
        initial_q=initial,
        iterations=iterations,
    )
    return cal_result, pose_fn, ik_fn


def _scale_and_setup_model(
    osim_path: Path, full: TourCapture, output_dir: Path
) -> tuple[Path, SegmentScaleResult, object, object, list[str], list[bool], list[str]]:
    import opensim  # type: ignore[import-not-found]

    scale_res = estimate_segment_scales(full)
    scaled_osim = apply_segment_scaling(
        osim_path, scale_res.scale_factors, output_dir / "golf_humanoid_scaled.osim"
    )
    unlocked_base = write_model(
        tracking_variant(scaled_osim, {"WaistLeft": ("pelvis", (0.0, 0.0, 0.0))}, []),
        output_dir / "base_scaled_unlocked.osim",
    )
    base_model = opensim.Model(str(unlocked_base))
    base_state = base_model.initSystem()
    coords = base_model.getCoordinateSet()
    names = [coords.get(i).getName() for i in range(coords.getSize())]
    rotational = [
        coords.get(i).getMotionType() == opensim.Coordinate.Rotational
        for i in range(coords.getSize())
    ]
    bodies = sorted(set(GOLF_HUMANOID_MARKER_BODIES.values()))
    return scaled_osim, scale_res, base_model, base_state, names, rotational, bodies


def _finalize_receipt(
    receipt: dict,
    scale_res: SegmentScaleResult,
    cal_result: CalibrationResult,
    eval_outputs: dict,
    output_dir: Path,
) -> None:
    receipt.update(
        status="complete",
        qualification="full 654-frame kinematic feasibility on subject-scaled model; not dynamics",
        scaling={
            "scale_factors": scale_res.scale_factors,
            "measured_lengths_m": scale_res.measured_lengths_m,
            "nominal_lengths_m": scale_res.nominal_lengths_m,
            "rigid_residuals_m": scale_res.rigid_residuals_m,
            "provenance": scale_res.provenance,
            "scaled_osim_sha256": sha(eval_outputs["scaled_osim"]),
        },
        calibration={
            "best_iteration": cal_result.best_iteration,
            "rms_per_iteration_m": list(cal_result.rms_per_iteration_m),
            "per_marker_rms_m": cal_result.per_marker_rms_m,
        },
        shared_metrics=eval_outputs["shared"].as_dict(),
        overall_marker_rms_m=float(np.mean(eval_outputs["per_frame_rms"])),
        outputs={
            "scaled_model_sha256": sha(eval_outputs["scaled_osim"]),
            "calibrated_model_sha256": sha(eval_outputs["final_model_path"]),
            "ik_npz_sha256": sha(eval_outputs["ik_npz"]),
            "overlay_gif": eval_outputs["gif_name"],
        },
    )
    (output_dir / "receipt.json").write_text(
        json.dumps(receipt, indent=2, default=str) + "\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--osim", type=Path, required=True)
    parser.add_argument("--trc", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cal-stride", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=4)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    receipt: dict = {
        "work_package": "OS-3b: segment scaling, calibration and full 654-frame IK",
        "epic": "#10003",
        "inputs": {"osim_sha256": sha(args.osim), "trc_sha256": sha(args.trc)},
        "cal_stride": args.cal_stride,
        "iterations": args.iterations,
        "status": "running",
    }
    (args.output / "receipt.json").write_text(
        json.dumps(receipt, indent=2, default=str) + "\n"
    )

    full = read_trc(args.trc)
    receipt["total_frames"] = full.frames

    scaled_osim, scale_res, base_model, base_state, names, rotational, bodies = (
        _scale_and_setup_model(args.osim, full, args.output)
    )

    cal_capture = subsample(full, args.cal_stride)
    receipt["calibration_frames"] = cal_capture.frames

    cal_result, pose_fn, ik_fn = _prepare_calibration_and_run(
        scaled_osim,
        cal_capture,
        names,
        rotational,
        bodies,
        base_model,
        base_state,
        args.iterations,
    )

    full_q, pred_m, per_frame_rms, shared, final_model_path, gif_name = (
        _solve_ik_and_evaluate(
            scaled_osim,
            cal_result,
            full,
            names,
            rotational,
            pose_fn,
            ik_fn,
            args.output,
        )
    )

    eval_outputs = {
        "scaled_osim": scaled_osim,
        "final_model_path": final_model_path,
        "ik_npz": args.output / "ik_full_654.npz",
        "gif_name": gif_name,
        "per_frame_rms": per_frame_rms,
        "shared": shared,
    }
    _finalize_receipt(receipt, scale_res, cal_result, eval_outputs, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
