"""OS-3: calibrate body-fixed marker offsets and solve OpenSim IK on the capture.

Runs where ``import opensim`` succeeds. Wires the tested alternating
calibration (``marker_calibration``) to real OpenSim forward kinematics and
the InverseKinematicsTool, on a frame subsample chosen by ``--stride``.
Writes the model with its calibrated MarkerSet, the IK motion, and a receipt
with per-iteration marker RMS. Segment lengths are not scaled here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from defusedxml import ElementTree as ET  # noqa: F401 - type name for annotations

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.engines.physics_engines.opensim.python.tour_matching import (  # noqa: E402
    GOLF_HUMANOID_MARKER_BODIES,
    MarkerPlacement,
    attach_marker_set,
    read_trc,
    unlock_coordinates,
    write_model,
    write_trc,
)
from src.engines.physics_engines.opensim.python.tour_matching.marker_calibration import (  # noqa: E402
    CalibrationResult,
    calibrate_marker_offsets,
)
from src.engines.physics_engines.opensim.python.tour_matching.marker_set import (  # noqa: E402
    locked_coordinates,
    parse_model,
)
from src.shared.python.motion_matching.tour_capture_contract import (  # noqa: E402
    TourCapture,
)

UNLOCK_FOR_GOLF = (
    "lumbar_extension",
    "lumbar_bending",
    "lumbar_rotation",
    "arm_flex_r",
    "arm_add_r",
    "arm_rot_r",
    "elbow_flex_r",
    "pro_sup_r",
    "wrist_flex_r",
    "wrist_dev_r",
    "arm_flex_l",
    "arm_add_l",
    "arm_rot_l",
    "elbow_flex_l",
    "pro_sup_l",
    "wrist_flex_l",
    "wrist_dev_l",
)


def tracking_variant(osim: Path, offsets, unlock) -> ET.ElementTree:
    """Parsed model with the golf tracking unlocks and a MarkerSet attached."""
    tree = parse_model(osim)
    unlock_coordinates(tree, unlock)
    attach_marker_set(tree, {k: MarkerPlacement(b, o) for k, (b, o) in offsets.items()})
    return tree


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def subsample(capture: TourCapture, stride: int) -> TourCapture:
    rows = np.arange(0, capture.frames, stride)
    time = capture.time_s[rows] - capture.time_s[rows][0]
    return TourCapture(
        time, capture.labels, capture.points_m[rows], capture.valid[rows]
    )


def main() -> int:  # noqa: PLR0915 - one bounded driver with explicit stages
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--osim", type=Path, required=True)
    parser.add_argument("--trc", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument(
        "--unlock",
        nargs="*",
        default=list(UNLOCK_FOR_GOLF),
        help="coordinates to unlock in the tracking variant (default: arms, lumbar)",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    receipt: dict = {
        "work_package": "OS-3: marker placement calibration and IK feasibility",
        "epic": "#10003",
        "inputs": {"osim_sha256": sha(args.osim), "trc_sha256": sha(args.trc)},
        "stride": args.stride,
        "iterations": args.iterations,
        "status": "running",
    }

    def save() -> None:
        (args.output / "receipt.json").write_text(
            json.dumps(receipt, indent=2, default=str) + "\n"
        )

    save()
    import opensim  # type: ignore[import-not-found]

    full = read_trc(args.trc)
    capture = subsample(full, args.stride)
    receipt["calibration_frames"] = capture.frames
    unlocked_base = write_model(
        tracking_variant(
            args.osim, {"WaistLeft": ("pelvis", (0.0, 0.0, 0.0))}, args.unlock
        ),
        args.output / "base_unlocked.osim",
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

    def pose_fn(q: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        for i, name in enumerate(names):
            coords.get(name).setValue(base_state, float(q[i]), False)
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

    def ik_fn(offsets, cap: TourCapture) -> np.ndarray:
        with tempfile.TemporaryDirectory() as tmp:
            osim_path = write_model(
                tracking_variant(args.osim, offsets, args.unlock),
                Path(tmp) / "with_markers.osim",
            )
            trc_path = write_trc(
                cap, Path(tmp) / "cal.trc", rate_hz=360.0 / args.stride
            )
            model = opensim.Model(str(osim_path))
            model.initSystem()
            tool = opensim.InverseKinematicsTool()
            tool.setModel(model)
            tool.setMarkerDataFileName(str(trc_path))
            tool.setStartTime(float(cap.time_s[0]))
            tool.setEndTime(float(cap.time_s[-1]))
            tool.setResultsDir(tmp)
            tool.setOutputMotionFileName(str(Path(tmp) / "ik.mot"))
            tasks = tool.getIKTaskSet()
            for label in cap.labels:
                task = opensim.IKMarkerTask()
                task.setName(label)
                task.setWeight(1.0)
                task.setApply(True)
                tasks.cloneAndAppend(task)
            tool.run()
            table = opensim.TimeSeriesTable(str(Path(tmp) / "ik.mot"))
            in_degrees = table.getTableMetaDataAsString("inDegrees") == "yes"
            labels = list(table.getColumnLabels())
            q = np.zeros((table.getNumRows(), len(names)))
            for i, name in enumerate(names):
                col = labels.index(name) if name in labels else None
                if col is None:
                    raise ValueError(f"IK output lacks coordinate {name}")
                column = np.array(
                    [table.getRowAtIndex(r)[col] for r in range(table.getNumRows())]
                )
                q[:, i] = (
                    np.radians(column) if (in_degrees and rotational[i]) else column
                )
            if q.shape[0] != cap.frames:
                raise ValueError("IK returned a different frame count")
            return q

    # Initial pose: default model pose with the pelvis translated to the waist centroid.
    initial = np.zeros(len(names))
    waist = [
        capture.index(k)
        for k in ("WaistLeft", "WaistRight", "WaistLBack", "WaistRBack")
    ]
    centroid = np.nanmean(capture.points_m[0, waist], axis=0)
    for axis, key in enumerate(("pelvis_tx", "pelvis_ty", "pelvis_tz")):
        initial[names.index(key)] = float(centroid[axis])
    receipt["initial_pelvis_translation_m"] = centroid.tolist()
    save()
    result: CalibrationResult = calibrate_marker_offsets(
        capture,
        GOLF_HUMANOID_MARKER_BODIES,
        pose_fn,
        ik_fn,
        initial_q=initial,
        iterations=args.iterations,
    )
    tree = tracking_variant(args.osim, result.offsets, args.unlock)
    model_out = write_model(tree, args.output / "golf_humanoid_tour_markers.osim")
    receipt["unlocked"] = list(args.unlock)
    receipt["still_locked"] = list(locked_coordinates(tree))
    np.savez_compressed(
        args.output / "calibration.npz",
        time_s=capture.time_s * args.stride / args.stride,
        q=result.q,
        coordinate_names=np.array(names),
    )
    receipt.update(
        rms_per_iteration_m=list(result.rms_per_iteration_m),
        best_iteration=result.best_iteration,
        per_marker_rms_m=result.per_marker_rms_m,
        offsets={
            k: {"body": b, "offset_m": list(o)} for k, (b, o) in result.offsets.items()
        },
        coordinate_names=names,
        q_range=[
            [float(np.min(result.q[:, i])), float(np.max(result.q[:, i]))]
            for i in range(len(names))
        ],
        outputs={
            "model_sha256": sha(model_out),
            "calibration_npz_sha256": sha(args.output / "calibration.npz"),
        },
        status="terminal",
        qualification="kinematic feasibility on a frame subsample; not dynamics, not acceptance",
    )
    save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
