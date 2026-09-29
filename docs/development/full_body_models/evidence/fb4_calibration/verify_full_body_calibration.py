"""Verification and evidence generator for full-body marker calibration and IK (FB-4, #10068).

Alternates marker placement and least-squares IK on a stride-20 subsample (33 frames, 3 iters),
solves the complete 654-frame IK trajectory, and archives machine-verifiable receipts,
calibrated marker offsets, and compressed .npz trajectories for MuJoCo, Pinocchio, and Drake.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from docs.development.full_body_models.evidence._gates import (
    FB4_CALIBRATION_THRESHOLDS,
    evaluate_gates,
)

FULL_BODY_SPEC_PATH = ROOT / "docs/development/full_body_models/full_body_spec_v1.json"
UPPER_SPEC_PATH = (
    ROOT
    / "docs/development/simscape_tour_matching/native_evidence/native_geometry_spec_9967.json"
)
C3D_PATH = ROOT / "data/C3D_TA_Driver.c3d"

HEAD_MARKER_LIMITATION_NOTE = (
    "Three head markers (HeadTop, HeadFront, HeadSide) are attached to the single rigid "
    "trunk/head torso segment ('Hub'). Because the skeletal spec models the torso rigidly "
    "without a cervical neck joint, head movements relative to the thorax produce higher "
    "rigid marker residual on Hub."
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def has_engine(engine_name: str) -> bool:
    if engine_name == "mujoco":
        try:
            import mujoco

            return hasattr(mujoco, "MjModel")
        except ImportError:
            return False
    elif engine_name == "pinocchio":
        try:
            import pinocchio

            return (
                type(pinocchio).__module__ != "unittest.mock"
                and hasattr(pinocchio, "Model")
                and hasattr(pinocchio, "SE3")
            )
        except ImportError:
            return False
    elif engine_name == "drake":
        try:
            import pydrake.all as drake_all

            return type(drake_all).__module__ != "unittest.mock" and hasattr(
                drake_all, "MultibodyPlant"
            )
        except ImportError:
            return False
    return False


def get_engine_version(engine_name: str) -> str:
    if engine_name == "mujoco":
        import mujoco

        return getattr(mujoco, "__version__", "unknown")
    if engine_name == "pinocchio":
        import pinocchio

        return getattr(pinocchio, "__version__", "unknown")
    if engine_name == "drake":
        try:
            import pydrake

            return getattr(pydrake, "__version__", "1.47.0")
        except (ImportError, AttributeError):
            return "unknown"
    return "unknown"


def _sync_files_to_remote(remote_base: str) -> None:
    dirs = [
        f"{remote_base}/docs/development/full_body_models",
        f"{remote_base}/docs/development/simscape_tour_matching/native_evidence",
        f"{remote_base}/docs/development/full_body_models/evidence/fb4_calibration",
        f"{remote_base}/data",
        f"{remote_base}/src/shared/python/body_part_viz/fitters",
        f"{remote_base}/src/shared/python/motion_matching",
        f"{remote_base}/src/shared/python/model_generation/builders",
        f"{remote_base}/src/shared/python/model_generation/core",
        f"{remote_base}/src/engines/physics_engines/opensim/python/tour_matching",
        f"{remote_base}/src/engines/physics_engines/mujoco/python",
        f"{remote_base}/src/engines/physics_engines/pinocchio/python",
        f"{remote_base}/src/engines/physics_engines/drake/python",
        f"{remote_base}/tests/unit/motion_matching",
    ]
    p0 = subprocess.run(
        ["ssh", "deskcomputer", "wsl", "-d", "Ubuntu-22.04", "mkdir", "-p", *dirs],
        capture_output=True,
    )
    if p0.returncode != 0:
        raise RuntimeError(f"Remote mkdir failed: {p0.stderr.decode()}")

    files_to_sync = [
        "docs/development/full_body_models/full_body_spec_v1.json",
        "docs/development/simscape_tour_matching/native_evidence/native_geometry_spec_9967.json",
        "data/C3D_TA_Driver.c3d",
        "src/shared/python/contracts.py",
        "src/shared/python/body_part_viz/fitters/_kabsch.py",
        "src/shared/python/motion_matching/tour_capture_contract.py",
        "src/shared/python/motion_matching/full_body_spec.py",
        "src/shared/python/motion_matching/contact_law.py",
        "src/shared/python/motion_matching/marker_projection.py",
        "src/shared/python/motion_matching/marker_calibration.py",
        "src/shared/python/motion_matching/full_body_ik.py",
        "src/shared/python/model_generation/builders/urdf_writer.py",
        "src/shared/python/model_generation/core/constants.py",
        "src/shared/python/model_generation/core/types.py",
        "src/shared/python/model_generation/core/contracts.py",
        "src/engines/physics_engines/opensim/python/tour_matching/marker_calibration.py",
        "src/engines/physics_engines/mujoco/python/full_body_mjcf.py",
        "src/engines/physics_engines/mujoco/python/full_body_model.py",
        "src/engines/physics_engines/mujoco/python/full_body_ik.py",
        "src/engines/physics_engines/mujoco/python/native_mjcf.py",
        "src/engines/physics_engines/mujoco/python/native_model.py",
        "src/engines/physics_engines/pinocchio/python/native_model.py",
        "src/engines/physics_engines/pinocchio/python/full_body_ik.py",
        "src/engines/physics_engines/drake/python/full_body_urdf.py",
        "src/engines/physics_engines/drake/python/full_body_model.py",
        "src/engines/physics_engines/drake/python/full_body_ik.py",
        "docs/development/full_body_models/evidence/fb4_calibration/verify_full_body_calibration.py",
    ]

    for rel in files_to_sync:
        local_path = ROOT / rel
        if not local_path.exists():
            continue
        remote_path = f"{remote_base}/{rel}"
        content = local_path.read_bytes()
        p = subprocess.run(
            ["ssh", "deskcomputer", "wsl", "-d", "Ubuntu-22.04", "tee", remote_path],
            input=content,
            capture_output=True,
        )
        if p.returncode != 0:
            raise RuntimeError(f"Failed to sync {rel}: {p.stderr.decode()}")

    # Clean __init__.py files in remote base
    subprocess.run(
        [
            "ssh",
            "deskcomputer",
            "wsl",
            "-d",
            "Ubuntu-22.04",
            "find",
            remote_base,
            "-name",
            "__init__.py",
            "-delete",
        ],
        capture_output=True,
    )


def run_remote_engine(engine_name: str, stride: int, iterations: int) -> dict[str, Any]:
    sys.stdout.write(
        f"Engine {engine_name} running on deskcomputer WSL Ubuntu-22.04...\n"
    )
    sys.stdout.flush()

    remote_base = "/home/dieterolson/fb4_calibration_run"
    _sync_files_to_remote(remote_base)

    script = (
        f"cd {remote_base}\n"
        f"export PYTHONPATH={remote_base}\n"
        f"python3 docs/development/full_body_models/evidence/fb4_calibration/verify_full_body_calibration.py "
        f"--engine {engine_name} --local --stride {stride} --iterations {iterations}\n"
    ).encode()

    res = subprocess.run(
        ["ssh", "deskcomputer", "wsl", "-d", "Ubuntu-22.04", "bash"],
        input=script,
        capture_output=True,
    )
    if res.returncode != 0:
        sys.stderr.write(
            f"Remote error:\n{res.stderr.decode('utf-8', errors='ignore')}\n"
            f"Remote stdout:\n{res.stdout.decode('utf-8', errors='ignore')}\n"
        )
        raise RuntimeError(
            f"Remote execution failed for {engine_name} with code {res.returncode}"
        )
    sys.stdout.write(res.stdout.decode("utf-8", errors="ignore"))

    # Fetch back the three generated artifacts
    engine_dir = HERE / engine_name
    engine_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("calibrated_offsets.json", "ik_trajectory.npz", "receipt.json"):
        cat_proc = subprocess.run(
            [
                "ssh",
                "deskcomputer",
                "wsl",
                "-d",
                "Ubuntu-22.04",
                "cat",
                f"{remote_base}/docs/development/full_body_models/evidence/fb4_calibration/{engine_name}/{filename}",
            ],
            capture_output=True,
        )
        if cat_proc.returncode != 0:
            raise RuntimeError(
                f"Failed to fetch {filename}: {cat_proc.stderr.decode()}"
            )
        (engine_dir / filename).write_bytes(cat_proc.stdout)

    return json.loads((engine_dir / "receipt.json").read_text(encoding="utf-8"))


def _init_adapter(engine_name: str, fb_spec: dict[str, Any]) -> Any:
    """Instantiate the engine-specific full-body IK adapter."""
    if engine_name == "mujoco":
        from src.engines.physics_engines.mujoco.python.full_body_ik import (
            MujocoFullBodyIK,
        )

        return MujocoFullBodyIK(fb_spec)
    if engine_name == "pinocchio":
        from src.engines.physics_engines.pinocchio.python.full_body_ik import (
            PinocchioFullBodyIK,
        )

        return PinocchioFullBodyIK(fb_spec)
    if engine_name == "drake":
        from src.engines.physics_engines.drake.python.full_body_ik import (
            DrakeFullBodyIK,
        )

        return DrakeFullBodyIK(fb_spec)
    raise ValueError(f"Unknown engine: {engine_name}")


def _subsample_calibration(
    adapter: Any,
    capture: Any,
    labels: Sequence[str],
    bodies: dict[str, str],
    stride: int,
    iterations: int,
) -> tuple[Any, Any, dict[str, Any]]:
    from src.shared.python.motion_matching.marker_calibration import (
        calibrate_marker_offsets,
    )
    from src.shared.python.motion_matching.tour_capture_contract import TourCapture

    t_sub = capture.time_s[::stride] - capture.time_s[0]
    p_sub = capture.points_m[::stride]
    v_sub = capture.valid[::stride]
    cap_sub = TourCapture(
        t_sub, capture.labels, p_sub, v_sub, capture.source_sha256
    ).subset(labels)

    sys.stdout.write(
        f"Calibrating 34 markers on {cap_sub.frames} frames ({iterations} iterations)...\n"
    )
    sys.stdout.flush()

    initial_q = np.zeros(len(adapter.coordinate_order), dtype=float)
    calib_res = calibrate_marker_offsets(
        cap_sub,
        bodies,
        adapter.pose_fn,
        adapter.ik_fn,
        initial_q=initial_q,
        iterations=iterations,
    )
    sub_frame_rms, sub_marker_rms, sub_total_rms = adapter.evaluate_trajectory_rms(
        calib_res.offsets, cap_sub, calib_res.q
    )
    sys.stdout.write(
        f"Subsample calibration complete: best iteration {calib_res.best_iteration}, "
        f"total RMS = {sub_total_rms * 1000.0:.2f} mm\n"
    )
    metrics = {
        "sub_frame_rms": sub_frame_rms,
        "sub_marker_rms": sub_marker_rms,
        "sub_total_rms": sub_total_rms,
    }
    return calib_res, cap_sub, metrics


def build_status_from_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate FB-4 calibration metrics against documented thresholds (#10960 P0-9)."""
    return evaluate_gates(metrics, FB4_CALIBRATION_THRESHOLDS)


def _build_receipt_dict(
    engine_name: str,
    calib_res: Any,
    traj_metrics: dict[str, Any],
    run_meta: dict[str, Any],
    offsets_path: Path,
    traj_path: Path,
) -> dict[str, Any]:
    sub_marker_rms = traj_metrics["sub_marker_rms"]
    full_frame_rms = traj_metrics["full_frame_rms"]
    q_full = traj_metrics["q_full"]
    gate_metrics = {
        "total_rms_m": float(traj_metrics["full_total_rms"]),
        "closure_max_error_m": float(traj_metrics["max_closure_err"]),
    }
    status_eval = build_status_from_metrics(gate_metrics)

    return {
        "work_package": "FB-4",
        "issue": "#10068",
        "epic": "#10062",
        "engine": engine_name,
        "status": status_eval["status"],
        "gate_evaluation": status_eval,
        "thresholds": dict(FB4_CALIBRATION_THRESHOLDS),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "executable": sys.executable,
            f"{engine_name}_version": get_engine_version(engine_name),
        },
        "inputs": {
            "full_body_spec_v1.json": sha256_file(FULL_BODY_SPEC_PATH),
            "native_geometry_spec_9967.json": sha256_file(UPPER_SPEC_PATH),
            "C3D_TA_Driver.c3d": sha256_file(C3D_PATH),
            "marker_calibration.py": sha256_file(
                ROOT / "src/shared/python/motion_matching/marker_calibration.py"
            ),
            "full_body_ik.py": sha256_file(
                ROOT / "src/shared/python/motion_matching/full_body_ik.py"
            ),
        },
        "stride_subsample": {
            "stride": run_meta["stride"],
            "num_frames": run_meta["cap_sub_frames"],
            "iterations": run_meta["iterations"],
            "best_iteration": calib_res.best_iteration,
            "total_rms_m": float(traj_metrics["sub_total_rms"]),
            "per_marker_rms_m": {k: float(v) for k, v in sub_marker_rms.items()},
        },
        "full_trajectory": {
            "stride": 1,
            "num_frames": q_full.shape[0],
            "total_rms_m": float(traj_metrics["full_total_rms"]),
            "mean_frame_rms_m": float(np.mean(full_frame_rms)),
            "max_frame_rms_m": float(np.max(full_frame_rms)),
            "closure_max_error_m": traj_metrics["max_closure_err"],
            "closure_mean_error_m": traj_metrics["mean_closure_err"],
        },
        "head_marker_limitation": HEAD_MARKER_LIMITATION_NOTE,
        "artifacts": {
            "calibrated_offsets_sha256": sha256_file(offsets_path),
            "ik_trajectory_sha256": sha256_file(traj_path),
        },
    }


def _save_artifacts(
    engine_name: str,
    adapter: Any,
    calib_res: Any,
    traj_metrics: dict[str, Any],
    run_meta: dict[str, Any],
) -> dict[str, Any]:
    out_dir = HERE / engine_name
    out_dir.mkdir(parents=True, exist_ok=True)

    offsets_path = out_dir / "calibrated_offsets.json"
    traj_path = out_dir / "ik_trajectory.npz"
    receipt_path = out_dir / "receipt.json"

    labels = run_meta["labels"]
    sub_marker_rms = traj_metrics["sub_marker_rms"]
    full_marker_rms = traj_metrics["full_marker_rms"]
    sub_total_rms = traj_metrics["sub_total_rms"]
    full_frame_rms = traj_metrics["full_frame_rms"]
    q_full = traj_metrics["q_full"]
    capture = run_meta["capture"]

    offsets_data = {
        "schema_version": "v1",
        "spec_sha256": sha256_file(FULL_BODY_SPEC_PATH),
        "engine": engine_name,
        "best_iteration": calib_res.best_iteration,
        "total_iterations": calib_res.iterations,
        "best_subsample_rms_m": float(sub_total_rms),
        "iterations_rms_history_m": list(calib_res.rms_per_iteration_m),
        "head_marker_limitation": HEAD_MARKER_LIMITATION_NOTE,
        "marker_offsets": {
            lbl: {
                "body": calib_res.offsets[lbl][0],
                "offset_m": list(calib_res.offsets[lbl][1]),
                "subsample_rms_m": float(sub_marker_rms[lbl]),
                "full_trajectory_rms_m": float(full_marker_rms[lbl]),
            }
            for lbl in sorted(labels)
        },
    }
    offsets_path.write_text(json.dumps(offsets_data, indent=2), encoding="utf-8")

    np.savez_compressed(
        traj_path,
        q=q_full,
        time_s=capture.time_s,
        rms_per_frame_m=full_frame_rms,
        coordinate_order=np.array(adapter.coordinate_order),
    )

    receipt_data = _build_receipt_dict(
        engine_name, calib_res, traj_metrics, run_meta, offsets_path, traj_path
    )
    receipt_path.write_text(json.dumps(receipt_data, indent=2), encoding="utf-8")
    sys.stdout.write(f"Saved receipt to {receipt_path}\n")
    return receipt_data


def run_local_engine(engine_name: str, stride: int, iterations: int) -> dict[str, Any]:
    from src.shared.python.motion_matching.full_body_spec import load_full_body_spec
    from src.shared.python.motion_matching.tour_capture_contract import (
        load_tour_capture,
        tracked_labels,
    )

    sys.stdout.write(f"=== Starting FB-4 Calibration: {engine_name.upper()} ===\n")
    sys.stdout.flush()

    upper_spec = json.loads(UPPER_SPEC_PATH.read_text(encoding="utf-8"))
    fb_spec = load_full_body_spec(FULL_BODY_SPEC_PATH, upper_spec)
    capture = load_tour_capture(C3D_PATH)
    labels = tracked_labels()
    if len(labels) != 34:
        raise ValueError(f"Expected exactly 34 tracked labels, found {len(labels)}")

    bodies = {lbl: fb_spec["marker_attachments"][lbl]["body"] for lbl in labels}
    adapter = _init_adapter(engine_name, fb_spec)

    calib_res, cap_sub, sub_m = _subsample_calibration(
        adapter, capture, labels, bodies, stride, iterations
    )

    # 2. Solve Full 654-Frame IK Trajectory with calibrated offsets
    sys.stdout.write(
        "Solving full 654-frame IK trajectory with calibrated offsets...\n"
    )
    sys.stdout.flush()
    cap_full_tracked = capture.subset(labels)
    q_full = adapter.ik_fn(
        calib_res.offsets,
        cap_full_tracked,
        initial_q=calib_res.q[0],
        closure_weight=10.0,
    )

    full_frame_rms, full_marker_rms, full_total_rms = adapter.evaluate_trajectory_rms(
        calib_res.offsets, cap_full_tracked, q_full
    )
    sys.stdout.write(
        f"Full trajectory IK complete: total RMS = {full_total_rms * 1000.0:.2f} mm, "
        f"mean frame RMS = {np.mean(full_frame_rms) * 1000.0:.2f} mm, "
        f"max frame RMS = {np.max(full_frame_rms) * 1000.0:.2f} mm\n"
    )

    # 3. Evaluate Dual-Grip Weld Loop Closure Residuals across trajectory
    closure_errs = [
        float(np.linalg.norm(adapter.closure_residuals(q_full[f])))
        for f in range(q_full.shape[0])
    ]

    traj_metrics = {
        "sub_marker_rms": sub_m["sub_marker_rms"],
        "sub_total_rms": sub_m["sub_total_rms"],
        "full_frame_rms": full_frame_rms,
        "full_marker_rms": full_marker_rms,
        "full_total_rms": full_total_rms,
        "q_full": q_full,
        "max_closure_err": float(np.max(closure_errs)),
        "mean_closure_err": float(np.mean(closure_errs)),
    }
    run_meta = {
        "labels": labels,
        "stride": stride,
        "iterations": iterations,
        "cap_sub_frames": cap_sub.frames,
        "capture": capture,
    }
    return _save_artifacts(engine_name, adapter, calib_res, traj_metrics, run_meta)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify and archive full-body marker calibration and IK (FB-4)."
    )
    parser.add_argument(
        "--engine",
        choices=["mujoco", "pinocchio", "drake", "all"],
        default="all",
        help="Engine to calibrate (default: all)",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Force execution in the current local environment",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=20,
        help="Subsample stride for calibration (default: 20)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of alternating calibration iterations (default: 3)",
    )
    args = parser.parse_args()

    engines = (
        ["mujoco", "pinocchio", "drake"] if args.engine == "all" else [args.engine]
    )

    for engine in engines:
        if args.local or has_engine(engine):
            run_local_engine(engine, args.stride, args.iterations)
        else:
            run_remote_engine(engine, args.stride, args.iterations)


if __name__ == "__main__":
    main()
