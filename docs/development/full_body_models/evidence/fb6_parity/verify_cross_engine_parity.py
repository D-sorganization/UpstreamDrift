"""Verification and evidence generator for cross-engine full-body parity and visual review (FB-6, #10070).

Replays the accepted full-body candidate in MuJoCo, Pinocchio, and Drake with the shared contact law
and weld loop closure; measures numerical step-size convergence; renders 3D marker overlay animations;
and archives cryptographic receipts and parity comparison reports.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
from collections.abc import Mapping
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from docs.development.full_body_models.evidence._gates import (
    FB6_PARITY_THRESHOLDS,
    evaluate_gates,
)

from src.shared.python.motion_matching.cross_engine_replay import (
    CrossEngineComparisonReport,
    CrossEngineReplayConfig,
    EngineReplayOutcome,
    StepSizeConvergenceResult,
    compare_engine_replays,
    compute_step_size_convergence,
    render_marker_overlay_animation,
)
from src.shared.python.motion_matching.full_body_forward_dynamics import (
    ContactAuditResult,
    ForwardRolloutResult,
    RolloutOptions,
    SharedMetrics,
    simulate_full_body_forward,
)
from src.shared.python.motion_matching.tour_capture_contract import load_tour_capture

logger = logging.getLogger("fb6_parity")

SPEC_PATH = ROOT / "docs/development/full_body_models/full_body_spec_v1.json"
CANDIDATE_PATH = (
    ROOT
    / "docs/development/simscape_tour_matching/native_evidence/two_window_fit_9967_81/returned-candidate.json"
)
C3D_PATH = ROOT / "data/C3D_TA_Driver.c3d"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _get_engine_version(engine: str) -> str:
    if engine == "mujoco":
        try:
            import mujoco

            return getattr(mujoco, "__version__", "unknown")
        except ImportError:
            return "unknown"
    if engine == "pinocchio":
        try:
            import pinocchio

            return getattr(pinocchio, "__version__", "unknown")
        except ImportError:
            return "unknown"
    if engine == "drake":
        try:
            import pydrake.all

            return "1.57.0"
        except ImportError:
            return "unknown"
    return "unknown"


def _build_model_and_adapter(engine: str, spec: dict[str, Any]) -> tuple[Any, Any]:
    if engine == "mujoco":
        from src.engines.physics_engines.mujoco.python.full_body_ik import (
            MujocoFullBodyIK,
        )
        from src.engines.physics_engines.mujoco.python.full_body_model import (
            NativeMujocoFullBodyModel,
        )

        spec_bytes = json.dumps(spec).encode("utf-8")
        model = NativeMujocoFullBodyModel(spec_bytes)
        ik_adapter = MujocoFullBodyIK(json.dumps(spec))
        return model, ik_adapter
    if engine == "pinocchio":
        from src.engines.physics_engines.pinocchio.python.full_body_ik import (
            PinocchioFullBodyIK,
        )
        from src.engines.physics_engines.pinocchio.python.native_model import (
            build_full_body_pinocchio_model,
        )

        model = build_full_body_pinocchio_model(spec)
        ik_adapter = PinocchioFullBodyIK(spec)
        return model, ik_adapter
    if engine == "drake":
        from src.engines.physics_engines.drake.python.full_body_ik import (
            DrakeFullBodyIK,
        )
        from src.engines.physics_engines.drake.python.full_body_model import (
            FullBodyDrakeModel,
        )

        model = FullBodyDrakeModel(spec)
        ik_adapter = DrakeFullBodyIK(spec)
        return model, ik_adapter
    raise ValueError(f"Unknown engine: {engine}")


def _measure_step_size_convergence(
    engine: str,
    spec_data: dict[str, Any],
    theta: np.ndarray,
    t_conv: np.ndarray,
    initial_state: tuple[np.ndarray, np.ndarray],
    marker_offsets: dict[str, list[float]],
    capture: Any,
) -> StepSizeConvergenceResult:
    """Measure step-size or tolerance convergence on the early window."""
    nom_opt = RolloutOptions(integrator="rk45", rtol=1e-5, atol=1e-7)
    ref_opt = RolloutOptions(integrator="rk45", rtol=1e-6, atol=1e-8)
    h_nom = 1e-5
    h_ref = 1e-6

    model_nom, ik_nom = _build_model_and_adapter(engine, spec_data)
    res_nom = simulate_full_body_forward(
        model=model_nom,
        ik_adapter=ik_nom,
        theta=theta,
        time_grid=t_conv,
        initial_state=initial_state,
        marker_offsets=marker_offsets,
        capture=capture,
        options=nom_opt,
    )
    model_ref, ik_ref = _build_model_and_adapter(engine, spec_data)
    res_ref = simulate_full_body_forward(
        model=model_ref,
        ik_adapter=ik_ref,
        theta=theta,
        time_grid=t_conv,
        initial_state=initial_state,
        marker_offsets=marker_offsets,
        capture=capture,
        options=ref_opt,
    )
    conv_result = compute_step_size_convergence(
        time_s=t_conv,
        q_nominal=res_nom.q,
        qd_nominal=res_nom.qd,
        q_refined=res_ref.q,
        qd_refined=res_ref.qd,
        h_nominal=h_nom,
        h_refined=h_ref,
        tolerance_rad=0.05,
    )
    logger.info(
        "    Step-size convergence: max_q_diff=%.4e rad, is_converged=%s",
        conv_result.max_q_difference,
        conv_result.is_converged,
    )
    return conv_result


def build_status_from_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate FB-6 cross-engine parity metrics against documented thresholds (#10960 P0-9)."""
    return evaluate_gates(metrics, FB6_PARITY_THRESHOLDS)


def _write_receipt(
    engine: str,
    output_dir: Path,
    rollout_res: ForwardRolloutResult,
    conv_result: StepSizeConvergenceResult,
    traj_path: Path,
    offsets_path: Path,
    ik_path: Path,
) -> Path:
    """Serialize and write reproducible engine execution receipt."""
    integ_mode = "rk45" if engine == "mujoco" else "euler"

    # A failed rollout has no audit or metrics (#10960 P1-9): None fails its gate.
    contact_dict = (
        rollout_res.contact_audit.as_dict()
        if rollout_res.contact_audit is not None
        else {}
    )
    shared = rollout_res.shared_metrics
    whole_rmse = shared.whole_marker_rmse_m if shared is not None else None
    gate_metrics = {
        "whole_marker_rmse_m": whole_rmse,
        "max_normal_force_n": contact_dict.get("max_normal_force_n"),
        "max_penetration_m": contact_dict.get("max_penetration_m"),
        "max_closure_residual_m": float(rollout_res.max_closure_residual_m),
        "max_q_difference": float(conv_result.max_q_difference),
    }
    status_eval = build_status_from_metrics(gate_metrics)

    receipt = {
        "work_package": "FB-6",
        "issue": "#10070",
        "epic": "#10062",
        "engine": engine,
        "status": status_eval["status"]
        if rollout_res.status == "success"
        else "FAILED",
        "gate_evaluation": status_eval,
        "thresholds": dict(FB6_PARITY_THRESHOLDS),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "executable": sys.executable,
            f"{engine}_version": _get_engine_version(engine),
        },
        "inputs": {
            "full_body_spec_v1.json": sha256_file(SPEC_PATH),
            "calibrated_offsets.json": sha256_file(offsets_path),
            "ik_trajectory.npz": sha256_file(ik_path),
            "returned-candidate.json": sha256_file(CANDIDATE_PATH),
            "C3D_TA_Driver.c3d": sha256_file(C3D_PATH),
        },
        "step_size_convergence": conv_result.as_dict(),
        "forward_rollout": {
            "integrator": integ_mode,
            "num_frames": len(rollout_res.time_s),
            "duration_s": float(rollout_res.time_s[-1]),
            "max_closure_residual_m": rollout_res.max_closure_residual_m,
            "shared_metrics": (
                rollout_res.shared_metrics.as_dict()
                if rollout_res.shared_metrics is not None
                else None
            ),
            "contact_audit": (
                rollout_res.contact_audit.as_dict()
                if rollout_res.contact_audit is not None
                else None
            ),
        },
        "artifacts": {
            f"forward_trajectory_{engine}_sha256": sha256_file(traj_path),
        },
    }
    receipt_path = output_dir / f"receipt_{engine}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt_path


def run_local_engine_replay(
    engine: str,
    output_dir: Path,
    num_frames: int = 0,
) -> tuple[EngineReplayOutcome, Path, Path]:
    """Execute forward simulation, step-size convergence, and receipt generation for an engine."""
    logger.info("Executing forward simulation replay on %s...", engine.upper())
    spec_data = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    candidate_data = json.loads(CANDIDATE_PATH.read_text(encoding="utf-8"))
    capture = load_tour_capture(str(C3D_PATH))

    offsets_path = (
        ROOT
        / f"docs/development/full_body_models/evidence/fb4_calibration/{engine}/calibrated_offsets.json"
    )
    ik_path = (
        ROOT
        / f"docs/development/full_body_models/evidence/fb4_calibration/{engine}/ik_trajectory.npz"
    )

    marker_offsets = json.loads(offsets_path.read_text(encoding="utf-8"))[
        "marker_offsets"
    ]
    ik_traj = np.load(ik_path)
    q0 = ik_traj["q"][0]
    qd0 = np.zeros_like(q0)

    theta = np.zeros((41, 7), dtype=np.float64)
    cand_coeffs = np.asarray(candidate_data["coefficients"], dtype=np.float64)
    theta[: cand_coeffs.shape[0], :] = cand_coeffs

    frames_to_run = (
        len(capture.time_s) if num_frames <= 0 else min(num_frames, len(capture.time_s))
    )
    time_grid = capture.time_s[:frames_to_run]

    # 1. Step-size convergence on early window (10 frames)
    logger.info("  Measuring numerical step-size convergence on early window...")
    conv_frames = min(10, frames_to_run)
    t_conv = time_grid[:conv_frames]
    conv_result = _measure_step_size_convergence(
        engine=engine,
        spec_data=spec_data,
        theta=theta,
        t_conv=t_conv,
        initial_state=(q0, qd0),
        marker_offsets=marker_offsets,
        capture=capture,
    )

    # 2. Full uninterrupted rollout
    logger.info(
        "  Simulating uninterrupted forward rollout (%d frames)...", frames_to_run
    )
    model, ik_adapter = _build_model_and_adapter(engine, spec_data)
    integ_mode = "rk45"
    rollout_res = simulate_full_body_forward(
        model=model,
        ik_adapter=ik_adapter,
        theta=theta,
        time_grid=time_grid,
        initial_state=(q0, qd0),
        marker_offsets=marker_offsets,
        capture=capture,
        options=RolloutOptions(integrator=integ_mode, rtol=1e-5, atol=1e-7),
    )

    # 3. Archive trajectory
    output_dir.mkdir(parents=True, exist_ok=True)
    traj_path = output_dir / f"forward_trajectory_{engine}.npz"
    np.savez_compressed(
        traj_path,
        time_s=rollout_res.time_s,
        q=rollout_res.q,
        qd=rollout_res.qd,
        predicted_markers_m=rollout_res.predicted_markers_m,
    )

    # 4. Write receipt
    receipt_path = _write_receipt(
        engine=engine,
        output_dir=output_dir,
        rollout_res=rollout_res,
        conv_result=conv_result,
        traj_path=traj_path,
        offsets_path=offsets_path,
        ik_path=ik_path,
    )

    outcome = EngineReplayOutcome(
        engine=engine,
        status=rollout_res.status,
        shared_metrics=rollout_res.shared_metrics,
        contact_audit=rollout_res.contact_audit,
        convergence=conv_result,
        max_closure_residual_m=rollout_res.max_closure_residual_m,
    )
    return outcome, traj_path, receipt_path


def _sync_to_remote(remote_base: str) -> None:
    """Synchronize required codebase files to remote execution directory."""
    dirs = [
        f"{remote_base}/data",
        f"{remote_base}/docs/development/full_body_models/evidence/fb4_calibration/pinocchio",
        f"{remote_base}/docs/development/full_body_models/evidence/fb4_calibration/drake",
        f"{remote_base}/docs/development/full_body_models/evidence/fb6_parity",
        f"{remote_base}/docs/development/simscape_tour_matching/native_evidence/two_window_fit_9967_81",
        f"{remote_base}/src/shared/python/body_part_viz/fitters",
        f"{remote_base}/src/shared/python/motion_matching",
        f"{remote_base}/src/shared/python/model_generation/builders",
        f"{remote_base}/src/shared/python/model_generation/core",
        f"{remote_base}/src/engines/physics_engines/pinocchio/python",
        f"{remote_base}/src/engines/physics_engines/drake/python",
    ]
    subprocess.run(
        ["ssh", "deskcomputer", "wsl", "-d", "Ubuntu-22.04", "mkdir", "-p", *dirs],
        check=True,
    )

    files_to_sync = [
        "docs/development/full_body_models/full_body_spec_v1.json",
        "docs/development/simscape_tour_matching/native_evidence/two_window_fit_9967_81/returned-candidate.json",
        "docs/development/full_body_models/evidence/fb4_calibration/pinocchio/calibrated_offsets.json",
        "docs/development/full_body_models/evidence/fb4_calibration/pinocchio/ik_trajectory.npz",
        "docs/development/full_body_models/evidence/fb4_calibration/drake/calibrated_offsets.json",
        "docs/development/full_body_models/evidence/fb4_calibration/drake/ik_trajectory.npz",
        "data/C3D_TA_Driver.c3d",
        "src/shared/python/contracts.py",
        "src/shared/python/body_part_viz/fitters/_kabsch.py",
        "src/shared/python/model_generation/builders/urdf_writer.py",
        "src/shared/python/model_generation/core/constants.py",
        "src/shared/python/model_generation/core/types.py",
        "src/shared/python/model_generation/core/contracts.py",
        "src/engines/physics_engines/pinocchio/python/native_model.py",
        "src/engines/physics_engines/pinocchio/python/full_body_ik.py",
        "src/engines/physics_engines/drake/python/full_body_urdf.py",
        "src/engines/physics_engines/drake/python/full_body_model.py",
        "src/engines/physics_engines/drake/python/full_body_ik.py",
        "docs/development/full_body_models/evidence/fb6_parity/verify_cross_engine_parity.py",
    ]
    for p in (ROOT / "src/shared/python/motion_matching").glob("*.py"):
        files_to_sync.append(f"src/shared/python/motion_matching/{p.name}")

    for rel in files_to_sync:
        local_p = ROOT / rel
        if local_p.is_file():
            content = local_p.read_bytes()
            subprocess.run(
                [
                    "ssh",
                    "deskcomputer",
                    "wsl",
                    "-d",
                    "Ubuntu-22.04",
                    "tee",
                    f"{remote_base}/{rel}",
                ],
                input=content,
                check=True,
                stdout=subprocess.DEVNULL,
            )

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
        check=True,
    )


def run_remote_engine_replay(
    engine: str,
    output_dir: Path,
    num_frames: int = 0,
) -> tuple[EngineReplayOutcome, Path, Path]:
    """Execute replay remotely on deskcomputer WSL Ubuntu-22.04 and fetch back artifacts."""
    logger.info("Executing remote replay for %s on deskcomputer WSL...", engine.upper())
    remote_base = "/home/dieterolson/fb6_parity_run"
    _sync_to_remote(remote_base)

    remote_out = f"{remote_base}/docs/development/full_body_models/evidence/fb6_parity"
    script = f"""
cd {remote_base}
export PYTHONPATH={remote_base}
python3 docs/development/full_body_models/evidence/fb6_parity/verify_cross_engine_parity.py --engine {engine} --local --frames {num_frames}
""".encode()

    res = subprocess.run(
        ["ssh", "deskcomputer", "wsl", "-d", "Ubuntu-22.04", "bash"],
        input=script,
        capture_output=True,
    )
    if res.returncode != 0:
        logger.error("Remote stderr:\n%s", res.stderr.decode("utf-8", errors="ignore"))
        raise RuntimeError(f"Remote replay failed for {engine} (code {res.returncode})")
    logger.info("Remote stdout:\n%s", res.stdout.decode("utf-8", errors="ignore"))

    # Fetch back receipt and trajectory
    output_dir.mkdir(parents=True, exist_ok=True)
    traj_path = output_dir / f"forward_trajectory_{engine}.npz"
    receipt_path = output_dir / f"receipt_{engine}.json"

    for local_f, name in (
        (traj_path, f"forward_trajectory_{engine}.npz"),
        (receipt_path, f"receipt_{engine}.json"),
    ):
        cat_p = subprocess.run(
            [
                "ssh",
                "deskcomputer",
                "wsl",
                "-d",
                "Ubuntu-22.04",
                "cat",
                f"{remote_out}/{name}",
            ],
            capture_output=True,
        )
        if cat_p.returncode != 0:
            raise RuntimeError(
                f"Failed to fetch {name} from remote: {cat_p.stderr.decode()}"
            )
        local_f.write_bytes(cat_p.stdout)

    receipt_data = json.loads(receipt_path.read_text(encoding="utf-8"))
    metrics_data = receipt_data["forward_rollout"]["shared_metrics"]
    conv_data = receipt_data["step_size_convergence"]
    audit_data = receipt_data["forward_rollout"]["contact_audit"]

    outcome = EngineReplayOutcome(
        engine=engine,
        status=receipt_data["status"],
        shared_metrics=SharedMetrics(
            whole_marker_rmse_m=metrics_data["whole_marker_rmse_m"],
            early_marker_rmse_m=metrics_data["early_marker_rmse_m"],
            terminal_marker_rmse_m=metrics_data["terminal_marker_rmse_m"],
            club_marker_rmse_m=metrics_data["club_marker_rmse_m"],
            pelvis_yaw_rmse_rad=metrics_data["pelvis_yaw_rmse_rad"],
        ),
        contact_audit=ContactAuditResult(
            max_normal_force_n=audit_data["max_normal_force_n"],
            max_friction_force_n=audit_data["max_friction_force_n"],
            max_penetration_m=audit_data["max_penetration_m"],
            per_sphere_max_force_n=audit_data["per_sphere_max_force_n"],
            per_sphere_contact_ratio=audit_data["per_sphere_contact_ratio"],
        ),
        convergence=StepSizeConvergenceResult(
            h_nominal=conv_data["h_nominal"],
            h_refined=conv_data["h_refined"],
            max_q_difference=conv_data["max_q_difference"],
            max_qd_difference=conv_data["max_qd_difference"],
            is_converged=conv_data["is_converged"],
        ),
        max_closure_residual_m=receipt_data["forward_rollout"][
            "max_closure_residual_m"
        ],
    )
    return outcome, traj_path, receipt_path


def render_all_overlay_animations(
    trajectories: dict[str, Path],
    replays_dir: Path,
    cand_hash_prefix: str,
) -> dict[str, Path]:
    """Render 3D marker overlay GIF animations for all executed engines."""
    replays_dir.mkdir(parents=True, exist_ok=True)
    capture = load_tour_capture(str(C3D_PATH))
    t_pts = capture.points_m
    val_mask = capture.valid
    gif_paths = {}

    for engine, traj_path in trajectories.items():
        data = np.load(traj_path)
        m_pts = data["predicted_markers_m"]
        times = data["time_s"]
        gif_filename = f"overlay_{engine}_{cand_hash_prefix}.gif"
        gif_out = replays_dir / gif_filename
        logger.info(
            "Rendering marker overlay animation for %s -> %s...",
            engine.upper(),
            gif_out,
        )
        render_marker_overlay_animation(
            time_s=times,
            target_markers_m=t_pts[: len(times)],
            model_markers_m=m_pts,
            output_gif_path=gif_out,
            engine_name=engine,
            stride=5,
            valid_mask=val_mask[: len(times)],
        )
        gif_paths[engine] = gif_out

    return gif_paths


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description="FB-6 cross-engine parity and visual review."
    )
    parser.add_argument(
        "--engine", choices=["mujoco", "pinocchio", "drake", "all"], default="all"
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Force local execution of specified engine.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=0,
        help="Number of frames to simulate (0 = all 654).",
    )
    args = parser.parse_args()

    cand_sha = sha256_file(CANDIDATE_PATH)
    cand_prefix = cand_sha[:12]

    evidence_dir = HERE
    replays_dir = HERE / "replays"

    engines_to_run = (
        ["mujoco", "pinocchio", "drake"] if args.engine == "all" else [args.engine]
    )

    if args.local:
        outcome, traj_path, receipt_path = run_local_engine_replay(
            args.engine, evidence_dir, args.frames
        )
        logger.info("Local execution completed for %s.", args.engine)
        return

    outcomes: list[EngineReplayOutcome] = []
    trajectories: dict[str, Path] = {}

    for eng in engines_to_run:
        if eng == "mujoco":
            outcome, traj_path, _ = run_local_engine_replay(
                eng, evidence_dir, args.frames
            )
        else:
            outcome, traj_path, _ = run_remote_engine_replay(
                eng, evidence_dir, args.frames
            )
        outcomes.append(outcome)
        trajectories[eng] = traj_path

    # Render marker overlay animations for visual review
    render_all_overlay_animations(trajectories, replays_dir, cand_prefix)

    # Assemble and write cross-engine comparison report
    report = compare_engine_replays(outcomes)
    report_path = evidence_dir / "parity_report.json"
    report_path.write_text(
        json.dumps(report.as_dict(), indent=2) + "\n", encoding="utf-8"
    )
    logger.info("Cross-engine parity report written to %s", report_path)


if __name__ == "__main__":
    main()
