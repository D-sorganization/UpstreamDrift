"""OS-7: Moco horizon ladder to G1 and the full swing (MS-42 phase A, #10341).

Runs where ``import opensim`` succeeds (ControlTower ``opensim-10003`` venv).
For each rung of the ladder 0.1 -> 0.3 -> 0.6 -> 0.85 -> full capture:

1. Windows the tour TRC, drops sparse markers, trims trailing gaps and fills
   short interior gaps (Moco splines need finite references).
2. Solves a MocoStudy (marker + IK-state tracking, effort) on the scaled
   tour-marker model with normalised CoordinateActuators, warm-started from
   the previous rung's solution and the OS-3b IK beyond it.
3. Replays the fitted open-loop controls once, uninterrupted, through
   ``opensim.Manager`` and scores the five shared metrics on the ORIGINAL
   validity mask (filled samples are never scored).
4. Writes per-rung artefacts, a playback GIF and a receipt; the top-level
   receipt is the longest converged rung at or below G1 (0.85 s).

A kinematic (collocation) number is reported next to the replay number so the
divergence of the open-loop replay is visible; acceptance is the replay.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
import multiprocessing as mp
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.engines.physics_engines.opensim.python.tour_matching.moco_g1 import (  # noqa: E402
    assemble_warm_start,
    fill_marker_gaps,
    gate_table,
    horizon_ladder,
    mesh_intervals_for,
    read_sto,
    retain_markers,
    trim_trailing_invalid,
    validate_os7_receipt,
    window_capture,
    write_sto,
)
from src.engines.physics_engines.opensim.python.tour_matching.moco_tracking import (  # noqa: E402
    LadderRungConfig,
    apply_guess,
    build_rung_study,
    guess_grid,
    normalise_actuators,
    replay_uninterrupted,
    solution_markers,
    solve_rung,
)
from src.engines.physics_engines.opensim.python.tour_matching.trc import (  # noqa: E402
    read_trc,
    write_trc,
)
from src.engines.physics_engines.opensim.python.tour_matching.visualization import (  # noqa: E402
    animate_marker_overlay,
)
from src.shared.python.motion_matching.tour_capture_contract import (  # noqa: E402
    MARKER_SEGMENTS,
    TourCapture,
)
from src.shared.python.motion_matching.tour_metrics import (  # noqa: E402
    SharedMetrics,
    compute_shared_metrics,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("os7_moco_g1")

G1_S = 0.85
CAPTURE_RATE_HZ = 360.0
RESIDUAL_PREFIX = "tau_pelvis_"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True
        ).strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--trc", type=Path, required=True)
    parser.add_argument("--ik-states", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--max-iterations", type=int, default=300)
    parser.add_argument("--mesh-per-second", type=float, default=100.0)
    parser.add_argument("--max-gap-frames", type=int, default=40)
    parser.add_argument("--min-valid-fraction", type=float, default=0.9)
    parser.add_argument("--replay-timeout-s", type=float, default=1800.0)
    parser.add_argument("--rungs", type=float, nargs="*", default=None)
    parser.add_argument("--warm-start", type=Path, default=None)
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="warm-start the next rung from a capped (unconverged) trajectory",
    )
    return parser.parse_args()


def _prepare_reference(
    capture: TourCapture, horizon_s: float, args: argparse.Namespace, rung_dir: Path
) -> tuple[TourCapture, tuple[str, ...], dict[str, Any]]:
    """Window, prune, trim and gap-fill; returns (scored capture, labels, report)."""
    window = window_capture(capture, horizon_s)
    kept, sparse = retain_markers(window, args.min_valid_fraction)
    trimmed, dropped_tail = trim_trailing_invalid(window, kept)
    filled, gaps = fill_marker_gaps(trimmed, args.max_gap_frames)
    tracked, holes = retain_markers(filled.subset(kept), 1.0)
    write_trc(
        filled.subset(tracked), rung_dir / "reference.trc", rate_hz=CAPTURE_RATE_HZ
    )
    report = {
        "frames": trimmed.frames,
        "trailing_frames_dropped": dropped_tail,
        "sparse_markers_dropped": sparse,
        "gap_filled_frames": gaps,
        "markers_with_unfilled_holes_dropped": holes,
        "tracked_markers": list(tracked),
    }
    return trimmed, tracked, report


def _warm_start(
    study: Any,
    previous: tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]] | None,
    ik: tuple[np.ndarray, dict[str, np.ndarray]],
) -> tuple[str, np.ndarray, dict[str, np.ndarray]]:
    guess, grid, state_names, control_names = guess_grid(study)
    states, controls = assemble_warm_start(
        grid,
        state_names=state_names,
        control_names=control_names,
        previous=previous,
        ik=ik,
    )
    apply_guess(study, guess, states, controls)
    label = "previous rung + IK beyond" if previous is not None else "IK, zero controls"
    return label, grid, states


def _replay_worker(payload: dict[str, Any]) -> None:
    """Child-process body: uninterrupted replay written to ``payload['npz']``."""
    markers, values, wall = replay_uninterrupted(
        payload["model"],
        payload["controls"],
        payload["initial_states"],
        np.asarray(payload["sample_times"]),
        payload["labels"],
    )
    np.savez_compressed(
        payload["npz"],
        markers_m=markers,
        wall_clock_s=wall,
        coordinate_names=np.array(list(values)),
        coordinate_values=np.column_stack([values[n] for n in values]),
    )


def _replay_with_timeout(payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    """Run the replay in a child so a stalled integrator cannot hang the ladder."""
    process = mp.get_context("spawn").Process(target=_replay_worker, args=(payload,))
    process.start()
    process.join(timeout_s)
    if process.is_alive():
        process.terminate()
        process.join()
        return {"status": "timeout", "timeout_s": timeout_s}
    if process.exitcode != 0 or not Path(payload["npz"]).is_file():
        return {"status": "failed", "exit_code": process.exitcode}
    with np.load(payload["npz"]) as data:
        return {
            "status": "ok",
            "markers": data["markers_m"],
            "wall_clock_s": float(data["wall_clock_s"]),
            "names": list(data["coordinate_names"]),
            "values": data["coordinate_values"],
        }


def _safe_metrics(capture: TourCapture, markers: np.ndarray) -> dict[str, Any]:
    try:
        metrics = compute_shared_metrics(capture, markers)
    except ValueError as error:  # non-finite replay markers = divergence
        return {"metrics": None, "gates": None, "error": str(error)}
    return {"metrics": metrics.as_dict(), "gates": gate_table(metrics)}


def _residual_effort(controls: dict[str, np.ndarray], forces: dict[str, float]) -> dict:
    """RMS actuator effort (N m or N) split into pelvis residuals and joints."""
    out: dict[str, float] = {}
    for name, column in controls.items():
        short = name.rsplit("/", 1)[-1]
        out[short] = float(np.sqrt(np.mean(column**2)) * forces.get(short, 1.0))
    residual = {k: v for k, v in out.items() if k.startswith(RESIDUAL_PREFIX)}
    return {
        "pelvis_residual_rms": residual,
        "pelvis_residual_rms_max": max(residual.values()) if residual else 0.0,
        "joint_rms_max": max(
            (v for k, v in out.items() if k not in residual), default=0.0
        ),
    }


def _evaluate_rung(
    scored: TourCapture,
    labels: tuple[str, ...],
    rung_dir: Path,
    moco_model: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Collocation metrics, uninterrupted replay metrics, replay.mot and GIF."""
    states_t, states = read_sto(rung_dir / "states.sto")
    sol = solution_markers(moco_model, states_t, states, scored.time_s, labels)
    evaluation: dict[str, Any] = {"solution": _safe_metrics(scored, sol)}
    payload = {
        "model": str(moco_model),
        "controls": str(rung_dir / "controls.sto"),
        "initial_states": {n: float(c[0]) for n, c in states.items()},
        "sample_times": scored.time_s.tolist(),
        "labels": list(labels),
        "npz": str(rung_dir / "replay_markers.npz"),
    }
    replay = _replay_with_timeout(payload, args.replay_timeout_s)
    evaluation["replay_status"] = replay["status"]
    if replay["status"] != "ok":
        evaluation["replay"] = {"metrics": None, "gates": None, "error": replay}
        return evaluation
    evaluation["replay"] = _safe_metrics(scored, replay["markers"])
    evaluation["replay_wall_clock_s"] = replay["wall_clock_s"]
    columns = {n: replay["values"][:, i] for i, n in enumerate(replay["names"])}
    write_sto(rung_dir / "replay.mot", scored.time_s, columns, in_degrees=False)
    finite = np.isfinite(replay["markers"]).all(axis=(1, 2))
    if finite.any():
        stride = max(1, scored.frames // 60)
        animate_marker_overlay(
            scored.points_m,
            np.nan_to_num(replay["markers"]),
            scored.valid & finite[:, None],
            scored.time_s,
            rung_dir / "playback.gif",
            stride=stride,
            title=f"OS-7 Moco replay {scored.time_s[-1]:.3f} s",
        )
    return evaluation


def _run_rung(
    capture: TourCapture,
    horizon_s: float,
    context: dict[str, Any],
    previous: tuple | None,
) -> tuple[dict[str, Any], tuple | None]:
    args: argparse.Namespace = context["args"]
    warm_unconverged = bool(context.get("warm_unconverged", False))
    rung_dir = args.outdir / "rungs" / f"{int(round(horizon_s * 1000)):04d}ms"
    rung_dir.mkdir(parents=True, exist_ok=True)
    scored, labels, reference = _prepare_reference(capture, horizon_s, args, rung_dir)
    config = LadderRungConfig(
        horizon_s=float(scored.time_s[-1]),
        mesh_intervals=mesh_intervals_for(
            float(scored.time_s[-1]), args.mesh_per_second
        ),
        max_iterations=args.max_iterations,
    )
    club = set(MARKER_SEGMENTS["club"])
    weights = {
        label: (config.club_marker_weight if label in club else 1.0) for label in labels
    }
    study = build_rung_study(
        context["moco_model"],
        rung_dir / "reference.trc",
        args.ik_states,
        config,
        weights,
    )
    warm, grid, guess_states = _warm_start(study, previous, context["ik"])
    guess_markers = solution_markers(
        context["moco_model"], grid, guess_states, scored.time_s, capture.labels
    )
    logger.info(
        "Rung %.3f s: %d mesh intervals, warm start %s",
        config.horizon_s,
        config.mesh_intervals,
        warm,
    )
    solve = solve_rung(study, rung_dir)
    logger.info(
        "Rung %.3f s: %s after %d iterations in %.0f s",
        config.horizon_s,
        solve["status"],
        solve["iterations"],
        solve["wall_clock_s"],
    )
    controls_t, controls = read_sto(rung_dir / "controls.sto")
    evaluation = _evaluate_rung(
        scored, capture.labels, rung_dir, context["moco_model"], args
    )
    receipt = {
        "deliverable": "OS-7",
        "issue": "#10341 (MS-42 phase A)",
        "horizon_s": config.horizon_s,
        "requested_horizon_s": horizon_s,
        "mesh_intervals": config.mesh_intervals,
        "config": config.__dict__,
        "warm_start": warm,
        "guess_metrics": _safe_metrics(scored, guess_markers),
        "reference": reference,
        "marker_weights": weights,
        "solver_status": solve["status"],
        "solver_success": solve["success"],
        "iterations": solve["iterations"],
        "wall_clock_s": solve["wall_clock_s"],
        "objective": solve["objective"],
        "effort": _residual_effort(controls, context["forces"]),
        "solution_metrics": evaluation["solution"],
        "replay_status": evaluation["replay_status"],
        "replay_metrics": evaluation["replay"],
        "replay_wall_clock_s": evaluation.get("replay_wall_clock_s"),
    }
    (rung_dir / "receipt.json").write_text(
        json.dumps(receipt, indent=2, default=str) + "\n"
    )
    states_t, states = read_sto(rung_dir / "states.sto")
    usable = solve["success"] or args.continue_on_failure
    receipt["warm_start_from_unconverged"] = bool(
        previous is not None and warm_unconverged
    )
    next_previous = (states_t, states, controls) if usable else None
    return receipt, next_previous


def _row(receipt: dict[str, Any]) -> dict[str, Any]:
    replay = receipt["replay_metrics"]["metrics"] or {}
    solution = receipt["solution_metrics"]["metrics"] or {}
    return {
        "horizon_s": receipt["horizon_s"],
        "mesh_intervals": receipt["mesh_intervals"],
        "solver_status": receipt["solver_status"],
        "iterations": receipt["iterations"],
        "wall_clock_s": round(receipt["wall_clock_s"], 1),
        "solution_whole_mm": _mm(solution.get("whole_marker_rmse_m")),
        "replay_status": receipt["replay_status"],
        "replay_whole_mm": _mm(replay.get("whole_marker_rmse_m")),
        "replay_early_mm": _mm(replay.get("early_marker_rmse_m")),
        "replay_terminal_mm": _mm(replay.get("terminal_marker_rmse_m")),
        "replay_club_mm": _mm(replay.get("club_marker_rmse_m")),
        "replay_pelvis_yaw_deg": (
            None
            if replay.get("pelvis_yaw_rmse_rad") is None
            else round(float(np.degrees(replay["pelvis_yaw_rmse_rad"])), 2)
        ),
        "gates_all_passed": (receipt["replay_metrics"]["gates"] or {}).get(
            "all_passed"
        ),
    }


def _mm(value: float | None) -> float | None:
    return None if value is None else round(1000.0 * float(value), 1)


def _write_top_level(receipts: list[dict[str, Any]], context: dict[str, Any]) -> Path:
    """Top-level receipt = longest converged rung at or below G1, plus the table."""
    args: argparse.Namespace = context["args"]
    converged = [
        r for r in receipts if r["solver_success"] and r["horizon_s"] <= G1_S + 1e-6
    ]
    headline = converged[-1] if converged else receipts[-1]
    document = dict(headline)
    document.update(
        {
            "title": "OpenSim Moco G1 dynamic tracking of the tour driver swing",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit(),
            "model_path": str(args.model),
            "model_sha256": sha(args.model),
            "moco_model_sha256": sha(context["moco_model"]),
            "trc_sha256": sha(args.trc),
            "ik_states_sha256": sha(args.ik_states),
            "actuator_optimal_forces": context["forces"],
            "gates": headline["replay_metrics"]["gates"],
            "per_horizon": [_row(r) for r in receipts],
            "g1_target_s": G1_S,
            "headline_is_g1": abs(headline["horizon_s"] - G1_S) < 0.02,
            "qualification": (
                "Moco direct collocation on the OS-3b scaled tour-marker model; the "
                "five metrics under replay_metrics come from one uninterrupted "
                "open-loop replay of the fitted controls. Ground reaction is inferred "
                "through pelvis residual actuators (no contact model, no force plates); "
                "acceptance requires replay_metrics.gates.all_passed, nothing else."
            ),
        }
    )
    validate_os7_receipt(document)
    (args.outdir / "receipt.json").write_text(
        json.dumps(document, indent=2, default=str) + "\n"
    )
    rung_dir = (
        args.outdir
        / "rungs"
        / f"{int(round(headline['requested_horizon_s'] * 1000)):04d}ms"
    )
    for name in ("solution.sto", "replay.mot", "playback.gif"):
        if (rung_dir / name).is_file():
            shutil.copyfile(rung_dir / name, args.outdir / name)
    return args.outdir / "receipt.json"


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    capture = read_trc(args.trc)
    ladder = horizon_ladder(float(capture.time_s[-1]))
    if args.rungs:
        ladder = tuple(h for h in ladder if any(abs(h - r) < 1e-6 for r in args.rungs))
    moco_model = args.outdir / "golf_humanoid_scaled_tour_markers_moco.osim"
    forces = normalise_actuators(args.model, moco_model, LadderRungConfig(1.0, 1))
    context = {
        "args": args,
        "moco_model": moco_model,
        "forces": forces,
        "ik": read_sto(args.ik_states),
    }
    previous = None
    if args.warm_start is not None:
        st_t, st = read_sto(args.warm_start / "states.sto")
        _, ct = read_sto(args.warm_start / "controls.sto")
        previous = (st_t, st, ct)
    receipts: list[dict[str, Any]] = []
    for horizon_s in ladder:
        receipt, previous = _run_rung(capture, horizon_s, context, previous)
        receipts.append(receipt)
        _write_top_level(receipts, context)
        context["warm_unconverged"] = not receipt["solver_success"]
        if previous is None:
            logger.warning("Rung %.3f s did not converge; ladder stops here", horizon_s)
            break
    logger.info(
        "Per-horizon table: %s", json.dumps([_row(r) for r in receipts], indent=1)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
