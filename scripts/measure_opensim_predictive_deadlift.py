"""Measure OpenSim predictive Moco simulation of deadlift (OSL-7, #13044).

Usage (headless, from repository root, real OpenSim 4.6)::

    QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg MUJOCO_GL=egl SDL_VIDEODRIVER=dummy \\
        python3 scripts/measure_opensim_predictive_deadlift.py \\
        --out-dir docs/development/lifting/evidence/opensim_predictive

Executes staged predictive simulations of the barbell deadlift using OpenSim Moco:
- Stage 1: Torque-driven simulation (ModOpRemoveMuscles, CoordinateActuators)
- Stage 2: Muscle-driven simulation (only if Stage 1 converged)
All outputs are explicitly labelled exploratory.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from typing import Any

# Ensure headless matplotlib
os.environ["MPLBACKEND"] = "Agg"
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.engines.physics_engines.opensim.python.msk_lift_predictive import (  # noqa: E402
    TRACKED_SOLUTION_UNAVAILABLE_REASON,
    PredictiveDeadliftConfig,
    build_predictive_deadlift_study,
    compute_bundle_metrics,
    format_predictive_receipt,
    replay_bar_path_fk,
    solve_predictive_deadlift,
)

logger = logging.getLogger("measure_opensim_predictive_deadlift")
DEFAULT_OUT_DIR = ROOT / "docs/development/lifting/evidence/opensim_predictive"
SCRATCH_LOG_ENV = "AGY_OSL7_LOGS"
DEFAULT_LOG_DIR = (
    Path(os.environ[SCRATCH_LOG_ENV])
    if SCRATCH_LOG_ENV in os.environ
    else Path(tempfile.gettempdir()) / "osl7-logs"
)
BUNDLE_PATH = ROOT / "docs/development/lifting/evidence/deadlift/deadlift_bundle.npz"


def _git(*args: str) -> str | None:
    """Run git command and return stdout, or ``None`` when git is unusable.

    Container jobs mount a worktree whose ``.git`` points outside the
    container; there the commit comes from ``UPSTREAMDRIFT_SOURCE_COMMIT``.
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(ROOT), *args],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
    except (subprocess.CalledProcessError, OSError):
        return os.environ.get("UPSTREAMDRIFT_SOURCE_COMMIT")
    return out.stdout.strip()


@contextmanager
def _capture_output_to_file(file_path: Path):
    """Capture C and Python stdout and stderr to a log file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        stdout_fd = sys.stdout.fileno()
        stderr_fd = sys.stderr.fileno()
        saved_stdout = os.dup(stdout_fd)
        saved_stderr = os.dup(stderr_fd)
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(f.fileno(), stdout_fd)
        os.dup2(f.fileno(), stderr_fd)
        try:
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved_stdout, stdout_fd)
            os.dup2(saved_stderr, stderr_fd)
            os.close(saved_stdout)
            os.close(saved_stderr)


def _parse_solve_log(log_path: Path) -> dict[str, Any]:
    """Extract solve summary directly from Ipopt / MocoCasADi log output.

    Every field is parsed directly; unparsed fields remain None / nan.
    """
    if not log_path.is_file():
        return {
            "converged": False,
            "solver_status": "log_file_not_found",
            "iterations": 0,
            "objective": float("nan"),
            "constraint_violation": float("nan"),
            "unscaled_constraint_violation": None,
            "wall_clock_s": 0.0,
        }

    text = log_path.read_text(encoding="utf-8")
    if "Solve_Succeeded" in text:
        status = "Solve_Succeeded"
        converged = True
    elif "Maximum_Iterations_Exceeded" in text:
        status = "Maximum_Iterations_Exceeded"
        converged = False
    elif "Infeasible_Problem_Detected" in text:
        status = "Infeasible_Problem_Detected"
        converged = False
    elif "Restoration_Failed" in text:
        status = "Restoration_Failed"
        converged = False
    else:
        status = "Unknown_Or_Interrupted"
        converged = False

    iterations: int | None = None
    objective: float | None = None
    cv: float | None = None
    unscaled_cv: float | None = None
    wall_clock: float | None = None
    last_inf_pr: float | None = None

    m_iter = re.search(r"Number of Iterations\.\.\.\.:\s+(\d+)", text)
    if m_iter:
        iterations = int(m_iter.group(1))

    m_obj = re.search(
        r"Objective\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.:\s+([+-]?[\d.eE+-]+)", text
    )
    if m_obj:
        objective = float(m_obj.group(1))

    m_cv = re.search(
        r"Constraint violation\.\.\.\.:\s+([+-]?[\d.eE+-]+)\s+([+-]?[\d.eE+-]+)",
        text,
    )
    if m_cv:
        cv = float(m_cv.group(1))
        unscaled_cv = float(m_cv.group(2))

    m_time = re.search(r"Elapsed real time:\s*([\d.]+)\s*second", text)
    if m_time:
        wall_clock = float(m_time.group(1))

    if iterations is None:
        iter_matches = list(
            re.finditer(
                r"^\s*(\d+)\s+([+-]?[\d.eE+-]+)\s+([+-]?[\d.eE+-]+)\s+([+-]?[\d.eE+-]+)",
                text,
                re.MULTILINE,
            )
        )
        if iter_matches:
            last = iter_matches[-1]
            iterations = int(last.group(1))
            objective = float(last.group(2))
            last_inf_pr = float(last.group(3))
            status = "Timeout_Interrupted"

    return {
        "converged": converged,
        "solver_status": status,
        "iterations": iterations,
        "objective": objective,
        "constraint_violation": cv,
        "unscaled_constraint_violation": unscaled_cv,
        # IPOPT prints no final block when the run is killed: report the last
        # iterate's primal infeasibility (inf_pr) under its own name instead.
        "last_iterate_inf_pr": last_inf_pr,
        "wall_clock_s": wall_clock,
    }


def _plot_real_bar_path(
    out_path: Path,
    pred_times: np.ndarray,
    pred_bar_pos: np.ndarray,
    bundle_path: Path,
) -> None:
    """Plot real measured bar path against bundle kinematics."""
    from src.shared.python.motion_matching.same_input import InputBundle

    bundle = InputBundle.load(bundle_path)
    bundle_ik = np.asarray(bundle.provenance["ik_bar_path_m"], dtype=float)
    bundle_times = np.asarray(
        bundle.provenance["sagittal_solution"]["time_s"], dtype=float
    )

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    ax.plot(
        bundle_times,
        bundle_ik[:, 2],
        label="Bundle Kinematics (AN-1)",
        color="black",
        linestyle="--",
        linewidth=1.5,
    )
    ax.plot(
        pred_times,
        pred_bar_pos[:, 1],
        label="Predictive Moco (Stage 1)",
        color="#1f77b4",
        linewidth=2,
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Vertical Bar Position (m)")
    ax.set_title("Barbell Deadlift Vertical Bar Trajectory (OSL-7)")
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("Saved %s", out_path)


def _write_readme(out_dir: Path, receipt: dict[str, Any]) -> None:
    """Write documentation README for predictive deadlift evidence."""
    comp = receipt["comparison_table"]
    q = comp["quantities"]
    stage1 = receipt.get("stage1_torque", {})
    stage2 = receipt.get("stage2_muscle", {})

    readme_content = f"""# Predictive Moco Deadlift Simulation (OSL-7, #13044)

Predictive musculoskeletal simulation of a conventional barbell deadlift
using OpenSim Moco with foot contact, bar carried at hands, and effort goal.

## Result Status: Exploratory

### Stage 1: Torque-Driven Solve
- **Actuation:** `CoordinateActuator`s (1000 N·m optimal force) on free coordinates; pelvis residuals (optimal force 1, weight 100); muscles removed
- **Solver Status:** `{stage1.get("solver_status", "unknown")}`
- **Converged:** `{stage1.get("converged", False)}`
- **Iterations:** {stage1.get("iterations")}
- **Objective:** {stage1.get("objective", float("nan"))}
- **Scaled Constraint Violation:** {stage1.get("constraint_violation")} (last-iterate inf_pr: {stage1.get("last_iterate_inf_pr")})
- **Wall Clock Time:** {stage1.get("wall_clock_s")} s
- **Total Weight (m g):** {receipt.get("total_weight_n")} N
- **Residual Peak:** {receipt.get("residual_peak_n")} N
- **Residual RMS:** {receipt.get("residual_rms_n")} N
- **Residual Ratio:** {receipt.get("residual_ratio")}
- **Residual Assisted:** {receipt.get("residual_assisted")}

### Stage 2: Muscle-Driven Solve
- **Actuation:** 80 Rajagopal lower-limb muscles (DeGroote-Fregly 2016)
- **Status:** `{stage2.get("status", stage2.get("solver_status", "not run"))}`
- **Reason:** {stage2.get("reason", "N/A")}

## Kinematic Comparison

| Quantity | Bundle Kinematics | Predictive Solution | Unit |
| :--- | :--- | :--- | :--- |
| Bar Start Height | {q["bar_start_height_m"]["bundle_kinematics"]} | {q["bar_start_height_m"]["predictive"]} | m |
| Bar Final Height | {q["bar_final_height_m"]["bundle_kinematics"]} | {q["bar_final_height_m"]["predictive"]} | m |
| Bar Vertical Rise | {q["bar_vertical_rise_m"]["bundle_kinematics"]} | {q["bar_vertical_rise_m"]["predictive"]} | m |
| Peak Bar Velocity | {q["peak_vertical_velocity_m_s"]["bundle_kinematics"]} | {q["peak_vertical_velocity_m_s"]["predictive"]} | m/s |
| Lift Duration | {q["lift_duration_s"]["bundle_kinematics"]} | {q["lift_duration_s"]["predictive"]} | s |

### Tracked OpenSim Solution Status
- **Status:** `{comp["tracked_opensim"]["status"]}`
- **Reason:** {comp["tracked_opensim"]["reason"]}

## Reproduction Command

```bash
{receipt["reproduction_command"]}
```
"""
    readme_path = out_dir / "README.md"
    readme_path.write_text(readme_content, encoding="utf-8")
    logger.info("Saved %s", readme_path)


def _load_guess(path: Path) -> Any:
    """Load a saved Moco trajectory to warm-start a solve.

    Raises:
        ValueError: If *path* is not an existing file.
    """
    if not Path(path).is_file():
        raise ValueError(f"guess trajectory not found: {path}")
    from src.shared.python.engine_core.opensim_runtime import import_opensim

    return import_opensim().MocoTrajectory(str(path))


def run_stage1_solve(
    config: PredictiveDeadliftConfig,
    log_file: Path,
    guess_path: Path | None = None,
) -> dict[str, Any]:
    """Execute Stage 1 torque-driven solve with stdout/stderr captured to log_file.

    With *guess_path*, the solve is warm-started from that saved trajectory
    (for example an unconverged earlier run) instead of the smooth-step guess;
    the summary records the path under ``warm_start``.
    """
    logger.info("Building Stage 1 torque-driven study...")
    study, meta = build_predictive_deadlift_study(config)
    initial_guess = None if guess_path is None else _load_guess(guess_path)

    logger.info("Solving Stage 1 with Moco (log -> %s)...", log_file)
    with _capture_output_to_file(log_file):
        summary = solve_predictive_deadlift(
            config,
            study=study,
            initial_guess=initial_guess,
            setup_pose=meta.get("setup_pose"),
            lockout_pose=meta.get("lockout_pose"),
        )
    summary["warm_start"] = None if guess_path is None else str(guess_path)

    # Enhance summary from parsed log
    parsed = _parse_solve_log(log_file)
    summary["iterations"] = parsed["iterations"]
    for key in ("objective", "constraint_violation", "last_iterate_inf_pr"):
        if parsed[key] is not None:
            summary[key] = parsed[key]
    summary["unscaled_constraint_violation"] = parsed.get(
        "unscaled_constraint_violation"
    )
    return summary


def stage2_guess_from_stage1(study2: Any, stage1_solution: Any) -> Any:
    """Muscle-study guess seeded from the torque-driven stage-1 solution.

    The two problems differ (torque actuators vs muscles, activation states),
    so the stage-1 trajectory cannot be the guess directly.  The guess keeps
    stage 1's time grid, its coordinate states, and every control both
    problems share (the pelvis residuals and lumbar actuators); muscle
    activations and excitations keep the solver's default guess.
    """
    from src.shared.python.engine_core.opensim_runtime import import_opensim

    osim = import_opensim()
    solver = osim.MocoCasADiSolver.safeDownCast(study2.updSolver())
    guess = solver.createGuess()
    guess.resampleWithNumTimes(stage1_solution.getNumTimes())
    guess.setTime(stage1_solution.getTime())
    guess.insertStatesTrajectory(stage1_solution.exportToStatesTable(), True)
    stage1_controls = set(stage1_solution.getControlNames())
    for name in guess.getControlNames():
        if name in stage1_controls:
            values = stage1_solution.getControl(name).to_numpy()
            guess.setControl(name, osim.Vector(values))
    return guess


def _run_stage2_if_converged(
    stage1_summary: dict[str, Any],
    stage1_solution: Any,
    mesh_intervals: int,
    max_iterations: int,
    stage2_log: Path,
    out_dir: Path,
    residual: dict[str, Any],
) -> dict[str, Any]:
    """Execute Stage 2 muscle-driven solve only if Stage 1 converged."""
    if not stage1_summary.get("converged", False):
        return {
            "status": "not run",
            "reason": (
                f"Stage 1 torque-driven solve did not converge "
                f"({stage1_summary.get('solver_status', 'unknown')})"
            ),
        }

    logger.info("Stage 1 converged! Running Stage 2 muscle-driven solve...")
    cfg_stage2 = PredictiveDeadliftConfig(
        actuation_mode="muscle",
        mesh_intervals=mesh_intervals,
        max_iterations=max_iterations,
        convergence_tolerance=1e-2,
        constraint_tolerance=1e-2,
        duration_bounds=(0.5, 3.0),
        bar_mass_kg=100.0,
        body_mass_kg=80.0,
        height_m=1.78,
        effort_weight=1.0,
        terminal_lockout_weight=10.0,
        exploratory=True,
        **residual,
    )
    study2, _ = build_predictive_deadlift_study(cfg_stage2)
    guess2 = stage2_guess_from_stage1(study2, stage1_solution)
    with _capture_output_to_file(stage2_log):
        stage2_summary = solve_predictive_deadlift(
            cfg_stage2, study=study2, initial_guess=guess2
        )
    parsed2 = _parse_solve_log(stage2_log)
    stage2_summary["iterations"] = parsed2["iterations"]
    for key in ("objective", "constraint_violation", "last_iterate_inf_pr"):
        if parsed2[key] is not None:
            stage2_summary[key] = parsed2[key]
    stage2_solution = stage2_summary.pop("solution", None)
    if stage2_solution is not None:
        sto2_path = out_dir / "predictive_solution_stage2.sto"
        stage2_solution.unseal()
        stage2_solution.write(str(sto2_path))
    return stage2_summary


def _write_receipt_artifacts(out_dir: Path, receipt: dict[str, Any]) -> None:
    """Write receipts.json formatted with prettier and generate README."""
    receipt_path = out_dir / "receipts.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    logger.info("Saved %s", receipt_path)
    try:
        subprocess.run(
            ["npx", "--yes", "prettier@3", "--write", str(receipt_path)],
            check=False,
            capture_output=True,
            timeout=30,
        )
    except (subprocess.SubprocessError, OSError):
        pass
    _write_readme(out_dir, receipt)


def run_measurement(
    out_dir: Path,
    log_dir: Path = DEFAULT_LOG_DIR,
    mesh_intervals: int = 15,
    max_iterations: int = 3000,
    from_log: Path | None = None,
    residual: dict[str, float] | None = None,
    solver: dict[str, str] | None = None,
    stage1_guess: Path | None = None,
) -> dict[str, Any]:
    """Execute staged predictive deadlift measurement and generate honest evidence."""
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    residual = residual or {}
    options: dict[str, Any] = {**residual, **(solver or {})}
    stage1_log = log_dir / "stage1_torque.log"
    stage2_log = log_dir / "stage2_muscle.log"

    cfg_stage1 = PredictiveDeadliftConfig(
        actuation_mode="torque",
        torque_optimal_force_nm=1000.0,
        mesh_intervals=mesh_intervals,
        max_iterations=max_iterations,
        convergence_tolerance=1e-2,
        constraint_tolerance=1e-2,
        duration_bounds=(0.5, 3.0),
        bar_mass_kg=100.0,
        body_mass_kg=80.0,
        height_m=1.78,
        effort_weight=1.0,
        terminal_lockout_weight=10.0,
        exploratory=True,
        **options,
    )

    t_start = time.perf_counter()
    if from_log and from_log.is_file():
        logger.info("Parsing Stage 1 results from existing log: %s", from_log)
        stage1_summary = _parse_solve_log(from_log)
        stage1_summary["actuation_mode"] = "torque"
        stage1_solution = None
    else:
        stage1_summary = run_stage1_solve(cfg_stage1, stage1_log, stage1_guess)
        stage1_solution = stage1_summary.pop("solution", None)

    # Save Stage 1 solution file if available
    sto_path = out_dir / "predictive_solution_stage1.sto"
    if stage1_solution is not None:
        stage1_solution.unseal()
        stage1_solution.write(str(sto_path))
        if sto_path.is_file() and sto_path.stat().st_size > 2 * 1024 * 1024:
            scratch_sto = log_dir / "predictive_solution_stage1.sto"
            sto_path.rename(scratch_sto)
            logger.info("Solution exceeds 2 MB; moved to scratchpad: %s", scratch_sto)

    stage2_summary = _run_stage2_if_converged(
        stage1_summary,
        stage1_solution,
        mesh_intervals,
        max_iterations,
        stage2_log,
        out_dir,
        options,
    )

    # Format receipt
    receipt = format_predictive_receipt(
        cfg_stage1, stage1_summary, bundle_path=BUNDLE_PATH
    )
    receipt["stage1_torque"] = stage1_summary
    receipt["stage2_muscle"] = stage2_summary
    receipt["source_commit"] = _git("rev-parse", "HEAD")
    receipt["total_script_wall_clock_s"] = time.perf_counter() - t_start

    _write_receipt_artifacts(out_dir, receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for predictive deadlift measurement."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out", type=Path, default=None, help="Output receipts.json path"
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument(
        "--from-log",
        type=Path,
        default=None,
        help="Parse results from existing solve log",
    )
    parser.add_argument("--mesh-intervals", type=int, default=15)
    parser.add_argument("--max-iterations", type=int, default=3000)
    parser.add_argument(
        "--residual-force-scale",
        type=float,
        default=2.0,
        help="Pelvis residual optimal force as a multiple of total weight",
    )
    parser.add_argument("--residual-effort-weight", type=float, default=10.0)
    parser.add_argument(
        "--fd-scheme", choices=("forward", "central"), default="forward"
    )
    parser.add_argument(
        "--dynamics-mode", choices=("explicit", "implicit"), default="explicit"
    )
    parser.add_argument(
        "--stage1-guess",
        type=Path,
        default=None,
        help="Warm-start stage 1 from a saved .sto trajectory",
    )
    args = parser.parse_args(argv)

    out_dir = args.out_dir
    if args.out:
        out_dir = args.out.parent if args.out.suffix == ".json" else args.out

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    run_measurement(
        out_dir=out_dir,
        log_dir=args.log_dir,
        mesh_intervals=args.mesh_intervals,
        max_iterations=args.max_iterations,
        from_log=args.from_log,
        residual={
            "residual_force_scale": args.residual_force_scale,
            "residual_effort_weight": args.residual_effort_weight,
        },
        solver={
            "finite_difference_scheme": args.fd_scheme,
            "multibody_dynamics_mode": args.dynamics_mode,
        },
        stage1_guess=args.stage1_guess,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
