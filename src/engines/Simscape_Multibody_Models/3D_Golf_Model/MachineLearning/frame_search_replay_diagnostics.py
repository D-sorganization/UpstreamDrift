"""Replay diagnostics for the frame-search polynomial export pipeline.

This module addresses #3979: after frame-search polynomial export, drive the
existing polynomial replay path (when available) and emit the canonical
visual + numeric feedback loop already used for the surrogate path:

* target vs simulated club trajectory
* impact-window residuals
* torque effort / mechanical work proxies

The replay step is delegated to ``replay_matching_workflow.py`` if it exists
on the search path; otherwise we record the documented next step and continue.

A canonical Metrics record is emitted via the central Metrics module from
#4046 if it is importable; otherwise we fall back to a plain JSON sidecar.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

# numpy>=2.0 renamed trapz -> trapezoid; fall back for older versions.
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
if _trapz is None:  # pragma: no cover - both names always exist on supported numpy
    raise RuntimeError("numpy must provide trapezoid or trapz")

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_REPLAY_SCRIPT = SCRIPT_DIR / "replay_matching_workflow.py"
EVALUATE_SCRIPT = SCRIPT_DIR / "evaluate_matching_workflow.py"

CLUB_POSITION_COLUMNS = (
    "ClubLogs_CHGlobalPosition_1",
    "ClubLogs_CHGlobalPosition_2",
    "ClubLogs_CHGlobalPosition_3",
)
CLUB_VELOCITY_COLUMNS = (
    "ClubLogs_CHGlobalVelocity_1",
    "ClubLogs_CHGlobalVelocity_2",
    "ClubLogs_CHGlobalVelocity_3",
)
TARGET_POSITION_COLUMNS = ("clubface_x", "clubface_y", "clubface_z")
TARGET_VELOCITY_COLUMNS = ("clubface_vx", "clubface_vy", "clubface_vz")


@dataclass
class ReplayInputs:
    """Inputs required to drive a polynomial replay diagnostic run."""

    polynomial_mat: Path
    target_csv: Path
    simulated_club_csv: Path | None
    torque_csv: Path
    output_dir: Path
    impact_window: tuple[float, float] = (-0.02, 0.02)
    replay_script: Path | None = None


@dataclass
class ReplayDiagnostics:
    """Numeric output of the replay diagnostic run."""

    replay_executed: bool
    replay_command: list[str] = field(default_factory=list)
    replay_note: str = ""
    impact_window: tuple[float, float] = (-0.02, 0.02)
    position_error_rms: dict[str, float] = field(default_factory=dict)
    impact_position_error: dict[str, float] = field(default_factory=dict)
    velocity_error_rms: dict[str, float] = field(default_factory=dict)
    torque_effort: dict[str, float] = field(default_factory=dict)
    mechanical_work_available: bool = False
    plots: list[str] = field(default_factory=list)
    metrics_emitted_via: str = "json"


def _try_import_metrics_module() -> Any | None:
    """Attempt to import the canonical Metrics module from #4046."""
    candidates = (
        "ud.metrics",
        "upstream_drift.metrics",
        "src.metrics",
        "metrics",
    )
    for name in candidates:
        try:
            module = __import__(name, fromlist=["record_metric"])
        except Exception:  # noqa: BLE001 — best-effort discovery
            continue
        if hasattr(module, "record_metric") or hasattr(module, "emit"):
            return module
    return None


def _select_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    available = [c for c in columns if c in frame.columns]
    if not available:
        return np.zeros((len(frame), 0), dtype=float)
    return frame[list(available)].to_numpy(dtype=float)


def _interp_to(
    time_a: np.ndarray, time_b: np.ndarray, values: np.ndarray
) -> np.ndarray:
    if values.ndim == 1:
        return np.interp(time_a, time_b, values)
    out = np.empty((len(time_a), values.shape[1]), dtype=float)
    for col in range(values.shape[1]):
        out[:, col] = np.interp(time_a, time_b, values[:, col])
    return out


def run_replay(inputs: ReplayInputs) -> tuple[bool, list[str], str]:
    """Invoke ``replay_matching_workflow.py`` when present.

    Returns
    -------
    (executed, command, note)
        ``executed`` is True if the replay script ran. ``command`` is the
        argv that was (or would be) used. ``note`` is a human-readable
        explanation when the replay was skipped.
    """
    script = inputs.replay_script or DEFAULT_REPLAY_SCRIPT
    command = [
        sys.executable,
        str(script),
        "--polynomial-mat",
        str(inputs.polynomial_mat),
        "--target-csv",
        str(inputs.target_csv),
        "--torque-csv",
        str(inputs.torque_csv),
        "--output-dir",
        str(inputs.output_dir),
    ]
    if not script.exists():
        note = (
            f"Replay script {script} not found; documented next step: run "
            f"`{' '.join(command)}` once the script lands."
        )
        LOGGER.info(note)
        return (False, command, note)
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except (subprocess.CalledProcessError, OSError) as exc:
        note = f"Replay script invocation failed: {exc}"
        LOGGER.warning(note)
        return (False, command, note)
    return (True, command, "ok")


def compute_trajectory_residuals(
    target: pd.DataFrame,
    simulated: pd.DataFrame,
    impact_window: tuple[float, float],
) -> dict[str, dict[str, float]]:
    """Compute target-vs-simulated trajectory residuals.

    Both frames must contain a ``time`` column. The simulated frame is
    interpolated onto the target time grid for direct comparison.
    """
    if "time" not in target.columns or "time" not in simulated.columns:
        raise ValueError("Both target and simulated frames require a 'time' column")

    t_target = target["time"].to_numpy(dtype=float)
    t_sim = simulated["time"].to_numpy(dtype=float)

    target_pos = _select_columns(target, TARGET_POSITION_COLUMNS)
    target_vel = _select_columns(target, TARGET_VELOCITY_COLUMNS)
    sim_pos = _select_columns(simulated, CLUB_POSITION_COLUMNS)
    sim_vel = _select_columns(simulated, CLUB_VELOCITY_COLUMNS)

    pos_rms: dict[str, float] = {}
    vel_rms: dict[str, float] = {}
    impact_err: dict[str, float] = {}

    if target_pos.size and sim_pos.size and target_pos.shape[1] == sim_pos.shape[1]:
        sim_pos_on_target = _interp_to(t_target, t_sim, sim_pos)
        residual = sim_pos_on_target - target_pos
        for axis_index, axis in enumerate(("x", "y", "z")[: residual.shape[1]]):
            pos_rms[axis] = float(np.sqrt(np.mean(residual[:, axis_index] ** 2)))
        # Impact window: identified as (impact_window[0], impact_window[1])
        # relative to the impact reference (assumed t=0 by convention).
        mask = (t_target >= impact_window[0]) & (t_target <= impact_window[1])
        if mask.any():
            impact_residual = residual[mask]
            for axis_index, axis in enumerate(
                ("x", "y", "z")[: impact_residual.shape[1]]
            ):
                impact_err[axis] = float(np.max(np.abs(impact_residual[:, axis_index])))

    if target_vel.size and sim_vel.size and target_vel.shape[1] == sim_vel.shape[1]:
        sim_vel_on_target = _interp_to(t_target, t_sim, sim_vel)
        residual = sim_vel_on_target - target_vel
        for axis_index, axis in enumerate(("x", "y", "z")[: residual.shape[1]]):
            vel_rms[axis] = float(np.sqrt(np.mean(residual[:, axis_index] ** 2)))

    return {
        "position_rms": pos_rms,
        "velocity_rms": vel_rms,
        "impact_position_error": impact_err,
    }


def compute_torque_effort(torque_frame: pd.DataFrame) -> dict[str, float]:
    """Compute torque effort proxies: integrated |torque| and squared sum."""
    if "time" not in torque_frame.columns:
        raise ValueError("Torque frame requires a 'time' column")
    time = torque_frame["time"].to_numpy(dtype=float)
    if len(time) < 2:
        return {"effort_l1": 0.0, "effort_l2_sq": 0.0, "peak_abs": 0.0}
    torque_columns = [c for c in torque_frame.columns if c != "time"]
    if not torque_columns:
        return {"effort_l1": 0.0, "effort_l2_sq": 0.0, "peak_abs": 0.0}
    torque = torque_frame[torque_columns].to_numpy(dtype=float)
    effort_l1 = float(_trapz(np.einsum('ij->i', np.abs(torque)), time))
    effort_l2_sq = float(_trapz(np.einsum('ij,ij->i', torque, torque), time))
    peak = float(np.max(np.abs(torque)))
    return {"effort_l1": effort_l1, "effort_l2_sq": effort_l2_sq, "peak_abs": peak}


def _try_render_plots(
    target: pd.DataFrame,
    simulated: pd.DataFrame,
    output_dir: Path,
) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001 — matplotlib optional
        LOGGER.info("matplotlib unavailable, skipping plots: %s", exc)
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    plots: list[str] = []
    fig, ax = plt.subplots(figsize=(6, 4))
    if "time" in target.columns:
        for col in TARGET_POSITION_COLUMNS:
            if col in target.columns:
                ax.plot(target["time"], target[col], label=f"target {col[-1]}")
    if "time" in simulated.columns:
        for col in CLUB_POSITION_COLUMNS:
            if col in simulated.columns:
                ax.plot(
                    simulated["time"],
                    simulated[col],
                    "--",
                    label=f"sim {col[-1]}",
                )
    ax.set_xlabel("time (s)")
    ax.set_ylabel("club head position (m)")
    ax.legend(loc="best", fontsize=8)
    ax.set_title("Frame-search replay: target vs simulated club trajectory")
    plot_path = output_dir / "trajectory_comparison.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    plots.append(str(plot_path))
    return plots


def _emit_metrics(record: dict[str, Any], output_dir: Path) -> str:
    metrics_module = _try_import_metrics_module()
    if metrics_module is not None:
        emit_fn = getattr(metrics_module, "record_metric", None) or getattr(
            metrics_module, "emit", None
        )
        if emit_fn is not None:
            try:
                emit_fn("frame_search_replay", record)
                return f"module:{metrics_module.__name__}"
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Metrics module emit failed: %s", exc)
    output_dir.mkdir(parents=True, exist_ok=True)
    fallback = output_dir / "frame_search_replay_metrics.json"
    fallback.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return f"json:{fallback}"


def run_replay_diagnostics(
    inputs: ReplayInputs,
) -> ReplayDiagnostics:
    """End-to-end diagnostic for the frame-search polynomial replay path.

    This function:
        1. Calls ``replay_matching_workflow.py`` if available.
        2. Loads simulated club CSV (provided directly or from the replay run).
        3. Computes trajectory residuals, impact-window residuals, and torque
           effort metrics.
        4. Renders a basic plot when matplotlib is installed.
        5. Emits a canonical Metrics record (or JSON fallback).

    The function is defensive: missing simulated CSV produces a partial
    diagnostic with ``replay_executed=False`` rather than raising.
    """
    inputs.output_dir.mkdir(parents=True, exist_ok=True)
    executed, command, note = run_replay(inputs)

    diagnostics = ReplayDiagnostics(
        replay_executed=executed,
        replay_command=command,
        replay_note=note,
        impact_window=inputs.impact_window,
    )

    sim_csv = inputs.simulated_club_csv
    if sim_csv is None or not sim_csv.exists():
        LOGGER.warning(
            "Simulated club CSV unavailable (%s); skipping trajectory residuals",
            sim_csv,
        )
    else:
        target = pd.read_csv(inputs.target_csv)
        simulated = pd.read_csv(sim_csv)
        residuals = compute_trajectory_residuals(
            target, simulated, inputs.impact_window
        )
        diagnostics.position_error_rms = residuals["position_rms"]
        diagnostics.velocity_error_rms = residuals["velocity_rms"]
        diagnostics.impact_position_error = residuals["impact_position_error"]
        diagnostics.plots = _try_render_plots(target, simulated, inputs.output_dir)

    if inputs.torque_csv.exists():
        torque_frame = pd.read_csv(inputs.torque_csv)
        diagnostics.torque_effort = compute_torque_effort(torque_frame)
        diagnostics.mechanical_work_available = any(
            c.startswith("qdot") or c.endswith("_qdot") for c in torque_frame.columns
        )

    record = asdict(diagnostics)
    record["polynomial_mat"] = str(inputs.polynomial_mat)
    record["target_csv"] = str(inputs.target_csv)
    record["simulated_club_csv"] = str(sim_csv) if sim_csv else None
    record["torque_csv"] = str(inputs.torque_csv)
    diagnostics.metrics_emitted_via = _emit_metrics(record, inputs.output_dir)

    summary_path = inputs.output_dir / "frame_search_replay_summary.json"
    summary_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    if shutil.which("matlab") is None:
        LOGGER.info(
            "MATLAB not on PATH; replay diagnostics returning Python-only metrics."
        )
    return diagnostics


def main(argv: list[str] | None = None) -> int:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--polynomial-mat", type=Path, required=True)
    parser.add_argument("--target-csv", type=Path, required=True)
    parser.add_argument("--simulated-club-csv", type=Path)
    parser.add_argument("--torque-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replay-script", type=Path)
    parser.add_argument("--impact-start", type=float, default=-0.02)
    parser.add_argument("--impact-end", type=float, default=0.02)
    args = parser.parse_args(argv)

    diagnostics = run_replay_diagnostics(
        ReplayInputs(
            polynomial_mat=args.polynomial_mat,
            target_csv=args.target_csv,
            simulated_club_csv=args.simulated_club_csv,
            torque_csv=args.torque_csv,
            output_dir=args.output_dir,
            impact_window=(args.impact_start, args.impact_end),
            replay_script=args.replay_script,
        )
    )
    LOGGER.info("%s", json.dumps(asdict(diagnostics), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
