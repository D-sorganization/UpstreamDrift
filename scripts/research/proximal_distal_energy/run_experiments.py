"""Run the proximal-to-distal timing analyses.

Drives the backend-neutral double-pendulum golf model (ODE reference
backend) with parameterized torque programs, and records:

- E1: a wrist-torque timing sweep (onset time x profile x shoulder torque)
  scored by clubhead speed at impact;
- E2: pointwise ZTCF / drift-control decompositions along representative
  trajectories (via ``simulation_backends.ztcf_zvcf``);
- E4: Robertson-Winter style wrist-interface power accounting.

Outputs (all under ``docs/research/proximal_distal_energy_transfer/``):

- ``data/e1_sweep.json`` — every sweep row plus provenance
- ``data/representative_traces.npz`` — full traces for the representative
  swings used by the figure generator
- ``data/results_summary.json`` — headline numbers cited in the report

Usage::

    python3 -m scripts.research.proximal_distal_energy.run_experiments
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from scripts.research.proximal_distal_energy.swing_model import (
    PlanarInertials,
    clubhead_speed,
    find_impact,
    first_club_vertical_crossing,
    segment_kinetic_energies,
    wrist_interface_powers,
)
from scripts.research.proximal_distal_energy.torque_programs import (
    TorqueProgram,
    drive_only_program,
    passive_program,
    restrain_then_drive_program,
)
from src.shared.python.simulation_backends import GolfModelParams, make_backend
from src.shared.python.simulation_backends.protocol import DynamicsProvider, SimState
from src.shared.python.simulation_backends.ztcf_zvcf import drift_and_control_split

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = REPO_ROOT / "docs" / "research" / "proximal_distal_energy_transfer"
DATA_DIR = OUTPUT_ROOT / "data"

DT = 1.0e-3
HORIZON = 900
INITIAL_Q = (-2.2, -1.57)
SHOULDER_TORQUES = (60.0, 100.0)
WRIST_DRIVE = 15.0
RESTRAIN_LEVELS = (5.0, 10.0)
ONSET_GRID = tuple(np.round(np.arange(0.0, 0.351, 0.025), 4))


def _git_sha() -> str:
    """Best-effort short commit SHA for provenance stamping."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip()


def rollout_program(
    params: GolfModelParams, program: TorqueProgram
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Roll one torque program out from the top-of-backswing state.

    Returns ``(t, q, v, u)`` arrays with ``horizon + 1`` samples each.
    """
    controls = program.controls(HORIZON, DT)
    return rollout_controls(params, controls)


def rollout_controls(
    params: GolfModelParams,
    controls: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Roll out an explicit control history from the registered initial state."""
    control_array = np.asarray(controls, dtype=float)
    if control_array.shape != (HORIZON, 2):
        raise ValueError(f"controls must have shape ({HORIZON}, 2)")
    backend = make_backend("ode", params)
    backend.reset(SimState(q=np.array(INITIAL_Q, dtype=float), v=np.zeros(2), time=0.0))
    trace = backend.rollout(controls=control_array, horizon=HORIZON, dt=DT)
    u = trace.u if trace.u is not None else np.zeros_like(trace.v)
    return trace.t, trace.q, trace.v, u


def counterfactual_split(
    params: GolfModelParams, q: np.ndarray, v: np.ndarray, u: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Pointwise drift/control club-relevant accelerations along a trace.

    Returns ``(drift, control)`` arrays of shape ``(T, 2)`` in joint space;
    the club's absolute angular acceleration split is the row sum.
    """
    provider = make_backend("ode", params)
    if not isinstance(provider, DynamicsProvider):
        raise TypeError("the ODE backend must expose analytical dynamics primitives")
    drift = np.empty_like(v)
    control = np.empty_like(v)
    for k in range(q.shape[0]):
        drift[k], control[k] = drift_and_control_split(provider, q[k], v[k], u[k])
    return drift, control


def trace_accelerations(
    params: GolfModelParams,
    q: np.ndarray,
    v: np.ndarray,
    u: np.ndarray,
) -> np.ndarray:
    """Evaluate exact backend forward dynamics at every recorded sample."""
    if not (len(q) == len(v) == len(u)):
        raise ValueError("q, v, and u must have equal sample counts")
    backend = make_backend("ode", params)
    return np.asarray(
        [
            backend.forward_dynamics(qk, vk, uk)
            for qk, vk, uk in zip(q, v, u, strict=True)
        ],
        dtype=float,
    )


def _split_integrals(
    series: np.ndarray,
    t: np.ndarray,
    *,
    split: float,
) -> tuple[float, float]:
    """Integrate before/after an exact shared, interpolated split boundary."""
    values = np.asarray(series, dtype=float)
    times = np.asarray(t, dtype=float)
    if values.shape != times.shape:
        raise ValueError("series and t must be one-dimensional arrays of equal shape")
    if times.ndim != 1 or len(times) < 2 or np.any(np.diff(times) <= 0.0):
        raise ValueError("t must contain at least two strictly increasing samples")
    if not times[0] <= split <= times[-1]:
        raise ValueError("split must lie within the sampled interval")

    split_value = float(np.interp(split, times, values))
    before = times < split
    after = times > split
    early_t = np.concatenate((times[before], [split]))
    early_y = np.concatenate((values[before], [split_value]))
    late_t = np.concatenate(([split], times[after]))
    late_y = np.concatenate(([split_value], values[after]))
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is None:
        raise RuntimeError("NumPy must provide trapezoid integration")
    early = 0.0 if len(early_t) < 2 else float(trapezoid(early_y, early_t))
    late = 0.0 if len(late_t) < 2 else float(trapezoid(late_y, late_t))
    return early, late


def _phase_energy_budget(
    inertials: PlanarInertials,
    t: np.ndarray,
    q: np.ndarray,
    v: np.ndarray,
    qdd: np.ndarray,
    u: np.ndarray,
    t_impact: float,
) -> dict[str, float]:
    """Early-half / late-half energy accounting up to impact."""
    mask = t <= t_impact
    t_c, q_c, v_c, qdd_c, u_c = (
        t[mask],
        q[mask],
        v[mask],
        qdd[mask],
        u[mask],
    )
    half = t_impact / 2.0

    _, e_club = segment_kinetic_energies(inertials, q_c, v_c)
    powers = wrist_interface_powers(inertials, t_c, q_c, v_c, qdd_c, u_c)
    e_club_half = float(np.interp(half, t_c, e_club))

    shoulder_power = u_c[:, 0] * v_c[:, 0]
    wrist_power = powers["muscle_moment_power"]
    wrist_early, wrist_late = _split_integrals(wrist_power, t_c, split=half)
    shoulder_early, shoulder_late = _split_integrals(shoulder_power, t_c, split=half)
    force_early, force_late = _split_integrals(
        powers["joint_force_power"], t_c, split=half
    )
    return {
        "club_ke_gain_early_j": e_club_half - float(e_club[0]),
        "club_ke_gain_late_j": float(e_club[-1]) - e_club_half,
        "wrist_actuator_work_early_j": wrist_early,
        "wrist_actuator_work_late_j": wrist_late,
        "shoulder_work_early_j": shoulder_early,
        "shoulder_work_late_j": shoulder_late,
        "joint_force_transfer_early_j": force_early,
        "joint_force_transfer_late_j": force_late,
    }


def run_sweep(params: GolfModelParams, inertials: PlanarInertials) -> list[dict]:
    """E1: sweep wrist onset time x profile x shoulder torque."""
    rows: list[dict] = []
    for tau_s in SHOULDER_TORQUES:
        programs: list[TorqueProgram] = [passive_program(tau_s)]
        for onset in ONSET_GRID:
            programs.append(drive_only_program(tau_s, WRIST_DRIVE, float(onset)))
            for restrain in RESTRAIN_LEVELS:
                programs.append(
                    restrain_then_drive_program(
                        tau_s, WRIST_DRIVE, restrain, float(onset)
                    )
                )
        for program in programs:
            t, q, v, u = rollout_program(params, program)
            candidate = first_club_vertical_crossing(t, q, v, inertials)
            impact = find_impact(t, q, v, inertials)
            row: dict[str, Any] = {
                "profile": program.name.split("@")[0],
                "shoulder_torque_nm": program.shoulder_torque_nm,
                "wrist_drive_nm": program.wrist_drive_nm,
                "wrist_restrain_nm": program.wrist_restrain_nm,
                "onset_s": (None if np.isinf(program.onset_s) else program.onset_s),
            }
            if impact is None:
                status = (
                    "no_club_vertical_crossing"
                    if candidate is None
                    else "crossing_outside_registered_delivery_zone"
                )
                row.update(
                    {
                        "impact_status": status,
                        "t_impact_s": None,
                        "clubhead_speed_mps": None,
                        "theta1_at_impact_rad": None,
                        "candidate_t_crossing_s": (
                            None if candidate is None else candidate[0]
                        ),
                        "candidate_clubhead_speed_mps": (
                            None if candidate is None else candidate[1]
                        ),
                        "candidate_theta1_at_crossing_rad": (
                            None if candidate is None else candidate[2]
                        ),
                    }
                )
            else:
                row.update(
                    {
                        "impact_status": "accepted_registered_delivery_zone",
                        "t_impact_s": impact[0],
                        "clubhead_speed_mps": impact[1],
                        "theta1_at_impact_rad": impact[2],
                        "candidate_t_crossing_s": impact[0],
                        "candidate_clubhead_speed_mps": impact[1],
                        "candidate_theta1_at_crossing_rad": impact[2],
                    }
                )
            rows.append(row)
            logger.info("sweep row: %s", row)
    return rows


def _best_row(rows: list[dict], profile: str, tau_s: float) -> dict:
    candidates = [
        r
        for r in rows
        if r["profile"] == profile
        and r["shoulder_torque_nm"] == tau_s
        and r["clubhead_speed_mps"] is not None
    ]
    return max(candidates, key=lambda r: r["clubhead_speed_mps"])


def collect_representatives(
    params: GolfModelParams,
    inertials: PlanarInertials,
    rows: list[dict],
    tau_s: float,
) -> dict[str, dict]:
    """Full trace + analysis bundles for the representative swings."""
    best_drive = _best_row(rows, "drive_only", tau_s)
    best_restrain = _best_row(rows, "restrain_then_drive", tau_s)
    reps = {
        "passive": passive_program(tau_s),
        "early_drive": drive_only_program(tau_s, WRIST_DRIVE, 0.0),
        "best_drive": drive_only_program(
            tau_s, WRIST_DRIVE, float(best_drive["onset_s"])
        ),
        "best_restrain": restrain_then_drive_program(
            tau_s,
            WRIST_DRIVE,
            float(best_restrain["wrist_restrain_nm"]),
            float(best_restrain["onset_s"]),
        ),
    }
    bundles: dict[str, dict] = {}
    for label, program in reps.items():
        t, q, v, u = rollout_program(params, program)
        impact = find_impact(t, q, v, inertials)
        drift, control = counterfactual_split(params, q, v, u)
        qdd = drift + control
        e_arm, e_club = segment_kinetic_energies(inertials, q, v)
        powers = wrist_interface_powers(inertials, t, q, v, qdd, u)
        bundle = {
            "program": {
                "name": program.name,
                "shoulder_torque_nm": program.shoulder_torque_nm,
                "wrist_drive_nm": program.wrist_drive_nm,
                "wrist_restrain_nm": program.wrist_restrain_nm,
                "onset_s": (None if np.isinf(program.onset_s) else program.onset_s),
            },
            "t": t,
            "q": q,
            "v": v,
            "u": u,
            "drift": drift,
            "control": control,
            "e_arm": e_arm,
            "e_club": e_club,
            "clubhead_speed": clubhead_speed(inertials, q, v),
            "impact": impact,
            "powers": powers,
        }
        if impact is not None:
            bundle["budget"] = _phase_energy_budget(
                inertials, t, q, v, qdd, u, impact[0]
            )
        bundles[label] = bundle
    return bundles


def _save_outputs(rows: list[dict], bundles: dict[str, dict]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    provenance = {
        "git_sha": _git_sha(),
        "dt_s": DT,
        "horizon": HORIZON,
        "initial_q_rad": list(INITIAL_Q),
        "shoulder_torques_nm": list(SHOULDER_TORQUES),
        "wrist_drive_nm": WRIST_DRIVE,
        "restrain_levels_nm": list(RESTRAIN_LEVELS),
        "onset_grid_s": [float(x) for x in ONSET_GRID],
        "backend": "ode",
        "model": "GolfModelParams.default()",
    }
    with (DATA_DIR / "e1_sweep.json").open("w", encoding="utf-8") as fh:
        json.dump({"provenance": provenance, "rows": rows}, fh, indent=1)

    arrays: dict[str, np.ndarray] = {}
    summary: dict[str, Any] = {"provenance": provenance, "representatives": {}}
    for label, bundle in bundles.items():
        for key in ("t", "q", "v", "u", "drift", "control", "e_arm", "e_club"):
            arrays[f"{label}__{key}"] = bundle[key]
        arrays[f"{label}__clubhead_speed"] = bundle["clubhead_speed"]
        for pkey, series in bundle["powers"].items():
            arrays[f"{label}__power__{pkey}"] = series
        summary["representatives"][label] = {
            "program": bundle["program"],
            "impact": bundle["impact"],
            "budget": bundle.get("budget"),
        }
    np.savez_compressed(DATA_DIR / "representative_traces.npz", **arrays)

    valid = [r for r in rows if r["clubhead_speed_mps"] is not None]
    passive_rows = {
        r["shoulder_torque_nm"]: r for r in valid if r["profile"] == "passive"
    }
    summary["headline"] = {
        "passive_baselines": {
            str(k): {
                "t_impact_s": r["t_impact_s"],
                "clubhead_speed_mps": r["clubhead_speed_mps"],
            }
            for k, r in passive_rows.items()
        },
        "best_overall": max(valid, key=lambda r: r["clubhead_speed_mps"]),
        "worst_valid": min(valid, key=lambda r: r["clubhead_speed_mps"]),
    }
    with (DATA_DIR / "results_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)


def main() -> None:
    """Run all experiments and persist data for the figure generator."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    params = GolfModelParams.default()
    inertials = PlanarInertials.from_params(params)
    rows = run_sweep(params, inertials)
    bundles = collect_representatives(params, inertials, rows, SHOULDER_TORQUES[0])
    _save_outputs(rows, bundles)
    logger.info("wrote %s", DATA_DIR)


if __name__ == "__main__":
    main()
