"""Generate the double-pendulum interaction-force evidence package."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from scripts.research.proximal_distal_energy.interaction_forces import (
    force_power_decomposition,
    matched_state_killswitch,
    reaction_force_decomposition,
)
from scripts.research.proximal_distal_energy.run_experiments import rollout_program
from scripts.research.proximal_distal_energy.swing_model import (
    PlanarInertials,
    find_impact,
)
from scripts.research.proximal_distal_energy.torque_programs import (
    restrain_then_drive_program,
)
from src.shared.python.simulation_backends import GolfModelParams, make_backend

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = REPO_ROOT / "docs" / "research" / "proximal_distal_energy_transfer"
DATA_DIR = OUTPUT_ROOT / "data"
DT = 1.0e-3


def _git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _integral(series: np.ndarray, t: np.ndarray, mask: np.ndarray) -> float:
    return float(np.trapezoid(series[mask], t[mask]))


def build_evidence() -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Compute force, power, geometry, and matched-state evidence."""
    params = GolfModelParams.default()
    inertials = PlanarInertials.from_params(params)
    program = restrain_then_drive_program(60.0, 15.0, 10.0, 0.10)
    t, q, v, u = rollout_program(params, program)
    impact = find_impact(t, q, v, inertials)
    if impact is None:
        raise ValueError("Reference program did not reach a valid impact")
    impact_index = int(np.searchsorted(t, impact[0], side="right") - 1)

    backend = make_backend("ode", params)
    qdd = np.vstack(
        [
            backend.forward_dynamics(qk, vk, uk)
            for qk, vk, uk in zip(q, v, u, strict=True)
        ]
    )
    drift_qdd = np.vstack(
        [
            backend.forward_dynamics(qk, vk, np.zeros(2))
            for qk, vk in zip(q, v, strict=True)
        ]
    )
    forces = reaction_force_decomposition(inertials, q, v, qdd)
    drift_forces = reaction_force_decomposition(inertials, q, v, drift_qdd)
    control_force = forces.total - drift_forces.total
    powers = force_power_decomposition(inertials, q, v, forces)
    drift_powers = force_power_decomposition(inertials, q, v, drift_forces)

    cut_index = int(round(0.72 * impact_index))
    horizon = min(120, t.size - cut_index - 1)
    killswitch = matched_state_killswitch(
        params,
        t,
        q,
        v,
        u,
        cut_index=cut_index,
        horizon=horizon,
        dt=DT,
    )
    zero_qdd = np.vstack(
        [
            backend.forward_dynamics(qk, vk, np.zeros(2))
            for qk, vk in zip(
                killswitch.zero_torque.q, killswitch.zero_torque.v, strict=True
            )
        ]
    )
    zero_forces = reaction_force_decomposition(
        inertials,
        killswitch.zero_torque.q,
        killswitch.zero_torque.v,
        zero_qdd,
    )

    theta2_grid = np.linspace(-np.pi, np.pi, 721)
    arrays: dict[str, np.ndarray] = {
        "t": t,
        "q": q,
        "v": v,
        "u": u,
        "qdd": qdd,
        "force_total": forces.total,
        "force_drift": drift_forces.total,
        "force_control": control_force,
        "power_total": powers.total,
        "power_drift": drift_powers.total,
        "theta2_grid": theta2_grid,
        "geometry_distal_tangential": np.cos(theta2_grid),
        "geometry_distal_centripetal": -np.sin(theta2_grid),
        "killswitch_t": killswitch.commanded.t + t[cut_index],
        "killswitch_commanded_q": killswitch.commanded.q,
        "killswitch_commanded_v": killswitch.commanded.v,
        "killswitch_zero_q": killswitch.zero_torque.q,
        "killswitch_zero_v": killswitch.zero_torque.v,
        "killswitch_zero_force": zero_forces.total,
    }
    for name, values in forces.components.items():
        arrays[f"force_component__{name}"] = values
    for name, values in powers.components.items():
        arrays[f"power_component__{name}"] = values

    to_impact = np.arange(t.size) <= impact_index
    late = to_impact & (t >= 0.5 * impact[0])
    positive = np.maximum(powers.total, 0.0)
    negative = np.minimum(powers.total, 0.0)
    summary: dict[str, Any] = {
        "provenance": {
            "git_sha": _git_sha(),
            "backend": "ode",
            "integrator": "fixed-step RK4",
            "dt_s": DT,
            "model": "GolfModelParams.default()",
            "program": program.name,
            "force_frame": "swing-plane Cartesian: x target-side, y upward",
            "counterfactual_contract": {
                "pointwise": "state held fixed; acceleration recomputed at zero torque",
                "killswitch": "commanded and zero-torque futures integrated from one matched state",
            },
        },
        "impact": {
            "time_s": impact[0],
            "clubhead_speed_m_s": impact[1],
            "index_before_crossing": impact_index,
        },
        "force": {
            "peak_total_n_to_impact": float(
                np.max(np.linalg.norm(forces.total[to_impact], axis=1))
            ),
            "peak_drift_n_to_impact": float(
                np.max(np.linalg.norm(drift_forces.total[to_impact], axis=1))
            ),
            "rms_control_fraction_to_impact": float(
                np.sqrt(np.mean(np.sum(control_force[to_impact] ** 2, axis=1)))
                / np.sqrt(np.mean(np.sum(forces.total[to_impact] ** 2, axis=1)))
            ),
        },
        "transfer": {
            "net_work_to_impact_j": _integral(powers.total, t, to_impact),
            "net_work_late_half_j": _integral(powers.total, t, late),
            "positive_work_to_impact_j": _integral(positive, t, to_impact),
            "negative_work_to_impact_j": _integral(negative, t, to_impact),
            "drift_work_to_impact_j": _integral(drift_powers.total, t, to_impact),
        },
        "killswitch": {
            "cut_time_s": float(t[cut_index]),
            "cut_index": cut_index,
            "horizon_s": horizon * DT,
            "terminal_q_separation_rad": float(
                np.linalg.norm(
                    killswitch.commanded.q[-1] - killswitch.zero_torque.q[-1]
                )
            ),
            "terminal_v_separation_rad_s": float(
                np.linalg.norm(
                    killswitch.commanded.v[-1] - killswitch.zero_torque.v[-1]
                )
            ),
        },
    }
    return arrays, summary


def main() -> None:
    """Write deterministic NPZ and JSON evidence artifacts."""
    arrays, summary = build_evidence()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DATA_DIR / "interaction_force_mechanisms.npz", **arrays)
    with (DATA_DIR / "interaction_force_summary.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
