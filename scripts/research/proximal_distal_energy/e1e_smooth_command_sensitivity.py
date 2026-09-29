"""Test whether finite command rise time changes the E1 grid ordering.

This is a command-filter sensitivity study, not a muscle activation model and
not a continuous optimal-control solution. It preserves the preloaded command
at the top of the downswing and applies a declared first-order transition to
subsequent changes.
"""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scripts.research.proximal_distal_energy.run_experiments import (
    DATA_DIR,
    DT,
    HORIZON,
    ONSET_GRID,
    RESTRAIN_LEVELS,
    SHOULDER_TORQUES,
    WRIST_DRIVE,
    _git_sha,
    rollout_controls,
)
from scripts.research.proximal_distal_energy.swing_model import (
    PlanarInertials,
    find_impact,
    first_club_vertical_crossing,
)
from scripts.research.proximal_distal_energy.torque_programs import (
    TorqueProgram,
    drive_only_program,
    passive_program,
    restrain_then_drive_program,
)
from src.shared.python.simulation_backends import GolfModelParams

OUTPUT_ROOT = (
    Path(__file__).resolve().parents[3]
    / "docs/research/proximal_distal_energy_transfer"
)
FIG_DIR = OUTPUT_ROOT / "figures"
TIME_CONSTANTS_S = (0.0, 0.020, 0.035, 0.050)


def first_order_command_filter(
    controls: np.ndarray,
    *,
    dt: float,
    time_constant_s: float,
) -> np.ndarray:
    """Apply an exact-discrete first-order filter after the initial preload."""
    raw = np.asarray(controls, dtype=float)
    if raw.ndim != 2 or raw.shape[0] < 1:
        raise ValueError("controls must be a non-empty two-dimensional array")
    if dt <= 0.0 or time_constant_s < 0.0:
        raise ValueError("dt must be positive and time_constant_s nonnegative")
    if time_constant_s == 0.0:
        return raw.copy()
    alpha = 1.0 - float(np.exp(-dt / time_constant_s))
    filtered = np.empty_like(raw)
    filtered[0] = raw[0]
    for index in range(1, len(raw)):
        filtered[index] = filtered[index - 1] + alpha * (
            raw[index] - filtered[index - 1]
        )
    return filtered


def _programs(shoulder_torque_nm: float) -> list[TorqueProgram]:
    programs = [passive_program(shoulder_torque_nm)]
    for onset in ONSET_GRID:
        programs.append(
            drive_only_program(shoulder_torque_nm, WRIST_DRIVE, float(onset))
        )
        programs.extend(
            restrain_then_drive_program(
                shoulder_torque_nm,
                WRIST_DRIVE,
                restraint,
                float(onset),
            )
            for restraint in RESTRAIN_LEVELS
        )
    return programs


def evaluate_smooth_command_sensitivity() -> dict[str, Any]:
    """Run all 92 programs under four registered command time constants."""
    params = GolfModelParams.default()
    inertials = PlanarInertials.from_params(params)
    rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for time_constant in TIME_CONSTANTS_S:
        tau_key = f"{int(round(time_constant * 1000.0))}_ms"
        summaries[tau_key] = {}
        for shoulder_torque in SHOULDER_TORQUES:
            shoulder_rows: list[dict[str, Any]] = []
            for program in _programs(shoulder_torque):
                controls = first_order_command_filter(
                    program.controls(HORIZON, DT),
                    dt=DT,
                    time_constant_s=time_constant,
                )
                t, q, v, u = rollout_controls(params, controls)
                candidate = first_club_vertical_crossing(t, q, v, inertials)
                impact = find_impact(t, q, v, inertials)
                status = (
                    "accepted_registered_delivery_zone"
                    if impact is not None
                    else (
                        "no_club_vertical_crossing"
                        if candidate is None
                        else "crossing_outside_registered_delivery_zone"
                    )
                )
                row = {
                    "command_time_constant_s": time_constant,
                    "profile": program.name.split("@")[0],
                    "shoulder_torque_nm": shoulder_torque,
                    "wrist_restrain_nm": program.wrist_restrain_nm,
                    "onset_s": None if np.isinf(program.onset_s) else program.onset_s,
                    "impact_status": status,
                    "impact_speed_mps": None if impact is None else impact[1],
                    "candidate_t_crossing_s": (
                        None if candidate is None else candidate[0]
                    ),
                    "candidate_speed_mps": None if candidate is None else candidate[1],
                    "candidate_theta1_rad": None if candidate is None else candidate[2],
                }
                rows.append(row)
                shoulder_rows.append(row)

            valid = [
                row for row in shoulder_rows if row["impact_speed_mps"] is not None
            ]
            passive = next(row for row in valid if row["profile"] == "passive")
            early = next(
                row
                for row in valid
                if row["profile"] == "drive_only" and row["onset_s"] == 0.0
            )
            best_drive = max(
                (row for row in valid if row["profile"] == "drive_only"),
                key=lambda row: row["impact_speed_mps"],
            )
            best_restrain = max(
                (row for row in valid if row["profile"] == "restrain_then_drive"),
                key=lambda row: row["impact_speed_mps"],
            )
            ordered = [
                float(early["impact_speed_mps"]),
                float(passive["impact_speed_mps"]),
                float(best_drive["impact_speed_mps"]),
                float(best_restrain["impact_speed_mps"]),
            ]
            summaries[tau_key][f"shoulder_{int(shoulder_torque)}"] = {
                "attempted_programs": len(shoulder_rows),
                "valid_programs": len(valid),
                "early_drive_speed_mps": ordered[0],
                "passive_speed_mps": ordered[1],
                "grid_selected_drive": best_drive,
                "grid_selected_restrain": best_restrain,
                "registered_ordering_preserved": all(
                    earlier < later for earlier, later in pairwise(ordered)
                ),
            }

    return {
        "schema_version": "proximal-distal-e1e-smooth-command-v1",
        "git_sha": _git_sha(),
        "interpretation": "command_filter_sensitivity_not_muscle_or_optimal_control",
        "time_constants_s": list(TIME_CONSTANTS_S),
        "summaries": summaries,
        "rows": rows,
    }


def write_outputs(result: dict[str, Any]) -> None:
    """Write deterministic JSON and a compact publication figure."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "e1e_smooth_command_sensitivity.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), sharey=False)
    labels = (
        ("early_drive_speed_mps", "Early Drive", "#c0392b"),
        ("passive_speed_mps", "Passive Wrist", "#666666"),
        ("grid_selected_drive", "Grid-Selected Late Drive", "#1f77b4"),
        ("grid_selected_restrain", "Grid-Selected Restrain Then Drive", "#2ca02c"),
    )
    x = np.asarray(TIME_CONSTANTS_S) * 1000.0
    for ax, shoulder_torque in zip(axes, SHOULDER_TORQUES, strict=True):
        records = [
            result["summaries"][f"{int(round(tau * 1000.0))}_ms"][
                f"shoulder_{int(shoulder_torque)}"
            ]
            for tau in TIME_CONSTANTS_S
        ]
        for key, label, color in labels:
            values = [
                (
                    record[key]
                    if isinstance(record[key], float)
                    else record[key]["impact_speed_mps"]
                )
                for record in records
            ]
            ax.plot(x, values, marker="o", label=label, color=color)
        ax.set_title(f"Shoulder Torque {shoulder_torque:.0f} N·m")
        ax.set_xlabel("Command Time Constant [ms]")
        ax.set_ylabel("Clubhead Speed at Registered Crossing [m/s]")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle("Finite Command Rise-Time Sensitivity of the Tested E1 Programs")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_e1e_smooth_command_sensitivity.pdf")
    fig.savefig(FIG_DIR / "fig_e1e_smooth_command_sensitivity.svg")
    plt.close(fig)


def main() -> None:
    result = evaluate_smooth_command_sensitivity()
    write_outputs(result)
    print(DATA_DIR / "e1e_smooth_command_sensitivity.json")


if __name__ == "__main__":
    main()
