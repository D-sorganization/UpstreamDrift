"""Run the matched-state pointwise-versus-killswitch evidence ensemble."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from scripts.research.proximal_distal_energy.counterfactual_ensemble import (
    evaluate_killswitch_case,
    evaluate_killswitch_ensemble,
)
from scripts.research.proximal_distal_energy.run_experiments import rollout_program
from scripts.research.proximal_distal_energy.swing_model import PlanarInertials
from scripts.research.proximal_distal_energy.torque_programs import (
    restrain_then_drive_program,
)
from src.shared.python.simulation_backends import GolfModelParams

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = REPO_ROOT / "docs" / "research" / "proximal_distal_energy_transfer"
DATA_DIR = OUTPUT_ROOT / "data"
CUT_TIMES = (0.080, 0.099, 0.100, 0.101, 0.160, 0.220, 0.280, 0.320)
HORIZONS = (0.020, 0.040, 0.080, 0.120)
TIMESTEPS = (0.0005, 0.001, 0.002)
PROGRAM = restrain_then_drive_program(60.0, 15.0, 10.0, 0.10)
SCHEMA_VERSION = "matched-state-counterfactual-ensemble-v2"
JSON_PATH = DATA_DIR / "counterfactual_ensemble.json"
NPZ_PATH = DATA_DIR / "counterfactual_selected_traces.npz"


def _source_hashes() -> dict[str, str]:
    """Hash the complete declared computational and WSCG input closure."""
    paths = (
        "docs/research/proximal_distal_energy_transfer/data/wscg_2024_hand_force_series.csv",
        "scripts/research/proximal_distal_energy/counterfactual_ensemble.py",
        "scripts/research/proximal_distal_energy/double_pendulum_attribution.py",
        "scripts/research/proximal_distal_energy/interaction_forces.py",
        "scripts/research/proximal_distal_energy/run_counterfactual_ensemble.py",
        "scripts/research/proximal_distal_energy/run_experiments.py",
        "scripts/research/proximal_distal_energy/swing_model.py",
        "scripts/research/proximal_distal_energy/torque_programs.py",
        "src/engines/pendulum_models/python/double_pendulum_model/physics/double_pendulum.py",
        "src/shared/python/simulation_backends/factory.py",
        "src/shared/python/simulation_backends/model_params.py",
        "src/shared/python/simulation_backends/ode_backend.py",
        "src/shared/python/simulation_backends/protocol.py",
    )
    return {
        path: hashlib.sha256((REPO_ROOT / path).read_bytes()).hexdigest()
        for path in paths
    }


def _wscg_delta_consistency() -> dict[str, dict[str, float]]:
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    csv_path = DATA_DIR / "wscg_2024_hand_force_series.csv"
    with csv_path.open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            grouped[row["series"]].append((float(row["time_s"]), float(row["value"])))
    pairs = (
        ("LeadHandAxial", "LeadHandCFAxial", "LHDiffAxial"),
        ("LeadHandNormal", "LeadHandCFNormal", "LHDiffNormal"),
        ("TrailHeadAxial", "TrailHandCFAxial", "THDiffAxial"),
        ("TrailHandNormal", "TrailHandCFNormal", "THDiffNormal"),
    )
    results: dict[str, dict[str, float]] = {}
    for base_name, counterfactual_name, delta_name in pairs:
        base_t, base_y = map(np.asarray, zip(*grouped[base_name], strict=True))
        cf_t, cf_y = map(np.asarray, zip(*grouped[counterfactual_name], strict=True))
        delta_t, delta_y = map(np.asarray, zip(*grouped[delta_name], strict=True))
        reconstructed = np.interp(delta_t, base_t, base_y) - np.interp(
            delta_t, cf_t, cf_y
        )
        residual = reconstructed - delta_y
        results[delta_name] = {
            "max_abs_residual_n": float(np.max(np.abs(residual))),
            "rmse_n": float(np.sqrt(np.mean(residual**2))),
        }
    return results


def _variant_params() -> dict[str, GolfModelParams]:
    baseline = GolfModelParams.default()
    return {
        "baseline": baseline,
        "gravity_disabled": baseline.model_copy(update={"gravity_enabled": False}),
        "damping_disabled": baseline.model_copy(
            update={"damping_shoulder": 0.0, "damping_wrist": 0.0}
        ),
    }


def _selected_trace_arrays(
    params: GolfModelParams, t: np.ndarray, q: np.ndarray, v: np.ndarray
) -> dict[str, np.ndarray]:
    inertials = PlanarInertials.from_params(params)
    arrays: dict[str, np.ndarray] = {}
    for cut_time in (0.12, 0.22, 0.30):
        q0 = np.array([np.interp(cut_time, t, q[:, j]) for j in range(2)])
        v0 = np.array([np.interp(cut_time, t, v[:, j]) for j in range(2)])
        case = evaluate_killswitch_case(
            params,
            inertials,
            PROGRAM,
            source_time_s=cut_time,
            source_q=q0,
            source_v=v0,
            horizon_s=0.08,
            dt_s=0.001,
        )
        prefix = f"cut_{cut_time:.2f}"
        arrays[f"{prefix}__time"] = case.commanded.t + cut_time
        arrays[f"{prefix}__commanded_q"] = case.commanded.q
        arrays[f"{prefix}__zero_q"] = case.zero_torque.q
        arrays[f"{prefix}__commanded_v"] = case.commanded.v
        arrays[f"{prefix}__zero_v"] = case.zero_torque.v
        arrays[f"{prefix}__commanded_force"] = case.commanded_force
        arrays[f"{prefix}__zero_force"] = case.zero_torque_force
        arrays[f"{prefix}__commanded_power"] = case.commanded_force_power
        arrays[f"{prefix}__zero_power"] = case.zero_torque_force_power
    return arrays


def _positive_zero_command_power_duration_s(arrays: dict[str, np.ndarray]) -> float:
    """Return the longest initial positive-power duration for selected cuts."""
    durations: list[float] = []
    for cut_time in (0.12, 0.22, 0.30):
        prefix = f"cut_{cut_time:.2f}"
        time = arrays[f"{prefix}__time"]
        power = arrays[f"{prefix}__zero_power"]
        nonpositive = np.flatnonzero(power <= 0.0)
        end = int(nonpositive[0]) if nonpositive.size else power.size - 1
        durations.append(float(time[end] - time[0]))
    return max(durations)


def build_outputs() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Return the JSON evidence record and representative trace arrays."""
    params = GolfModelParams.default()
    inertials = PlanarInertials.from_params(params)
    t, q, v, _ = rollout_program(params, PROGRAM)
    rows = evaluate_killswitch_ensemble(
        params,
        inertials,
        PROGRAM,
        t,
        q,
        v,
        cut_times_s=CUT_TIMES,
        horizons_s=HORIZONS,
        timesteps_s=TIMESTEPS,
    )

    variant_rows: list[dict[str, float | str]] = []
    for name, variant in _variant_params().items():
        variant_inertials = PlanarInertials.from_params(variant)
        vt, vq, vv, _ = rollout_program(variant, PROGRAM)
        evaluated = evaluate_killswitch_ensemble(
            variant,
            variant_inertials,
            PROGRAM,
            vt,
            vq,
            vv,
            cut_times_s=(0.12, 0.20, 0.28, 0.32),
            horizons_s=(0.08,),
            timesteps_s=(0.001,),
        )
        variant_rows.extend({"variant": name, **row} for row in evaluated)

    sensitivity: dict[str, dict[str, float]] = {}
    metrics = (
        "terminal_q_distance_rad",
        "terminal_v_distance_rad_s",
        "terminal_force_distance_n",
        "force_work_difference_j",
        "terminal_clubhead_speed_difference_m_s",
    )
    for timestep in (0.001, 0.002):
        timestep_result: dict[str, float] = {}
        for metric in metrics:
            differences: list[float] = []
            for cut in CUT_TIMES:
                for horizon in HORIZONS:
                    selected = [
                        row
                        for row in rows
                        if row["cut_time_s"] == cut and row["horizon_s"] == horizon
                    ]
                    fine = next(row for row in selected if row["dt_s"] == 0.0005)
                    candidate = next(row for row in selected if row["dt_s"] == timestep)
                    differences.append(abs(candidate[metric] - fine[metric]))
            timestep_result[f"max_abs_{metric}"] = max(differences)
        sensitivity[f"{1e3 * timestep:g}_ms_versus_0_5_ms"] = timestep_result

    arrays = _selected_trace_arrays(params, t, q, v)

    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source_sha256": _source_hashes(),
        "provenance": {
            "backend": "ode",
            "integrator": "fixed-step RK4",
            "program": PROGRAM.name,
            "cut_times_s": list(CUT_TIMES),
            "horizons_s": list(HORIZONS),
            "timesteps_s": list(TIMESTEPS),
            "pointwise_contract": "Recompute acceleration and force at a fixed source state.",
            "killswitch_contract": "Integrate commanded and zero-torque futures from one matched state.",
            "wscg_convention": "DELTA = BASE - counterfactual after time alignment.",
            "state_interpolation": "componentwise linear interpolation of the smooth registered source trace",
            "terminal_control_sampling": "commands sampled at all state times including the terminal endpoint",
        },
        "rows": rows,
        "variant_rows": variant_rows,
        "timestep_sensitivity": sensitivity,
        "wscg_delta_consistency": _wscg_delta_consistency(),
        "selected_trace_diagnostics": {
            "longest_initial_positive_zero_command_force_power_duration_s": (
                _positive_zero_command_power_duration_s(arrays)
            ),
            "selected_cut_times_s": [0.12, 0.22, 0.30],
        },
        "claim_boundaries": {
            "legacy_matlab_sampler": "re-simulates to each cut and records the first torque-off sample; not a forward zero-torque future",
            "variant_rows": "whole-model scenario variants with different pre-cut source traces; not an additive mechanism partition",
            "human_causal_strategy": "not_tested",
            "bilateral_hand_allocation": "not_tested",
        },
    }
    return record, arrays


def write_outputs(output_dir: Path = DATA_DIR) -> tuple[Path, Path]:
    """Write deterministic JSON and NPZ counterfactual evidence."""
    record, arrays = build_outputs()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / JSON_PATH.name
    npz_path = output_dir / NPZ_PATH.name
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    np.savez_compressed(npz_path, **arrays)
    return json_path, npz_path


def main() -> None:
    """Write deterministic JSON and NPZ counterfactual evidence."""
    write_outputs()


if __name__ == "__main__":
    main()
