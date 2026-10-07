"""End-to-end musculoskeletal swing pipeline and receipt (issue #11617).

Order of operations (each step is a function of the previous one's files):

1. build the muscle model fitted to the golf humanoid and write it,
2. map the matched swing kinematics (OpenSim IK states) onto the model
   coordinates 1:1 by name, low-pass filter, and cut the analysis window,
3. estimate the ground reactions needed for dynamic consistency,
4. solve muscle redundancy with StaticOptimization,
5. summarise activations next to the reserve and stand-in torques and write a
   receipt that states the limitations plainly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import logging
from pathlib import Path
import time
from typing import Any

import numpy as np

from src.engines.physics_engines.opensim.python import (
    musculoskeletal_grf as grf,
    musculoskeletal_solvers as solvers,
    musculoskeletal_swing as swing,
)
from src.shared.python.contracts import require

logger = logging.getLogger(__name__)

RECEIPT_SCHEMA = "msk-swing/opensim-static-optimization-v1"
LIMITATIONS: tuple[str, ...] = (
    "The base model has 80 lower-limb muscles only: there are no trunk, shoulder, "
    "arm or forearm muscles. Lumbar, shoulder, elbow, forearm and wrist demand is "
    "carried by upper_* joint-torque actuators, not muscles.",
    "No ground-reaction forces were measured. They are estimated from the "
    "kinematics (force-free inverse dynamics + friction-limited NNLS on eight sole "
    "points); the split between feet is statically indeterminate, so leg-muscle "
    "forces depend on that assumption.",
    "Kinematics are marker-based IK of a tour-average swing with no foot markers; "
    "large trunk/arm angles and fast transitions make joint accelerations "
    "(hence demands) unreliable, especially after the downswing begins.",
    "Muscle strength is the generic Rajagopal-Lai-Uhlrich set scaled with the "
    "segments; it is not calibrated to this golfer.",
    "StaticOptimization has no activation dynamics and no tendon history; it is a "
    "per-frame redundancy resolution, not a forward-dynamics replay.",
)


@dataclass(frozen=True)
class PipelineConfig:
    """Inputs of the musculoskeletal swing pipeline."""

    golf_model: Path
    states_file: Path
    out_dir: Path
    window: solvers.SolveWindow
    cutoff_hz: float = 15.0
    so_step: int = 2
    phase_split_s: float = 1.08
    base_model: Path | None = None
    extra_notes: tuple[str, ...] = field(default_factory=tuple)

    def validate(self) -> None:
        """Raise ``ValueError``/``FileNotFoundError`` on unusable inputs."""
        require(self.cutoff_hz > 0, "cutoff_hz must be positive")
        require(self.so_step >= 1, "so_step must be >= 1")
        for p in (self.golf_model, self.states_file):
            if not Path(p).is_file():
                raise FileNotFoundError(f"input not found: {p}")


def sha256_file(path: str | Path) -> str:
    """SHA-256 of a file's bytes."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _phase_rms(times: np.ndarray, values: np.ndarray, split: float) -> dict[str, float]:
    """RMS of ``values`` before and after ``split`` (empty phase -> 0.0)."""
    out: dict[str, float] = {}
    for name, mask in (
        ("before_split", times < split),
        ("after_split", times >= split),
    ):
        out[name] = float(np.sqrt(np.mean(values[mask] ** 2))) if mask.any() else 0.0
    return out


def prepare_inputs(cfg: PipelineConfig) -> dict[str, Any]:
    """Steps 1-3: model, mapped/filtered kinematics and estimated loads.

    Returns a dict of paths and metrics; raises ``MappingError`` if any model
    coordinate has no source column.
    """
    cfg.validate()
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model, info = swing.build_musculoskeletal_model(cfg.golf_model, cfg.base_model)
    model_path = out / "msk_model.osim"
    model.printToXML(str(model_path))
    times, cols = swing.read_states_table(cfg.states_file)
    names = [c.getName() for c in model.getCoordinateSet()]
    mapping = swing.map_coordinates(list(cols), names)
    if mapping.unmapped_model:
        raise swing.MappingError(
            f"unmapped model coordinates: {mapping.unmapped_model}"
        )
    smooth = swing.smooth_kinematics(
        times, {n: cols[mapping.mapped[n]] for n in names}, cfg.cutoff_hz
    )
    sel = (times >= cfg.window.t_start - 1e-9) & (times <= cfg.window.t_end + 1e-9)
    require(int(sel.sum()) > 12, "analysis window contains too few frames")
    tw = times[sel]
    qw = {k: v[sel] for k, v in smooth.items()}
    coords_path = swing.write_sto(out / "coords.sto", tw, qw)
    est = grf.estimate_ground_reactions(model, tw, qw)
    xml = grf.write_external_loads(
        out, tw, est["forces"], est["points"], spins=est["spins"]
    )
    total = est["forces"].sum(axis=1)
    bw = info["total_mass_kg"] * 9.80665
    return {
        "model": model,
        "model_path": model_path,
        "coords_path": coords_path,
        "loads_xml": xml,
        "times": tw,
        "info": info,
        "mapping": mapping,
        "grf": {
            "estimator": "force-free ID root wrench + friction-limited NNLS, 8 sole points",
            "friction_coefficient": grf.DEFAULT_FRICTION,
            "contact_height_tolerance_m": grf.DEFAULT_CONTACT_HEIGHT_TOL_M,
            "floor_height_m": est["floor_y"],
            "vertical_force_bw_mean": float(np.mean(total[:, 1]) / bw),
            "vertical_force_bw_peak": float(np.max(total[:, 1]) / bw),
            "horizontal_force_bw_peak": float(
                np.max(np.hypot(total[:, 0], total[:, 2])) / bw
            ),
            "unexplained_wrench_rms": float(
                np.sqrt(np.mean(est["residual_norm"] ** 2))
            ),
            "unexplained_wrench_max": float(np.max(est["residual_norm"])),
            "unexplained_wrench_rms_by_phase": _phase_rms(
                tw, est["residual_norm"], cfg.phase_split_s
            ),
            "required_wrench_rms": float(
                np.sqrt(np.mean(np.sum(est["required"] ** 2, axis=1)))
            ),
        },
    }


def solve_and_summarise(cfg: PipelineConfig, prep: dict[str, Any]) -> dict[str, Any]:
    """Steps 4-5: StaticOptimization and the activation/reserve summary."""
    out = Path(cfg.out_dir)
    start = time.time()
    act_path = solvers.run_static_optimization(
        prep["model_path"],
        prep["coords_path"],
        prep["loads_xml"],
        out / "so",
        cfg.window,
        step_interval=cfg.so_step,
    )
    wall = time.time() - start
    t, acts = solvers.activation_table_to_arrays(act_path)
    muscle_names = [m.getName() for m in prep["model"].getMuscles()]
    muscle_names = [m for m in muscle_names if m in acts]
    summary = solvers.summarize_solution(
        t,
        acts,
        muscle_names=muscle_names,
        optimal_forces=prep["info"]["actuator_optimal_force"],
    )

    def summarize_phase(tt: np.ndarray, aa: dict[str, np.ndarray]) -> dict[str, Any]:
        sub = solvers.summarize_solution(
            tt,
            aa,
            muscle_names=muscle_names,
            optimal_forces=prep["info"]["actuator_optimal_force"],
        )
        return {
            "t_range_s": [float(tt[0]), float(tt[-1])],
            "per_group": sub["per_group"],
            "reserve_rms_max": sub["reserve_rms_max"],
            "reserve_torques": sub["reserve_torques"],
            "upper_body_torques": sub["upper_body_torques"],
            "muscles_saturated_fraction": sub["muscles_saturated_fraction"],
        }

    summary["phases"] = {}
    for name, mask in (
        ("before_split", t < cfg.phase_split_s),
        ("after_split", t >= cfg.phase_split_s),
    ):
        if int(mask.sum()) > 1:
            summary["phases"][name] = summarize_phase(
                t[mask], {k: v[mask] for k, v in acts.items()}
            )
    summary["wall_clock_s"] = wall
    summary["activation_file"] = str(act_path)
    return summary


def build_receipt(
    cfg: PipelineConfig, prep: dict[str, Any], summary: dict[str, Any]
) -> dict[str, Any]:
    """Assemble the receipt dict (JSON-serialisable)."""
    info = prep["info"]
    mapping = prep["mapping"]
    top = sorted(summary["per_muscle"].items(), key=lambda kv: -kv[1]["peak"])[:20]
    return {
        "schema": RECEIPT_SCHEMA,
        "issue": "#11617",
        "epic": "#11605",
        "status": "STATIC_OPTIMIZATION_ONLY_NOT_QUALIFIED",
        "window_s": [cfg.window.t_start, cfg.window.t_end],
        "inputs": {
            "golf_model": str(cfg.golf_model),
            "golf_model_sha256": sha256_file(cfg.golf_model),
            "states_file": str(cfg.states_file),
            "states_file_sha256": sha256_file(cfg.states_file),
            "base_model": info["base_model"],
            "base_model_sha256": sha256_file(info["base_model"]),
            "musculoskeletal_model_sha256": sha256_file(prep["model_path"]),
        },
        "model": {
            "n_muscles": info["n_muscles"],
            "muscle_type": "Millard2012EquilibriumMuscle",
            "n_coordinates": info["n_coordinates"],
            "total_mass_kg": info["total_mass_kg"],
            "body_scale_factors": info["body_scale_factors"],
            "unlocked_coordinates": list(swing.UNLOCKED_COORDINATES),
            "upper_body_actuators": "upper_* CoordinateActuator (optimal force "
            f"{swing.UPPER_OPTIMAL_FORCE:g}; torques reported in N m)",
            "reserve_actuators": "reserve_* CoordinateActuator (leg optimal force "
            f"{swing.RESERVE_OPTIMAL_FORCE:g}, pelvis residual "
            f"{swing.ROOT_OPTIMAL_FORCE:g}; reported in N or N m)",
        },
        "coordinate_mapping": {
            "method": "1:1 by coordinate name from OpenSim IK states",
            "mapped": len(mapping.mapped),
            "of": info["n_coordinates"],
            "unmapped_model_coordinates": list(mapping.unmapped_model),
            "unmapped_source_columns": list(mapping.unmapped_source),
            "notes": [
                "knee_angle_*_beta follow the patellofemoral coupler constraint",
                "mtp_angle_* are identically zero in the source and locked",
                "kinematics low-passed at "
                f"{cfg.cutoff_hz} Hz (zero-phase 2nd-order Butterworth)",
            ],
        },
        "ground_reaction_estimate": prep["grf"],
        "solver": {
            "name": "opensim.StaticOptimization (AnalyzeTool)",
            "activation_exponent": solvers.DEFAULT_ACTIVATION_EXPONENT,
            "use_muscle_physiology": True,
            "step_interval": cfg.so_step,
            "wall_clock_s": summary["wall_clock_s"],
            "converged": True,
            "convergence_note": (
                "AnalyzeTool completed; per-frame optimiser failures, if any, are "
                "visible as reserve use and saturated muscles below"
            ),
        },
        "results": {
            "per_group": summary["per_group"],
            "top_muscles_by_peak": dict(top),
            "reserve_rms_max": summary["reserve_rms_max"],
            "reserve_torques": summary["reserve_torques"],
            "upper_body_torques": summary["upper_body_torques"],
            "muscles_saturated_fraction": summary["muscles_saturated_fraction"],
            "phases": summary["phases"],
            "phase_split_s": cfg.phase_split_s,
        },
        "moco_inverse": {
            "status": "NOT_RUN_TO_CONVERGENCE",
            "note": "see limitations in the README; direct collocation with 80 "
            "muscles took more than a minute per iteration on 4 cores even for a "
            "10-interval pilot, so a full-swing solve was infeasible here",
        },
        "limitations": list(LIMITATIONS) + list(cfg.extra_notes),
    }


def run_pipeline(cfg: PipelineConfig, receipt_path: str | Path) -> dict[str, Any]:
    """Run all steps and write the receipt; return it."""
    prep = prepare_inputs(cfg)
    summary = solve_and_summarise(cfg, prep)
    receipt = build_receipt(cfg, prep, summary)
    path = Path(receipt_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt
