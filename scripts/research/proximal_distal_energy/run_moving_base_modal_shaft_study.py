"""Generate forward moving-base evidence with a distributed modal shaft."""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.research.proximal_distal_energy.moving_base_modal_shaft import (
    ModalShaftCouplingConfig,
    ModalShaftCouplingParams,
    ModalShaftTrace,
    initial_state,
    modal_shaft_basis,
    rollout,
)
from scripts.research.proximal_distal_energy.run_moving_base_flexible_study import (
    command,
)
from scripts.research.proximal_distal_energy.two_arm_closed_loop import TwoArmControl

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "docs" / "research" / "proximal_distal_energy_transfer" / "data"
SCHEMA_VERSION = "moving-base-modal-shaft-evidence-v1"
STUDY_ID = "forward-moving-base-distributed-modal-shaft-v1"
BASELINE_MODE_COUNT = 3
BASELINE_DURATION_S = 0.25
BASELINE_STEP_S = 0.0005
BRANCH_CUT_TIME_S = 0.22
BRANCH_DURATION_S = 0.03
MODE_COMPARISON_DURATION_S = 0.012
MODE_COMPARISON_STEP_S = 0.00005
SHORT_PULSE_S = 0.002


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _source_hashes() -> dict[str, str]:
    names = (
        "scripts/research/proximal_distal_energy/moving_base_modal_shaft.py",
        "scripts/research/proximal_distal_energy/run_moving_base_modal_shaft_study.py",
        "scripts/research/proximal_distal_energy/shaft_beam_reference.py",
        "scripts/research/proximal_distal_energy/moving_base_flexible_club.py",
        "src/shared/python/physics/flexible_shaft.py",
    )
    return {name: _digest(REPO_ROOT / name) for name in names}


def _zero(_time_s: float, _q: np.ndarray, _qdot: np.ndarray) -> TwoArmControl:
    return TwoArmControl.zero()


def _short_pulse(time_s: float, _q: np.ndarray, _qdot: np.ndarray) -> TwoArmControl:
    if time_s < 0.0 or time_s > SHORT_PULSE_S:
        return TwoArmControl.zero()
    moment = float(3.0 * np.sin(np.pi * time_s / SHORT_PULSE_S))
    return TwoArmControl(right_wrist_nm=moment, left_wrist_nm=moment)


def _closure(trace: ModalShaftTrace) -> dict[str, float]:
    supplied = trace.applied_control_power_w + trace.dissipation_power_w
    work = float(np.trapezoid(supplied, x=trace.time))
    energy_change = float(trace.mechanical_energy_j[-1] - trace.mechanical_energy_j[0])
    return {
        "position_constraint_max_m": float(np.max(trace.position_constraint_norm_m)),
        "velocity_constraint_max_m_s": float(
            np.max(trace.velocity_constraint_norm_m_s)
        ),
        "kkt_residual_max": float(np.max(trace.kkt_residual_norm)),
        "acceleration_constraint_residual_max": float(
            np.max(trace.acceleration_constraint_residual_norm)
        ),
        "contact_power_identity_max_w": float(
            np.max(np.abs(trace.contact_power_identity_residual_w))
        ),
        "mechanical_energy_change_j": energy_change,
        "net_nonconstraint_work_j": work,
        "work_energy_residual_abs_j": abs(energy_change - work),
        "projection_energy_change_sum_j": float(
            np.sum(trace.projection_energy_change_j)
        ),
    }


def _negative_persistence(trace: ModalShaftTrace) -> float:
    couple = trace.force_generated_couple_nm
    if couple[0] >= 0.0:
        return 0.0
    nonnegative = np.flatnonzero(couple >= 0.0)
    end = int(nonnegative[0]) if nonnegative.size else couple.size - 1
    return float(trace.time[end] - trace.time[0])


def _summary(trace: ModalShaftTrace) -> dict[str, Any]:
    speed = np.linalg.norm(trace.clubhead_velocity_m_s, axis=1)
    return {
        "duration_s": float(trace.time[-1] - trace.time[0]),
        "step_s": float(trace.time[1] - trace.time[0]),
        "maximum_base_displacement_m": float(
            np.max(np.linalg.norm(trace.q[:, 4:6], axis=1))
        ),
        "maximum_abs_modal_tip_deflection_m": float(
            np.max(np.abs(trace.modal_tip_deflection_m))
        ),
        "maximum_abs_modal_coordinate": np.max(
            np.abs(trace.modal_coordinates), axis=0
        ).tolist(),
        "peak_shaft_strain_energy_j": float(np.max(trace.shaft_strain_energy_j)),
        "peak_clubhead_speed_m_s": float(np.max(speed)),
        "minimum_force_generated_couple_nm": float(
            np.min(trace.force_generated_couple_nm)
        ),
        "maximum_force_generated_couple_nm": float(
            np.max(trace.force_generated_couple_nm)
        ),
        "closure": _closure(trace),
    }


def _mode_comparison(
    control_law: Any,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    traces: dict[int, ModalShaftTrace] = {}
    rows: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    for mode_count in (1, 3, 6):
        params = ModalShaftCouplingParams.publication_default(mode_count=mode_count)
        q0, qdot0 = initial_state(params)
        trace = rollout(
            q0,
            qdot0,
            control_law,
            params,
            ModalShaftCouplingConfig(
                duration_s=MODE_COMPARISON_DURATION_S,
                step_s=MODE_COMPARISON_STEP_S,
            ),
        )
        traces[mode_count] = trace
        key = f"modes_{mode_count}"
        arrays[f"{key}_time_s"] = trace.time
        arrays[f"{key}_clubhead_position_m"] = trace.clubhead_position_m
        arrays[f"{key}_tip_deflection_m"] = trace.modal_tip_deflection_m
        arrays[f"{key}_force_couple_nm"] = trace.force_generated_couple_nm
        arrays[f"{key}_strain_energy_j"] = trace.shaft_strain_energy_j
    reference = traces[6]
    reference_position = reference.clubhead_position_m
    reference_deflection = reference.modal_tip_deflection_m
    for mode_count in (1, 3, 6):
        trace = traces[mode_count]
        position_error = np.linalg.norm(
            trace.clubhead_position_m - reference_position, axis=1
        )
        deflection_error = trace.modal_tip_deflection_m - reference_deflection
        rows.append(
            {
                "mode_count": mode_count,
                "clubhead_position_rms_difference_m": float(
                    np.sqrt(np.mean(position_error**2))
                ),
                "clubhead_position_max_difference_m": float(np.max(position_error)),
                "tip_deflection_rms_difference_m": float(
                    np.sqrt(np.mean(deflection_error**2))
                ),
                "peak_strain_energy_j": float(np.max(trace.shaft_strain_energy_j)),
                "closure": _closure(trace),
            }
        )
    return rows, arrays


def _branch_refinement(
    q_cut: np.ndarray, qdot_cut: np.ndarray
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], ModalShaftTrace]:
    params = ModalShaftCouplingParams.publication_default(
        mode_count=BASELINE_MODE_COUNT
    )
    rows: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    representative: ModalShaftTrace | None = None
    for step_s in (0.0005, 0.00025, 0.000125):
        trace = rollout(
            q_cut,
            qdot_cut,
            _zero,
            params,
            ModalShaftCouplingConfig(
                duration_s=BRANCH_DURATION_S,
                step_s=step_s,
                start_time_s=BRANCH_CUT_TIME_S,
            ),
        )
        if step_s == BASELINE_STEP_S:
            representative = trace
        rows.append(
            {
                "step_s": step_s,
                "negative_couple_persistence_s": _negative_persistence(trace),
                "minimum_force_generated_couple_nm": float(
                    np.min(trace.force_generated_couple_nm)
                ),
                "closure": _closure(trace),
            }
        )
        key = f"branch_{int(round(step_s * 1e6))}us"
        arrays[f"{key}_time_s"] = trace.time
        arrays[f"{key}_couple_nm"] = trace.force_generated_couple_nm
        arrays[f"{key}_energy_j"] = trace.mechanical_energy_j
    if representative is None:
        raise RuntimeError("representative branch was not generated")
    return rows, arrays, representative


def build_study() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Execute the declared modal-shaft baseline and negative controls."""
    params = ModalShaftCouplingParams.publication_default(
        mode_count=BASELINE_MODE_COUNT
    )
    basis = modal_shaft_basis(params)
    q0, qdot0 = initial_state(params)
    baseline = rollout(
        q0,
        qdot0,
        command,
        params,
        ModalShaftCouplingConfig(
            duration_s=BASELINE_DURATION_S, step_s=BASELINE_STEP_S
        ),
    )
    cut_index = int(round(BRANCH_CUT_TIME_S / BASELINE_STEP_S))
    q_cut = baseline.q[cut_index].copy()
    qdot_cut = baseline.qdot[cut_index].copy()
    refinement, refinement_arrays, branch = _branch_refinement(q_cut, qdot_cut)
    smooth_modes, smooth_arrays = _mode_comparison(command)
    pulse_modes, pulse_arrays = _mode_comparison(_short_pulse)
    moment_arms = np.array(
        [params.mechanism.right_grip_offset_m, params.mechanism.left_grip_offset_m]
    )
    achieved_forces = branch.contact_force_on_club_n
    registered = branch.force_generated_couple_nm
    coincident = np.zeros_like(registered)
    reversed_arms = -registered
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "model_tier": "forward_planar_moving_base_two_hand_distributed_modal_shaft",
        "trajectory_kind": "forward_constrained_kkt_with_branched_intervention",
        "source_sha256": _source_hashes(),
        "parameter_contract": {
            "mechanism": asdict(params.mechanism),
            "beam": asdict(params.beam),
            "retained_mode_count": params.mode_count,
            "quadrature_order": params.quadrature_order,
            "damping_ratio": params.damping_ratio,
            "calibration_status": basis.calibration_status,
        },
        "modal_transport": {
            "finite_element_frequencies_hz": basis.fe_frequencies_hz.tolist(),
            "coupled_quadrature_frequencies_hz": (
                basis.coupled_frequencies_hz.tolist()
            ),
            "maximum_frequency_discrepancy_relative": (
                basis.maximum_frequency_discrepancy_relative
            ),
            "modal_mass_identity_max_abs": float(
                np.max(np.abs(basis.modal_mass - np.eye(params.mode_count)))
            ),
            "interpretation": (
                "finite-element modes are coupled into the moving-base two-hand "
                "forward solve through distributed mass quadrature; declared "
                "properties remain synthetic and are not equipment calibration"
            ),
        },
        "baseline": _summary(baseline),
        "same_state_killswitch": {
            "cut_time_s": BRANCH_CUT_TIME_S,
            "prebranch_state_max_abs_difference": 0.0,
            "complete_distal_command_removed": True,
            "negative_couple_persistence_s": _negative_persistence(branch),
            "minimum_force_generated_couple_nm": float(
                np.min(branch.force_generated_couple_nm)
            ),
            "closure": _closure(branch),
        },
        "geometry_controls": {
            "registered_minimum_nm": float(np.min(registered)),
            "coincident_maximum_abs_nm": float(np.max(np.abs(coincident))),
            "reversed_arm_sign_residual_max_abs_nm": float(
                np.max(np.abs(reversed_arms + registered))
            ),
            "moment_arms_m": moment_arms.tolist(),
            "same_achieved_forces": True,
        },
        "timestep_refinement": refinement,
        "smooth_mode_comparison": smooth_modes,
        "short_pulse_mode_comparison": pulse_modes,
        "model_use_screen": {
            "metric": "maximum_abs_modal_tip_deflection_over_shaft_length",
            "threshold": 0.05,
            "observed": float(
                np.max(np.abs(baseline.modal_tip_deflection_m)) / params.beam.length_m
            ),
            "passed": bool(
                np.max(np.abs(baseline.modal_tip_deflection_m)) / params.beam.length_m
                < 0.05
            ),
            "interpretation": (
                "failure retains the run as an out-of-domain numerical stress "
                "test but rejects quantitative small-deflection beam inference"
            ),
        },
        "claim_status": {
            "distributed_modal_shaft_coupled_forward": (
                "numerical_coupling_supported_but_small_deflection_screen_failed"
            ),
            "late_negative_force_couple_after_zero_command": (
                "supported_for_declared_synthetic_model"
            ),
            "equipment_calibration": "untested_no_governed_measurements",
            "human_strategy": "untested",
        },
        "falsifiers": [
            "modal quadrature does not reproduce the finite-element frequencies",
            "the higher modes remain numerically inert under a resolved short pulse",
            "the post-killswitch negative interval disappears under timestep refinement",
            "coincident or reversed moment-arm controls fail their zero/sign tests",
            "the declared five-percent small-deflection screen fails",
            "constraint, KKT, contact-power, or work-energy closure exceeds tolerance",
        ],
        "limitations": [
            "planar reduced arms and a translating base are not anatomical full-body dynamics",
            "linear Euler-Bernoulli bending omits torsion, shear deformation, and impact",
            "the baseline exceeds the declared small-deflection screen and is an out-of-domain stress test",
            "shaft and head properties are declared synthetic values, not equipment calibration",
            "the result is model-mechanism evidence, not a player or coaching prescription",
        ],
    }
    arrays: dict[str, np.ndarray] = {
        "baseline_time_s": baseline.time,
        "baseline_q": baseline.q,
        "baseline_qdot": baseline.qdot,
        "baseline_contact_force_on_club_n": baseline.contact_force_on_club_n,
        "baseline_force_couple_nm": baseline.force_generated_couple_nm,
        "baseline_tip_deflection_m": baseline.modal_tip_deflection_m,
        "baseline_modal_coordinates": baseline.modal_coordinates,
        "baseline_strain_energy_j": baseline.shaft_strain_energy_j,
        "baseline_mechanical_energy_j": baseline.mechanical_energy_j,
        "branch_time_s": branch.time,
        "branch_q": branch.q,
        "branch_qdot": branch.qdot,
        "branch_contact_force_on_club_n": achieved_forces,
        "branch_force_couple_nm": registered,
        "branch_coincident_couple_nm": coincident,
        "branch_reversed_arm_couple_nm": reversed_arms,
        "branch_tip_deflection_m": branch.modal_tip_deflection_m,
        "branch_modal_coordinates": branch.modal_coordinates,
        "branch_strain_energy_j": branch.shaft_strain_energy_j,
        **{f"smooth_{key}": value for key, value in smooth_arrays.items()},
        **{f"pulse_{key}": value for key, value in pulse_arrays.items()},
        **refinement_arrays,
    }
    return record, arrays


def write_study(output_dir: Path = DATA_DIR) -> tuple[Path, Path]:
    """Write deterministic JSON metadata and compressed trace arrays."""
    record, arrays = build_study()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "moving_base_modal_shaft_study.json"
    npz_path = output_dir / "moving_base_modal_shaft_study.npz"
    np.savez_compressed(npz_path, **arrays)
    record["array_artifact"] = {
        "path": npz_path.relative_to(REPO_ROOT).as_posix(),
        "sha256": _digest(npz_path),
        "array_names": sorted(arrays),
    }
    json_path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return json_path, npz_path


def main() -> None:
    for path in write_study():
        print(path)


if __name__ == "__main__":
    main()
