"""Generate deterministic evidence for epic #8497."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from scripts.research.proximal_distal_energy.torque_allocation_preload import (
    RoleReversalProgram,
    TransmissionChannel,
    evaluate_continuous_role_reversal,
    evaluate_role_reversal,
    matched_allocation_sweep,
)

ROOT = Path(__file__).resolve().parents[3]
ARTICLE = ROOT / "docs" / "research" / "proximal_distal_energy_transfer"
DATA = ARTICLE / "data"


def _trace_record(trace) -> dict[str, float]:
    return {
        "arm_zero_transmission_duration_s": trace.arm_zero_transmission_duration_s,
        "wrist_zero_transmission_duration_s": trace.wrist_zero_transmission_duration_s,
        "arm_zero_transmission_duration_bounds_s": list(
            trace.arm_zero_transmission_duration_bounds_s
        ),
        "wrist_zero_transmission_duration_bounds_s": list(
            trace.wrist_zero_transmission_duration_bounds_s
        ),
        "temporal_resolution_s": trace.temporal_resolution_s,
        "net_torque_error_impulse_nms": trace.net_torque_error_impulse_nms,
        "arm_reversal_delay_s": trace.arm_reversal_delay_s,
        "wrist_reversal_delay_s": trace.wrist_reversal_delay_s,
    }


def main() -> None:
    """Write JSON summary and NPZ arrays from source calculations."""
    DATA.mkdir(parents=True, exist_ok=True)
    angles = np.deg2rad(np.linspace(-12.0, 24.0, 19))
    fractions = np.linspace(0.0, 1.0, 21)
    sweep = matched_allocation_sweep(angles, fractions, 8.0)
    channel = TransmissionChannel(
        stiffness_nm_rad=600.0,
        dead_zone_rad=0.012,
        time_constant_s=0.018,
    )
    programs = (
        RoleReversalProgram.persistent_direction(),
        RoleReversalProgram.opposite_role_reversal(),
    )
    traces = {
        f"{program.name}_{'preloaded' if preload else 'relaxed'}": evaluate_role_reversal(
            program,
            arm_channel=channel,
            wrist_channel=channel,
            duration_s=0.12,
            step_s=0.0001,
            initialize_at_preload=preload,
        )
        for program in programs
        for preload in (True, False)
    }
    continuous_traces = {
        program.name: evaluate_continuous_role_reversal(
            program,
            arm_channel=channel,
            wrist_channel=channel,
            preparation_duration_s=0.18,
            post_transition_duration_s=0.12,
            step_s=0.0001,
        )
        for program in programs
    }
    dead_zones = np.array([0.0, 0.004, 0.012, 0.024])
    time_constants = np.array([0.008, 0.018, 0.035])
    sensitivity_error = np.empty((dead_zones.size, time_constants.size, 2))
    sensitivity_zero_duration = np.empty_like(sensitivity_error)
    for dead_zone_index, dead_zone in enumerate(dead_zones):
        for time_index, time_constant in enumerate(time_constants):
            varied = TransmissionChannel(
                stiffness_nm_rad=600.0,
                dead_zone_rad=float(dead_zone),
                time_constant_s=float(time_constant),
            )
            for program_index, program in enumerate(programs):
                trace = evaluate_role_reversal(
                    program,
                    arm_channel=varied,
                    wrist_channel=varied,
                    duration_s=0.12,
                    step_s=0.0001,
                    initialize_at_preload=True,
                )
                sensitivity_error[dead_zone_index, time_index, program_index] = (
                    trace.net_torque_error_impulse_nms
                )
                sensitivity_zero_duration[
                    dead_zone_index, time_index, program_index
                ] = (
                    trace.arm_zero_transmission_duration_s
                    + trace.wrist_zero_transmission_duration_s
                )
    equivalence_tolerance_nms = 1e-10
    sensitivity_difference = sensitivity_error[:, :, 1] - sensitivity_error[:, :, 0]
    persistent_favored = sensitivity_difference > equivalence_tolerance_nms
    role_reversal_favored = sensitivity_difference < -equivalence_tolerance_nms
    equivalent = np.abs(sensitivity_difference) <= equivalence_tolerance_nms
    summary = {
        "schema_version": "1.0.0",
        "epic": "https://github.com/D-sorganization/UpstreamDrift/issues/8497",
        "claim_boundary": (
            "The constrained sweep compares generalized actuator allocations; "
            "it neither identifies muscles nor equates proximal joint torque with "
            "scapular retraction. The transmission study is phenomenological."
        ),
        "matched_task": {
            "target_net_control_moment_nm": 8.0,
            "maximum_moment_closure_error_nm": float(
                np.max(
                    np.abs(
                        sweep.direct_wrist_moment_nm
                        + sweep.grip_force_couple_nm
                        - sweep.net_control_moment_nm
                    )
                )
            ),
            "maximum_task_error_nm": float(
                np.max(np.abs(sweep.net_control_moment_nm - 8.0))
            ),
            "minimum_hand_force_rms_n": float(np.min(sweep.hand_force_rms_n)),
            "maximum_hand_force_rms_n": float(np.max(sweep.hand_force_rms_n)),
            "minimum_joint_torque_norm_nm": float(np.min(sweep.joint_torque_norm_nm)),
            "maximum_joint_torque_norm_nm": float(np.max(sweep.joint_torque_norm_nm)),
        },
        "transmission_channel": asdict(channel),
        "programs": {program.name: asdict(program) for program in programs},
        "transmission_results": {
            name: _trace_record(trace) for name, trace in traces.items()
        },
        "continuous_preparation_results": {
            "preparation_duration_s": 0.18,
            "initial_state": "relaxed_zero_deflection",
            "transition_contract": (
                "Internal deflection and transmitted torque remain continuous; "
                "only desired channel commands change at time zero."
            ),
            "programs": {
                name: _trace_record(trace) for name, trace in continuous_traces.items()
            },
        },
        "transmission_sensitivity": {
            "dead_zone_rad": dead_zones.tolist(),
            "time_constant_s": time_constants.tolist(),
            "equivalence_tolerance_nms": equivalence_tolerance_nms,
            "persistent_direction_favored_case_count": int(
                np.count_nonzero(persistent_favored)
            ),
            "equivalent_case_count": int(np.count_nonzero(equivalent)),
            "role_reversal_favored_case_count": int(
                np.count_nonzero(role_reversal_favored)
            ),
            "case_count": int(dead_zones.size * time_constants.size),
            "interpretation": (
                "The persistent-direction advantage is conditional on the "
                "declared transmission family. Under the explicit equivalence "
                "tolerance, all three zero-dead-zone cases are equivalent and "
                "the nine positive-dead-zone cases favor persistent direction."
            ),
        },
        "registered_hypotheses": {
            "RA-H1": "Allocation extremes can match the same club task.",
            "RA-H2": "Allocation changes internal force and effort signatures.",
            "RA-H3": "Sign reversal across a dead zone creates transmission delay.",
            "RA-H4": "Preload improves early torque continuity for a fixed program.",
            "RA-H5": "No universal allocation optimum is asserted without cost weights.",
        },
    }
    arrays = {
        "club_angles_rad": sweep.club_angles_rad,
        "wrist_fractions": sweep.wrist_fractions,
        "net_control_moment_nm": sweep.net_control_moment_nm,
        "direct_wrist_moment_nm": sweep.direct_wrist_moment_nm,
        "grip_force_couple_nm": sweep.grip_force_couple_nm,
        "hand_force_rms_n": sweep.hand_force_rms_n,
        "hand_force_resultant_n": sweep.hand_force_resultant_n,
        "joint_torque_norm_nm": sweep.joint_torque_norm_nm,
        "sensitivity_dead_zone_rad": dead_zones,
        "sensitivity_time_constant_s": time_constants,
        "sensitivity_error_impulse_nms": sensitivity_error,
        "sensitivity_reversal_minus_persistent_nms": sensitivity_difference,
        "sensitivity_zero_duration_s": sensitivity_zero_duration,
    }
    for name, trace in traces.items():
        arrays[f"{name}_time_s"] = trace.time_s
        arrays[f"{name}_desired_net_torque_nm"] = trace.desired_net_torque_nm
        arrays[f"{name}_transmitted_net_torque_nm"] = trace.transmitted_net_torque_nm
        arrays[f"{name}_transmitted_arm_torque_nm"] = trace.transmitted_arm_torque_nm
        arrays[f"{name}_transmitted_wrist_torque_nm"] = (
            trace.transmitted_wrist_torque_nm
        )
    for name, trace in continuous_traces.items():
        prefix = f"continuous_{name}"
        arrays[f"{prefix}_time_s"] = trace.time_s
        arrays[f"{prefix}_desired_net_torque_nm"] = trace.desired_net_torque_nm
        arrays[f"{prefix}_transmitted_net_torque_nm"] = trace.transmitted_net_torque_nm
        arrays[f"{prefix}_transmitted_arm_torque_nm"] = trace.transmitted_arm_torque_nm
        arrays[f"{prefix}_transmitted_wrist_torque_nm"] = (
            trace.transmitted_wrist_torque_nm
        )
    (DATA / "torque_allocation_preload_study.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(DATA / "torque_allocation_preload_study.npz", **arrays)


if __name__ == "__main__":
    main()
