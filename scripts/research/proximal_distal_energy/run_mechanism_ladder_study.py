"""Build common-observable evidence across higher-order mechanism tiers."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from scripts.research.proximal_distal_energy.flexible_shaft_study import (
    FlexibleShaftParams,
)
from scripts.research.proximal_distal_energy.mechanism_ladder import (
    InteractionSample,
    closed_loop_grip_jacobian,
    embed_planar_sample,
    mobile_hub_force_shift,
    rotation_matrix,
)
from src.engines.common.jacobian_diagnostics import compute_constraint_diagnostics

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = REPO_ROOT / "docs" / "research" / "proximal_distal_energy_transfer"
DATA_DIR = OUTPUT_ROOT / "data"


def _git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short=10", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


def _reference_arrays() -> tuple[dict[str, np.ndarray], dict]:
    arrays = dict(np.load(DATA_DIR / "shaft_contribution_traces.npz"))
    record = json.loads(
        (DATA_DIR / "shaft_contribution_study.json").read_text(encoding="utf-8")
    )
    return arrays, record


def _three_link_samples(
    arrays: dict[str, np.ndarray], record: dict
) -> tuple[list[InteractionSample], int]:
    prefix = "flexible_reference__"
    time = arrays[f"{prefix}time"]
    wrist = arrays[f"{prefix}wrist2"]
    velocity = np.gradient(wrist, time, axis=0, edge_order=2)
    force = arrays[f"{prefix}shaft_force"]
    state = arrays[f"{prefix}state"]
    params = FlexibleShaftParams.reference()
    elastic_moment = -params.shaft_stiffness_nm_rad * state[:, 2]
    damping_moment = -params.shaft_damping_nms_rad * state[:, 5]
    omega = state[:, 3] + state[:, 4] + state[:, 5]
    samples = [
        embed_planar_sample(
            model_tier="three-link-planar",
            time_s=float(time[index]),
            reference_point_xy_m=wrist[index],
            force_xy_n=force[index],
            couple_z_nm=float(elastic_moment[index] + damping_moment[index]),
            linear_velocity_xy_m_s=velocity[index],
            angular_velocity_z_rad_s=float(omega[index]),
        )
        for index in range(time.size)
    ]
    flexible = next(
        row
        for row in record["variant_summaries"]
        if row["name"] == "flexible_reference"
    )
    impact_time = flexible["impact"]["impact_time_s"]
    impact_index = int(np.argmin(np.abs(time - impact_time)))
    return samples, impact_index


def _frame_and_transport_audits(samples: list[InteractionSample]) -> dict[str, float]:
    rotation_power = []
    rotation_force = []
    rotation_couple = []
    transport_power = []
    for index, sample in enumerate(samples[::20]):
        axis = np.array([1.0 + index % 3, -0.4 + 0.01 * index, 0.7])
        transform = rotation_matrix(axis, 0.031 * index)
        rotated = sample.rotate(transform, frame=f"rotated-{index}")
        rotation_power.append(abs(rotated.total_power_w - sample.total_power_w))
        rotation_force.append(
            abs(np.linalg.norm(rotated.force_n) - np.linalg.norm(sample.force_n))
        )
        rotation_couple.append(
            abs(np.linalg.norm(rotated.couple_nm) - np.linalg.norm(sample.couple_nm))
        )
        new_point = sample.reference_point_m + np.array([0.2, -0.1, 0.05])
        moved = sample.transport(new_point)
        transport_power.append(abs(moved.total_power_w - sample.total_power_w))
    return {
        "maximum_rotation_power_residual_w": max(rotation_power),
        "maximum_transport_power_residual_w": max(transport_power),
        "maximum_rotation_force_norm_residual_n": max(rotation_force),
        "maximum_rotation_couple_norm_residual_nm": max(rotation_couple),
    }


def _mobile_hub_cases(samples: list[InteractionSample]) -> tuple[list[dict], dict]:
    time = np.array([sample.time_s for sample in samples])
    angular_frequency = 2.0 * np.pi * 1.25
    supported_mass = FlexibleShaftParams.reference().distal_head_mass_kg
    rows = []
    trace_arrays: dict[str, np.ndarray] = {}
    base_force = np.stack([sample.force_n for sample in samples])
    base_velocity = np.stack([sample.linear_velocity_m_s for sample in samples])
    base_couple_power = np.array([sample.couple_power_w for sample in samples])
    base_power = np.array([sample.total_power_w for sample in samples])
    for amplitude in (0.0, 0.025, 0.05, 0.10):
        phase = 0.35
        hub_velocity = np.column_stack(
            [
                amplitude * angular_frequency * np.cos(angular_frequency * time),
                amplitude
                * angular_frequency
                * np.cos(2.0 * angular_frequency * time + phase),
                np.zeros_like(time),
            ]
        )
        hub_acceleration = np.column_stack(
            [
                -amplitude * angular_frequency**2 * np.sin(angular_frequency * time),
                -2.0
                * amplitude
                * angular_frequency**2
                * np.sin(2.0 * angular_frequency * time + phase),
                np.zeros_like(time),
            ]
        )
        force_shift = np.stack(
            [
                mobile_hub_force_shift(supported_mass, value)
                for value in hub_acceleration
            ]
        )
        force = base_force + force_shift
        velocity = base_velocity + hub_velocity
        power = np.einsum("ij,ij->i", force, velocity) + base_couple_power
        key = f"hub_{int(round(amplitude * 1000)):03d}mm"
        trace_arrays[f"{key}__force"] = force
        trace_arrays[f"{key}__velocity"] = velocity
        trace_arrays[f"{key}__power"] = power
        rows.append(
            {
                "amplitude_m": amplitude,
                "frequency_hz": 1.25,
                "supported_mass_kg": supported_mass,
                "maximum_force_shift_n": float(
                    np.max(np.linalg.norm(force_shift, axis=1))
                ),
                "maximum_power_difference_w": float(np.max(np.abs(power - base_power))),
                "signed_power_difference_integral_j": float(
                    np.trapezoid(power - base_power, time)
                ),
            }
        )
    return rows, trace_arrays


def _closed_loop_diagnostics(samples: list[InteractionSample]) -> tuple[dict, dict]:
    ranks = []
    nullspaces = []
    conditions = []
    velocity_residuals = []
    singular_values = []
    for index, _sample in enumerate(samples[::5]):
        phase = index / max(1, len(samples[::5]) - 1)
        jacobian = closed_loop_grip_jacobian(
            lead_angle_rad=-1.15 + 1.9 * phase,
            trail_angle_rad=-0.90 + 1.65 * phase,
            grip_angle_rad=-0.45 + 1.1 * phase,
            lead_arm_length_m=0.75,
            trail_arm_length_m=0.78,
            grip_separation_m=0.25,
        )
        diagnostic = compute_constraint_diagnostics(jacobian, expected_dof=1)
        ranks.append(diagnostic.constraint_rank)
        nullspaces.append(diagnostic.nullspace_dim)
        conditions.append(diagnostic.condition_number)
        singular_values.append(diagnostic.condition_number)
        if diagnostic.nullspace_basis.size:
            residual = jacobian @ diagnostic.nullspace_basis[:, 0]
            velocity_residuals.append(float(np.linalg.norm(residual)))
    summary = {
        "sample_count": len(ranks),
        "minimum_rank": min(ranks),
        "maximum_rank": max(ranks),
        "minimum_nullspace_dimension": min(nullspaces),
        "maximum_nullspace_dimension": max(nullspaces),
        "minimum_condition_number": min(conditions),
        "maximum_condition_number": max(conditions),
        "maximum_constraint_velocity_residual": max(velocity_residuals),
        "interpretation": (
            "Rank and conditioning describe kinematic constraint transmission; "
            "they do not determine contact-force magnitude without dynamics."
        ),
    }
    arrays = {
        "closed_loop__phase": np.linspace(0.0, 1.0, len(ranks)),
        "closed_loop__condition_number": np.asarray(singular_values),
    }
    return summary, arrays


def build_study() -> tuple[dict, dict[str, np.ndarray]]:
    """Return the complete model-ladder record and plotting arrays."""
    arrays, shaft_record = _reference_arrays()
    spatial_record = json.loads(
        (DATA_DIR / "spatial_full_body_study.json").read_text(encoding="utf-8")
    )
    spatial_forward_record = json.loads(
        (DATA_DIR / "spatial_forward_contact_study.json").read_text(encoding="utf-8")
    )
    samples, impact_index = _three_link_samples(arrays, shaft_record)
    audits = _frame_and_transport_audits(samples)
    hub_rows, hub_arrays = _mobile_hub_cases(samples)
    loop_summary, loop_arrays = _closed_loop_diagnostics(samples)
    impact = samples[impact_index]
    flexible_summary = next(
        row
        for row in shaft_record["variant_summaries"]
        if row["name"] == "flexible_reference"
    )
    time = np.array([sample.time_s for sample in samples])
    three_force = np.stack([sample.force_n for sample in samples])
    three_power = np.array([sample.total_power_w for sample in samples])
    output_arrays = {
        "time": time,
        "three_link__force": three_force,
        "three_link__power": three_power,
        "three_link__point": np.stack([sample.reference_point_m for sample in samples]),
        **hub_arrays,
        **loop_arrays,
    }
    record = {
        "provenance": {
            "git_sha": _git_sha(),
            "source_trace": "shaft_contribution_traces.npz:flexible_reference",
            "common_schema": "InteractionSample v1",
            "interpretation_boundary": (
                "The three-link, prescribed mobile-hub, closed-loop geometry, "
                "rigid-frame audits, and reduced full-body common-state inverse "
                "dynamics are executed mechanism tests. Reduced MuJoCo/Pinocchio "
                "forward contact is also executed; articulated anatomical "
                "forward contact remains unexecuted."
            ),
        },
        "three_link_reference": {
            "delivery_time_s": flexible_summary["impact"]["impact_time_s"],
            "nearest_recorded_sample_time_s": impact.time_s,
            "interface_force_at_delivery_n": float(np.linalg.norm(impact.force_n)),
            "interface_couple_at_delivery_nm": float(impact.couple_nm[2]),
            "interface_force_power_at_delivery_w": impact.force_power_w,
            "interface_couple_power_at_delivery_w": impact.couple_power_w,
            "interface_total_power_at_delivery_w": impact.total_power_w,
        },
        "frame_and_transport_audits": audits,
        "mobile_hub_cases": hub_rows,
        "closed_loop_diagnostics": loop_summary,
        "model_discrepancy_table": [
            {
                "tier": "double_pendulum_planar",
                "status": "executed_in_prior_chapter",
                "added_mechanism": "single wrist interaction force and power",
                "surviving_result": "force magnitude alone does not determine transfer",
                "boundary": "fixed hub, rigid links, one distal interface",
            },
            {
                "tier": "three_link_planar",
                "status": "executed",
                "added_mechanism": "second internal interface and elastic coordinate",
                "surviving_result": "reference-explicit force and couple power remain additive",
                "boundary": "point masses and one linear torsional mode",
            },
            {
                "tier": "mobile_hub_inverse_dynamics",
                "status": "executed",
                "added_mechanism": "prescribed hub translation",
                "surviving_result": "hub acceleration changes reaction force through supported mass",
                "boundary": "prescribed motion comparison, not a forward torso model",
            },
            {
                "tier": "two_hand_closed_loop_geometry",
                "status": "executed",
                "added_mechanism": "four grip constraints and one feasible nullspace coordinate",
                "surviving_result": "constraint rank organizes admissible motion, not force magnitude",
                "boundary": "geometry/rank audit; archived forces remain the dynamic evidence",
            },
            {
                "tier": "rotated_3d_wrench_audit",
                "status": "executed",
                "added_mechanism": "arbitrary proper 3-D frame rotations and wrench transport",
                "surviving_result": "force norm, couple norm, and total power are frame invariant",
                "boundary": "3-D representation of a planar trajectory, not out-of-plane dynamics",
            },
            {
                "tier": "reduced_full_body_common_state_inverse_dynamics",
                "status": "executed",
                "added_mechanism": "nonplanar body and free-club inverse dynamics in two formulations",
                "surviving_result": (
                    "geometry sign response and same-state generalized action agree "
                    f"to {spatial_record['cross_formulation']['maximum_relative_inverse_dynamics_error']:.3e} relative error"
                ),
                "boundary": "prescribed hand loads and common state; not forward closed contact",
            },
            {
                "tier": "reduced_spatial_forward_cross_engine_contact",
                "status": "executed",
                "added_mechanism": (
                    "native MuJoCo/Pinocchio forward dynamics with paired "
                    "compliant contacts and a same-state driver killswitch"
                ),
                "surviving_result": (
                    "negative contact couple persists for "
                    f"{1e3 * spatial_forward_record['mechanism_tests']['same_state_killswitch_negative_duration_s']:.1f} ms"
                ),
                "boundary": (
                    "finite-mass translational hand carriages and rigid club; "
                    "not anatomy, tissue, equipment, or human validation"
                ),
            },
            {
                "tier": "articulated_full_body_forward_cross_engine_contact",
                "status": "not_executed",
                "added_mechanism": (
                    "subject-scaled articulated arms, calibrated grip, and "
                    "coupled distributed shaft"
                ),
                "surviving_result": "undetermined",
                "boundary": "must not be inferred from the reduced carriage model",
            },
        ],
    }
    return record, output_arrays


def main() -> None:
    """Write JSON and NPZ evidence artifacts."""
    record, arrays = build_study()
    (DATA_DIR / "mechanism_ladder_study.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(DATA_DIR / "mechanism_ladder_traces.npz", **arrays)


if __name__ == "__main__":
    main()
