"""Build common-observable evidence across higher-order mechanism tiers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

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
SCHEMA_VERSION = "mechanism-ladder-study-v2"
JSON_PATH = DATA_DIR / "mechanism_ladder_study.json"
NPZ_PATH = DATA_DIR / "mechanism_ladder_traces.npz"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_hashes() -> dict[str, str]:
    """Hash every direct data and executable dependency of this aggregate study."""
    paths = (
        DATA_DIR / "shaft_contribution_study.json",
        DATA_DIR / "shaft_contribution_traces.npz",
        DATA_DIR / "spatial_full_body_study.json",
        DATA_DIR / "spatial_forward_contact_study.json",
        REPO_ROOT / "scripts/research/proximal_distal_energy/mechanism_ladder.py",
        REPO_ROOT
        / "scripts/research/proximal_distal_energy/run_mechanism_ladder_study.py",
        REPO_ROOT / "scripts/research/proximal_distal_energy/flexible_shaft_study.py",
        REPO_ROOT / "src/engines/common/jacobian_diagnostics.py",
    )
    return {path.relative_to(REPO_ROOT).as_posix(): _sha256(path) for path in paths}


def _reference_arrays() -> tuple[dict[str, np.ndarray], dict]:
    arrays = dict(np.load(DATA_DIR / "shaft_contribution_traces.npz"))
    record = json.loads(
        (DATA_DIR / "shaft_contribution_study.json").read_text(encoding="utf-8")
    )
    return arrays, record


def _wrist2_velocity(state: np.ndarray, params: FlexibleShaftParams) -> np.ndarray:
    """Return exact second-joint velocity for relative triple-link coordinates."""
    checked = np.asarray(state, dtype=float)
    if checked.ndim != 2 or checked.shape[1] != 6:
        raise ValueError("state must have shape (samples, 6)")
    triple = params.triple()
    theta = checked[:, 0]
    phi = checked[:, 1]
    dtheta = checked[:, 3]
    dphi = checked[:, 4]
    absolute_second = theta + phi
    absolute_second_rate = dtheta + dphi
    return np.column_stack(
        (
            triple.L1 * np.cos(theta) * dtheta
            + triple.L2 * np.cos(absolute_second) * absolute_second_rate,
            triple.L1 * np.sin(theta) * dtheta
            + triple.L2 * np.sin(absolute_second) * absolute_second_rate,
        )
    )


def _three_link_samples(
    arrays: dict[str, np.ndarray], record: dict
) -> tuple[list[InteractionSample], int, dict[str, Any]]:
    prefix = "flexible_reference__"
    time = arrays[f"{prefix}time"]
    wrist = arrays[f"{prefix}wrist2"]
    force = arrays[f"{prefix}shaft_force"]
    state = arrays[f"{prefix}state"]
    params = FlexibleShaftParams.reference()
    velocity = _wrist2_velocity(state, params)
    gradient_velocity = np.gradient(wrist, time, axis=0, edge_order=2)
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
    velocity_audit = {
        "method": "analytic relative-coordinate second-joint kinematics",
        "position_gradient_comparison_method": (
            "numpy.gradient with second-order edges on the stored 0.5 ms positions"
        ),
        "maximum_position_gradient_discrepancy_m_s": float(
            np.max(np.linalg.norm(velocity - gradient_velocity, axis=1))
        ),
    }
    return samples, impact_index, velocity_audit


def _frame_and_transport_audits(samples: list[InteractionSample]) -> dict[str, Any]:
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
        "rotation_sample_count": len(rotation_power),
        "reference_translation_m": [0.2, -0.1, 0.05],
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
    samples, impact_index, velocity_audit = _three_link_samples(arrays, shaft_record)
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
    three_couple = np.stack([sample.couple_nm for sample in samples])
    three_velocity = np.stack([sample.linear_velocity_m_s for sample in samples])
    three_angular_velocity = np.stack(
        [sample.angular_velocity_rad_s for sample in samples]
    )
    three_power = np.array([sample.total_power_w for sample in samples])
    output_arrays = {
        "time": time,
        "three_link__force": three_force,
        "three_link__couple": three_couple,
        "three_link__velocity": three_velocity,
        "three_link__angular_velocity": three_angular_velocity,
        "three_link__power": three_power,
        "three_link__point": np.stack([sample.reference_point_m for sample in samples]),
        **hub_arrays,
        **loop_arrays,
    }
    record = {
        "schema_version": SCHEMA_VERSION,
        "study_id": "common-observable-model-ladder",
        "source_sha256": _source_hashes(),
        "provenance": {
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
        "kinematic_velocity_audit": velocity_audit,
        "frame_and_transport_audits": audits,
        "mobile_hub_contract": {
            "fundamental_frequency_hz": 1.25,
            "second_harmonic_frequency_hz": 2.5,
            "second_harmonic_position_amplitude_ratio": 0.5,
            "phase_rad": 0.35,
            "path_definition": ("x=A sin(omega t); y=(A/2) sin(2 omega t + phase)"),
            "comparison_type": (
                "prescribed inverse-dynamics perturbation at fixed relative trace"
            ),
        },
        "mobile_hub_cases": hub_rows,
        "closed_loop_diagnostics": loop_summary,
        "model_discrepancy_table": [
            {
                "tier": "double_pendulum_planar",
                "status": "executed_in_prior_chapter",
                "added_mechanism": "single wrist interaction force and power",
                "surviving_result": "force magnitude alone does not determine transfer",
                "boundary": "fixed hub, rigid links, one distal interface",
                "capabilities": ["interface_wrench"],
            },
            {
                "tier": "three_link_planar",
                "status": "executed",
                "added_mechanism": "second internal interface and elastic coordinate",
                "surviving_result": "reference-explicit force and couple power remain additive",
                "boundary": "point masses and one linear torsional mode",
                "capabilities": ["interface_wrench", "elastic_coordinate"],
            },
            {
                "tier": "mobile_hub_inverse_dynamics",
                "status": "executed",
                "added_mechanism": "prescribed hub translation",
                "surviving_result": "hub acceleration changes reaction force through supported mass",
                "boundary": "prescribed motion comparison, not a forward torso model",
                "capabilities": [
                    "interface_wrench",
                    "elastic_coordinate",
                    "prescribed_hub",
                ],
            },
            {
                "tier": "two_hand_closed_loop_geometry",
                "status": "executed",
                "added_mechanism": "four grip constraints and one feasible nullspace coordinate",
                "surviving_result": "constraint rank organizes admissible motion, not force magnitude",
                "boundary": "geometry/rank audit; archived forces remain the dynamic evidence",
                "capabilities": ["closed_loop_geometry"],
            },
            {
                "tier": "rotated_3d_wrench_audit",
                "status": "executed",
                "added_mechanism": "arbitrary proper 3-D frame rotations and wrench transport",
                "surviving_result": "force norm, couple norm, and total power are frame invariant",
                "boundary": "3-D representation of a planar trajectory, not out-of-plane dynamics",
                "capabilities": [
                    "interface_wrench",
                    "elastic_coordinate",
                    "frame_3d",
                ],
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
                "capabilities": ["frame_3d", "spatial_inverse_dynamics"],
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
                "capabilities": ["frame_3d", "forward_contact"],
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
                "capabilities": ["articulated_contact"],
            },
        ],
    }
    return record, output_arrays


def write_outputs(output_dir: Path = DATA_DIR) -> tuple[Path, Path]:
    """Write deterministic JSON and NPZ evidence artifacts."""
    record, arrays = build_study()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / JSON_PATH.name
    npz_path = output_dir / NPZ_PATH.name
    json_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(npz_path, **arrays)
    return json_path, npz_path


def main() -> None:
    """Write JSON and NPZ evidence artifacts."""
    write_outputs()


if __name__ == "__main__":
    main()
