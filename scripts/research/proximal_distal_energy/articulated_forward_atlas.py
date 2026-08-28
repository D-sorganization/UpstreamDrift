"""Cohort orchestration for bounded articulated bilateral attachment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from scripts.research.proximal_distal_energy.articulated_forward_contract import (
    ArticulatedForwardContactConfig,
    ForwardVariant,
    registered_variants,
)
from scripts.research.proximal_distal_energy.articulated_forward_integration import (
    ForwardIntegrationCase,
    integrate_articulated_contact,
)
from scripts.research.proximal_distal_energy.articulated_inertia_cross_engine import (
    finite_difference_kinematics,
)
from scripts.research.proximal_distal_energy.subject_scaled_spatial_geometry import (
    build_subject_scaled_model,
    default_synthetic_profiles,
)

FloatArray = NDArray[np.float64]
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "docs/research/proximal_distal_energy_transfer/data"
SOURCE_PATHS = (
    "docs/research/proximal_distal_energy_transfer/data/subject_scaled_closed_contact.json",
    "docs/research/proximal_distal_energy_transfer/data/subject_scaled_closed_contact.npz",
    "docs/research/proximal_distal_energy_transfer/data/articulated_contact_projection.json",
    "scripts/research/proximal_distal_energy/articulated_forward_contact.py",
    "scripts/research/proximal_distal_energy/articulated_forward_contract.py",
    "scripts/research/proximal_distal_energy/articulated_forward_integration.py",
    "scripts/research/proximal_distal_energy/articulated_forward_atlas.py",
    "scripts/research/proximal_distal_energy/run_articulated_forward_contact.py",
    "scripts/research/proximal_distal_energy/make_articulated_forward_contact_figure.py",
    "scripts/research/proximal_distal_energy/register_articulated_forward_contact_claims.py",
    "scripts/research/proximal_distal_energy/articulated_contact_projection.py",
    "scripts/research/proximal_distal_energy/articulated_inertia_cross_engine.py",
    "scripts/research/proximal_distal_energy/spatial_full_body.py",
    "scripts/research/proximal_distal_energy/subject_scaled_spatial_geometry.py",
    "tests/research/test_articulated_forward_contact.py",
)


@dataclass(frozen=True, slots=True)
class ForwardAuthority:
    time_s: FloatArray
    profile_index: NDArray[np.int_]
    grip_span_m: FloatArray
    solution_q: FloatArray


@dataclass(slots=True)
class _AtlasBuffers:
    maximum_force: FloatArray
    maximum_separation: FloatArray
    retained: NDArray[np.bool_]
    maximum_virtual_power: FloatArray
    maximum_dissipation: FloatArray
    maximum_energy_residual: FloatArray
    normalized_energy_residual: FloatArray
    final_club_speed: FloatArray
    final_q: FloatArray
    trajectory_relative_error: FloatArray
    force_relative_error: FloatArray


def load_forward_authority() -> ForwardAuthority:
    with np.load(DATA_DIR / "subject_scaled_closed_contact.npz") as source:
        authority = ForwardAuthority(
            time_s=np.asarray(source["time_s"], dtype=float),
            profile_index=np.asarray(source["case_profile_index"], dtype=int),
            grip_span_m=np.asarray(source["case_grip_span_m"], dtype=float),
            solution_q=np.asarray(source["solution_q"], dtype=float),
        )
        feasible = np.asarray(source["feasible"], dtype=bool)
    if authority.solution_q.shape != (18, 13, 20) or not np.all(feasible):
        raise RuntimeError("the closed-state authority is incomplete or infeasible")
    return authority


def _atlas_buffers(shape: tuple[int, ...], nq: int) -> _AtlasBuffers:
    parity_shape = shape[:-1]
    return _AtlasBuffers(
        maximum_force=np.empty(shape),
        maximum_separation=np.empty(shape),
        retained=np.empty(shape, dtype=bool),
        maximum_virtual_power=np.empty(shape),
        maximum_dissipation=np.empty(shape),
        maximum_energy_residual=np.empty(shape),
        normalized_energy_residual=np.empty(shape),
        final_club_speed=np.empty(shape),
        final_q=np.empty((*shape, nq)),
        trajectory_relative_error=np.empty(parity_shape),
        force_relative_error=np.empty(parity_shape),
    )


def _relative_error(left: FloatArray, right: FloatArray) -> float:
    scale = max(1.0, float(np.max(np.abs(left))), float(np.max(np.abs(right))))
    return float(np.max(np.abs(left - right)) / scale)


def build_forward_integration_case(
    authority: ForwardAuthority,
    config: ArticulatedForwardContactConfig,
    variant: ForwardVariant,
    case: int,
    sample: int,
    time_step_s: float,
    hand_contact_local_x_m: float,
    engine: str,
) -> ForwardIntegrationCase:
    velocity, _ = finite_difference_kinematics(
        authority.solution_q[case], authority.time_s
    )
    return ForwardIntegrationCase(
        q=authority.solution_q[case, sample],
        qd=velocity[sample],
        grip_span_m=float(authority.grip_span_m[case]),
        hand_contact_local_x_m=hand_contact_local_x_m,
        time_step_s=time_step_s,
        contact_stiffness=config.contact_stiffness * variant.stiffness_factor,
        contact_damping=config.contact_damping * variant.damping_factor,
        initial_club_displacement_m=(
            config.initial_club_displacement_m * variant.displacement_factor
        ),
        initial_club_velocity_m_s=(
            config.initial_club_velocity_m_s * variant.velocity_factor
        ),
        engine=engine,
    )


def _record_trace(
    buffers: _AtlasBuffers,
    slots: tuple[int, int, int, int],
    trace: dict[str, FloatArray | float],
    config: ArticulatedForwardContactConfig,
) -> None:
    force = np.asarray(trace["maximum_contact_force_n"])
    separation = np.asarray(trace["maximum_attachment_separation_m"])
    power = np.asarray(trace["virtual_power_residual_w"])
    dissipation = np.asarray(trace["contact_dissipation_power_w"])
    residual = np.asarray(trace["work_energy_residual_j"])
    total = np.asarray(trace["total_energy_j"])
    qd_trace, q_trace = np.asarray(trace["qd"]), np.asarray(trace["q"])
    buffers.maximum_force[slots] = np.max(force)
    buffers.maximum_separation[slots] = np.max(separation)
    buffers.retained[slots] = bool(np.all(separation <= config.retention_threshold_m))
    buffers.maximum_virtual_power[slots] = np.max(np.abs(power))
    buffers.maximum_dissipation[slots] = np.max(dissipation)
    buffers.maximum_energy_residual[slots] = np.max(np.abs(residual))
    buffers.normalized_energy_residual[slots] = np.max(np.abs(residual)) / max(
        1.0, float(np.ptp(total))
    )
    buffers.final_club_speed[slots] = np.linalg.norm(qd_trace[-1, 14:17])
    buffers.final_q[slots] = q_trace[-1]


def _run_state(
    authority: ForwardAuthority,
    buffers: _AtlasBuffers,
    config: ArticulatedForwardContactConfig,
    state_slot: int,
    state: tuple[int, int],
    variants: tuple[ForwardVariant, ...],
) -> None:
    case, sample = state
    profiles = default_synthetic_profiles()
    model, metadata = build_subject_scaled_model(
        profiles[authority.profile_index[case]]
    )
    hand_x = float(metadata["hand_contact_local_x_m"])
    engines = ("mujoco", "pinocchio")
    for variant_slot, variant in enumerate(variants):
        for step_slot, time_step in enumerate(config.time_steps_s):
            traces: dict[str, dict[str, FloatArray | float]] = {}
            for engine_slot, engine in enumerate(engines):
                integration_case = build_forward_integration_case(
                    authority,
                    config,
                    variant,
                    case,
                    sample,
                    time_step,
                    hand_x,
                    engine,
                )
                trace = integrate_articulated_contact(model, integration_case, config)
                traces[engine] = trace
                _record_trace(
                    buffers,
                    (state_slot, variant_slot, step_slot, engine_slot),
                    trace,
                    config,
                )
            parity_slot = (state_slot, variant_slot, step_slot)
            buffers.trajectory_relative_error[parity_slot] = _relative_error(
                np.asarray(traces["mujoco"]["q"]),
                np.asarray(traces["pinocchio"]["q"]),
            )
            buffers.force_relative_error[parity_slot] = _relative_error(
                np.asarray(traces["mujoco"]["maximum_contact_force_n"]),
                np.asarray(traces["pinocchio"]["maximum_contact_force_n"]),
            )


def _gates(
    buffers: _AtlasBuffers, config: ArticulatedForwardContactConfig
) -> tuple[NDArray[np.bool_], NDArray[np.bool_], FloatArray, bool]:
    cells = (
        buffers.retained
        & (buffers.maximum_virtual_power <= config.virtual_power_tolerance_w)
        & (buffers.maximum_dissipation <= config.positive_dissipation_tolerance_w)
        & (
            buffers.normalized_energy_residual
            <= config.normalized_energy_residual_tolerance
        )
    )
    parity = (
        buffers.trajectory_relative_error <= config.trajectory_relative_tolerance
    ) & (buffers.force_relative_error <= config.trajectory_relative_tolerance)
    refinement = np.max(buffers.normalized_energy_residual, axis=(0, 1, 3))
    refinement_passed = bool(
        np.all(np.diff(refinement) <= 0.0)
        and refinement[-1] <= config.refinement_ratio_limit * refinement[0]
    )
    return cells, parity, refinement, refinement_passed


def _arrays(
    authority: ForwardAuthority,
    buffers: _AtlasBuffers,
    states: tuple[tuple[int, int], ...],
    variants: tuple[ForwardVariant, ...],
    config: ArticulatedForwardContactConfig,
    gates: tuple[NDArray[np.bool_], NDArray[np.bool_], FloatArray, bool],
) -> dict[str, NDArray[Any]]:
    state_case = np.asarray([pair[0] for pair in states], dtype=int)
    state_sample = np.asarray([pair[1] for pair in states], dtype=int)
    return {
        "state_case_index": state_case,
        "state_sample_index": state_sample,
        "state_profile_index": authority.profile_index[state_case],
        "state_grip_span_m": authority.grip_span_m[state_case],
        "variant_names": np.asarray([variant.name for variant in variants]),
        "time_steps_s": np.asarray(config.time_steps_s),
        "engine_names": np.asarray(["mujoco", "pinocchio"]),
        "maximum_contact_force_n": buffers.maximum_force,
        "maximum_attachment_separation_m": buffers.maximum_separation,
        "retained": buffers.retained,
        "maximum_virtual_power_residual_w": buffers.maximum_virtual_power,
        "maximum_contact_dissipation_power_w": buffers.maximum_dissipation,
        "maximum_work_energy_residual_j": buffers.maximum_energy_residual,
        "normalized_work_energy_residual": buffers.normalized_energy_residual,
        "final_club_translation_speed_m_s": buffers.final_club_speed,
        "final_q": buffers.final_q,
        "trajectory_relative_error": buffers.trajectory_relative_error,
        "force_relative_error": buffers.force_relative_error,
        "cell_gates_passed": gates[0],
        "parity_gates_passed": gates[1],
        "refinement_worst_normalized_residual": gates[2],
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(
    authority: ForwardAuthority,
    buffers: _AtlasBuffers,
    states: tuple[tuple[int, int], ...],
    variants: tuple[ForwardVariant, ...],
    config: ArticulatedForwardContactConfig,
    gates: tuple[NDArray[np.bool_], NDArray[np.bool_], FloatArray, bool],
    engine_versions: dict[str, str],
) -> dict[str, Any]:
    state_cases = np.asarray([pair[0] for pair in states], dtype=int)
    all_passed = bool(np.all(gates[0]) and np.all(gates[1]) and gates[3])
    shape = buffers.maximum_force.shape
    return {
        "schema_version": "articulated-forward-contact/v1",
        "study_id": "bounded-subject-scaled-articulated-bilateral-contact",
        "model_tier": "bounded_articulated_bilateral_attachment_forward_dynamics",
        "design": {
            "engine_names": ["mujoco", "pinocchio"],
            "engine_versions": engine_versions,
            "bounded_horizon_s": config.duration_s,
            "state_count": len(states),
            "profile_count": int(np.unique(authority.profile_index[state_cases]).size),
            "grip_span_count": int(np.unique(authority.grip_span_m[state_cases]).size),
            "phase_count": len(config.sample_indices),
            "variant_count": len(variants),
            "time_step_count": len(config.time_steps_s),
            "trajectory_count": int(np.prod(shape)),
            "unilateral_collision_contact": False,
            "attachment_type": "bilateral Kelvin-Voigt point attachment",
            "active_driver_or_joint_torque": "none; motion is an initial condition",
        },
        "configuration": asdict(config),
        "variants": [asdict(variant) for variant in variants],
        "results": _result_summary(buffers, gates, all_passed),
        "interpretation": {
            "retention": "attachment separation remains below a declared screening threshold; this is not unilateral biological contact",
            "right_censoring": "no registered failure before 5 ms does not establish persistence beyond the bounded horizon",
            "adverse_controls": "stiffness, damping, velocity reversal, and zero-preload branches are reported without selecting a preferred mechanism",
        },
        "claim_boundary": {
            "supported": "the declared synthetic bilateral attachments can be advanced through the bounded articulated horizon subject to the registered gates",
            "distributed_grip_shaft_or_ground": "not_modeled",
            "late_downswing_or_impact": "not_established_by_5_ms_horizon",
            "anatomy_or_muscle_action": "not_calibrated_or_inferred",
            "slack_benefit_or_timing_economy": "untested",
            "human_transfer_or_strategy": "untested",
        },
        "next_gate": "extend the right-censored horizon with unilateral/typed slack, distributed grip and shaft compliance, ground coupling, and governed human wrenches",
        "source_sha256": {path: _sha256(REPO_ROOT / path) for path in SOURCE_PATHS},
    }


def _result_summary(
    buffers: _AtlasBuffers,
    gates: tuple[NDArray[np.bool_], NDArray[np.bool_], FloatArray, bool],
    all_passed: bool,
) -> dict[str, Any]:
    return {
        "maximum_contact_force_n": float(np.max(buffers.maximum_force)),
        "maximum_attachment_separation_m": float(np.max(buffers.maximum_separation)),
        "failed_retention_cell_count": int(np.count_nonzero(~buffers.retained)),
        "maximum_virtual_power_residual_w": float(
            np.max(buffers.maximum_virtual_power)
        ),
        "maximum_positive_dissipation_power_w": float(
            np.max(buffers.maximum_dissipation)
        ),
        "maximum_work_energy_residual_j": float(
            np.max(buffers.maximum_energy_residual)
        ),
        "maximum_normalized_work_energy_residual": float(
            np.max(buffers.normalized_energy_residual)
        ),
        "maximum_trajectory_relative_error": float(
            np.max(buffers.trajectory_relative_error)
        ),
        "maximum_force_relative_error": float(np.max(buffers.force_relative_error)),
        "refinement_worst_normalized_residual": gates[2].tolist(),
        "refinement_direction_passed": gates[3],
        "failed_cell_count": int(np.count_nonzero(~gates[0])),
        "failed_parity_count": int(np.count_nonzero(~gates[1])),
        "all_registered_gates_passed": all_passed,
    }


def run_articulated_forward_contact_atlas(
    config: ArticulatedForwardContactConfig = ArticulatedForwardContactConfig(),
) -> tuple[dict[str, Any], dict[str, NDArray[Any]]]:
    """Run the preregistered cohort, refinement, and adverse-control matrix."""

    try:
        import mujoco
        import pinocchio as pin
    except ImportError as error:  # pragma: no cover - native runtime gate
        raise RuntimeError("MuJoCo and robotics Pinocchio are required") from error
    authority = load_forward_authority()
    states = tuple(
        (case, sample)
        for case in config.case_indices
        for sample in config.sample_indices
    )
    variants = registered_variants()
    shape = (len(states), len(variants), len(config.time_steps_s), 2)
    buffers = _atlas_buffers(shape, authority.solution_q.shape[2])
    for state_slot, state in enumerate(states):
        _run_state(authority, buffers, config, state_slot, state, variants)
    gates = _gates(buffers, config)
    arrays = _arrays(authority, buffers, states, variants, config, gates)
    versions = {
        "mujoco": str(mujoco.__version__),
        "pinocchio": str(pin.__version__),  # type: ignore[attr-defined]
    }
    return (
        _record(authority, buffers, states, variants, config, gates, versions),
        arrays,
    )


__all__ = [
    "ForwardAuthority",
    "build_forward_integration_case",
    "load_forward_authority",
    "run_articulated_forward_contact_atlas",
]
