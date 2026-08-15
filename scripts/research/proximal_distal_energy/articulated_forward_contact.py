"""Bounded forward bilateral-contact falsifier for the articulated tree.

The hand--grip elements are bilateral Kelvin--Voigt attachments, not unilateral
collision contacts.  The study advances independently assembled MuJoCo and
robotics-Pinocchio dynamics through a short, preregistered horizon and audits
attachment separation, power, energy, refinement, and adverse controls.  It
does not represent calibrated anatomy, tissue, equipment, or human technique.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Any
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from scripts.research.proximal_distal_energy.articulated_contact_projection import (
    ArticulatedContactProjectionConfig,
    evaluate_contact_projection,
)
from scripts.research.proximal_distal_energy.articulated_inertia_cross_engine import (
    build_pinocchio_articulated_model,
    finite_difference_kinematics,
)
from scripts.research.proximal_distal_energy.spatial_full_body import (
    SpatialModel,
    forward_kinematics,
    mass_matrix,
    mujoco_mass_matrix_and_bias,
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
class ArticulatedForwardContactConfig:
    """Preoutcome horizon, cohort, perturbation, and acceptance gates."""

    duration_s: float = 0.005
    time_steps_s: tuple[float, ...] = (0.001, 0.0005, 0.00025)
    case_indices: tuple[int, ...] = (0, 4, 8, 9, 13, 17)
    sample_indices: tuple[int, ...] = (0, 6, 12)
    contact_stiffness: float = 1800.0
    contact_damping: float = 18.0
    initial_club_displacement_m: float = 1.0e-3
    initial_club_velocity_m_s: float = 5.0e-2
    retention_threshold_m: float = 1.0e-2
    virtual_power_tolerance_w: float = 1.0e-10
    positive_dissipation_tolerance_w: float = 1.0e-12
    trajectory_relative_tolerance: float = 1.0e-7
    normalized_energy_residual_tolerance: float = 2.0e-2
    refinement_ratio_limit: float = 0.8

    def __post_init__(self) -> None:
        positive = (
            "duration_s",
            "contact_stiffness",
            "initial_club_displacement_m",
            "initial_club_velocity_m_s",
            "retention_threshold_m",
            "virtual_power_tolerance_w",
            "positive_dissipation_tolerance_w",
            "trajectory_relative_tolerance",
            "normalized_energy_residual_tolerance",
            "refinement_ratio_limit",
        )
        for name in positive:
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        steps = np.asarray(self.time_steps_s, dtype=float)
        if (
            steps.ndim != 1
            or steps.size < 2
            or np.any(~np.isfinite(steps))
            or np.any(steps <= 0.0)
            or np.any(np.diff(steps) >= 0.0)
            or np.any(
                np.abs(self.duration_s / steps - np.rint(self.duration_s / steps))
                > 1e-10
            )
        ):
            raise ValueError(
                "time_steps_s must be finite, positive, strictly decreasing, "
                "and divide duration_s"
            )
        if not np.isfinite(self.contact_damping) or self.contact_damping < 0.0:
            raise ValueError("contact_damping must be finite and nonnegative")
        if not 0.0 < self.refinement_ratio_limit < 1.0:
            raise ValueError("refinement_ratio_limit must lie in (0, 1)")
        for name, values, upper in (
            ("case_indices", self.case_indices, 18),
            ("sample_indices", self.sample_indices, 13),
        ):
            if (
                not values
                or len(set(values)) != len(values)
                or any(
                    not isinstance(value, int) or not 0 <= value < upper
                    for value in values
                )
            ):
                raise ValueError(f"{name} must contain unique in-range integers")


@dataclass(frozen=True, slots=True)
class ForwardVariant:
    """One factor at a time from the nominal attachment perturbation."""

    name: str
    stiffness_factor: float = 1.0
    damping_factor: float = 1.0
    displacement_factor: float = 1.0
    velocity_factor: float = 1.0


def registered_variants() -> tuple[ForwardVariant, ...]:
    """Return nominal, null, reversal, and one-factor adverse branches."""

    return (
        ForwardVariant("nominal"),
        ForwardVariant("stiffness_low", stiffness_factor=0.5),
        ForwardVariant("stiffness_high", stiffness_factor=2.0),
        ForwardVariant("damping_low", damping_factor=0.5),
        ForwardVariant("damping_high", damping_factor=2.0),
        ForwardVariant("velocity_reversed", velocity_factor=-1.0),
        ForwardVariant("zero_preload", displacement_factor=0.0, velocity_factor=0.0),
    )


def mechanical_energy(model: SpatialModel, q: FloatArray, qd: FloatArray) -> float:
    """Return articulated kinetic plus gravitational potential energy."""

    position = np.asarray(q, dtype=float)
    velocity = np.asarray(qd, dtype=float)
    if position.shape != (model.nq,) or velocity.shape != (model.nq,):
        raise ValueError("q and qd must match the articulated model dimension")
    matrix = mass_matrix(model, position)
    kinetic = 0.5 * float(velocity @ matrix @ velocity)
    kinematics = forward_kinematics(model, position)
    potential = sum(
        body.mass_kg * 9.80665 * kinematics.body_position_m[index, 2]
        for index, body in enumerate(model.bodies)
    )
    return kinetic + float(potential)


def _native_operator(
    engine: str, model: SpatialModel
) -> Callable[[FloatArray, FloatArray], tuple[FloatArray, FloatArray]]:
    if engine == "mujoco":
        return lambda q, qd: mujoco_mass_matrix_and_bias(model, q, qd)
    if engine != "pinocchio":
        raise ValueError("engine must be 'mujoco' or 'pinocchio'")
    try:
        import pinocchio as pin
    except ImportError as error:  # pragma: no cover - native runtime gate
        raise RuntimeError("robotics Pinocchio is required") from error
    native = build_pinocchio_articulated_model(pin, model)
    data = native.createData()

    def evaluate(q: FloatArray, qd: FloatArray) -> tuple[FloatArray, FloatArray]:
        matrix = np.asarray(pin.crba(native, data, q)).copy()
        bias = np.asarray(pin.nonLinearEffects(native, data, q, qd)).copy()
        return matrix, bias

    return evaluate


def integrate_articulated_contact(
    model: SpatialModel,
    q: FloatArray,
    qd: FloatArray,
    *,
    grip_span_m: float,
    hand_contact_local_x_m: float,
    time_step_s: float,
    contact_stiffness: float,
    contact_damping: float,
    initial_club_displacement_m: float,
    initial_club_velocity_m_s: float,
    engine: str,
    config: ArticulatedForwardContactConfig = ArticulatedForwardContactConfig(),
) -> dict[str, FloatArray | float]:
    """Advance one engine with semi-implicit Euler and a named energy ledger."""

    if not isinstance(config, ArticulatedForwardContactConfig):
        raise TypeError("config must be an ArticulatedForwardContactConfig")
    scalar_values = {
        "time_step_s": time_step_s,
        "contact_stiffness": contact_stiffness,
        "contact_damping": contact_damping,
        "initial_club_displacement_m": initial_club_displacement_m,
        "initial_club_velocity_m_s": initial_club_velocity_m_s,
    }
    if any(not np.isfinite(value) for value in scalar_values.values()):
        raise ValueError("integration scalars must be finite")
    if time_step_s <= 0.0 or contact_stiffness <= 0.0 or contact_damping < 0.0:
        raise ValueError("step and stiffness must be positive; damping nonnegative")
    step_count = int(round(config.duration_s / time_step_s))
    if not np.isclose(step_count * time_step_s, config.duration_s):
        raise ValueError("time_step_s must divide the configured duration")

    position = np.asarray(q, dtype=float).copy()
    velocity = np.asarray(qd, dtype=float).copy()
    if position.shape != (model.nq,) or velocity.shape != (model.nq,):
        raise ValueError("q and qd must match the articulated model dimension")
    position[14] += initial_club_displacement_m
    velocity[14] += initial_club_velocity_m_s
    contact_config = ArticulatedContactProjectionConfig(
        contact_stiffness=contact_stiffness,
        contact_damping=contact_damping,
    )
    native_operator = _native_operator(engine, model)
    sample_count = step_count + 1
    time = np.arange(sample_count, dtype=float) * time_step_s
    positions = np.empty((sample_count, model.nq))
    velocities = np.empty_like(positions)
    force = np.empty(sample_count)
    separation = np.empty(sample_count)
    virtual_power = np.empty(sample_count)
    dissipation = np.empty(sample_count)
    strain = np.empty(sample_count)
    mechanical = np.empty(sample_count)

    for index in range(sample_count):
        snapshot = evaluate_contact_projection(
            model,
            position,
            velocity,
            grip_span_m=grip_span_m,
            hand_contact_local_x_m=hand_contact_local_x_m,
            perturb_contact=False,
            config=contact_config,
        )
        positions[index] = position
        velocities[index] = velocity
        force[index] = snapshot.maximum_contact_force_n
        separation[index] = snapshot.maximum_attachment_separation_m
        virtual_power[index] = snapshot.virtual_power_residual_w
        dissipation[index] = snapshot.contact_dissipation_power_w
        strain[index] = snapshot.attachment_strain_energy_j
        mechanical[index] = mechanical_energy(model, position, velocity)
        if index < step_count:
            matrix, bias = native_operator(position, velocity)
            acceleration = np.linalg.solve(
                matrix, snapshot.generalized_contact_force - bias
            )
            velocity = velocity + time_step_s * acceleration
            position = position + time_step_s * velocity

    cumulative_dissipation = np.zeros(sample_count)
    cumulative_dissipation[1:] = np.cumsum(
        0.5 * (dissipation[1:] + dissipation[:-1]) * time_step_s
    )
    total = mechanical + strain
    return {
        "time_s": time,
        "q": positions,
        "qd": velocities,
        "maximum_contact_force_n": force,
        "maximum_attachment_separation_m": separation,
        "virtual_power_residual_w": virtual_power,
        "contact_dissipation_power_w": dissipation,
        "attachment_strain_energy_j": strain,
        "mechanical_energy_j": mechanical,
        "total_energy_j": total,
        "cumulative_dissipation_j": cumulative_dissipation,
        "work_energy_residual_j": total - total[0] - cumulative_dissipation,
    }


def _relative_error(left: FloatArray, right: FloatArray) -> float:
    scale = max(1.0, float(np.max(np.abs(left))), float(np.max(np.abs(right))))
    return float(np.max(np.abs(left - right)) / scale)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_articulated_forward_contact_atlas(
    config: ArticulatedForwardContactConfig = ArticulatedForwardContactConfig(),
) -> tuple[dict[str, Any], dict[str, NDArray[Any]]]:
    """Run the preregistered cohort, refinement, and adverse-control matrix."""

    try:
        import mujoco
        import pinocchio as pin
    except ImportError as error:  # pragma: no cover - native runtime gate
        raise RuntimeError("MuJoCo and robotics Pinocchio are required") from error
    with np.load(DATA_DIR / "subject_scaled_closed_contact.npz") as source:
        source_time = np.asarray(source["time_s"], dtype=float)
        profile_index = np.asarray(source["case_profile_index"], dtype=int)
        grip_span = np.asarray(source["case_grip_span_m"], dtype=float)
        solution_q = np.asarray(source["solution_q"], dtype=float)
        feasible = np.asarray(source["feasible"], dtype=bool)
    if solution_q.shape != (18, 13, 20) or not np.all(feasible):
        raise RuntimeError("the closed-state authority is incomplete or infeasible")

    state_pairs = tuple(
        (case, sample)
        for case in config.case_indices
        for sample in config.sample_indices
    )
    variants = registered_variants()
    engines = ("mujoco", "pinocchio")
    shape = (len(state_pairs), len(variants), len(config.time_steps_s), len(engines))
    maximum_force = np.empty(shape)
    maximum_separation = np.empty(shape)
    retained = np.empty(shape, dtype=bool)
    maximum_virtual_power = np.empty(shape)
    maximum_dissipation = np.empty(shape)
    maximum_energy_residual = np.empty(shape)
    normalized_energy_residual = np.empty(shape)
    final_club_speed = np.empty(shape)
    final_q = np.empty((*shape, solution_q.shape[2]))
    trajectory_relative_error = np.empty(shape[:-1])
    force_relative_error = np.empty(shape[:-1])
    profiles = default_synthetic_profiles()

    for state_slot, (case, sample) in enumerate(state_pairs):
        model, metadata = build_subject_scaled_model(profiles[profile_index[case]])
        velocity, _ = finite_difference_kinematics(solution_q[case], source_time)
        for variant_slot, variant in enumerate(variants):
            for step_slot, time_step in enumerate(config.time_steps_s):
                traces: dict[str, dict[str, FloatArray | float]] = {}
                for engine_slot, engine in enumerate(engines):
                    trace = integrate_articulated_contact(
                        model,
                        solution_q[case, sample],
                        velocity[sample],
                        grip_span_m=float(grip_span[case]),
                        hand_contact_local_x_m=float(
                            metadata["hand_contact_local_x_m"]
                        ),
                        time_step_s=time_step,
                        contact_stiffness=(
                            config.contact_stiffness * variant.stiffness_factor
                        ),
                        contact_damping=config.contact_damping * variant.damping_factor,
                        initial_club_displacement_m=(
                            config.initial_club_displacement_m
                            * variant.displacement_factor
                        ),
                        initial_club_velocity_m_s=(
                            config.initial_club_velocity_m_s * variant.velocity_factor
                        ),
                        engine=engine,
                        config=config,
                    )
                    traces[engine] = trace
                    force = np.asarray(trace["maximum_contact_force_n"])
                    separation = np.asarray(trace["maximum_attachment_separation_m"])
                    power = np.asarray(trace["virtual_power_residual_w"])
                    dissipation = np.asarray(trace["contact_dissipation_power_w"])
                    residual = np.asarray(trace["work_energy_residual_j"])
                    total = np.asarray(trace["total_energy_j"])
                    maximum_force[state_slot, variant_slot, step_slot, engine_slot] = (
                        np.max(force)
                    )
                    maximum_separation[
                        state_slot, variant_slot, step_slot, engine_slot
                    ] = np.max(separation)
                    retained[state_slot, variant_slot, step_slot, engine_slot] = bool(
                        np.all(separation <= config.retention_threshold_m)
                    )
                    maximum_virtual_power[
                        state_slot, variant_slot, step_slot, engine_slot
                    ] = np.max(np.abs(power))
                    maximum_dissipation[
                        state_slot, variant_slot, step_slot, engine_slot
                    ] = np.max(dissipation)
                    maximum_energy_residual[
                        state_slot, variant_slot, step_slot, engine_slot
                    ] = np.max(np.abs(residual))
                    scale = max(1.0, float(np.ptp(total)))
                    normalized_energy_residual[
                        state_slot, variant_slot, step_slot, engine_slot
                    ] = np.max(np.abs(residual)) / scale
                    qd_trace = np.asarray(trace["qd"])
                    q_trace = np.asarray(trace["q"])
                    final_club_speed[
                        state_slot, variant_slot, step_slot, engine_slot
                    ] = np.linalg.norm(qd_trace[-1, 14:17])
                    final_q[state_slot, variant_slot, step_slot, engine_slot] = q_trace[
                        -1
                    ]
                trajectory_relative_error[state_slot, variant_slot, step_slot] = (
                    _relative_error(
                        np.asarray(traces["mujoco"]["q"]),
                        np.asarray(traces["pinocchio"]["q"]),
                    )
                )
                force_relative_error[state_slot, variant_slot, step_slot] = (
                    _relative_error(
                        np.asarray(traces["mujoco"]["maximum_contact_force_n"]),
                        np.asarray(traces["pinocchio"]["maximum_contact_force_n"]),
                    )
                )

    cell_gates = (
        retained
        & (maximum_virtual_power <= config.virtual_power_tolerance_w)
        & (maximum_dissipation <= config.positive_dissipation_tolerance_w)
        & (normalized_energy_residual <= config.normalized_energy_residual_tolerance)
    )
    parity_gates = (
        trajectory_relative_error <= config.trajectory_relative_tolerance
    ) & (force_relative_error <= config.trajectory_relative_tolerance)
    refinement_residual = np.max(normalized_energy_residual, axis=(0, 1, 3))
    refinement_direction_passed = bool(
        np.all(np.diff(refinement_residual) <= 0.0)
        and refinement_residual[-1]
        <= config.refinement_ratio_limit * refinement_residual[0]
    )
    state_case_index = np.asarray([pair[0] for pair in state_pairs], dtype=int)
    state_sample_index = np.asarray([pair[1] for pair in state_pairs], dtype=int)
    arrays: dict[str, NDArray[Any]] = {
        "state_case_index": state_case_index,
        "state_sample_index": state_sample_index,
        "state_profile_index": profile_index[state_case_index],
        "state_grip_span_m": grip_span[state_case_index],
        "variant_names": np.asarray([variant.name for variant in variants]),
        "time_steps_s": np.asarray(config.time_steps_s),
        "engine_names": np.asarray(engines),
        "maximum_contact_force_n": maximum_force,
        "maximum_attachment_separation_m": maximum_separation,
        "retained": retained,
        "maximum_virtual_power_residual_w": maximum_virtual_power,
        "maximum_contact_dissipation_power_w": maximum_dissipation,
        "maximum_work_energy_residual_j": maximum_energy_residual,
        "normalized_work_energy_residual": normalized_energy_residual,
        "final_club_translation_speed_m_s": final_club_speed,
        "final_q": final_q,
        "trajectory_relative_error": trajectory_relative_error,
        "force_relative_error": force_relative_error,
        "cell_gates_passed": cell_gates,
        "parity_gates_passed": parity_gates,
        "refinement_worst_normalized_residual": refinement_residual,
    }
    all_passed = bool(
        np.all(cell_gates) and np.all(parity_gates) and refinement_direction_passed
    )
    record: dict[str, Any] = {
        "schema_version": "articulated-forward-contact/v1",
        "study_id": "bounded-subject-scaled-articulated-bilateral-contact",
        "model_tier": "bounded_articulated_bilateral_attachment_forward_dynamics",
        "design": {
            "engine_names": list(engines),
            "engine_versions": {
                "mujoco": str(mujoco.__version__),
                "pinocchio": str(pin.__version__),
            },
            "bounded_horizon_s": config.duration_s,
            "state_count": len(state_pairs),
            "profile_count": int(np.unique(profile_index[state_case_index]).size),
            "grip_span_count": int(np.unique(grip_span[state_case_index]).size),
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
        "results": {
            "maximum_contact_force_n": float(np.max(maximum_force)),
            "maximum_attachment_separation_m": float(np.max(maximum_separation)),
            "failed_retention_cell_count": int(np.count_nonzero(~retained)),
            "maximum_virtual_power_residual_w": float(np.max(maximum_virtual_power)),
            "maximum_positive_dissipation_power_w": float(np.max(maximum_dissipation)),
            "maximum_work_energy_residual_j": float(np.max(maximum_energy_residual)),
            "maximum_normalized_work_energy_residual": float(
                np.max(normalized_energy_residual)
            ),
            "maximum_trajectory_relative_error": float(
                np.max(trajectory_relative_error)
            ),
            "maximum_force_relative_error": float(np.max(force_relative_error)),
            "refinement_worst_normalized_residual": refinement_residual.tolist(),
            "refinement_direction_passed": refinement_direction_passed,
            "failed_cell_count": int(np.count_nonzero(~cell_gates)),
            "failed_parity_count": int(np.count_nonzero(~parity_gates)),
            "all_registered_gates_passed": all_passed,
        },
        "interpretation": {
            "retention": (
                "attachment separation remains below a declared screening threshold; "
                "this is not unilateral biological contact"
            ),
            "right_censoring": (
                "no registered failure before 5 ms does not establish persistence "
                "beyond the bounded horizon"
            ),
            "adverse_controls": (
                "stiffness, damping, velocity reversal, and zero-preload branches "
                "are reported without selecting a preferred mechanism"
            ),
        },
        "claim_boundary": {
            "supported": (
                "the declared synthetic bilateral attachments can be advanced through "
                "the bounded articulated horizon subject to the registered gates"
            ),
            "distributed_grip_shaft_or_ground": "not_modeled",
            "late_downswing_or_impact": "not_established_by_5_ms_horizon",
            "anatomy_or_muscle_action": "not_calibrated_or_inferred",
            "slack_benefit_or_timing_economy": "untested",
            "human_transfer_or_strategy": "untested",
        },
        "next_gate": (
            "extend the right-censored horizon with unilateral/typed slack, distributed "
            "grip and shaft compliance, ground coupling, and governed human wrenches"
        ),
        "source_sha256": {path: _sha256(REPO_ROOT / path) for path in SOURCE_PATHS},
    }
    return record, arrays


__all__ = [
    "ArticulatedForwardContactConfig",
    "ForwardVariant",
    "integrate_articulated_contact",
    "mechanical_energy",
    "registered_variants",
    "run_articulated_forward_contact_atlas",
]
