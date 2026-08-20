"""Closed-contact feasibility for the subject-scaled spatial model tier.

This module solves bilateral point-contact inverse kinematics while holding the
six prescribed club coordinates fixed.  Passing the solve is a necessary
geometric gate, not evidence of anatomical realism, passive dynamics, human
strategy, or a beneficial transfer mechanism.  The declared joint limits and
spherical collision screen are engineering guards for this reduced tree.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from scripts.research.proximal_distal_energy.spatial_full_body import (
    Kinematics,
    SpatialModel,
    forward_kinematics,
    point_contact_jacobians,
    prescribed_state,
)
from scripts.research.proximal_distal_energy.subject_scaled_spatial_geometry import (
    SyntheticSubjectProfile,
    build_subject_scaled_model,
    default_synthetic_profiles,
)

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]
ACTIVE_DOF_COUNT = 14


@dataclass(frozen=True, slots=True)
class ClosedContactConfig:
    """Predeclared numerical and feasibility gates for one IK solve."""

    closure_tolerance_m: float = 5.0e-4
    closure_residual_scale_m: float = 1.0e-3
    regularization_weight: float = 1.0e-2
    maximum_function_evaluations: int = 1000
    solver_tolerance: float = 1.0e-11
    joint_limit_scale: float = 1.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.closure_tolerance_m) or not (
            0.0 < self.closure_tolerance_m <= 0.005
        ):
            raise ValueError("closure_tolerance_m must be in (0, 0.005]")
        if not np.isfinite(self.closure_residual_scale_m) or not (
            0.0 < self.closure_residual_scale_m <= 0.01
        ):
            raise ValueError("closure_residual_scale_m must be in (0, 0.01]")
        if not np.isfinite(self.regularization_weight) or not (
            self.regularization_weight > 0.0
        ):
            raise ValueError("regularization_weight must be finite and positive")
        if self.maximum_function_evaluations < 10:
            raise ValueError("maximum_function_evaluations must be at least 10")
        if not np.isfinite(self.solver_tolerance) or not (
            1.0e-14 <= self.solver_tolerance <= 1.0e-6
        ):
            raise ValueError("solver_tolerance must be in [1e-14, 1e-6]")
        if not np.isfinite(self.joint_limit_scale) or not (
            0.5 <= self.joint_limit_scale <= 1.5
        ):
            raise ValueError("joint_limit_scale must be in [0.5, 1.5]")


@dataclass(frozen=True, slots=True)
class ClosedContactSolution:
    """One solver result with every feasibility gate reported separately."""

    q: FloatArray
    hand_to_grip_distance_m: FloatArray
    solver_converged: bool
    contact_closed: bool
    joint_limits_satisfied: bool
    collision_free: bool
    feasible: bool
    minimum_joint_limit_margin_rad: float
    minimum_collision_clearance_m: float
    closest_collision_pair: tuple[str, str]
    constraint_jacobian_rank: int
    constraint_jacobian_singular_values: FloatArray
    function_evaluations: int
    cost: float


def engineering_joint_bounds(
    model: SpatialModel, joint_limit_scale: float = 1.0
) -> tuple[FloatArray, FloatArray]:
    """Return declared reduced-model bounds; club coordinates remain fixed.

    These broad bounds prevent periodic wraparound and extreme reduced-tree
    postures.  They are not clinical ranges of motion or subject-specific
    anatomical limits.
    """

    if model.nq != 20 or model.club_dof_indices.tolist() != list(range(14, 20)):
        raise ValueError("closed-contact bounds require the canonical 20-DOF tree")
    if not np.isfinite(joint_limit_scale) or not 0.5 <= joint_limit_scale <= 1.5:
        raise ValueError("joint_limit_scale must be in [0.5, 1.5]")
    lower = np.full(model.nq, -np.inf, dtype=np.float64)
    upper = np.full(model.nq, np.inf, dtype=np.float64)
    named_bounds = {
        "pelvis_yaw": (-1.20, 1.20),
        "pelvis_roll": (-0.60, 0.60),
        "torso_pitch": (-1.00, 1.00),
        "torso_yaw": (-1.50, 1.50),
        "lead_shoulder_x": (-2.80, 2.80),
        "lead_shoulder_y": (-2.80, 2.80),
        "lead_shoulder_z": (-2.80, 2.80),
        "lead_elbow": (-0.10, 2.70),
        "lead_wrist": (-1.50, 1.50),
        "trail_shoulder_x": (-2.80, 2.80),
        "trail_shoulder_y": (-2.80, 2.80),
        "trail_shoulder_z": (-2.80, 2.80),
        "trail_elbow": (-0.10, 2.70),
        "trail_wrist": (-1.50, 1.50),
    }
    for index, joint in enumerate(model.joints[:ACTIVE_DOF_COUNT]):
        try:
            base_lower, base_upper = named_bounds[joint.name]
            lower[index] = joint_limit_scale * base_lower
            upper[index] = joint_limit_scale * base_upper
        except KeyError as error:
            raise ValueError(f"no declared bound for {joint.name}") from error
    return lower, upper


def _contact_kinematics(
    model: SpatialModel,
    q: FloatArray,
    *,
    grip_span_m: float,
    hand_contact_local_x_m: float,
) -> tuple[FloatArray, FloatArray, Kinematics]:
    kin = forward_kinematics(model, q)
    hand_local = np.array([hand_contact_local_x_m, 0.0, 0.0])
    grip_locals = (
        np.array([0.0, grip_span_m / 2.0, -0.03]),
        np.array([0.0, -grip_span_m / 2.0, -0.03]),
    )
    errors: list[FloatArray] = []
    jacobians: list[FloatArray] = []
    for hand_joint, grip_local in zip(
        (model.lead_hand_joint, model.trail_hand_joint), grip_locals, strict=True
    ):
        hand_point, hand_jv, _ = point_contact_jacobians(
            model, kin, hand_joint, hand_local
        )
        grip_point, grip_jv, _ = point_contact_jacobians(
            model, kin, model.club_frame_joint, grip_local
        )
        errors.append(hand_point - grip_point)
        jacobians.append(hand_jv - grip_jv)
    return np.asarray(errors), np.vstack(jacobians), kin


def _collision_clearance(
    model: SpatialModel, kin: Kinematics
) -> tuple[float, tuple[str, str]]:
    physical = [
        (index, body)
        for index, body in enumerate(model.bodies)
        if not body.name.startswith("joint_carrier_")
    ]
    exempt = {
        frozenset(("lower_body", "pelvis_mass")),
        frozenset(("pelvis_mass", "torso_mass")),
        frozenset(("torso_mass", "lead_upper_arm")),
        frozenset(("torso_mass", "trail_upper_arm")),
        frozenset(("lead_upper_arm", "lead_forearm")),
        frozenset(("lead_forearm", "lead_hand")),
        frozenset(("trail_upper_arm", "trail_forearm")),
        frozenset(("trail_forearm", "trail_hand")),
        frozenset(("lead_hand", "club_grip_mass")),
        frozenset(("trail_hand", "club_grip_mass")),
    }
    minimum = np.inf
    closest = ("none", "none")
    for position, (first_index, first) in enumerate(physical):
        for second_index, second in physical[position + 1 :]:
            pair = frozenset((first.name, second.name))
            if pair in exempt:
                continue
            if first.region == second.region == "club":
                continue
            clearance = float(
                np.linalg.norm(
                    kin.body_position_m[first_index] - kin.body_position_m[second_index]
                )
                - first.radius_m
                - second.radius_m
            )
            if clearance < minimum:
                minimum = clearance
                closest = (first.name, second.name)
    if not np.isfinite(minimum):
        raise RuntimeError("collision screen produced no non-exempt body pairs")
    return minimum, closest


def _rank_and_singular_values(matrix: FloatArray) -> tuple[int, FloatArray]:
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    threshold = np.finfo(float).eps * max(matrix.shape) * singular_values[0]
    return int(np.count_nonzero(singular_values > threshold)), singular_values


def _finalize_solution(
    model: SpatialModel,
    q_reference: FloatArray,
    active: NDArray[np.int_],
    lower: FloatArray,
    upper: FloatArray,
    result: Any,
    contact_geometry: tuple[float, float],
    config: ClosedContactConfig,
) -> ClosedContactSolution:
    grip_span_m, hand_contact_local_x_m = contact_geometry
    q = q_reference.copy()
    q[active] = result.x
    errors, constraint_jacobian, kinematics = _contact_kinematics(
        model,
        q,
        grip_span_m=grip_span_m,
        hand_contact_local_x_m=hand_contact_local_x_m,
    )
    distances = np.linalg.norm(errors, axis=1)
    contact_closed = bool(np.max(distances) <= config.closure_tolerance_m)
    margins = np.minimum(result.x - lower[active], upper[active] - result.x)
    minimum_joint_margin = float(np.min(margins))
    limits_satisfied = minimum_joint_margin >= -1.0e-10
    collision_clearance, closest_pair = _collision_clearance(model, kinematics)
    collision_free = collision_clearance >= 0.0
    rank, singular_values = _rank_and_singular_values(constraint_jacobian)
    converged = bool(result.success and np.all(np.isfinite(result.x)))
    feasible = bool(
        converged
        and contact_closed
        and limits_satisfied
        and collision_free
        and rank == 6
    )
    return ClosedContactSolution(
        q=q,
        hand_to_grip_distance_m=distances,
        solver_converged=converged,
        contact_closed=contact_closed,
        joint_limits_satisfied=limits_satisfied,
        collision_free=collision_free,
        feasible=feasible,
        minimum_joint_limit_margin_rad=minimum_joint_margin,
        minimum_collision_clearance_m=collision_clearance,
        closest_collision_pair=closest_pair,
        constraint_jacobian_rank=rank,
        constraint_jacobian_singular_values=singular_values,
        function_evaluations=int(result.nfev),
        cost=float(result.cost),
    )


def solve_closed_contact_configuration(
    model: SpatialModel,
    *,
    q_reference: FloatArray,
    grip_span_m: float,
    hand_contact_local_x_m: float,
    q_seed: FloatArray | None = None,
    config: ClosedContactConfig = ClosedContactConfig(),
) -> ClosedContactSolution:
    """Solve the body/arm coordinates while preserving the club pose."""

    q_reference = np.asarray(q_reference, dtype=np.float64)
    if q_reference.shape != (model.nq,) or not np.all(np.isfinite(q_reference)):
        raise ValueError(f"q_reference must be finite with shape ({model.nq},)")
    if not np.isfinite(grip_span_m) or grip_span_m <= 0.0:
        raise ValueError("grip_span_m must be finite and positive")
    if not np.isfinite(hand_contact_local_x_m) or hand_contact_local_x_m <= 0.0:
        raise ValueError("hand_contact_local_x_m must be finite and positive")
    if not isinstance(config, ClosedContactConfig):
        raise TypeError("config must be a ClosedContactConfig")

    lower, upper = engineering_joint_bounds(model, config.joint_limit_scale)
    active = np.arange(ACTIVE_DOF_COUNT)
    seed = (
        q_reference.copy() if q_seed is None else np.asarray(q_seed, dtype=float).copy()
    )
    if seed.shape != (model.nq,) or not np.all(np.isfinite(seed)):
        raise ValueError(f"q_seed must be finite with shape ({model.nq},)")
    seed[model.club_dof_indices] = q_reference[model.club_dof_indices]
    x0 = np.clip(seed[active], lower[active], upper[active])

    def residual(active_q: FloatArray) -> FloatArray:
        q = q_reference.copy()
        q[active] = active_q
        errors, _, _ = _contact_kinematics(
            model,
            q,
            grip_span_m=grip_span_m,
            hand_contact_local_x_m=hand_contact_local_x_m,
        )
        return np.concatenate(
            (
                errors.ravel() / config.closure_residual_scale_m,
                config.regularization_weight * (active_q - q_reference[active]),
            )
        )

    def jacobian(active_q: FloatArray) -> FloatArray:
        q = q_reference.copy()
        q[active] = active_q
        _, contact_jacobian, _ = _contact_kinematics(
            model,
            q,
            grip_span_m=grip_span_m,
            hand_contact_local_x_m=hand_contact_local_x_m,
        )
        return np.vstack(
            (
                contact_jacobian[:, active] / config.closure_residual_scale_m,
                config.regularization_weight * np.eye(active.size),
            )
        )

    result = least_squares(
        residual,
        x0,
        jac=jacobian,
        bounds=(lower[active], upper[active]),
        ftol=config.solver_tolerance,
        xtol=config.solver_tolerance,
        gtol=config.solver_tolerance,
        max_nfev=config.maximum_function_evaluations,
    )
    return _finalize_solution(
        model,
        q_reference,
        active,
        lower,
        upper,
        result,
        (grip_span_m, hand_contact_local_x_m),
        config,
    )


def _atlas_inputs(
    profiles: Sequence[SyntheticSubjectProfile] | None = None,
    grip_spans_m: FloatArray | None = None,
    time_s: FloatArray | None = None,
) -> tuple[tuple[SyntheticSubjectProfile, ...], FloatArray, FloatArray]:
    selected_profiles = tuple(
        default_synthetic_profiles() if profiles is None else profiles
    )
    spans = np.asarray(
        [0.12, 0.18, 0.24] if grip_spans_m is None else grip_spans_m,
        dtype=np.float64,
    )
    times = np.asarray(
        np.linspace(0.0, 0.24, 13) if time_s is None else time_s,
        dtype=np.float64,
    )
    if not selected_profiles or not all(
        isinstance(profile, SyntheticSubjectProfile) for profile in selected_profiles
    ):
        raise ValueError("profiles must contain SyntheticSubjectProfile values")
    if (
        spans.ndim != 1
        or spans.size == 0
        or not np.all(np.isfinite(spans))
        or np.any(spans <= 0.0)
    ):
        raise ValueError("grip_spans_m must be one-dimensional, finite, and positive")
    if (
        times.ndim != 1
        or times.size == 0
        or not np.all(np.isfinite(times))
        or np.any(times < 0.0)
    ):
        raise ValueError("time_s must be one-dimensional, finite, and nonnegative")
    return selected_profiles, spans, times


def _atlas_record(
    selected_profiles: tuple[SyntheticSubjectProfile, ...],
    spans: FloatArray,
    times: FloatArray,
    profile_records: list[dict[str, Any]],
    config: ClosedContactConfig,
    arrays: dict[str, NDArray[Any]],
) -> dict[str, Any]:
    feasible = arrays["feasible"]
    distances = arrays["hand_to_grip_distance_m"]
    margin = arrays["minimum_joint_limit_margin_rad"]
    clearance = arrays["minimum_collision_clearance_m"]
    evaluations = arrays["function_evaluations"]
    change = arrays["adjacent_configuration_change_rad"]
    rank = arrays["constraint_jacobian_rank"]
    return {
        "schema_version": "subject-scaled-closed-contact/v1",
        "study_id": "subject-scaled-bilateral-closed-contact-feasibility",
        "model_tier": "reduced_articulated_subject_scaled_inverse_kinematics",
        "design": {
            "profile_count": len(selected_profiles),
            "grip_span_count": int(spans.size),
            "case_count": int(feasible.shape[0]),
            "time_sample_count": int(times.size),
            "grip_spans_m": spans.tolist(),
            "time_s": times.tolist(),
            "profiles": profile_records,
            "club_coordinates": "held_at_each_prescribed_reference_state",
            "solved_coordinates": [
                joint.name
                for joint in build_subject_scaled_model(selected_profiles[0])[0].joints[
                    :ACTIVE_DOF_COUNT
                ]
            ],
        },
        "registered_gates": {
            "solver": "bounded_trust_region_reflective_least_squares",
            "configuration": asdict(config),
            "joint_limits": "declared_reduced_model_engineering_bounds_not_clinical_ranges",
            "collision": "nonadjacent_physical_body_bounding_spheres",
            "constraint_rank_required": 6,
        },
        "results": {
            "feasible_sample_count": int(np.count_nonzero(feasible)),
            "total_sample_count": int(feasible.size),
            "feasible_fraction": float(np.mean(feasible)),
            "maximum_contact_error_m": float(np.max(distances)),
            "minimum_joint_limit_margin_rad": float(np.min(margin)),
            "minimum_collision_clearance_m": float(np.min(clearance)),
            "maximum_function_evaluations": int(np.max(evaluations)),
            "maximum_adjacent_configuration_change_rad": float(np.max(change)),
            "median_adjacent_configuration_change_rad": float(np.median(change)),
            "constraint_jacobian_rank_values": sorted(
                int(value) for value in np.unique(rank)
            ),
        },
        "claim_status": {
            "closed_contact_feasibility": "evaluated_in_declared_reduced_tree",
            "anatomical_feasibility": "not_established",
            "passive_transfer": "untested",
            "timing_or_slack_benefit": "untested",
            "human_strategy": "untested",
        },
        "limitations": [
            "Joint limits are broad engineering guards, not clinical or subject-specific ranges of motion.",
            "Collision uses reduced spherical bounds and exempt connected or intended-contact pairs; it is not mesh-level anatomical clearance.",
            "The scapula, forearm pronation-supination, multi-axis wrist, fingers, and distributed grip contact are absent.",
            "Inverse kinematics establishes no force, work, passivity, timing, robustness, slack, or delivery result.",
            "Synthetic de Leva design points are not participants or a population distribution.",
        ],
        "array_artifact": "subject_scaled_closed_contact.npz",
    }


def run_closed_contact_feasibility_atlas(
    *,
    profiles: Sequence[SyntheticSubjectProfile] | None = None,
    grip_spans_m: FloatArray | None = None,
    time_s: FloatArray | None = None,
    config: ClosedContactConfig = ClosedContactConfig(),
) -> tuple[dict[str, Any], dict[str, NDArray[Any]]]:
    """Run a deterministic continuation atlas across profiles, spans, and time."""

    selected_profiles, spans, times = _atlas_inputs(profiles, grip_spans_m, time_s)

    case_count = len(selected_profiles) * spans.size
    shape = (case_count, times.size)
    feasible = np.zeros(shape, dtype=bool)
    converged = np.zeros(shape, dtype=bool)
    closed = np.zeros(shape, dtype=bool)
    collision_free = np.zeros(shape, dtype=bool)
    distances = np.empty((*shape, 2))
    limit_margin = np.empty(shape)
    collision_clearance = np.empty(shape)
    rank = np.empty(shape, dtype=np.int64)
    function_evaluations = np.empty(shape, dtype=np.int64)
    solution_q = np.empty((*shape, 20))
    case_profile_index = np.empty(case_count, dtype=np.int64)
    case_grip_span = np.empty(case_count)
    profile_records: list[dict[str, Any]] = []

    case = 0
    for profile_index, profile in enumerate(selected_profiles):
        model, metadata = build_subject_scaled_model(profile)
        profile_records.append(metadata)
        hand_contact = float(metadata["hand_contact_local_x_m"])
        for span in spans:
            case_profile_index[case] = profile_index
            case_grip_span[case] = span
            previous: FloatArray | None = None
            for time_index, sample_time in enumerate(times):
                q_reference, _, _ = prescribed_state(model, float(sample_time))
                solution = solve_closed_contact_configuration(
                    model,
                    q_reference=q_reference,
                    grip_span_m=float(span),
                    hand_contact_local_x_m=hand_contact,
                    q_seed=previous,
                    config=config,
                )
                previous = solution.q if solution.contact_closed else None
                feasible[case, time_index] = solution.feasible
                converged[case, time_index] = solution.solver_converged
                closed[case, time_index] = solution.contact_closed
                collision_free[case, time_index] = solution.collision_free
                distances[case, time_index] = solution.hand_to_grip_distance_m
                limit_margin[case, time_index] = solution.minimum_joint_limit_margin_rad
                collision_clearance[case, time_index] = (
                    solution.minimum_collision_clearance_m
                )
                rank[case, time_index] = solution.constraint_jacobian_rank
                function_evaluations[case, time_index] = solution.function_evaluations
                solution_q[case, time_index] = solution.q
            case += 1

    adjacent_configuration_change = np.linalg.norm(
        np.diff(solution_q[:, :, :ACTIVE_DOF_COUNT], axis=1), axis=2
    )
    arrays: dict[str, NDArray[Any]] = {
        "time_s": times,
        "grip_spans_m": spans,
        "case_profile_index": case_profile_index,
        "case_grip_span_m": case_grip_span,
        "feasible": feasible,
        "solver_converged": converged,
        "contact_closed": closed,
        "collision_free": collision_free,
        "hand_to_grip_distance_m": distances,
        "minimum_joint_limit_margin_rad": limit_margin,
        "minimum_collision_clearance_m": collision_clearance,
        "constraint_jacobian_rank": rank,
        "function_evaluations": function_evaluations,
        "solution_q": solution_q,
        "adjacent_configuration_change_rad": adjacent_configuration_change,
    }
    return _atlas_record(
        selected_profiles, spans, times, profile_records, config, arrays
    ), arrays


__all__ = [
    "ClosedContactConfig",
    "ClosedContactSolution",
    "engineering_joint_bounds",
    "run_closed_contact_feasibility_atlas",
    "solve_closed_contact_configuration",
]
