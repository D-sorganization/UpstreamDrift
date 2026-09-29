"""Paired fixed-shoulder and scapula-on-ellipsoid contact geometry screen.

This is a subject-scaled kinematic falsification tier.  It holds the prescribed
trunk and club pose fixed, solves only the two arms, and compares the existing
fixed shoulder centers with four scapular coordinates per side: protraction,
elevation, upward rotation, and winging.  The first two move the shoulder center
on a declared thoracic ellipsoid; the latter two precede glenohumeral rotation.

The construction is informed by scapulothoracic models, but it is not a
subject-specific OpenSim shoulder, muscle model, clinical range assessment, or
human technique identifier.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
import math
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from scripts.research.proximal_distal_energy.spatial_full_body import (
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
Side = Literal["lead", "trail"]


@dataclass(frozen=True, slots=True)
class ScapulothoracicConfig:
    """Predeclared geometry, range, and numerical contract."""

    ellipsoid_radii_m: tuple[float, float, float] = (0.18, 0.25, 0.30)
    closure_tolerance_m: float = 5.0e-4
    closure_residual_scale_m: float = 1.0e-3
    regularization_weight: float = 1.0e-2
    protraction_limit_rad: float = math.radians(20.0)
    elevation_limit_rad: float = math.radians(15.0)
    upward_rotation_limit_rad: float = math.radians(30.0)
    winging_limit_rad: float = math.radians(15.0)
    finite_difference_step_rad: float = 1.0e-6
    maximum_function_evaluations: int = 600

    def __post_init__(self) -> None:
        radii = np.asarray(self.ellipsoid_radii_m, dtype=float)
        if radii.shape != (3,) or not np.all(np.isfinite(radii)) or np.any(radii <= 0):
            raise ValueError("ellipsoid_radii_m must contain three positive values")
        if not 0.0 < self.closure_tolerance_m <= 0.005:
            raise ValueError("closure_tolerance_m must be in (0, 0.005]")
        if not 0.0 < self.closure_residual_scale_m <= 0.01:
            raise ValueError("closure_residual_scale_m must be in (0, 0.01]")
        if not 0.0 < self.regularization_weight <= 0.1:
            raise ValueError("regularization_weight must be in (0, 0.1]")
        for name in (
            "protraction_limit_rad",
            "elevation_limit_rad",
            "upward_rotation_limit_rad",
            "winging_limit_rad",
            "finite_difference_step_rad",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.maximum_function_evaluations < 100:
            raise ValueError("maximum_function_evaluations must be at least 100")


@dataclass(frozen=True, slots=True)
class _GeometryContext:
    torso_origin_m: FloatArray
    torso_rotation: FloatArray
    grip_points_m: FloatArray
    upper_arm_m: float
    forearm_plus_hand_m: float
    elbow_drop_m: float
    linear_scale: float


@dataclass(frozen=True, slots=True)
class _SolveResult:
    coordinates: FloatArray
    hand_points_m: FloatArray
    shoulder_points_m: FloatArray
    max_contact_error_m: float
    contact_closed: bool
    solver_termination_success: bool
    contact_jacobian_rank: int
    contact_jacobian_nullity: int
    contact_jacobian_singular_values: FloatArray
    minimum_coordinate_bound_margin_rad: float


def _rotation(axis: FloatArray, angle: float) -> FloatArray:
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    skew = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def ellipsoid_surface_point(
    config: ScapulothoracicConfig,
    side: Side,
    protraction_rad: float,
    elevation_rad: float,
    *,
    linear_scale: float,
) -> FloatArray:
    """Return the declared shoulder center on the thoracic ellipsoid."""
    if side not in {"lead", "trail"}:
        raise ValueError("side must be 'lead' or 'trail'")
    if not np.isfinite(linear_scale) or linear_scale <= 0.0:
        raise ValueError("linear_scale must be finite and positive")
    a, b, c = np.asarray(config.ellipsoid_radii_m) * linear_scale
    azimuth = (math.pi / 2.0 if side == "lead" else -math.pi / 2.0) + protraction_rad
    base_elevation = math.asin(0.18 / config.ellipsoid_radii_m[2])
    latitude = base_elevation + elevation_rad
    return np.array(
        [
            a * math.cos(latitude) * math.cos(azimuth),
            b * math.cos(latitude) * math.sin(azimuth),
            c * math.sin(latitude),
        ]
    )


def _arm_hand_point(
    context: _GeometryContext,
    side: Side,
    scapular: FloatArray,
    arm: FloatArray,
    config: ScapulothoracicConfig,
) -> tuple[FloatArray, FloatArray]:
    shoulder_local = ellipsoid_surface_point(
        config,
        side,
        float(scapular[0]),
        float(scapular[1]),
        linear_scale=context.linear_scale,
    )
    shoulder = context.torso_origin_m + context.torso_rotation @ shoulder_local
    scapular_rotation = _rotation(np.array([0.0, 0.0, 1.0]), scapular[2]) @ _rotation(
        np.array([1.0, 0.0, 0.0]), scapular[3]
    )
    shoulder_rotation = (
        _rotation(np.array([1.0, 0.0, 0.0]), arm[0])
        @ _rotation(np.array([0.0, 1.0, 0.0]), arm[1])
        @ _rotation(np.array([0.0, 0.0, 1.0]), arm[2])
    )
    upper_rotation = context.torso_rotation @ scapular_rotation @ shoulder_rotation
    elbow = shoulder + upper_rotation @ np.array(
        [context.upper_arm_m, 0.0, context.elbow_drop_m]
    )
    forearm_rotation = upper_rotation @ _rotation(np.array([0.0, 1.0, 0.0]), arm[3])
    hand = elbow + forearm_rotation @ np.array([context.forearm_plus_hand_m, 0.0, 0.0])
    return hand, shoulder


def _positions(
    coordinates: FloatArray,
    context: _GeometryContext,
    config: ScapulothoracicConfig,
    *,
    include_scapula: bool,
) -> tuple[FloatArray, FloatArray]:
    block_size = 8 if include_scapula else 4
    hands: list[FloatArray] = []
    shoulders: list[FloatArray] = []
    for index, side in enumerate(("lead", "trail")):
        block = coordinates[index * block_size : (index + 1) * block_size]
        scapular = block[:4] if include_scapula else np.zeros(4)
        arm = block[4:] if include_scapula else block
        hand, shoulder = _arm_hand_point(context, side, scapular, arm, config)
        hands.append(hand)
        shoulders.append(shoulder)
    return np.asarray(hands), np.asarray(shoulders)


def _contact_jacobian(
    coordinates: FloatArray,
    context: _GeometryContext,
    config: ScapulothoracicConfig,
    *,
    include_scapula: bool,
) -> FloatArray:
    step = config.finite_difference_step_rad
    jacobian = np.empty((6, coordinates.size))
    for index in range(coordinates.size):
        delta = np.zeros_like(coordinates)
        delta[index] = step
        plus, _ = _positions(
            coordinates + delta, context, config, include_scapula=include_scapula
        )
        minus, _ = _positions(
            coordinates - delta, context, config, include_scapula=include_scapula
        )
        jacobian[:, index] = ((plus - minus) / (2.0 * step)).ravel()
    return jacobian


def _rank(jacobian: FloatArray) -> tuple[int, FloatArray]:
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    threshold = np.finfo(float).eps * max(jacobian.shape) * singular_values[0]
    return int(np.count_nonzero(singular_values > threshold)), singular_values


def _bounds(
    config: ScapulothoracicConfig, *, include_scapula: bool
) -> tuple[FloatArray, FloatArray]:
    arm_lower = np.array([-2.8, -2.8, -2.8, -0.1])
    arm_upper = np.array([2.8, 2.8, 2.8, 2.7])
    if not include_scapula:
        return np.tile(arm_lower, 2), np.tile(arm_upper, 2)
    scap_upper = np.array(
        [
            config.protraction_limit_rad,
            config.elevation_limit_rad,
            config.upward_rotation_limit_rad,
            config.winging_limit_rad,
        ]
    )
    lower = np.concatenate((-scap_upper, arm_lower, -scap_upper, arm_lower))
    upper = np.concatenate((scap_upper, arm_upper, scap_upper, arm_upper))
    return lower, upper


def _solve(
    context: _GeometryContext,
    reference: FloatArray,
    config: ScapulothoracicConfig,
    *,
    include_scapula: bool,
    seed: FloatArray | None = None,
    fallback: FloatArray | None = None,
) -> _SolveResult:
    lower, upper = _bounds(config, include_scapula=include_scapula)
    initial = np.clip(reference if seed is None else seed, lower, upper)

    def residual(values: FloatArray) -> FloatArray:
        hands, _ = _positions(values, context, config, include_scapula=include_scapula)
        contact = (hands - context.grip_points_m).ravel()
        return np.concatenate(
            (
                contact / config.closure_residual_scale_m,
                config.regularization_weight * (values - reference),
            )
        )

    result = least_squares(
        residual,
        initial,
        bounds=(lower, upper),
        ftol=1.0e-11,
        xtol=1.0e-11,
        gtol=1.0e-11,
        max_nfev=config.maximum_function_evaluations,
    )
    candidates = (
        (initial, result.x)
        if fallback is None
        else (initial, result.x, np.clip(fallback, lower, upper))
    )
    coordinate = min(
        candidates,
        key=lambda values: float(
            np.max(
                np.linalg.norm(
                    _positions(
                        values, context, config, include_scapula=include_scapula
                    )[0]
                    - context.grip_points_m,
                    axis=1,
                )
            )
        ),
    )
    hands, shoulders = _positions(
        coordinate, context, config, include_scapula=include_scapula
    )
    error = float(np.max(np.linalg.norm(hands - context.grip_points_m, axis=1)))
    jacobian = _contact_jacobian(
        coordinate, context, config, include_scapula=include_scapula
    )
    rank, singular_values = _rank(jacobian)
    bound_margin = np.minimum(coordinate - lower, upper - coordinate)
    return _SolveResult(
        coordinates=coordinate,
        hand_points_m=hands,
        shoulder_points_m=shoulders,
        max_contact_error_m=error,
        contact_closed=bool(error <= config.closure_tolerance_m),
        solver_termination_success=bool(
            result.success and np.all(np.isfinite(result.x))
        ),
        contact_jacobian_rank=rank,
        contact_jacobian_nullity=int(coordinate.size - rank),
        contact_jacobian_singular_values=singular_values,
        minimum_coordinate_bound_margin_rad=float(np.min(bound_margin)),
    )


def _context(
    model: SpatialModel,
    q_reference: FloatArray,
    metadata: dict[str, Any],
    grip_span_m: float,
) -> _GeometryContext:
    kin = forward_kinematics(model, q_reference)
    grip_points = []
    for local in (
        np.array([0.0, grip_span_m / 2.0, -0.03]),
        np.array([0.0, -grip_span_m / 2.0, -0.03]),
    ):
        point, _, _ = point_contact_jacobians(model, kin, model.club_frame_joint, local)
        grip_points.append(point)
    lengths = metadata["segment_lengths_m"]
    return _GeometryContext(
        torso_origin_m=kin.joint_position_m[3],
        torso_rotation=kin.joint_rotation[3],
        grip_points_m=np.asarray(grip_points),
        upper_arm_m=float(lengths["upper_arm"]),
        forearm_plus_hand_m=float(lengths["forearm"] + 0.5 * lengths["hand"]),
        elbow_drop_m=float(-0.05 * metadata["linear_scale_from_1_75_m"]),
        linear_scale=float(metadata["linear_scale_from_1_75_m"]),
    )


def _fixed_reference(q: FloatArray) -> FloatArray:
    return np.concatenate((q[4:8], q[9:13]))


def _scapular_reference(fixed: FloatArray) -> FloatArray:
    return np.concatenate((np.zeros(4), fixed[:4], np.zeros(4), fixed[4:]))


def _embedded_fixed_seed(fixed_solution: _SolveResult) -> FloatArray:
    values = fixed_solution.coordinates
    return np.concatenate((np.zeros(4), values[:4], np.zeros(4), values[4:]))


def _validated_design(
    profiles: Sequence[SyntheticSubjectProfile] | None,
    grip_spans_m: FloatArray | None,
    time_s: FloatArray | None,
    adverse_grip_span_m: float,
) -> tuple[tuple[SyntheticSubjectProfile, ...], FloatArray, FloatArray]:
    selected = tuple(default_synthetic_profiles() if profiles is None else profiles)
    spans = np.asarray([0.12, 0.18, 0.24] if grip_spans_m is None else grip_spans_m)
    times = np.asarray(np.array([0.0, 0.12, 0.24]) if time_s is None else time_s)
    if not selected or not all(
        isinstance(item, SyntheticSubjectProfile) for item in selected
    ):
        raise ValueError("profiles must contain SyntheticSubjectProfile values")
    if (
        spans.ndim != 1
        or spans.size == 0
        or np.any(~np.isfinite(spans))
        or np.any(spans <= 0)
    ):
        raise ValueError("grip_spans_m must be one-dimensional, finite, and positive")
    if (
        times.ndim != 1
        or times.size == 0
        or np.any(~np.isfinite(times))
        or np.any(times < 0)
    ):
        raise ValueError("time_s must be one-dimensional, finite, and nonnegative")
    if not np.isfinite(adverse_grip_span_m) or adverse_grip_span_m <= float(
        np.max(spans)
    ):
        raise ValueError("adverse_grip_span_m must exceed every registered grip span")
    return selected, spans, times


def _empty_result_arrays(shape: tuple[int, int]) -> dict[str, FloatArray]:
    return {
        "fixed_max_contact_error_m": np.empty(shape),
        "scapular_max_contact_error_m": np.empty(shape),
        "fixed_contact_jacobian_rank": np.empty(shape, dtype=np.int64),
        "scapular_contact_jacobian_rank": np.empty(shape, dtype=np.int64),
        "fixed_contact_jacobian_nullity": np.empty(shape, dtype=np.int64),
        "scapular_contact_jacobian_nullity": np.empty(shape, dtype=np.int64),
        "fixed_solver_termination_success": np.empty(shape, dtype=bool),
        "scapular_solver_termination_success": np.empty(shape, dtype=bool),
        "fixed_minimum_bound_margin_rad": np.empty(shape),
        "scapular_minimum_bound_margin_rad": np.empty(shape),
        "scapular_shoulder_excursion_m": np.empty((*shape, 2)),
        "scapular_coordinates_rad": np.empty((*shape, 2, 4)),
    }


def _store_pair(
    arrays: dict[str, FloatArray],
    case: int,
    time_index: int,
    fixed: _SolveResult,
    scapular: _SolveResult,
    neutral_shoulders: FloatArray,
) -> None:
    prefix_to_result = {"fixed": fixed, "scapular": scapular}
    for prefix, result in prefix_to_result.items():
        arrays[f"{prefix}_max_contact_error_m"][case, time_index] = (
            result.max_contact_error_m
        )
        arrays[f"{prefix}_contact_jacobian_rank"][case, time_index] = (
            result.contact_jacobian_rank
        )
        arrays[f"{prefix}_contact_jacobian_nullity"][case, time_index] = (
            result.contact_jacobian_nullity
        )
        arrays[f"{prefix}_solver_termination_success"][case, time_index] = (
            result.solver_termination_success
        )
        arrays[f"{prefix}_minimum_bound_margin_rad"][case, time_index] = (
            result.minimum_coordinate_bound_margin_rad
        )
    arrays["scapular_shoulder_excursion_m"][case, time_index] = np.linalg.norm(
        scapular.shoulder_points_m - neutral_shoulders, axis=1
    )
    arrays["scapular_coordinates_rad"][case, time_index] = np.asarray(
        (scapular.coordinates[:4], scapular.coordinates[8:12])
    )


def _run_registered_cases(
    selected: tuple[SyntheticSubjectProfile, ...],
    spans: FloatArray,
    times: FloatArray,
    config: ScapulothoracicConfig,
) -> tuple[list[dict[str, Any]], dict[str, FloatArray]]:
    arrays = _empty_result_arrays((len(selected) * spans.size, times.size))
    profile_records: list[dict[str, Any]] = []
    case = 0
    for profile in selected:
        model, metadata = build_subject_scaled_model(profile)
        profile_records.append(metadata)
        for span in spans:
            fixed_previous: FloatArray | None = None
            scapular_previous: FloatArray | None = None
            for time_index, sample_time in enumerate(times):
                q, _, _ = prescribed_state(model, float(sample_time))
                context = _context(model, q, metadata, float(span))
                fixed_reference = _fixed_reference(q)
                fixed = _solve(
                    context,
                    fixed_reference,
                    config,
                    include_scapula=False,
                    seed=fixed_previous,
                )
                fixed_previous = fixed.coordinates
                scap_reference = _scapular_reference(fixed_reference)
                embedded = _embedded_fixed_seed(fixed)
                scapular = _solve(
                    context,
                    scap_reference,
                    config,
                    include_scapula=True,
                    seed=(
                        scapular_previous if scapular_previous is not None else embedded
                    ),
                    fallback=embedded,
                )
                scapular_previous = scapular.coordinates
                neutral_shoulders = _positions(
                    scap_reference, context, config, include_scapula=True
                )[1]
                _store_pair(
                    arrays,
                    case,
                    time_index,
                    fixed,
                    scapular,
                    neutral_shoulders,
                )
            case += 1
    return profile_records, arrays


def _run_adverse_control(
    profile: SyntheticSubjectProfile,
    times: FloatArray,
    adverse_grip_span_m: float,
    config: ScapulothoracicConfig,
) -> dict[str, Any]:
    adverse_model, adverse_metadata = build_subject_scaled_model(profile)
    adverse_q, _, _ = prescribed_state(adverse_model, float(times[len(times) // 2]))
    adverse_context = _context(
        adverse_model, adverse_q, adverse_metadata, adverse_grip_span_m
    )
    adverse_fixed = _solve(
        adverse_context,
        _fixed_reference(adverse_q),
        config,
        include_scapula=False,
    )
    adverse_scapular = _solve(
        adverse_context,
        _scapular_reference(_fixed_reference(adverse_q)),
        config,
        include_scapula=True,
        seed=_embedded_fixed_seed(adverse_fixed),
    )
    return {
        "grip_span_m": adverse_grip_span_m,
        "fixed_contact_closed": adverse_fixed.contact_closed,
        "scapular_contact_closed": adverse_scapular.contact_closed,
        "scapular_max_contact_error_m": adverse_scapular.max_contact_error_m,
    }


def _result_record(
    selected: tuple[SyntheticSubjectProfile, ...],
    spans: FloatArray,
    times: FloatArray,
    profile_records: list[dict[str, Any]],
    arrays: dict[str, FloatArray],
    adverse_control: dict[str, Any],
    config: ScapulothoracicConfig,
) -> dict[str, Any]:
    shape = arrays["fixed_max_contact_error_m"].shape
    return {
        "schema_version": "scapulothoracic-contact-screen/v1",
        "study_id": "paired-fixed-and-scapula-on-ellipsoid-arm-only-contact-screen",
        "model": {
            "tier": "subject_scaled_kinematic_geometry_screen",
            "scapular_coordinates_per_side": 4,
            "coordinates": ["protraction", "elevation", "upward_rotation", "winging"],
            "config": asdict(config),
            "fixed_trunk_and_club": True,
        },
        "design": {
            "profile_count": len(selected),
            "grip_span_count": int(spans.size),
            "time_sample_count": int(times.size),
            "paired_state_count": int(np.prod(shape)),
            "profiles": profile_records,
            "grip_spans_m": spans.tolist(),
            "time_s": times.tolist(),
        },
        "results": {
            "fixed_contact_closed_count": int(
                np.count_nonzero(
                    arrays["fixed_max_contact_error_m"] <= config.closure_tolerance_m
                )
            ),
            "scapular_contact_closed_count": int(
                np.count_nonzero(
                    arrays["scapular_max_contact_error_m"] <= config.closure_tolerance_m
                )
            ),
            "scapular_never_worse_than_nested_fixed": bool(
                np.all(
                    arrays["scapular_max_contact_error_m"]
                    <= arrays["fixed_max_contact_error_m"] + 1.0e-9
                )
            ),
            "maximum_scapular_shoulder_excursion_m": float(
                np.max(arrays["scapular_shoulder_excursion_m"])
            ),
            "fixed_solver_termination_success_count": int(
                np.count_nonzero(arrays["fixed_solver_termination_success"])
            ),
            "scapular_solver_termination_success_count": int(
                np.count_nonzero(arrays["scapular_solver_termination_success"])
            ),
            "scapular_qualified_contact_count": int(
                np.count_nonzero(
                    (
                        arrays["scapular_max_contact_error_m"]
                        <= config.closure_tolerance_m
                    )
                    & arrays["scapular_solver_termination_success"]
                )
            ),
            "scapular_bound_active_count": int(
                np.count_nonzero(arrays["scapular_minimum_bound_margin_rad"] <= 1.0e-6)
            ),
        },
        "adverse_control": adverse_control,
        "boundaries": {
            "model_identity": "ellipsoid_surface_kinematic_surrogate_not_exact_seth_opensim_model",
            "scapular_or_glenohumeral_allocation": "structurally_nonunique_from_contact_geometry",
            "muscle_force_or_activation": "not_identified",
            "clinical_range_or_injury": "not_evaluated",
            "human_strategy": "not_identified",
            "forward_force_power_or_transfer": "not_evaluated",
        },
        "array_artifact": "scapulothoracic_contact_screen.npz",
    }


def run_scapulothoracic_contact_screen(
    *,
    profiles: Sequence[SyntheticSubjectProfile] | None = None,
    grip_spans_m: FloatArray | None = None,
    time_s: FloatArray | None = None,
    adverse_grip_span_m: float = 2.0,
    config: ScapulothoracicConfig = ScapulothoracicConfig(),
) -> tuple[dict[str, Any], dict[str, FloatArray]]:
    """Run paired arm-only closure with fixed and mobile scapular geometry."""
    selected, spans, times = _validated_design(
        profiles, grip_spans_m, time_s, adverse_grip_span_m
    )
    profile_records, arrays = _run_registered_cases(selected, spans, times, config)
    adverse_control = _run_adverse_control(
        selected[0], times, adverse_grip_span_m, config
    )
    record = _result_record(
        selected,
        spans,
        times,
        profile_records,
        arrays,
        adverse_control,
        config,
    )
    return record, arrays


__all__ = [
    "ScapulothoracicConfig",
    "ellipsoid_surface_point",
    "run_scapulothoracic_contact_screen",
]
