"""Subject-scaled geometry audit for the reduced spatial full-body tier.

The existing 20-coordinate common-state study proves spatial wrench transport
and inverse-dynamics parity, but it prescribes bilateral loads.  This module
tests the next necessary condition: whether canonical anthropometric variants
place the articulated hand contact points on the declared club grip points and
whether the local two-contact constraint remains well conditioned.

The profiles are deterministic synthetic design points generated with the
repository's canonical de Leva estimator.  They are not participants or a
population sample.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
import sys
from typing import Any

import numpy as np
from numpy.typing import NDArray

# Keep the canonical shared package importable when this study is executed with
# ``python -m`` outside pytest's configured pythonpath.
_SHARED_PYTHON = str(Path(__file__).resolve().parents[3] / "src/shared/python")
if _SHARED_PYTHON not in sys.path:
    sys.path.insert(0, _SHARED_PYTHON)

from anthropometrics.estimators import DeLevaEstimator
from scripts.research.proximal_distal_energy.bilateral_wrench_identifiability import (
    audit_linear_map,
    internal_axial_measurement,
    point_force_wrench_map,
)
from scripts.research.proximal_distal_energy.spatial_full_body import (
    BodySpec,
    JointSpec,
    SpatialModel,
    build_spatial_model,
    evaluate_hand_wrenches,
    forward_kinematics,
    point_contact_jacobians,
    prescribed_state,
)

FloatArray = NDArray[np.float64]
REGIONS = ("pelvis", "torso", "lead_arm", "trail_arm", "club")
REFERENCE_HEIGHT_M = 1.75
REFERENCE_MASS_KG = 75.0


@dataclass(frozen=True, slots=True)
class SyntheticSubjectProfile:
    """One deterministic anthropometric design point in SI units."""

    profile_id: str
    height_m: float
    mass_kg: float
    sex: str

    def __post_init__(self) -> None:
        if not isinstance(self.profile_id, str) or not self.profile_id.strip():
            raise ValueError("profile_id must be a non-empty string")
        if not np.isfinite(self.height_m) or self.height_m <= 0.0:
            raise ValueError("height_m must be finite and positive")
        if not np.isfinite(self.mass_kg) or self.mass_kg <= 0.0:
            raise ValueError("mass_kg must be finite and positive")
        if self.sex not in {"M", "F"}:
            raise ValueError("sex must be 'M' or 'F' for this declared design")


@dataclass(frozen=True, slots=True)
class ContactGeometrySnapshot:
    """Closure, rank, conditioning, and load results at one achieved state."""

    hand_to_grip_distance_m: FloatArray
    constraint_jacobian: FloatArray
    constraint_jacobian_rank: int
    constraint_jacobian_singular_values: FloatArray
    constraint_jacobian_minimum_singular_value: float
    constraint_jacobian_condition_number: float
    point_force_wrench_map_rank: int
    point_force_wrench_map_nullity: int
    augmented_point_force_wrench_map_rank: int
    force_generated_couple_nm: float
    regional_generalized_load_norm: FloatArray


def default_synthetic_profiles() -> tuple[SyntheticSubjectProfile, ...]:
    """Return six fixed stature/sex-table design points.

    Mass follows a declared BMI of 24 kg/m² to avoid treating the profiles as a
    sampled human distribution.
    """

    heights = (1.55, 1.75, 1.95)
    return tuple(
        SyntheticSubjectProfile(
            profile_id=f"{sex.lower()}-{height:.2f}",
            height_m=height,
            mass_kg=24.0 * height**2,
            sex=sex,
        )
        for sex in ("F", "M")
        for height in heights
    )


def _segment_map(subject: Any) -> dict[str, Any]:
    return dict(subject.segments)


def _scaled_joint(
    joint: JointSpec,
    *,
    linear_scale: float,
    upper_arm_m: float,
    forearm_m: float,
) -> JointSpec:
    offset = np.asarray(joint.offset_m, dtype=float).copy()
    if (
        joint.name == "pelvis_yaw"
        or joint.name == "torso_pitch"
        or joint.name == "torso_yaw"
    ):
        offset[2] *= linear_scale
    elif joint.name in {"lead_shoulder_x", "trail_shoulder_x"}:
        offset *= linear_scale
    elif joint.name in {"lead_elbow", "trail_elbow"}:
        offset = np.array([upper_arm_m, 0.0, -0.05 * linear_scale])
    elif joint.name in {"lead_wrist", "trail_wrist"}:
        offset = np.array([forearm_m, 0.0, 0.0])
    return replace(joint, offset_m=offset)


def _scaled_body(
    body: BodySpec,
    *,
    linear_scale: float,
    masses: dict[str, float],
    lengths: dict[str, float],
) -> BodySpec:
    if body.region == "club" or body.name.startswith("joint_carrier_"):
        return body
    mass_lookup = {
        "lower_body": masses["lower_body"],
        "pelvis_mass": masses["pelvis"],
        "torso_mass": masses["torso"],
        "lead_upper_arm": masses["upper_arm"],
        "trail_upper_arm": masses["upper_arm"],
        "lead_forearm": masses["forearm"],
        "trail_forearm": masses["forearm"],
        "lead_hand": masses["hand"],
        "trail_hand": masses["hand"],
    }
    com = np.asarray(body.com_offset_m, dtype=float) * linear_scale
    if "upper_arm" in body.name:
        com = np.array([0.5 * lengths["upper_arm"], 0.0, -0.025 * linear_scale])
    elif "forearm" in body.name:
        com = np.array([0.5 * lengths["forearm"], 0.0, 0.0])
    elif "hand" in body.name:
        com = np.array([0.5 * lengths["hand"], 0.0, 0.0])
    return replace(
        body,
        mass_kg=mass_lookup[body.name],
        radius_m=body.radius_m * linear_scale,
        com_offset_m=com,
    )


def build_subject_scaled_model(
    profile: SyntheticSubjectProfile,
) -> tuple[SpatialModel, dict[str, Any]]:
    """Scale the canonical spatial tree from one synthetic profile.

    Postcondition: the returned tree preserves the 20-coordinate topology and
    records every dimensional value used in the transformation.
    """

    if not isinstance(profile, SyntheticSubjectProfile):
        raise TypeError("profile must be a SyntheticSubjectProfile")
    subject = DeLevaEstimator().estimate(
        subject_id=profile.profile_id,
        height_m=profile.height_m,
        mass_kg=profile.mass_kg,
        sex=profile.sex,
    )
    segments = _segment_map(subject)
    lengths = {
        "upper_arm": float(segments["left_upper_arm"].length_m),
        "forearm": float(segments["left_forearm"].length_m),
        "hand": float(segments["left_hand"].length_m),
    }
    masses = {
        "upper_arm": float(segments["left_upper_arm"].mass_kg),
        "forearm": float(segments["left_forearm"].mass_kg),
        "hand": float(segments["left_hand"].mass_kg),
        "pelvis": float(segments["pelvis"].mass_kg),
        "torso": float(
            sum(segments[name].mass_kg for name in ("head", "neck", "thorax", "lumbar"))
        ),
    }
    represented = (
        masses["pelvis"]
        + masses["torso"]
        + 2.0 * (masses["upper_arm"] + masses["forearm"] + masses["hand"])
    )
    masses["lower_body"] = float(profile.mass_kg - represented)
    if masses["lower_body"] <= 0.0:
        raise ValueError("profile leaves no positive lower-body remainder")

    base = build_spatial_model()
    linear_scale = profile.height_m / REFERENCE_HEIGHT_M
    joints = tuple(
        _scaled_joint(
            joint,
            linear_scale=linear_scale,
            upper_arm_m=lengths["upper_arm"],
            forearm_m=lengths["forearm"],
        )
        for joint in base.joints
    )
    bodies = tuple(
        _scaled_body(
            body,
            linear_scale=linear_scale,
            masses=masses,
            lengths=lengths,
        )
        for body in base.bodies
    )
    model = SpatialModel(
        joints=joints,
        bodies=bodies,
        club_dof_indices=base.club_dof_indices.copy(),
        lead_hand_joint=base.lead_hand_joint,
        trail_hand_joint=base.trail_hand_joint,
        club_frame_joint=base.club_frame_joint,
    )
    metadata: dict[str, Any] = {
        "profile": asdict(profile),
        "estimator": subject.source_method,
        "linear_scale_from_1_75_m": linear_scale,
        "segment_lengths_m": lengths,
        "represented_body_masses_kg": masses,
        "hand_contact_local_x_m": 0.5 * lengths["hand"],
        "model_sha256": model.canonical_hash,
    }
    return model, metadata


def _rank_and_singular_values(matrix: FloatArray) -> tuple[int, FloatArray]:
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    threshold = np.finfo(float).eps * max(matrix.shape) * singular_values[0]
    return int(np.count_nonzero(singular_values > threshold)), singular_values


def contact_geometry_snapshot(
    model: SpatialModel,
    *,
    time_s: float,
    grip_span_m: float,
    hand_contact_local_x_m: float,
) -> ContactGeometrySnapshot:
    """Evaluate bilateral closure and local contact geometry at one state."""

    if not np.isfinite(time_s) or time_s < 0.0:
        raise ValueError("time_s must be finite and nonnegative")
    if not np.isfinite(grip_span_m) or grip_span_m <= 0.0:
        raise ValueError("grip_span_m must be finite and positive")
    if not np.isfinite(hand_contact_local_x_m) or hand_contact_local_x_m <= 0.0:
        raise ValueError("hand_contact_local_x_m must be finite and positive")

    q, _, _ = prescribed_state(model, time_s)
    kin = forward_kinematics(model, q)
    grip_locals = (
        np.array([0.0, grip_span_m / 2.0, -0.03]),
        np.array([0.0, -grip_span_m / 2.0, -0.03]),
    )
    hand_local = np.array([hand_contact_local_x_m, 0.0, 0.0])
    hand_joints = (model.lead_hand_joint, model.trail_hand_joint)
    grip_points: list[FloatArray] = []
    hand_points: list[FloatArray] = []
    constraint_rows: list[FloatArray] = []
    club_jacobians: list[FloatArray] = []
    hand_jacobians: list[FloatArray] = []
    for hand_joint, grip_local in zip(hand_joints, grip_locals, strict=True):
        grip_point, grip_jv, _ = point_contact_jacobians(
            model, kin, model.club_frame_joint, grip_local
        )
        hand_point, hand_jv, _ = point_contact_jacobians(
            model, kin, hand_joint, hand_local
        )
        grip_points.append(grip_point)
        hand_points.append(hand_point)
        club_jacobians.append(grip_jv)
        hand_jacobians.append(hand_jv)
        constraint_rows.append(hand_jv - grip_jv)
    grip_array = np.asarray(grip_points)
    hand_array = np.asarray(hand_points)
    constraint = np.vstack(constraint_rows)
    rank, singular_values = _rank_and_singular_values(constraint)
    minimum = float(singular_values[-1])
    condition = float(singular_values[0] / minimum)

    wrench_map = point_force_wrench_map(grip_array, np.mean(grip_array, axis=0))
    wrench_audit = audit_linear_map(wrench_map)
    axial = internal_axial_measurement(grip_array)
    augmented_audit = audit_linear_map(np.vstack((wrench_map, axial)))
    sample = evaluate_hand_wrenches(
        model,
        time_s,
        coincident_hands=False,
        grip_span_m=grip_span_m,
    )
    generalized_load = (
        club_jacobians[0].T @ sample.lead_force_n
        + club_jacobians[1].T @ sample.trail_force_n
        - hand_jacobians[0].T @ sample.lead_force_n
        - hand_jacobians[1].T @ sample.trail_force_n
    )
    regional_norm = np.array(
        [
            np.linalg.norm(
                generalized_load[
                    [
                        index
                        for index, joint in enumerate(model.joints)
                        if joint.region == region
                    ]
                ]
            )
            for region in REGIONS
        ]
    )
    return ContactGeometrySnapshot(
        hand_to_grip_distance_m=np.linalg.norm(hand_array - grip_array, axis=1),
        constraint_jacobian=constraint,
        constraint_jacobian_rank=rank,
        constraint_jacobian_singular_values=singular_values,
        constraint_jacobian_minimum_singular_value=minimum,
        constraint_jacobian_condition_number=condition,
        point_force_wrench_map_rank=wrench_audit.rank,
        point_force_wrench_map_nullity=wrench_audit.nullity,
        augmented_point_force_wrench_map_rank=augmented_audit.rank,
        force_generated_couple_nm=sample.force_generated_couple_nm,
        regional_generalized_load_norm=regional_norm,
    )


def run_subject_scaled_geometry_atlas() -> tuple[dict[str, Any], dict[str, FloatArray]]:
    """Run the deterministic six-profile, three-grip-span atlas."""

    profiles = default_synthetic_profiles()
    spans = np.array([0.12, 0.18, 0.24])
    time = np.linspace(0.0, 0.24, 61)
    case_count = len(profiles) * spans.size
    distances = np.empty((case_count, time.size, 2))
    singular_values = np.empty((case_count, time.size, 6))
    condition = np.empty((case_count, time.size))
    couple = np.empty((case_count, time.size))
    regional_load = np.empty((case_count, time.size, len(REGIONS)))
    case_profile = np.empty(case_count, dtype=float)
    case_span = np.empty(case_count)
    rank_values: set[int] = set()
    augmented_rank_values: set[int] = set()
    constraint_rank_values: set[int] = set()
    profile_records: list[dict[str, Any]] = []

    case = 0
    for profile_index, profile in enumerate(profiles):
        model, metadata = build_subject_scaled_model(profile)
        profile_records.append(metadata)
        hand_contact = float(metadata["hand_contact_local_x_m"])
        for span in spans:
            case_profile[case] = profile_index
            case_span[case] = span
            for time_index, sample_time in enumerate(time):
                snapshot = contact_geometry_snapshot(
                    model,
                    time_s=float(sample_time),
                    grip_span_m=float(span),
                    hand_contact_local_x_m=hand_contact,
                )
                distances[case, time_index] = snapshot.hand_to_grip_distance_m
                singular_values[case, time_index] = (
                    snapshot.constraint_jacobian_singular_values
                )
                condition[case, time_index] = (
                    snapshot.constraint_jacobian_condition_number
                )
                couple[case, time_index] = snapshot.force_generated_couple_nm
                regional_load[case, time_index] = (
                    snapshot.regional_generalized_load_norm
                )
                rank_values.add(snapshot.point_force_wrench_map_rank)
                augmented_rank_values.add(
                    snapshot.augmented_point_force_wrench_map_rank
                )
                constraint_rank_values.add(snapshot.constraint_jacobian_rank)
            case += 1

    ratio = couple.reshape(len(profiles), spans.size, time.size) / spans[None, :, None]
    span_invariance = float(np.max(np.abs(ratio - ratio[:, :1, :])))
    record: dict[str, Any] = {
        "schema_version": "subject-scaled-spatial-geometry/v1",
        "study_id": "subject-scaled-spatial-contact-geometry-atlas",
        "model_tier": "prescribed_common_state_subject_scaled_articulated_geometry",
        "design": {
            "profile_count": len(profiles),
            "grip_span_count": int(spans.size),
            "case_count": case_count,
            "time_sample_count": int(time.size),
            "grip_spans_m": spans.tolist(),
            "mass_design": "BMI 24 kg/m^2 at each stature",
            "profiles": profile_records,
            "profile_interpretation": (
                "deterministic synthetic de Leva design points, not a population"
            ),
        },
        "closure_tests": {
            "minimum_hand_to_grip_distance_m": float(np.min(distances)),
            "median_hand_to_grip_distance_m": float(np.median(distances)),
            "maximum_hand_to_grip_distance_m": float(np.max(distances)),
            "contact_closure_tolerance_m": 0.005,
            "all_samples_close_contact": bool(np.all(distances <= 0.005)),
        },
        "geometry_tests": {
            "point_force_map_rank_values": sorted(rank_values),
            "augmented_map_rank_values": sorted(augmented_rank_values),
            "constraint_jacobian_rank_values": sorted(constraint_rank_values),
            "constraint_condition_number_minimum": float(np.min(condition)),
            "constraint_condition_number_median": float(np.median(condition)),
            "constraint_condition_number_maximum": float(np.max(condition)),
            "couple_per_span_invariance_residual": span_invariance,
        },
        "claim_status": {
            "grip_span_couple_scaling": (
                "supported_for_prescribed_point_forces_in_declared_spatial_tier"
            ),
            "local_contact_constraint_rank": (
                "evaluated_at_prescribed_states_without_contact_closure"
            ),
            "subject_scaled_anatomical_contact": (
                "not_established_prescribed_state_fails_contact_closure"
            ),
            "human_strategy": "untested",
            "human_or_coaching_inference": "unsupported",
        },
        "limitations": [
            "Synthetic profiles are deterministic design points, not participants or population estimates.",
            "The prescribed common-state trajectories are not solved to close both anatomical hands on the club.",
            "The local constraint Jacobian can be full row rank at a geometrically open contact state.",
            "The forces remain prescribed and do not establish passive anatomical contact dynamics.",
            "de Leva segment regressions do not encode subject-specific scapular, wrist, tissue, or grip geometry.",
        ],
        "array_artifact": "subject_scaled_spatial_geometry.npz",
    }
    arrays: dict[str, FloatArray] = {
        "time_s": time,
        "grip_spans_m": spans,
        "case_profile_index": case_profile,
        "case_grip_span_m": case_span,
        "hand_to_grip_distance_m": distances,
        "constraint_jacobian_singular_values": singular_values,
        "constraint_jacobian_condition_number": condition,
        "force_generated_couple_nm": couple,
        "regional_generalized_load_norm": regional_load,
    }
    return record, arrays


__all__ = [
    "ContactGeometrySnapshot",
    "SyntheticSubjectProfile",
    "build_subject_scaled_model",
    "contact_geometry_snapshot",
    "default_synthetic_profiles",
    "run_subject_scaled_geometry_atlas",
]
