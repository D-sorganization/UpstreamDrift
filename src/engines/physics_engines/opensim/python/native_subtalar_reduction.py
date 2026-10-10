"""Source-bound bilateral locked-zero subtalar CustomJoint reduction.

Only the declared mechanical zero-transform chart is reduced. Muscle state
preparation, contact, physiological suitability, and Moco solving are separate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Protocol

from defusedxml import ElementTree
import numpy as np

from .native_mtp_reduction import (
    _constraint_residual,
    _finite_native_array,
    _max_difference,
    _perturbed_pose,
    _pose_error,
    _sha,
    _transform,
)


@dataclass(frozen=True)
class ZeroCustomJointProfile:
    """Exact bilateral source names; never a generic joint-type allowlist."""

    label: str
    joints: tuple[str, str]
    coordinates: tuple[str, str]


SUBTALAR_PROFILE = ZeroCustomJointProfile(
    "subtalar",
    ("subtalar_r", "subtalar_l"),
    ("subtalar_angle_r", "subtalar_angle_l"),
)


class ZeroCustomJointRequest(Protocol):
    @property
    def source_model_path(self) -> Path: ...

    @property
    def source_sha256(self) -> str: ...

    @property
    def derived_model_path(self) -> Path: ...

    @property
    def declared_target_rad(self) -> tuple[tuple[str, float], ...]: ...


_AXES = (
    "rotation1",
    "rotation2",
    "rotation3",
    "translation1",
    "translation2",
    "translation3",
)
_TOL = 1e-10


@dataclass(frozen=True)
class ZeroSubtalarReductionRequest:
    """Exact source, fresh destination and explicit bilateral zero targets."""

    source_model_path: Path
    source_sha256: str
    derived_model_path: Path
    declared_target_rad: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        if not isinstance(
            self.declared_target_rad, tuple
        ) or self.declared_target_rad != (
            ("subtalar_angle_r", 0.0),
            ("subtalar_angle_l", 0.0),
        ):
            raise ValueError("exact bilateral zero-target declaration required")
        if not re.fullmatch(r"[0-9a-f]{64}", self.source_sha256):
            raise ValueError("source SHA-256 must be lowercase hex")
        if self.source_model_path.resolve() == self.derived_model_path.resolve():
            raise ValueError("source and derived paths must differ")


@dataclass(frozen=True)
class ZeroSubtalarReductionReceipt:
    """Sampled, source/runtime-bound mechanical comparison after native reload."""

    source_sha256: str
    derived_sha256: str
    reducer_sha256: str
    native_version: str
    native_simulation_sha256: str
    native_simbody_sha256: str
    native_actuators_sha256: str
    comparison_helper_sha256: str
    removed_coordinate_names: tuple[str, str]
    max_body_transform_error: float
    max_body_velocity_error: float
    max_muscle_length_error: float
    max_muscle_speed_error: float
    max_projected_mass_error: float
    max_projected_total_force_error: float
    max_full_lift_mass_error: float
    max_full_lift_force_error: float
    max_constraint_residual: float
    mobility_lift_rank: int
    mobility_lift_condition: float
    observed_pose_count: int
    native_reload_verified: bool


def _reject_removed_coordinate_references(
    source: Path,
    profile: ZeroCustomJointProfile = SUBTALAR_PROFILE,
) -> None:
    root = ElementTree.parse(source).getroot()
    targets = {
        node
        for node in root.iter()
        if node.tag == "CustomJoint" and node.get("name") in profile.joints
    }
    if len(targets) != 2:
        raise ValueError("bilateral declared CustomJoint source declaration required")
    for parent in root.iter():
        for child in list(parent):
            if child in targets:
                parent.remove(child)
    names = "|".join(re.escape(name) for name in profile.coordinates)
    pattern = re.compile(rf"(?<![A-Za-z0-9_])(?:{names})(?![A-Za-z0-9_])")
    if any(
        pattern.search(value)
        for node in root.iter()
        for value in (*node.attrib.values(), node.text or "")
    ):
        raise ValueError(
            "unresolved reference or dependency on removed declared CustomJoint coordinate"
        )


def _source_axis_profile(source: Path, profile: ZeroCustomJointProfile) -> None:
    root = ElementTree.parse(source).getroot()
    joints = {
        node.get("name"): node for node in root.iter() if node.tag == "CustomJoint"
    }
    for joint_name, coordinate_name in zip(
        profile.joints, profile.coordinates, strict=True
    ):
        joint = joints.get(joint_name)
        if joint is None:
            raise ValueError("exact declared CustomJoint is missing")
        coordinate = joint.find("coordinates/Coordinate")
        axes = joint.findall("SpatialTransform/TransformAxis")
        if (
            coordinate is None
            or coordinate.get("name") != coordinate_name
            or len(axes) != 6
        ):
            raise ValueError(
                "declared CustomJoint one-coordinate spatial transform required"
            )
        if [axis.get("name") for axis in axes] != list(_AXES):
            raise ValueError("declared CustomJoint spatial-axis order changed")
        for index, axis in enumerate(axes):
            values = _finite_native_array(
                np.fromstring(axis.findtext("axis", ""), sep=" ")
            )
            if values.shape != (3,) or np.linalg.norm(values) < _TOL:
                raise ValueError("invalid declared CustomJoint spatial axis")
            if index == 0:
                function = axis.find("LinearFunction/coefficients")
                if (
                    axis.findtext("coordinates", "").strip() != coordinate_name
                    or function is None
                ):
                    raise ValueError(
                        "declared CustomJoint rotation law or coordinate changed"
                    )
                coefficients = _finite_native_array(
                    np.fromstring(function.text or "", sep=" ")
                )
                if coefficients.shape != (2,) or not np.array_equal(
                    coefficients, [1.0, 0.0]
                ):
                    raise ValueError(
                        "declared CustomJoint rotation law must be identity through zero"
                    )
            else:
                constant = axis.find("Constant/value")
                if axis.findtext("coordinates", "").strip() or constant is None:
                    raise ValueError(
                        "declared CustomJoint nonrotational axis is not constant"
                    )
                values = _finite_native_array(
                    np.fromstring(constant.text or "", sep=" ")
                )
                if values.shape != (1,) or values[0] != 0.0:
                    raise ValueError(
                        "declared CustomJoint nonrotational axis is not zero"
                    )


def _admit_native_joint(
    model: Any,
    state: Any,
    profile: ZeroCustomJointProfile,
) -> None:
    for joint_name, coordinate_name in zip(
        profile.joints, profile.coordinates, strict=True
    ):
        joint = model.getJointSet().get(joint_name)
        if joint.getConcreteClassName() != "CustomJoint" or joint.numCoordinates() != 1:
            raise ValueError("exact one-coordinate declared CustomJoint required")
        coordinate = joint.get_coordinates(0)
        if coordinate.getName() != coordinate_name:
            raise ValueError("declared CustomJoint native coordinate mapping changed")
        if not coordinate.getDefaultLocked() or not coordinate.getLocked(state):
            raise ValueError("declared CustomJoint target is not locked")
        if coordinate.getDefaultIsPrescribed() or coordinate.isPrescribed(state):
            raise ValueError("prescribed declared CustomJoint target unsupported")
        if coordinate.getDefaultClamped() != coordinate.getClamped(state):
            raise ValueError(
                "declared CustomJoint clamp option changed during initialization"
            )
        if any(
            abs(value) > _TOL
            for value in (
                coordinate.getDefaultValue(),
                coordinate.getValue(state),
                coordinate.getDefaultSpeedValue(),
                coordinate.getSpeedValue(state),
            )
        ):
            raise ValueError(
                "nonzero declared CustomJoint source or achieved target cannot be welded"
            )
        if coordinate.getClamped(state) and not (
            coordinate.getRangeMin() < -_TOL and coordinate.getRangeMax() > _TOL
        ):
            raise ValueError(
                "zero declared CustomJoint target lies on clamped boundary"
            )
        model.realizePosition(state)
        if (
            _max_difference(
                _transform(joint.getParentFrame(), state),
                _transform(joint.getChildFrame(), state),
            )
            > _TOL
        ):
            raise ValueError(
                "native declared CustomJoint frame transform at zero is not identity"
            )


def _common_state_pairs(
    original: Any,
    derived: Any,
    profile: ZeroCustomJointProfile,
) -> tuple[tuple[Any, Any], ...]:
    source_state = original.initSystem()
    reduced_state = derived.initSystem()
    source_ids = {
        original.getStateVariableNames().get(i)
        for i in range(original.getStateVariableNames().getSize())
    }
    reduced_ids = {
        derived.getStateVariableNames().get(i)
        for i in range(derived.getStateVariableNames().getSize())
    }
    removed = source_ids - reduced_ids
    if (
        reduced_ids - source_ids
        or len(removed) != 4
        or any(
            not any(
                f"/{name}/{kind}" in identifier
                for name in profile.coordinates
                for kind in ("value", "speed")
            )
            for identifier in removed
        )
    ):
        raise ValueError("derived state changed beyond bilateral declared CustomJoint")
    for identifier in reduced_ids:
        value = original.getStateVariableValue(source_state, identifier)
        derived.setStateVariableValue(reduced_state, identifier, value)
        if derived.getStateVariableValue(reduced_state, identifier) != value:
            raise ValueError("common continuous state did not restore exactly")
    names = {coordinate.getName() for coordinate in derived.getCoordinateSet()}
    pairs: tuple[tuple[Any, Any], ...] = ((source_state, reduced_state),)
    for name in ("ankle_angle_r", "knee_angle_r", "pelvis_tilt"):
        if name in names:
            changed = _perturbed_pose(
                original, derived, source_state, reduced_state, name
            )
            if changed is not None:
                pairs += (changed,)
    return pairs


def _check_inventory(
    original: Any,
    derived: Any,
    profile: ZeroCustomJointProfile,
) -> tuple[str, ...]:
    for title, before, after in (
        ("body", original.getBodySet(), derived.getBodySet()),
        ("muscle", original.getMuscles(), derived.getMuscles()),
        ("actuator", original.getActuators(), derived.getActuators()),
        ("constraint", original.getConstraintSet(), derived.getConstraintSet()),
    ):
        if tuple(item.dump() for item in before) != tuple(
            item.dump() for item in after
        ):
            raise ValueError(f"derived {title} properties changed")
    before_joints = original.getJointSet()
    after_joints = derived.getJointSet()
    if {joint.getName() for joint in before_joints} != {
        joint.getName() for joint in after_joints
    }:
        raise ValueError("derived joint inventory changed")
    for joint in before_joints:
        matched = after_joints.get(joint.getName())
        if joint.getName() in profile.joints:
            if matched.getConcreteClassName() != "WeldJoint":
                raise ValueError("only declared CustomJoint joints may be welded")
        elif joint.dump() != matched.dump():
            raise ValueError("unrelated joint property changed")
    names = tuple(coordinate.getName() for coordinate in derived.getCoordinateSet())
    if set(names) != {
        coordinate.getName() for coordinate in original.getCoordinateSet()
    } - set(profile.coordinates):
        raise ValueError(
            "derived coordinate inventory changed beyond declared CustomJoint"
        )
    return names


def _compare_native(
    original: Any,
    derived: Any,
    profile: ZeroCustomJointProfile,
) -> tuple[float, ...]:
    names = _check_inventory(original, derived, profile)
    pairs = _common_state_pairs(original, derived, profile)
    if len(pairs) < 3:
        raise ValueError("insufficient nonzero common native comparison poses")
    errors = np.zeros(9)
    rank = 0
    condition = 1.0
    for source_state, reduced_state in pairs:
        observed = _pose_error(original, derived, names, source_state, reduced_state)
        speed_error = max(
            (
                abs(
                    muscle.getLengtheningSpeed(source_state)
                    - derived.getMuscles()
                    .get(muscle.getName())
                    .getLengtheningSpeed(reduced_state)
                )
                for muscle in original.getMuscles()
            ),
            default=0.0,
        )
        if not np.isfinite(speed_error):
            raise ValueError("nonfinite native muscle lengthening speed")
        errors = np.maximum(
            errors,
            _finite_native_array(
                np.asarray((*observed[:2], speed_error, *observed[2:8]), dtype=float)
            ),
        )
        rank = int(observed[8])
        condition = max(condition, float(observed[9]))
    if np.max(errors) > _TOL or _constraint_residual(pairs[0][0]) > _TOL:
        raise ValueError(
            "fresh native declared CustomJoint derivation changed sampled mechanics"
        )
    return (*[float(value) for value in errors], rank, condition, len(pairs))


def derive_zero_custom_joint_model(
    request: ZeroCustomJointRequest,
    profile: ZeroCustomJointProfile,
) -> ZeroSubtalarReductionReceipt:
    """Weld only exact declared zero CustomJoints and independently reload."""
    import opensim as osim
    from . import native_mtp_reduction as comparison

    source = request.source_model_path
    output = request.derived_model_path
    reducer_hash = _sha(Path(__file__))
    helper_hash = _sha(Path(comparison.__file__))
    if not source.is_file() or _sha(source) != request.source_sha256:
        raise ValueError("source model hash mismatch or source missing")
    if output.exists():
        raise FileExistsError(output)
    if not output.parent.is_dir():
        raise ValueError("derived model parent directory missing")
    if request.declared_target_rad != tuple(
        (name, 0.0) for name in profile.coordinates
    ):
        raise ValueError("joint profile and declared zero targets differ")
    _reject_removed_coordinate_references(source, profile)
    _source_axis_profile(source, profile)
    original = osim.Model(str(source))
    _admit_native_joint(original, original.initSystem(), profile)
    derived = osim.Model(original)
    names = osim.StdVectorString()
    for joint_name in profile.joints:
        names.append(joint_name)
    osim.ModOpReplaceJointsWithWelds(names).operate(derived, "")
    derived.printToXML(str(output))
    output_hash = _sha(output)
    try:
        observed = _compare_native(original, osim.Model(str(output)), profile)
        if (
            _sha(source),
            _sha(output),
            _sha(Path(__file__)),
            _sha(Path(comparison.__file__)),
        ) != (
            request.source_sha256,
            output_hash,
            reducer_hash,
            helper_hash,
        ):
            raise ValueError(
                "source, derived output or reduction policy changed during verification"
            )
    except Exception:
        output.unlink(missing_ok=True)
        raise
    return ZeroSubtalarReductionReceipt(
        source_sha256=request.source_sha256,
        derived_sha256=output_hash,
        reducer_sha256=reducer_hash,
        native_version=osim.GetVersionAndDate(),
        native_simulation_sha256=_sha(Path(osim._simulation.__file__)),
        native_simbody_sha256=_sha(Path(osim._simbody.__file__)),
        native_actuators_sha256=_sha(Path(osim._actuators.__file__)),
        comparison_helper_sha256=helper_hash,
        removed_coordinate_names=profile.coordinates,
        max_body_transform_error=observed[0],
        max_muscle_length_error=observed[1],
        max_muscle_speed_error=observed[2],
        max_projected_mass_error=observed[3],
        max_projected_total_force_error=observed[4],
        max_body_velocity_error=observed[5],
        max_constraint_residual=observed[6],
        max_full_lift_mass_error=observed[7],
        max_full_lift_force_error=observed[8],
        mobility_lift_rank=int(observed[9]),
        mobility_lift_condition=observed[10],
        observed_pose_count=int(observed[11]),
        native_reload_verified=True,
    )


def derive_zero_subtalar_model(
    request: ZeroSubtalarReductionRequest,
) -> ZeroSubtalarReductionReceipt:
    """Preserve the exact published subtalar profile and result contract."""
    return derive_zero_custom_joint_model(request, SUBTALAR_PROFILE)
