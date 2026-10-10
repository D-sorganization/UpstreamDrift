"""Source-bound bilateral zero-MTP reduction for derived OpenSim models.

This is a mechanical ModelFactory transform. It does not prepare muscle states,
qualify contact or physiology, or serialize an arbitrary locked-coordinate target.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Any

from defusedxml import ElementTree

import numpy as np

_JOINTS = ("mtp_r", "mtp_l")
_COORDINATES = ("mtp_angle_r", "mtp_angle_l")
_ZERO_TOL = 1e-12
_GEOMETRY_TOL = 1e-10


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_fresh_reduction_paths(
    source: Path, source_sha256: str, output: Path
) -> None:
    """Require unchanged source bytes and an unused derived-model destination."""
    if not source.is_file() or _sha(source) != source_sha256:
        raise ValueError("source model hash mismatch or source missing")
    if output.exists():
        raise FileExistsError(output)
    if not output.parent.is_dir():
        raise ValueError("derived model parent directory missing")


def _finite_native_array(values: np.ndarray) -> np.ndarray:
    if not np.all(np.isfinite(values)):
        raise ValueError("nonfinite native reduction observation")
    return values


def _max_difference(left: np.ndarray, right: np.ndarray) -> float:
    difference = _finite_native_array(np.abs(left - right))
    return float(np.max(difference, initial=0.0))


@dataclass(frozen=True)
class ZeroMtpReductionRequest:
    """Exact source artifact, fresh output and explicit zero lock targets."""

    source_model_path: Path
    source_sha256: str
    derived_model_path: Path
    declared_target_rad: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        if not isinstance(
            self.declared_target_rad, tuple
        ) or self.declared_target_rad != (("mtp_angle_r", 0.0), ("mtp_angle_l", 0.0)):
            raise ValueError("exact bilateral zero-target declaration required")
        if not re.fullmatch(r"[0-9a-f]{64}", self.source_sha256):
            raise ValueError("source SHA-256 must be lowercase hex")
        if self.source_model_path.resolve() == self.derived_model_path.resolve():
            raise ValueError("source and derived paths must differ")


@dataclass(frozen=True)
class ZeroMtpReductionReceipt:
    """Verified sampled mechanical equivalence on a freshly reloaded model."""

    source_sha256: str
    derived_sha256: str
    reducer_sha256: str
    native_version: str
    native_simulation_sha256: str
    native_simbody_sha256: str
    native_actuators_sha256: str
    removed_coordinate_names: tuple[str, str]
    max_body_transform_error: float
    max_muscle_length_error: float
    max_projected_mass_error: float
    max_full_lift_mass_error: float
    max_full_lift_force_error: float
    mobility_lift_rank: int
    mobility_lift_condition: float
    max_projected_total_force_error: float
    max_body_velocity_error: float
    max_constraint_residual: float
    observed_pose_count: int
    native_reload_verified: bool


def _reject_removed_coordinate_references(source: Path) -> None:
    """Reject any remaining XML field that names a deleted mobility.

    Unknown coordinate consumers are intentionally rejected too. This admits
    the exact source class only when no such consumer can be hidden downstream.
    """
    root = ElementTree.parse(source).getroot()
    targets = {
        node
        for node in root.iter()
        if node.tag == "PinJoint" and node.get("name") in _JOINTS
    }
    if len(targets) != len(_JOINTS):
        raise ValueError("bilateral MTP PinJoint source declaration required")
    for parent in root.iter():
        for child in list(parent):
            if child in targets:
                parent.remove(child)
    pattern = re.compile(
        r"(?<![A-Za-z0-9_])(?:mtp_angle_r|mtp_angle_l)(?![A-Za-z0-9_])"
    )
    for node in root.iter():
        if any(
            pattern.search(value) for value in (*node.attrib.values(), node.text or "")
        ):
            raise ValueError(
                "unresolved reference or dependency on removed MTP coordinate"
            )


def _coordinate_admission(model: Any, state: Any) -> None:
    for joint_name, name in zip(_JOINTS, _COORDINATES, strict=True):
        joint = model.getJointSet().get(joint_name)
        if joint.getConcreteClassName() != "PinJoint":
            raise ValueError("MTP reduction requires an exact PinJoint")
        coordinate = joint.get_coordinates(0)
        if coordinate.getName() != name:
            raise ValueError("MTP coordinate name does not match declared joint")
        if not coordinate.getDefaultLocked() or not coordinate.getLocked(state):
            raise ValueError(
                "MTP coordinate is not locked in source and initialized state"
            )
        if coordinate.getDefaultIsPrescribed() or coordinate.isPrescribed(state):
            raise ValueError("prescribed MTP coordinate is unsupported")
        if coordinate.getDefaultClamped() != coordinate.getClamped(state):
            raise ValueError("MTP clamped option changed during initialization")
        if coordinate.getClamped(state) and not (
            coordinate.getRangeMin() < -_ZERO_TOL
            and coordinate.getRangeMax() > _ZERO_TOL
        ):
            raise ValueError("zero MTP target lies on clamped range boundary")
        if (
            abs(coordinate.getDefaultValue()) > _ZERO_TOL
            or abs(coordinate.getValue(state)) > _ZERO_TOL
        ):
            raise ValueError(
                "nonzero source or achieved MTP lock target cannot be welded"
            )
        if (
            abs(coordinate.getDefaultSpeedValue()) > _ZERO_TOL
            or abs(coordinate.getSpeedValue(state)) > _ZERO_TOL
        ):
            raise ValueError("nonzero source or achieved MTP speed cannot be welded")


def _transform(frame: Any, state: Any) -> np.ndarray:
    transform = frame.getTransformInGround(state)
    return _finite_native_array(
        np.array(
            [transform.R().get(i, j) for i in range(3) for j in range(3)]
            + [transform.p().get(i) for i in range(3)],
            dtype=float,
        )
    )


def _mass_on_common_coordinates(
    model: Any, state: Any, names: tuple[str, ...]
) -> np.ndarray:
    mass = _native_mass(model, state)
    projection = _speed_projection(model, state, names)
    return projection.T @ mass @ projection


def _native_mass(model: Any, state: Any) -> np.ndarray:
    import opensim as osim

    matrix = osim.Matrix()
    model.getMatterSubsystem().calcM(state, matrix)
    return _finite_native_array(
        np.array(
            [
                [matrix.get(i, j) for j in range(matrix.ncol())]
                for i in range(matrix.nrow())
            ]
        )
    )


def _speed_projection(model: Any, state: Any, names: tuple[str, ...]) -> np.ndarray:
    import opensim as osim

    columns: list[np.ndarray] = []
    for name in names:
        probe = osim.State(state)
        probe.updU().setToZero()
        model.getCoordinateSet().get(name).setSpeedValue(probe, 1.0)
        columns.append(np.array([probe.getU().get(i) for i in range(probe.getNU())]))
    if not columns:
        return np.zeros((state.getNU(), 0))
    return _finite_native_array(np.column_stack(columns))


def _projected_total_force(
    model: Any, state: Any, names: tuple[str, ...]
) -> np.ndarray:
    return _speed_projection(model, state, names).T @ _native_total_generalized_force(
        model, state
    )


def _native_total_generalized_force(model: Any, state: Any) -> np.ndarray:
    """Native body-force projection plus direct mobility force, before bias/reactions."""
    import opensim as osim

    model.realizeDynamics(state)
    body_forces = model.getRigidBodyForces(state)
    projected = osim.Vector()
    model.getMatterSubsystem().multiplyBySystemJacobianTranspose(
        state, body_forces, projected
    )
    mobility = model.getMobilityForces(state)
    if projected.size() != mobility.size():
        raise ValueError("native body and mobility force dimensions differ")
    return _finite_native_array(
        np.array([projected.get(i) + mobility.get(i) for i in range(mobility.size())])
    )


def _full_mobility_lift_errors(
    original: Any,
    derived: Any,
    original_state: Any,
    derived_state: Any,
    names: tuple[str, ...],
) -> tuple[float, float, int, float]:
    source_map = _speed_projection(original, original_state, names)
    reduced_map = _speed_projection(derived, derived_state, names)
    if derived_state.getNU() == 0:
        return 0.0, 0.0, 0, 1.0
    rank = int(np.linalg.matrix_rank(reduced_map))
    if rank != derived_state.getNU():
        raise ValueError(
            "common-coordinate velocity map does not span reduced mobility"
        )
    condition = float(np.linalg.cond(reduced_map))
    if not np.isfinite(condition):
        raise ValueError("nonfinite native mobility-lift condition")
    lift = source_map @ np.linalg.pinv(reduced_map)
    if _max_difference(source_map, lift @ reduced_map) > _GEOMETRY_TOL:
        raise ValueError("native source-to-derived mobility lift is incomplete")
    source_mass = _native_mass(original, original_state)
    reduced_mass = _native_mass(derived, derived_state)
    mass_error = _max_difference(reduced_mass, lift.T @ source_mass @ lift)
    source_force = _native_total_generalized_force(original, original_state)
    reduced_force = _native_total_generalized_force(derived, derived_state)
    force_error = _max_difference(reduced_force, lift.T @ source_force)
    return mass_error, force_error, rank, condition


def _body_velocity(body: Any, state: Any) -> np.ndarray:
    spatial = body.getVelocityInGround(state)
    return _finite_native_array(
        np.array([spatial.get(i).get(j) for i in range(2) for j in range(3)])
    )


def _constraint_residual(state: Any) -> float:
    values = np.array(
        [
            vector.get(i)
            for vector in (state.getQErr(), state.getUErr())
            for i in range(vector.size())
        ]
    )
    return float(np.max(np.abs(_finite_native_array(values)), initial=0.0))


def _check_inventory(original: Any, derived: Any) -> tuple[str, ...]:
    original_bodies = original.getBodySet()
    derived_bodies = derived.getBodySet()
    if tuple(body.getName() for body in original_bodies) != tuple(
        body.getName() for body in derived_bodies
    ):
        raise ValueError("derived body inventory changed")
    if tuple(body.dump() for body in original_bodies) != tuple(
        body.dump() for body in derived_bodies
    ):
        raise ValueError("derived body mass, COM, inertia or properties changed")
    original_muscles = original.getMuscles()
    derived_muscles = derived.getMuscles()
    if tuple(muscle.getName() for muscle in original_muscles) != tuple(
        muscle.getName() for muscle in derived_muscles
    ):
        raise ValueError("derived muscle inventory changed")
    if any(
        muscle.dump() != derived_muscles.get(muscle.getName()).dump()
        for muscle in original_muscles
    ):
        raise ValueError("derived muscle properties changed")
    if tuple(actuator.dump() for actuator in original.getActuators()) != tuple(
        actuator.dump() for actuator in derived.getActuators()
    ):
        raise ValueError("derived actuator inventory or properties changed")
    if tuple(constraint.dump() for constraint in original.getConstraintSet()) != tuple(
        constraint.dump() for constraint in derived.getConstraintSet()
    ):
        raise ValueError("derived constraint inventory or properties changed")
    names = tuple(coordinate.getName() for coordinate in derived.getCoordinateSet())
    if set(names) != {
        coordinate.getName() for coordinate in original.getCoordinateSet()
    } - set(_COORDINATES):
        raise ValueError("derived coordinate inventory changed beyond bilateral MTP")
    return names


def _comparison_poses(
    original: Any, derived: Any, names: tuple[str, ...]
) -> tuple[tuple[Any, Any], ...]:
    original_state = original.initSystem()
    derived_state = derived.initSystem()
    source_names = original.getStateVariableNames()
    reduced_names = derived.getStateVariableNames()
    source_ids = {source_names.get(i) for i in range(source_names.getSize())}
    reduced_ids = {reduced_names.get(i) for i in range(reduced_names.getSize())}
    removed_ids = source_ids - reduced_ids
    if (
        reduced_ids - source_ids
        or len(removed_ids) != 4
        or any(
            not any(
                f"/mtp_angle_{side}/{kind}" in identifier
                for side in ("r", "l")
                for kind in ("value", "speed")
            )
            for identifier in removed_ids
        )
    ):
        raise ValueError("derived continuous state changed beyond bilateral MTP")
    for identifier in reduced_ids:
        value = original.getStateVariableValue(original_state, identifier)
        derived.setStateVariableValue(derived_state, identifier, value)
        if derived.getStateVariableValue(derived_state, identifier) != value:
            raise ValueError(
                "reduced model did not restore exact common continuous state"
            )
    poses: tuple[tuple[Any, Any], ...] = ((original_state, derived_state),)
    for name in ("ankle_angle_r", "pelvis_tilt"):
        if name not in names:
            continue
        changed = _perturbed_pose(
            original, derived, original_state, derived_state, name
        )
        if changed is not None:
            poses += (changed,)
    return poses


def _perturbed_pose(
    original: Any,
    derived: Any,
    original_state: Any,
    derived_state: Any,
    name: str,
) -> tuple[Any, Any] | None:
    import opensim as osim

    original_coordinate = original.getCoordinateSet().get(name)
    if original_coordinate.getLocked(original_state):
        return None
    changed_original = osim.State(original_state)
    changed_derived = osim.State(derived_state)
    derived_coordinate = derived.getCoordinateSet().get(name)
    initial_value = original_coordinate.getValue(changed_original)
    changed_value = initial_value + 0.01
    if changed_value > original_coordinate.getRangeMax():
        changed_value = initial_value - 0.01
    if changed_value < original_coordinate.getRangeMin():
        return None
    original_coordinate.setValue(changed_original, changed_value, False)
    derived_coordinate.setValue(changed_derived, changed_value, False)
    original_coordinate.setSpeedValue(changed_original, 0.02)
    derived_coordinate.setSpeedValue(changed_derived, 0.02)
    return changed_original, changed_derived


def _pose_error(
    original: Any,
    derived: Any,
    names: tuple[str, ...],
    original_pose: Any,
    derived_pose: Any,
) -> tuple[float, float, float, float, float, float, float, float, int, float]:
    original_bodies = original.getBodySet()
    derived_bodies = derived.getBodySet()
    original_muscles = original.getMuscles()
    derived_muscles = derived.getMuscles()
    original.realizeDynamics(original_pose)
    derived.realizeDynamics(derived_pose)
    body_error = 0.0
    velocity_error = 0.0
    muscle_error = 0.0
    for body in original_bodies:
        matched = derived_bodies.get(body.getName())
        body_error = max(
            body_error,
            _max_difference(
                _transform(body, original_pose), _transform(matched, derived_pose)
            ),
        )
        velocity_error = max(
            velocity_error,
            _max_difference(
                _body_velocity(body, original_pose),
                _body_velocity(matched, derived_pose),
            ),
        )
    for muscle in original_muscles:
        matched = derived_muscles.get(muscle.getName())
        length_error = abs(
            muscle.getLength(original_pose) - matched.getLength(derived_pose)
        )
        if not np.isfinite(length_error):
            raise ValueError("nonfinite native muscle path length")
        muscle_error = max(
            muscle_error,
            length_error,
        )
    before_mass = _mass_on_common_coordinates(original, original_pose, names)
    after_mass = _mass_on_common_coordinates(derived, derived_pose, names)
    mass_error = _max_difference(before_mass, after_mass)
    before_force = _projected_total_force(original, original_pose, names)
    after_force = _projected_total_force(derived, derived_pose, names)
    force_error = _max_difference(before_force, after_force)
    residual = max(
        _constraint_residual(original_pose), _constraint_residual(derived_pose)
    )
    lift_mass, lift_force, rank, condition = _full_mobility_lift_errors(
        original, derived, original_pose, derived_pose, names
    )
    return (
        body_error,
        muscle_error,
        mass_error,
        force_error,
        velocity_error,
        residual,
        lift_mass,
        lift_force,
        rank,
        condition,
    )


def _native_comparison(
    original: Any, derived: Any
) -> tuple[float, float, float, float, float, float, float, float, int, float, int]:
    names = _check_inventory(original, derived)
    poses = _comparison_poses(original, derived, names)
    body_error = muscle_error = mass_error = 0.0
    force_error = velocity_error = 0.0
    constraint_residual = 0.0
    lift_mass_error = lift_force_error = 0.0
    rank = 0
    condition = 1.0
    for original_pose, derived_pose in poses:
        pose_errors = _pose_error(original, derived, names, original_pose, derived_pose)
        body_error = max(body_error, pose_errors[0])
        muscle_error = max(muscle_error, pose_errors[1])
        mass_error = max(mass_error, pose_errors[2])
        force_error = max(force_error, pose_errors[3])
        velocity_error = max(velocity_error, pose_errors[4])
        constraint_residual = max(constraint_residual, pose_errors[5])
        lift_mass_error = max(lift_mass_error, pose_errors[6])
        lift_force_error = max(lift_force_error, pose_errors[7])
        rank = pose_errors[8]
        condition = max(condition, pose_errors[9])
    if (
        max(
            body_error,
            muscle_error,
            mass_error,
            force_error,
            velocity_error,
            constraint_residual,
            lift_mass_error,
            lift_force_error,
        )
        > _GEOMETRY_TOL
    ):
        raise ValueError(
            "fresh native derived model does not preserve zero-MTP mechanics"
        )
    return (
        body_error,
        muscle_error,
        mass_error,
        force_error,
        velocity_error,
        constraint_residual,
        lift_mass_error,
        lift_force_error,
        rank,
        condition,
        len(poses),
    )


def derive_zero_mtp_model(request: ZeroMtpReductionRequest) -> ZeroMtpReductionReceipt:
    """Apply native zero-only bilateral weld and verify a fresh saved artifact."""
    import opensim as osim

    source = request.source_model_path
    output = request.derived_model_path
    reducer_sha256 = _sha(Path(__file__))
    require_fresh_reduction_paths(source, request.source_sha256, output)
    _reject_removed_coordinate_references(source)
    original = osim.Model(str(source))
    state = original.initSystem()
    _coordinate_admission(original, state)
    derived = osim.Model(original)
    names = osim.StdVectorString()
    for joint_name in _JOINTS:
        names.append(joint_name)
    osim.ModOpReplaceJointsWithWelds(names).operate(derived, "")
    derived.printToXML(str(output))
    if _sha(source) != request.source_sha256:
        output.unlink(missing_ok=True)
        raise ValueError("source model changed during reduction")
    try:
        reloaded = osim.Model(str(output))
        (
            body_error,
            muscle_error,
            mass_error,
            force_error,
            velocity_error,
            constraint_residual,
            lift_mass_error,
            lift_force_error,
            lift_rank,
            lift_condition,
            pose_count,
        ) = _native_comparison(original, reloaded)
    except Exception:
        output.unlink(missing_ok=True)
        raise
    if _sha(Path(__file__)) != reducer_sha256:
        output.unlink(missing_ok=True)
        raise ValueError("reduction implementation changed during native verification")
    if _sha(source) != request.source_sha256:
        output.unlink(missing_ok=True)
        raise ValueError("source model changed during native verification")
    return ZeroMtpReductionReceipt(
        source_sha256=request.source_sha256,
        derived_sha256=_sha(output),
        reducer_sha256=reducer_sha256,
        native_version=osim.GetVersionAndDate(),
        native_simulation_sha256=_sha(Path(osim._simulation.__file__)),
        native_simbody_sha256=_sha(Path(osim._simbody.__file__)),
        native_actuators_sha256=_sha(Path(osim._actuators.__file__)),
        removed_coordinate_names=_COORDINATES,
        max_body_transform_error=body_error,
        max_muscle_length_error=muscle_error,
        max_projected_mass_error=mass_error,
        max_full_lift_mass_error=lift_mass_error,
        max_full_lift_force_error=lift_force_error,
        mobility_lift_rank=lift_rank,
        mobility_lift_condition=lift_condition,
        max_projected_total_force_error=force_error,
        max_body_velocity_error=velocity_error,
        max_constraint_residual=constraint_residual,
        observed_pose_count=pose_count,
        native_reload_verified=True,
    )
