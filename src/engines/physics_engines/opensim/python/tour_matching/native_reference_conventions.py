"""Read-only native geometry and muscle-law observations at a declared state.

Coordinates and names from distinct model sources have no implicit anatomical
correspondence. This module records observations and explicit comparisons only.
Native initSystem includes initialization assembly. The observer performs no
additional assembly, equilibrium, integration, or source-parameter change.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, cast

from src.engines.physics_engines.opensim.python.tour_matching.native_constraint_state import (
    NativeConstraintStateAudit,
    audit_native_constraint_state,
    native_muscles,
    owned_native_source_state,
)


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(asdict(value), sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def _number(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("native reference values must be finite")
    return result


def _vec(value: Any) -> tuple[float, float, float]:
    return cast(
        tuple[float, float, float], tuple(_number(value.get(i)) for i in range(3))
    )


def _rotation(value: Any) -> tuple[tuple[float, float, float], ...]:
    return tuple(
        cast(
            tuple[float, float, float],
            tuple(_number(value.get(i, j)) for j in range(3)),
        )
        for i in range(3)
    )


def _relative_position(
    rotation: tuple[tuple[float, float, float], ...],
    position: tuple[float, float, float],
    origin: tuple[float, float, float],
) -> tuple[float, float, float]:
    return cast(
        tuple[float, float, float],
        tuple(
            sum(rotation[j][i] * (position[j] - origin[j]) for j in range(3))
            for i in range(3)
        ),
    )


def _relative_rotation(
    reference: tuple[tuple[float, float, float], ...],
    other: tuple[tuple[float, float, float], ...],
) -> tuple[tuple[float, float, float], ...]:
    return tuple(
        cast(
            tuple[float, float, float],
            tuple(
                sum(reference[k][i] * other[k][j] for k in range(3)) for j in range(3)
            ),
        )
        for i in range(3)
    )


@dataclass(frozen=True)
class ReferenceStateDeclaration:
    """Post-initSystem state or all named continuous values, with provenance."""

    reference_id: str
    provenance: str
    mode: str
    expected_source_sha256: str
    named_state: tuple[tuple[str, float], ...] = ()
    expected_native_time_seconds: float = 0.0

    def __post_init__(self) -> None:
        if not self.reference_id.strip() or not self.provenance.strip():
            raise ValueError("reference identity and provenance must be explicit")
        if not re.fullmatch(r"[0-9a-f]{64}", self.expected_source_sha256):
            raise ValueError("reference requires exact source SHA-256")
        if self.mode not in {"post-init-system", "complete-named-state"}:
            raise ValueError("unsupported reference state mode")
        if (self.mode == "post-init-system" and self.named_state) or (
            self.mode == "complete-named-state" and not self.named_state
        ):
            raise ValueError("complete named state required only for named-state mode")
        if len({name for name, _ in self.named_state}) != len(self.named_state):
            raise ValueError("duplicate named state variables")
        for name, value in self.named_state:
            if not name.startswith("/"):
                raise ValueError("named state paths must be absolute")
            _number(value)
        _number(self.expected_native_time_seconds)


@dataclass(frozen=True)
class FrameObservation:
    path: str
    concrete_class: str
    position_in_reference_m: tuple[float, float, float]
    rotation_in_reference: tuple[tuple[float, float, float], ...]
    serialized_sha256: str


@dataclass(frozen=True)
class JointObservation:
    path: str
    concrete_class: str
    parent_frame_path: str
    child_frame_path: str
    coordinate_paths: tuple[str, ...]
    serialized_sha256: str


@dataclass(frozen=True)
class CouplerObservation:
    path: str
    dependent_coordinate_path: str
    independent_coordinate_paths: tuple[str, ...]
    function_class: str
    enforced: bool
    serialized_sha256: str


@dataclass(frozen=True)
class PathPointObservation:
    path: str
    concrete_class: str
    parent_frame_path: str
    local_m: tuple[float, float, float]
    ground_m: tuple[float, float, float]
    active: bool
    serialized_sha256: str


@dataclass(frozen=True)
class RoutePointObservation:
    path: str
    concrete_class: str
    ground_m: tuple[float, float, float]
    active: bool
    wrap_length_m: float | None
    wrap_curve_native_m: tuple[tuple[float, float, float], ...]


@dataclass(frozen=True)
class WrapObservation:
    path: str
    object_name: str
    object_path: str
    object_class: str
    parent_frame_path: str
    translation_m: tuple[float, float, float]
    xyz_body_rotation_rad: tuple[float, float, float]
    serialized_sha256: str


@dataclass(frozen=True)
class MuscleObservation:
    path: str
    concrete_class: str
    path_length_m: float
    optimal_fiber_length_m: float
    tendon_slack_length_m: float
    pennation_angle_at_optimal_rad: float
    maximum_isometric_force_n: float
    ignore_tendon_compliance: bool
    ignore_activation_dynamics: bool
    ignore_passive_fiber_force: bool | None
    fiber_damping: float | None
    path_points: tuple[PathPointObservation, ...]
    wraps: tuple[WrapObservation, ...]
    current_route: tuple[RoutePointObservation, ...]
    serialized_sha256: str


@dataclass(frozen=True)
class NativeReferenceObservation:
    source_sha256: str
    native_state: NativeConstraintStateAudit
    reference: ReferenceStateDeclaration
    reference_frame_path: str
    native_assembly_accuracy: float
    frames: tuple[FrameObservation, ...]
    joints: tuple[JointObservation, ...]
    couplers: tuple[CouplerObservation, ...]
    muscles: tuple[MuscleObservation, ...]
    observer_sha256: str
    qualification: str = field(default="not-qualified-for-anatomy", init=False)

    @property
    def observation_sha256(self) -> str:
        return _sha(self)


def _frame_transform(frame: Any, state: Any) -> tuple[Any, Any]:
    transform = frame.getTransformInGround(state)
    return _vec(transform.p()), _rotation(transform.R())


def _observe_frames(model: Any, state: Any, reference_path: str, osim: Any):
    frames = [model.getGround()]
    for component in tuple(model.getComponentsList()):
        frame = osim.PhysicalFrame.safeDownCast(component)
        if frame is not None and frame.getAbsolutePathString() != "/ground":
            frames.append(frame)
    unique = {frame.getAbsolutePathString(): frame for frame in frames}
    if reference_path not in unique:
        raise ValueError("reference frame path is unknown")
    origin, rotation = _frame_transform(unique[reference_path], state)
    result = []
    for path, frame in sorted(unique.items()):
        position, frame_rotation = _frame_transform(frame, state)
        result.append(
            FrameObservation(
                path,
                frame.getConcreteClassName(),
                _relative_position(rotation, position, origin),
                _relative_rotation(rotation, frame_rotation),
                hashlib.sha256(frame.dump().encode()).hexdigest(),
            )
        )
    return tuple(result)


def _observe_joints(model: Any):
    return tuple(
        JointObservation(
            joint.getAbsolutePathString(),
            joint.getConcreteClassName(),
            joint.getParentFrame().getAbsolutePathString(),
            joint.getChildFrame().getAbsolutePathString(),
            tuple(
                joint.get_coordinates(i).getAbsolutePathString()
                for i in range(joint.numCoordinates())
            ),
            hashlib.sha256(joint.dump().encode()).hexdigest(),
        )
        for joint in model.getJointSet()
    )


def _observe_couplers(model: Any, state: Any, osim: Any):
    coordinate_set = tuple(model.getCoordinateSet())
    coordinates = {
        coordinate.getName(): coordinate.getAbsolutePathString()
        for coordinate in coordinate_set
    }
    if len(coordinates) != len(coordinate_set):
        raise ValueError(
            "ambiguous short coordinate names in native coupler resolution"
        )
    result = []
    for component in tuple(model.getComponentsList()):
        coupler = osim.CoordinateCouplerConstraint.safeDownCast(component)
        if coupler is None:
            continue
        names = coupler.getIndependentCoordinateNames()
        dependent = coupler.getDependentCoordinateName()
        if dependent not in coordinates or any(
            names.get(i) not in coordinates for i in range(names.getSize())
        ):
            raise ValueError("coupler coordinate dependency is unresolved")
        result.append(
            CouplerObservation(
                coupler.getAbsolutePathString(),
                coordinates[dependent],
                tuple(coordinates[names.get(i)] for i in range(names.getSize())),
                coupler.getFunction().getConcreteClassName(),
                bool(coupler.isEnforced(state)),
                hashlib.sha256(coupler.dump().encode()).hexdigest(),
            )
        )
    return tuple(result)


def _path_point(point: Any, state: Any) -> PathPointObservation:
    return PathPointObservation(
        point.getAbsolutePathString(),
        point.getConcreteClassName(),
        point.getParentFrame().getAbsolutePathString(),
        _vec(point.getLocation(state)),
        _vec(point.getLocationInGround(state)),
        bool(point.isActive(state)),
        hashlib.sha256(point.dump().encode()).hexdigest(),
    )


def _route_point(point: Any, state: Any, osim: Any) -> RoutePointObservation:
    wrapped = osim.PathWrapPoint.safeDownCast(point)
    curve = wrapped.getWrapPath(state) if wrapped is not None else None
    return RoutePointObservation(
        point.getAbsolutePathString(),
        point.getConcreteClassName(),
        _vec(point.getLocationInGround(state)),
        bool(point.isActive(state)),
        _number(wrapped.getWrapLength(state)) if wrapped is not None else None,
        tuple(_vec(curve.get(i)) for i in range(curve.getSize()))
        if curve is not None
        else (),
    )


def _wrap(wrap: Any) -> WrapObservation:
    obj = wrap.getWrapObject()
    if obj is None:
        raise ValueError("unresolved native wrap object")
    return WrapObservation(
        wrap.getAbsolutePathString(),
        wrap.getWrapObjectName(),
        obj.getAbsolutePathString(),
        obj.getConcreteClassName(),
        obj.getFrame().getAbsolutePathString(),
        _vec(obj.get_translation()),
        _vec(obj.get_xyz_body_rotation()),
        hashlib.sha256(obj.dump().encode()).hexdigest(),
    )


def _observe_muscles(model: Any, state: Any, osim: Any):
    result = []
    for muscle in native_muscles(model, osim):
        path = muscle.getGeometryPath()
        points = path.getPathPointSet()
        wraps = path.getWrapSet()
        route = path.getCurrentPath(state)
        dgf = osim.DeGrooteFregly2016Muscle.safeDownCast(muscle)
        result.append(
            MuscleObservation(
                muscle.getAbsolutePathString(),
                muscle.getConcreteClassName(),
                _number(path.getLength(state)),
                _number(muscle.getOptimalFiberLength()),
                _number(muscle.getTendonSlackLength()),
                _number(muscle.getPennationAngleAtOptimalFiberLength()),
                _number(muscle.getMaxIsometricForce()),
                bool(muscle.getIgnoreTendonCompliance(state)),
                bool(muscle.getIgnoreActivationDynamics(state)),
                bool(dgf.get_ignore_passive_fiber_force()) if dgf is not None else None,
                _number(dgf.get_fiber_damping()) if dgf is not None else None,
                tuple(
                    _path_point(points.get(i), state) for i in range(points.getSize())
                ),
                tuple(_wrap(wraps.get(i)) for i in range(wraps.getSize())),
                tuple(
                    _route_point(route.get(i), state, osim)
                    for i in range(route.getSize())
                ),
                hashlib.sha256(muscle.dump().encode()).hexdigest(),
            )
        )
    return tuple(result)


def audit_source_reference(
    model_path: Path, reference: ReferenceStateDeclaration, reference_frame_path: str
) -> NativeReferenceObservation:
    """Observe one exact XML source at a documented native reference state."""
    import opensim as osim

    source_sha = hashlib.sha256(Path(model_path).read_bytes()).hexdigest()
    if source_sha != reference.expected_source_sha256:
        raise ValueError("source SHA-256 identity differs from reference declaration")
    if not reference_frame_path.startswith("/"):
        raise ValueError("reference frame path must be absolute")
    initial = (
        dict(reference.named_state)
        if reference.mode == "complete-named-state"
        else None
    )
    with owned_native_source_state(Path(model_path), initial) as (
        model,
        state,
        loaded_source_sha,
    ):
        if loaded_source_sha != source_sha:
            raise ValueError("source SHA-256 changed during native load")
        native = audit_native_constraint_state(model, state)
        if native.time_seconds != reference.expected_native_time_seconds:
            raise ValueError(
                "native initialized clock differs from declared reference time"
            )
        model.realizePosition(state)
        assembly_accuracy = _number(model.get_assembly_accuracy())
        frames = _observe_frames(model, state, reference_frame_path, osim)
        joints = _observe_joints(model)
        couplers = _observe_couplers(model, state, osim)
        muscles = _observe_muscles(model, state, osim)
        after = audit_native_constraint_state(model, state)
        if native != after:
            raise ValueError(
                "native state or model mutated during reference observation"
            )
    return NativeReferenceObservation(
        source_sha,
        native,
        reference,
        reference_frame_path,
        assembly_accuracy,
        frames,
        joints,
        couplers,
        muscles,
        hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )


@dataclass(frozen=True)
class NativeReferenceCorrespondence:
    """Explicit selective mapping and rigid registration, not anatomical identity."""

    left_observation_sha256: str
    right_observation_sha256: str
    frame_pairs: tuple[tuple[str, str], ...]
    joint_pairs: tuple[tuple[str, str], ...]
    coordinate_pairs: tuple[tuple[str, str], ...]
    muscle_pairs: tuple[tuple[str, str], ...]
    reference_rotation: tuple[tuple[float, float, float], ...]
    reference_translation_m: tuple[float, float, float]
    provenance: str

    def __post_init__(self) -> None:
        if not self.provenance.strip():
            raise ValueError("correspondence provenance required")
        if not any(
            (
                self.frame_pairs,
                self.joint_pairs,
                self.coordinate_pairs,
                self.muscle_pairs,
            )
        ):
            raise ValueError("at least one explicit correspondence pair required")
        for pairs in (
            self.frame_pairs,
            self.joint_pairs,
            self.coordinate_pairs,
            self.muscle_pairs,
        ):
            if any(
                len(pair) != 2 or not all(p.startswith("/") for p in pair)
                for pair in pairs
            ):
                raise ValueError("correspondence paths must be absolute pairs")
            if len({pair[0] for pair in pairs}) != len(pairs) or len(
                {pair[1] for pair in pairs}
            ) != len(pairs):
                raise ValueError("correspondence paths must be one-to-one")
        rotation = self.reference_rotation
        if len(rotation) != 3 or any(len(row) != 3 for row in rotation):
            raise ValueError("reference rotation must be 3x3")
        if any(
            abs(sum(rotation[k][i] * rotation[k][j] for k in range(3)) - (i == j))
            > 1e-10
            for i in range(3)
            for j in range(3)
        ):
            raise ValueError("reference rotation must be orthonormal")
        determinant = sum(
            rotation[0][j]
            * (
                rotation[1][(j + 1) % 3] * rotation[2][(j + 2) % 3]
                - rotation[1][(j + 2) % 3] * rotation[2][(j + 1) % 3]
            )
            for j in range(3)
        )
        if abs(determinant - 1.0) > 1e-10:
            raise ValueError("reference rotation must be proper")
        if len(self.reference_translation_m) != 3:
            raise ValueError("reference translation must be three-dimensional")
        for row in rotation:
            for value in row:
                _number(value)
        for value in self.reference_translation_m:
            _number(value)


@dataclass(frozen=True)
class ReferenceDelta:
    left_path: str
    right_path: str
    delta_m: float | tuple[float, float, float]


@dataclass(frozen=True)
class FrameOrientationDelta:
    left_path: str
    right_path: str
    angle_rad: float


@dataclass(frozen=True)
class NativeReferenceComparison:
    correspondence_sha256: str
    left_observation_sha256: str
    right_observation_sha256: str
    frame_offsets_m: tuple[ReferenceDelta, ...]
    frame_orientation_deltas_rad: tuple[FrameOrientationDelta, ...]
    muscle_length_deltas_m: tuple[ReferenceDelta, ...]
    mapped_joint_pairs_not_compared: tuple[tuple[str, str], ...]
    mapped_coordinate_pairs_not_compared: tuple[tuple[str, str], ...]
    qualification: str = field(default="not-qualified-for-anatomy", init=False)


def _mapped(
    pairs: tuple[tuple[str, str], ...], left: dict[str, Any], right: dict[str, Any]
):
    for first, second in pairs:
        if first not in left or second not in right:
            raise ValueError("unknown explicit correspondence path")
    return ((first, second, left[first], right[second]) for first, second in pairs)


def _orientation_angle(
    registration: tuple[tuple[float, float, float], ...],
    left_frame: FrameObservation,
    right_frame: FrameObservation,
) -> float:
    predicted = tuple(
        tuple(
            sum(
                registration[i][k] * left_frame.rotation_in_reference[k][j]
                for k in range(3)
            )
            for j in range(3)
        )
        for i in range(3)
    )
    cosine = (
        sum(
            predicted[i][j] * right_frame.rotation_in_reference[i][j]
            for i in range(3)
            for j in range(3)
        )
        - 1.0
    ) / 2.0
    return math.acos(max(-1.0, min(1.0, cosine)))


def compare_native_references(
    left: NativeReferenceObservation,
    right: NativeReferenceObservation,
    correspondence: NativeReferenceCorrespondence,
) -> NativeReferenceComparison:
    """Compare only explicitly mapped observations in declared rigid frames."""
    if (
        left.observation_sha256 != correspondence.left_observation_sha256
        or right.observation_sha256 != correspondence.right_observation_sha256
    ):
        raise ValueError("reference observation identity mismatch")
    matched_frames = tuple(
        _mapped(
            correspondence.frame_pairs,
            {v.path: v for v in left.frames},
            {v.path: v for v in right.frames},
        )
    )
    frames = tuple(
        ReferenceDelta(
            a,
            b,
            tuple(
                right_pos.position_in_reference_m[i]
                - sum(
                    correspondence.reference_rotation[i][j]
                    * left_pos.position_in_reference_m[j]
                    for j in range(3)
                )
                - correspondence.reference_translation_m[i]
                for i in range(3)
            ),
        )
        for a, b, left_pos, right_pos in matched_frames
    )
    orientations = tuple(
        FrameOrientationDelta(
            a, b, _orientation_angle(correspondence.reference_rotation, lf, rf)
        )
        for a, b, lf, rf in matched_frames
    )
    muscles = tuple(
        ReferenceDelta(a, b, right_muscle.path_length_m - left_muscle.path_length_m)
        for a, b, left_muscle, right_muscle in _mapped(
            correspondence.muscle_pairs,
            {v.path: v for v in left.muscles},
            {v.path: v for v in right.muscles},
        )
    )
    tuple(
        _mapped(
            correspondence.joint_pairs,
            {v.path: v for v in left.joints},
            {v.path: v for v in right.joints},
        )
    )
    tuple(
        _mapped(
            correspondence.coordinate_pairs,
            {v.path: v for v in left.native_state.coordinates},
            {v.path: v for v in right.native_state.coordinates},
        )
    )
    return NativeReferenceComparison(
        _sha(correspondence),
        left.observation_sha256,
        right.observation_sha256,
        frames,
        orientations,
        muscles,
        correspondence.joint_pairs,
        correspondence.coordinate_pairs,
    )


__all__ = [
    "NativeReferenceCorrespondence",
    "NativeReferenceComparison",
    "NativeReferenceObservation",
    "ReferenceStateDeclaration",
    "audit_source_reference",
    "compare_native_references",
]
