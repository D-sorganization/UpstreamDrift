"""OpenSim force/torque overlay source (ADR-0052, FTO-15, #11300).

Builds a world-frame (Z-up) :class:`ForceTorqueFrame` from an OpenSim state:

* ``JOINT_REACTION`` for every joint, from
  ``Joint.calcReactionOnChildExpressedInGround`` (the wrench the parent applies
  to the child at the child-frame origin).
* ``JOINT_ACTUATOR`` for every ``CoordinateActuator`` on a rotational pin or
  single-rotation custom joint, as ``actuation * axis`` about the joint origin.
  Translational coordinates, couplings and multi-rotation custom joints are
  omitted (their axis is ambiguous), never reported as zero.
* ``CONTACT`` for every sphere/half-space ``HuntCrossleyForce`` and
  ``SmoothSphereHalfSpaceForce``, applied at the sphere's lowest point.
* ``MUSCLE`` wrenches (FTO-16, #11301): for every enabled ``Muscle`` the tendon
  force along the path's effective end directions, one wrench at the origin and
  one at the insertion (labels ``muscle:<name>:origin`` / ``:insertion``).
* Axial loads of each joint-child segment from its proximal reaction.

``grip:*`` wrenches are deliberately absent: the OpenSim grip model still
returns a placeholder (#11161, #10286), so grip is unavailable, not zero.

OpenSim ground is Y-up by convention; ADR-0026 world is Z-up. The model's
gravity declares its up axis: Y-up models are rotated by the single constant
:data:`R_ZUP_FROM_OPENSIM_GROUND` and already-Z-up models (gravity -Z) are not. (The
pre-existing ``coord_map._R_YUP_TO_ZUP`` is an axis swap with determinant -1,
a reflection, which would flip torque handedness, so it is not reused.)

Note: the segment-axis, torque-wrench and world-conversion helpers below are
minimal private stand-ins for FTO-2 (#11287) and FTO-11, which are not on main
yet; replace them with ``force_overlay`` modules once those land.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from typing import Any

import numpy as np
import opensim

from src.shared.python.biomechanics.grip_extraction import unavailable_analysis
from src.shared.python.biomechanics.grip_wrench import GripAnalysis
from src.shared.python.body_part_viz import AxialLoadFrame
from src.shared.python.body_part_viz.axial_loads import (
    axial_force_from_proximal_reaction,
)
from src.shared.python.force_overlay import (
    ForceTorqueFrame,
    OverlayWrench,
    WrenchKind,
)
from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.motion_matching.force_torque import (
    SpatialWrench,
    transform_wrench,
)

__all__ = ["R_ZUP_FROM_OPENSIM_GROUND", "OpenSimForceTorqueSource"]

logger = get_logger(__name__)

# Rotation by +90 deg about x: OpenSim (x, y, z) -> world (x, -z, y), so the
# OpenSim up axis (0, 1, 0) becomes world up (0, 0, 1). det = +1.
R_ZUP_FROM_OPENSIM_GROUND: np.ndarray = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float64
)
R_ZUP_FROM_OPENSIM_GROUND.setflags(write=False)

_ENGINE = "opensim"
_REACTION_SOURCE = "opensim:Joint.calcReactionOnChildExpressedInGround"
_ACTUATOR_SOURCE = "opensim:CoordinateActuator.getActuation"
_CONTACT_SOURCE = "opensim:ContactForce.getRecordValues"
_MUSCLE_SOURCE = "opensim:Muscle.getTendonForce"
_MIN_SEGMENT_LENGTH_M = 1e-12
_MIN_DIRECTION_NORM = 1e-9  # PointForceDirection with no force transfer
_HALF_SPACE_OUTWARD_LOCAL = np.array([-1.0, 0.0, 0.0])  # Simbody HalfSpace normal


def _label(prefix: str, name: str) -> str:
    return f"{prefix}:{re.sub(r'[^A-Za-z0-9_.:-]', '_', name)}"


def _vec3(v: Any) -> tuple[float, float, float]:
    """Convert an OpenSim/Simbody ``Vec3`` to a plain tuple."""
    return (float(v.get(0)), float(v.get(1)), float(v.get(2)))


def _mat33(rotation: Any) -> np.ndarray:
    """Convert an OpenSim ``Rotation`` to a 3x3 array."""
    m = rotation.asMat33()
    return np.array([[m.get(i, j) for j in range(3)] for i in range(3)])


def _rotation_to_z_up(model: Any) -> np.ndarray:
    """Rotation taking the model's declared up axis (against gravity) to +Z.

    OpenSim models are Y-up by convention, but generated full-body models use
    gravity ``(0, 0, -g)`` and are already Z-up. A Y-up model gets exactly
    :data:`R_ZUP_FROM_OPENSIM_GROUND`; a zero-gravity model has no declared
    axis and is treated as Y-up. Postcondition: proper rotation (det +1).
    """
    gravity = np.asarray(_vec3(model.get_gravity()))
    norm = float(np.linalg.norm(gravity))
    if norm < 1e-12:
        return R_ZUP_FROM_OPENSIM_GROUND
    up = -gravity / norm
    target = np.array([0.0, 0.0, 1.0])
    axis = np.cross(up, target)
    sin, cos = float(np.linalg.norm(axis)), float(up @ target)
    if sin < 1e-12:
        if cos > 0:
            return np.eye(3)
        return np.diag([1.0, -1.0, -1.0])  # up is -Z: half turn about x
    k = axis / sin
    skew = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + sin * skew + (1 - cos) * (skew @ skew)


def _tuple3(v: np.ndarray) -> tuple[float, float, float]:
    return (float(v[0]), float(v[1]), float(v[2]))


def _base_name(frame: Any) -> str:
    return str(frame.findBaseFrame().getName())


def _geometry_names(contact_force: Any) -> list[str]:
    """Contact geometry names referenced by a HuntCrossleyForce's parameters.

    ``ContactParameters`` is not wrapped by the Python bindings, so the names
    are read through the generic property interface.
    """
    names: list[str] = []
    sets = contact_force.getPropertyByName("contact_parameters")
    for i in range(sets.size()):
        objects = sets.getValueAsObject(i).getPropertyByName("objects")
        for j in range(objects.size()):
            geometry = objects.getValueAsObject(j).getPropertyByName("geometry")
            names.extend(geometry.toString().strip("()").split())
    return names


class OpenSimForceTorqueSource:
    """Compute world-frame force/torque overlays for an OpenSim model.

    Preconditions: ``model`` is an initialized ``opensim.Model`` (``initSystem``
    has been called) and states passed to :meth:`sample` belong to it.
    Postconditions: ``sample`` returns a frame in the Z-up world; unsupported
    joints, coordinates and forces are omitted, never zero-filled. Muscle
    wrenches are included by default (a model without muscles has none); pass
    ``include_muscles=False`` to skip them.
    """

    def __init__(self, model: Any, *, include_muscles: bool = True) -> None:
        if not isinstance(model, opensim.Model):
            raise TypeError("model must be an opensim.Model")
        self._model = model
        self._include_muscles = bool(include_muscles)
        self._to_world = _rotation_to_z_up(model)

    def _world(self, v: Iterable[float]) -> np.ndarray:
        """Express an OpenSim ground-frame vector or point in the Z-up world."""
        return self._to_world @ np.asarray(tuple(v), dtype=np.float64)

    @property
    def model(self) -> Any:
        """The wrapped ``opensim.Model``."""
        return self._model

    #: Why grip wrenches cannot be reported (GCV-8, #11714).
    GRIP_UNAVAILABLE_REASON = (
        "OpenSim grip model is a placeholder (#11161, #10286): no constraint "
        "multipliers or per-hand wrenches are computed"
    )

    def grip_analysis(self, state: Any = None) -> GripAnalysis:
        """Explicitly unavailable grip analysis: ``None`` values, never zero."""
        return unavailable_analysis(
            self.GRIP_UNAVAILABLE_REASON, metadata={"engine": _ENGINE}
        )

    def sample(self, state: Any) -> ForceTorqueFrame:
        """Return the overlay frame for ``state``.

        The state is realized to the Acceleration stage here (idempotent;
        reactions need accelerations, so a state realized only to Dynamics
        would otherwise read stale cache values).

        Raises:
            TypeError: If ``state`` is not an ``opensim.State``.
            ValueError: If the state time is not finite.
        """
        if not isinstance(state, opensim.State):
            raise TypeError("state must be an opensim.State")
        time_s = float(state.getTime())
        if not math.isfinite(time_s):
            raise ValueError("state time must be finite")
        self._model.realizeAcceleration(state)

        wrenches: list[OverlayWrench] = []
        wrenches.extend(self._reactions(state))
        wrenches.extend(self._actuators(state))
        wrenches.extend(self._contacts(state))
        if self._include_muscles:
            wrenches.extend(self.muscle_wrenches(state))
        axial = self._axial_loads(state)
        loads = (
            AxialLoadFrame(time_s=time_s, values_n=axial, source=_REACTION_SOURCE)
            if axial
            else None
        )
        return ForceTorqueFrame(
            time_s=time_s, engine=_ENGINE, wrenches=tuple(wrenches), axial_loads=loads
        )

    def _joints(self) -> list[Any]:
        joints = self._model.getJointSet()
        return [joints.get(i) for i in range(joints.getSize())]

    def _reaction_on_child(self, joint: Any, state: Any) -> tuple[np.ndarray, ...]:
        """World-frame ``(point, force, torque)`` of the parent-on-child reaction."""
        spatial = joint.calcReactionOnChildExpressedInGround(state)
        point = joint.getChildFrame().getPositionInGround(state)
        return (
            self._world(_vec3(point)),
            self._world(_vec3(spatial.get(1))),
            self._world(_vec3(spatial.get(0))),
        )

    def _reactions(self, state: Any) -> list[OverlayWrench]:
        wrenches = []
        for joint in self._joints():
            point, force, torque = self._reaction_on_child(joint, state)
            wrenches.append(
                OverlayWrench(
                    kind=WrenchKind.JOINT_REACTION,
                    label=_label("reaction", joint.getName()),
                    body=_base_name(joint.getChildFrame()),
                    point_m=_tuple3(point),
                    force_n=_tuple3(force),
                    torque_nm=_tuple3(torque),
                    source=_REACTION_SOURCE,
                )
            )
        return wrenches

    def _actuator_axis(self, coordinate: Any, state: Any) -> np.ndarray | None:
        """Unit rotation axis of ``coordinate`` in OpenSim ground, else None."""
        joint = coordinate.getJoint()
        pin = opensim.PinJoint.safeDownCast(joint)
        if pin is not None:
            return _mat33(joint.getChildFrame().getRotationInGround(state))[:, 2]
        custom = opensim.CustomJoint.safeDownCast(joint)
        if custom is None:
            return None
        axis = self._single_rotation_axis(custom, coordinate)
        if axis is None:
            return None
        return _mat33(joint.getParentFrame().getRotationInGround(state)) @ axis

    @staticmethod
    def _single_rotation_axis(custom: Any, coordinate: Any) -> np.ndarray | None:
        """Parent-frame axis if ``coordinate`` is the joint's only rotation.

        With several rotations the generalized force depends on how they
        compose, so the axis is ambiguous and the actuator is omitted. The
        coordinate must also map 1:1 onto the angle (LinearFunction slope 1).
        """
        transform = custom.get_SpatialTransform()
        driven = []
        for k in range(3):  # rotation1..rotation3
            axis = transform.getTransformAxis(k)
            names = axis.getCoordinateNames()
            if names.size() > 0:
                driven.append((axis, [names.getValue(i) for i in range(names.size())]))
        if len(driven) != 1 or driven[0][1] != [coordinate.getName()]:
            return None
        function = opensim.LinearFunction.safeDownCast(driven[0][0].getFunction())
        if function is None or not math.isclose(function.getSlope(), 1.0):
            return None
        vector = _vec3(driven[0][0].getAxis())
        return np.asarray(vector) / np.linalg.norm(vector)

    def _actuators(self, state: Any) -> list[OverlayWrench]:
        wrenches = []
        forces = self._model.getForceSet()
        for i in range(forces.getSize()):
            actuator = opensim.CoordinateActuator.safeDownCast(forces.get(i))
            if actuator is None or not actuator.appliesForce(state):
                continue
            coordinate = actuator.getCoordinate()
            axis = self._actuator_axis(coordinate, state)
            if axis is None:
                logger.info(
                    "Omitting actuator %s: coordinate %s has no unambiguous rotation "
                    "axis",
                    actuator.getName(),
                    coordinate.getName(),
                )
                continue
            joint = coordinate.getJoint()
            point = joint.getChildFrame().getPositionInGround(state)
            moment = float(actuator.getActuation(state)) * axis
            wrenches.append(
                OverlayWrench(
                    kind=WrenchKind.JOINT_ACTUATOR,
                    label=_label("actuator", f"{joint.getName()}.{actuator.getName()}"),
                    body=_base_name(joint.getChildFrame()),
                    point_m=_tuple3(self._world(_vec3(point))),
                    torque_nm=_tuple3(self._world(moment)),
                    source=_ACTUATOR_SOURCE,
                )
            )
        return wrenches

    def muscle_wrenches(self, state: Any) -> tuple[OverlayWrench, ...]:
        """``MUSCLE`` wrenches at the first and last effective path points.

        For each enabled muscle the tendon force (N, from
        ``Muscle.getTendonForce``) times the unit direction OpenSim reports at
        the end points of ``GeometryPath.getPointForceDirections`` (the force
        the path applies to the body, in ground), applied at the attachment in
        ground and rotated to the Z-up world. Torque is unavailable (``None``).
        Wrapping surfaces and via points shape the path and the end directions
        but **only the end attachments are drawn** (leading or trailing points
        on the same body as their neighbour carry no force and are skipped); the full path polyline is
        a recorded follow-up. Complements the scalar fiber forces of
        ``muscle_analysis.get_muscle_forces``, which is unchanged.

        Preconditions: ``state`` is realized to Dynamics or later (``sample``
        does this). Postcondition: every force is finite with a norm equal to
        the tendon force, which is >= 0 (tendons only pull); a muscle whose path
        has no point geometry (``FunctionBasedPath``, ``Scholz2015GeometryPath``)
        or fewer than two effective points is omitted, never
        reported as zero.

        Raises:
            AssertionError: If a tendon force is negative or not finite.
        """
        wrenches: list[OverlayWrench] = []
        forces = self._model.getForceSet()
        for i in range(forces.getSize()):
            muscle = opensim.Muscle.safeDownCast(forces.get(i))
            if muscle is None or not muscle.appliesForce(state):
                continue
            wrenches.extend(self._muscle_end_wrenches(muscle, state))
        return tuple(wrenches)

    def _muscle_end_wrenches(self, muscle: Any, state: Any) -> list[OverlayWrench]:
        # OpenSim 4.5+ paths may be FunctionBasedPath / Scholz2015GeometryPath,
        # which have no points: only a plain GeometryPath is drawable.
        path = opensim.GeometryPath.safeDownCast(muscle.getPath())
        if path is None:
            logger.info(
                "Omitting muscle %s: path has no point geometry", muscle.getName()
            )
            return []
        tendon_n = float(muscle.getTendonForce(state))
        if not math.isfinite(tendon_n) or tendon_n < 0.0:
            raise AssertionError(
                f"muscle {muscle.getName()!r}: tendon force must be finite and "
                f">= 0 N, got {tendon_n}"
            )
        directions = opensim.ArrayPointForceDirection()
        path.getPointForceDirections(state, directions)
        # The array only stores pointers; the caller owns each point.
        points = [directions.get(k) for k in range(directions.getSize())]
        try:
            return self._end_wrenches(muscle, tendon_n, points, state)
        finally:
            for point in points:
                point.thisown = True  # freed when the last reference is dropped

    def _end_wrenches(
        self, muscle: Any, tendon_n: float, points: list[Any], state: Any
    ) -> list[OverlayWrench]:
        # Consecutive path points on one body exchange force internally, so
        # OpenSim reports a zero direction there; the effective ends are the
        # first and last points that transmit force to a body.
        carrying = [
            pfd
            for pfd in points
            if np.linalg.norm(_vec3(pfd.direction())) > _MIN_DIRECTION_NORM
        ]
        if len(carrying) < 2:
            logger.info("Omitting muscle %s: path has < 2 points", muscle.getName())
            return []
        wrenches = []
        for end, pfd in (("origin", carrying[0]), ("insertion", carrying[-1])):
            frame = pfd.frame()
            point = frame.findStationLocationInGround(state, pfd.point())
            direction = np.asarray(_vec3(pfd.direction()))
            wrenches.append(
                OverlayWrench(
                    kind=WrenchKind.MUSCLE,
                    label=f"{_label('muscle', muscle.getName())}:{end}",
                    body=_base_name(frame),
                    point_m=_tuple3(self._world(_vec3(point))),
                    force_n=_tuple3(self._world(tendon_n * direction)),
                    source=_MUSCLE_SOURCE,
                )
            )
        return wrenches

    def _contact_geometry(self, force: Any) -> tuple[Any, Any] | None:
        """``(sphere, half_space)`` of a supported contact force, else None."""
        if opensim.SmoothSphereHalfSpaceForce.safeDownCast(force) is not None:
            geometry = [force.getConnectee("sphere"), force.getConnectee("half_space")]
        else:
            catalog = self._model.getContactGeometrySet()
            names = _geometry_names(force)
            geometry = [catalog.get(n) for n in names if catalog.contains(n)]
        spheres = [g for g in map(opensim.ContactSphere.safeDownCast, geometry) if g]
        planes = [g for g in map(opensim.ContactHalfSpace.safeDownCast, geometry) if g]
        if len(spheres) != 1 or len(planes) != 1:
            return None
        return spheres[0], planes[0]

    @staticmethod
    def _record(force: Any, state: Any, prefix: str) -> np.ndarray | None:
        """Body force then torque (ground frame) from the force's report values."""
        labels = force.getRecordLabels()
        names = [labels.get(i) for i in range(labels.size())]
        key = f"{prefix}.force.X"
        if key not in names:
            return None
        start = names.index(key)
        values = force.getRecordValues(state)
        return np.array([values.get(start + i) for i in range(6)])

    def _contacts(self, state: Any) -> list[OverlayWrench]:
        wrenches = []
        forces = self._model.getForceSet()
        for i in range(forces.getSize()):
            force = forces.get(i)
            smooth = opensim.SmoothSphereHalfSpaceForce.safeDownCast(force) is not None
            if not smooth and opensim.HuntCrossleyForce.safeDownCast(force) is None:
                continue
            wrench = self._contact_wrench(force, smooth, state)
            if wrench is None:
                logger.info("Omitting contact force %s: unsupported", force.getName())
                continue
            wrenches.append(wrench)
        return wrenches

    def _contact_wrench(
        self, force: Any, smooth: bool, state: Any
    ) -> OverlayWrench | None:
        pair = self._contact_geometry(force)
        if pair is None:
            return None
        sphere, plane = pair
        base = sphere.getFrame().findBaseFrame()
        body = str(base.getName())
        record = self._record(
            force, state, f"{force.getName()}.{'Sphere' if smooth else body}"
        )
        if record is None:
            return None
        # Record torque is about the body origin, in ground (Simbody body force).
        origin = self._world(_vec3(base.getPositionInGround(state)))
        total = SpatialWrench(
            application_frame="world",
            point_m=_tuple3(origin),
            force_n=_tuple3(self._world(record[:3])),
            torque_nm=_tuple3(self._world(record[3:])),
        )
        bottom = self._world(self._sphere_bottom(sphere, plane, state))
        moved = transform_wrench(total, "world", _tuple3(bottom))
        return OverlayWrench(
            kind=WrenchKind.CONTACT,
            label=_label("contact", force.getName()),
            body=body,
            point_m=moved.point_m,
            force_n=moved.force_n,
            torque_nm=moved.torque_nm,
            source=_CONTACT_SOURCE,
        )

    @staticmethod
    def _sphere_bottom(sphere: Any, plane: Any, state: Any) -> np.ndarray:
        """Sphere centre minus radius along the plane normal, in ground."""
        centre = np.asarray(
            _vec3(
                sphere.getFrame().findStationLocationInGround(
                    state, sphere.get_location()
                )
            )
        )
        frame_rotation = _mat33(plane.getFrame().getRotationInGround(state))
        transform = plane.getTransform()  # keep alive: R() views into it
        local_rotation = _mat33(transform.R())
        normal = frame_rotation @ local_rotation @ _HALF_SPACE_OUTWARD_LOCAL
        return centre - float(sphere.getRadius()) * normal

    def _segment_distal_point(
        self, body: str, children: dict[str, list[Any]], state: Any
    ) -> np.ndarray | None:
        """Distal end of ``body`` in ground, or None when ambiguous.

        Exactly one child joint: that joint's origin on this body. No child
        joints: the body mass centre. Several (branching): ambiguous, None.
        """
        joints = children.get(body, [])
        if len(joints) == 1:
            return np.asarray(
                _vec3(joints[0].getParentFrame().getPositionInGround(state))
            )
        if joints:
            return None
        owner = opensim.Body.safeDownCast(self._model.getBodySet().get(body))
        return np.asarray(
            _vec3(owner.findStationLocationInGround(state, owner.getMassCenter()))
        )

    def _axial_loads(self, state: Any) -> dict[str, float | None]:
        joints = self._joints()
        children: dict[str, list[Any]] = {}
        for joint in joints:
            children.setdefault(_base_name(joint.getParentFrame()), []).append(joint)
        loads: dict[str, float | None] = {}
        for joint in joints:
            body = _base_name(joint.getChildFrame())
            proximal = np.asarray(
                _vec3(joint.getChildFrame().getPositionInGround(state))
            )
            distal = self._segment_distal_point(body, children, state)
            if (
                distal is None
                or np.linalg.norm(distal - proximal) <= _MIN_SEGMENT_LENGTH_M
            ):
                loads[body] = None  # ambiguous or degenerate axis: unavailable
                continue
            _, force, _ = self._reaction_on_child(joint, state)
            loads[body] = axial_force_from_proximal_reaction(
                force, self._world(proximal), self._world(distal)
            )
        return loads
