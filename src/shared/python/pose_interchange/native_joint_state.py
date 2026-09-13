"""Lossless native joint-state interchange, independent of engine array layouts.

All three native engine providers consume named SI primitive coordinates. This
adapter converts that shared boundary to joint-local quaternions and physical
angular vectors. It does not replace joints or qualify spherical-joint dynamics.
Only rotational triples within ONE native joint are grouped; bodies, offsets,
closure and actuation remain part of the hashed specification.
"""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from .joint_chart import SerialRotationChart
from .se3 import is_valid_se3


@dataclass(frozen=True)
class NativeRotationGroup:
    """Exact primitive sequence and fixed frames; angles in radians.

    Parent angular vectors mean the native joint BASE frame, before its serial
    rotations, not world or the parent body's solid reference frame.
    """

    name: str
    coordinates: tuple[str, ...]
    axes: str
    parent_body: str
    child_body: str
    parent_to_base: tuple[tuple[float, ...], ...]
    child_to_follower: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class RotationState:
    """Intrinsic follower-to-base rotation and base-expressed angular vectors.

    The wxyz quaternion excludes the fixed child-to-follower transform and any
    preceding bushing translations; it is not the child's complete body pose.
    """

    quaternion_wxyz: tuple[float, ...]
    omega_parent_rad_s: tuple[float, ...]
    alpha_parent_rad_s2: tuple[float, ...]
    moment_parent_nm: tuple[float, ...]


@dataclass(frozen=True)
class NativeManifoldState:
    """Native rotational groups plus untouched SI scalar primitives.

    Scalar tuple entries are position, rate, acceleration, conjugate effort.
    Prismatic units are m, m/s, m/s², N; rotational units rad, rad/s, rad/s², Nm.
    Identity uses a canonical JSON fingerprint, not the raw model-file hash.
    """

    specification_sha256: str
    rotations: Mapping[str, RotationState]
    scalars: Mapping[str, tuple[float, float, float, float]]
    convention_tag: str = "native-joint-manifold-v1"


class NativeJointStateAdapter:
    """Convert named native engine states without assuming q/v array ordering."""

    def __init__(self, specification: Mapping[str, Any]) -> None:
        if specification.get("schema_version") != 1:
            raise ValueError("Unsupported native specification version")
        self.specification_sha256 = hashlib.sha256(
            json.dumps(
                specification, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest()
        self.coordinate_order = tuple(specification["coordinate_order"])
        inventory: dict[str, str] = {}
        groups = []
        names = set()
        for joint in specification["joints"]:
            if joint["name"] in names:
                raise ValueError("Duplicate native joint name")
            names.add(joint["name"])
            for key in ("parent_to_base", "child_to_follower"):
                if not is_valid_se3(joint[key]):
                    raise ValueError("Invalid native fixed frame")
            primitives = joint["primitives"]
            for primitive in primitives:
                coordinate, kind = primitive["coordinate"], primitive["primitive"]
                if coordinate in inventory or kind not in (
                    "Px",
                    "Py",
                    "Pz",
                    "Rx",
                    "Ry",
                    "Rz",
                ):
                    raise ValueError(
                        "Duplicate or unsupported native primitive inventory"
                    )
                inventory[coordinate] = kind
            # Accept only a pure rotational triple or a bushing's translation
            # prefix followed by that triple. Never cross a native body edge.
            kinds = [p["primitive"] for p in primitives]
            prefix = kinds[:-3]
            suffix = kinds[-3:]
            if (
                len(suffix) == 3
                and set(suffix) == {"Rx", "Ry", "Rz"}
                and all(k.startswith("P") for k in prefix)
            ):
                groups.append(
                    NativeRotationGroup(
                        joint["name"],
                        tuple(p["coordinate"] for p in primitives[-3:]),
                        "".join(k[1].upper() for k in suffix),
                        joint["parent"],
                        joint["child"],
                        tuple(tuple(row) for row in joint["parent_to_base"]),
                        tuple(tuple(row) for row in joint["child_to_follower"]),
                    )
                )
        if len(self.coordinate_order) != len(set(self.coordinate_order)) or set(
            inventory
        ) != set(self.coordinate_order):
            raise ValueError("Native coordinate inventory mismatch")
        self.groups = tuple(groups)
        self.primitive_types = inventory
        grouped = {name for group in groups for name in group.coordinates}
        self.scalar_coordinates = tuple(
            n for n in self.coordinate_order if n not in grouped
        )

    def _validate(self, values: Mapping[str, float]) -> None:
        if set(values) != set(self.coordinate_order):
            raise ValueError("Native coordinate inventory mismatch")
        if not np.isfinite(list(values.values())).all():
            raise ValueError("Native state must be finite")

    def export(
        self,
        coordinates: Mapping[str, float],
        rates: Mapping[str, float],
        accelerations: Mapping[str, float],
        efforts: Mapping[str, float],
    ) -> NativeManifoldState:
        """Export engine-neutral named native state; no dynamics evaluation.

        Efforts must already be conjugate to native primitives. Upstream
        actuator or force-frame values must pass through the actuation map first.
        """
        fields = (coordinates, rates, accelerations, efforts)
        for field in fields:
            self._validate(field)
        rotations = {}
        for group in self.groups:
            chart = SerialRotationChart(group.axes)
            q, v, a, tau = ([field[n] for n in group.coordinates] for field in fields)
            rotations[group.name] = RotationState(
                tuple(chart.quaternion(q)),
                tuple(chart.angular_velocity(q, v)),
                tuple(chart.angular_acceleration(q, v, a)),
                tuple(chart.parent_moment(q, tau)),
            )
        scalars = {
            n: (coordinates[n], rates[n], accelerations[n], efforts[n])
            for n in self.scalar_coordinates
        }
        return NativeManifoldState(self.specification_sha256, rotations, scalars)

    def restore(
        self,
        state: NativeManifoldState,
        reference_coordinates: Mapping[str, float],
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, float]]:
        """Recover native coordinates near a supplied branch reference.

        Singular inverse charts fail explicitly; a quaternion cannot create a
        unique native Euler rate at gimbal lock.
        """
        self._validate(reference_coordinates)
        if state.convention_tag != "native-joint-manifold-v1":
            raise ValueError("Unsupported native manifold convention")
        if state.specification_sha256 != self.specification_sha256:
            raise ValueError("Native specification identity mismatch")
        if set(state.rotations) != {g.name for g in self.groups} or set(
            state.scalars
        ) != set(self.scalar_coordinates):
            raise ValueError("Native manifold state inventory mismatch")
        fields: tuple[
            dict[str, float], dict[str, float], dict[str, float], dict[str, float]
        ] = ({}, {}, {}, {})
        for name, values in state.scalars.items():
            if len(values) != 4 or not np.isfinite(values).all():
                raise ValueError("Expected four finite native scalar values")
            for field, value in zip(fields, values, strict=True):
                field[name] = float(value)
        for group in self.groups:
            chart = SerialRotationChart(group.axes)
            rotation_state = state.rotations[group.name]
            q = chart.coordinates(
                rotation_state.quaternion_wxyz,
                [reference_coordinates[n] for n in group.coordinates],
            )
            v = chart.coordinate_rate(q, rotation_state.omega_parent_rad_s)
            a = chart.coordinate_acceleration(q, v, rotation_state.alpha_parent_rad_s2)
            tau = chart.coordinate_effort(q, rotation_state.moment_parent_nm)
            for field, vector in zip(fields, (q, v, a, tau), strict=True):
                field.update(zip(group.coordinates, map(float, vector), strict=True))
        q_result, v_result, a_result, tau_result = (
            {n: field[n] for n in self.coordinate_order} for field in fields
        )
        return q_result, v_result, a_result, tau_result
