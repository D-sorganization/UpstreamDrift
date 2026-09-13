"""Immutable native motion batches; conversion only, without file I/O.

Efforts are primitive-conjugate SI values, not raw world-frame polynomial inputs.
Grouped quaternions exclude fixed attachments and preceding translations: they
are not complete world body poses. Every original q sample remains its own branch
reference; no previous-output substitution or pseudoinverse is permitted.
"""

from dataclasses import dataclass, replace
from types import MappingProxyType

import numpy as np
from numpy.typing import ArrayLike

from .native_joint_state import (
    NativeJointStateAdapter,
    NativeManifoldState,
    NativeRotationGroup,
    RotationState,
)
from .se3 import is_valid_se3

Row = tuple[float, ...]
Matrix = tuple[Row, ...]
NativeFields = tuple[Matrix, Matrix, Matrix, Matrix]


def _row(value: ArrayLike, size: int) -> Row:
    array = np.asarray(value, dtype=float)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"Expected finite vector of length {size}")
    return tuple(map(float, array))


def _matrix(value: ArrayLike, rows: int, columns: int) -> Matrix:
    array = np.asarray(value, dtype=float)
    if array.shape != (rows, columns) or not np.isfinite(array).all():
        raise ValueError(f"Expected finite sample matrix ({rows}, {columns})")
    return tuple(tuple(map(float, row)) for row in array)


def _times(value: ArrayLike) -> Row:
    times = np.asarray(value, dtype=float)
    if (
        times.ndim != 1
        or len(times) == 0
        or not np.isfinite(times).all()
        or np.any(np.diff(times) <= 0)
    ):
        raise ValueError("Expected nonempty finite strictly increasing times")
    return tuple(map(float, times))


def _group(group: NativeRotationGroup) -> NativeRotationGroup:
    frames = []
    for frame in (group.parent_to_base, group.child_to_follower):
        matrix = _matrix(frame, 4, 4)
        if not is_valid_se3(matrix):
            raise ValueError("Invalid group fixed transform")
        frames.append(matrix)
    return replace(
        group,
        coordinates=tuple(group.coordinates),
        parent_to_base=frames[0],
        child_to_follower=frames[1],
    )


def _state(state: NativeManifoldState) -> NativeManifoldState:
    rotations = {}
    for name, rotation in state.rotations.items():
        quaternion = _row(rotation.quaternion_wxyz, 4)
        if not np.isclose(np.linalg.norm(quaternion), 1, rtol=0, atol=1e-10):
            raise ValueError("Expected unit quaternion")
        rotations[name] = RotationState(
            quaternion,
            _row(rotation.omega_parent_rad_s, 3),
            _row(rotation.alpha_parent_rad_s2, 3),
            _row(rotation.moment_parent_nm, 3),
        )
    scalars = {}
    for name, values in state.scalars.items():
        a, b, c, d = _row(values, 4)
        scalars[name] = (a, b, c, d)
    return NativeManifoldState(
        state.specification_sha256,
        MappingProxyType(rotations),
        MappingProxyType(scalars),
        state.convention_tag,
    )


@dataclass(frozen=True)
class NativeMotionSequence:
    """Owned immutable manifold samples with native branch and model metadata.

    Times are seconds; grouped angular vectors use the joint base frame. Scalar
    Px/Py/Pz channels use m, m/s, m/s², N; Rx/Ry/Rz use rad, rad/s, rad/s², Nm.
    The fingerprint hashes canonical JSON, not the original model-file bytes.
    """

    specification_sha256: str
    coordinate_order: tuple[str, ...]
    primitive_types: tuple[tuple[str, str], ...]
    rotation_groups: tuple[NativeRotationGroup, ...]
    times_s: Row
    native_reference: Matrix
    states: tuple[NativeManifoldState, ...]
    convention_tag: str = "native-motion-sequence-v1"

    def __post_init__(self) -> None:
        if self.convention_tag != "native-motion-sequence-v1":
            raise ValueError("Unsupported motion sequence convention")
        digest = self.specification_sha256
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Expected canonical specification SHA256")
        order = tuple(self.coordinate_order)
        if (
            not order
            or len(set(order)) != len(order)
            or any(not isinstance(n, str) or not n for n in order)
        ):
            raise ValueError("Expected unique nonempty coordinate names")
        primitives = tuple(tuple(pair) for pair in self.primitive_types)
        if (
            any(len(pair) != 2 for pair in primitives)
            or tuple(p[0] for p in primitives) != order
            or any(p[1] not in ("Px", "Py", "Pz", "Rx", "Ry", "Rz") for p in primitives)
        ):
            raise ValueError("Primitive inventory/order mismatch")
        groups = tuple(_group(g) for g in self.rotation_groups)
        grouped = [n for g in groups for n in g.coordinates]
        if (
            len({g.name for g in groups}) != len(groups)
            or len(set(grouped)) != len(grouped)
            or not set(grouped) <= set(order)
        ):
            raise ValueError("Rotation group inventory mismatch")
        kinds = {pair[0]: pair[1] for pair in primitives}
        for group in groups:
            if (
                len(group.axes) != 3
                or set(group.axes) != set("XYZ")
                or len(group.coordinates) != 3
                or [kinds[n] for n in group.coordinates]
                != ["R" + a.lower() for a in group.axes]
            ):
                raise ValueError("Invalid three-axis rotation group")
        times = _times(self.times_s)
        reference = _matrix(self.native_reference, len(times), len(order))
        if len(self.states) != len(times):
            raise ValueError("State count does not match times")
        states = []
        for index, state in enumerate(self.states):
            try:
                if (
                    state.specification_sha256 != digest
                    or state.convention_tag != "native-joint-manifold-v1"
                ):
                    raise ValueError("State identity/convention mismatch")
                if set(state.rotations) != {g.name for g in groups} or set(
                    state.scalars
                ) != set(order) - set(grouped):
                    raise ValueError("State inventory mismatch")
                states.append(_state(state))
            except ValueError as error:
                raise ValueError(
                    f"Invalid sample {index} at {times[index]:g} s: {error}"
                ) from error
        for name, value in [
            ("coordinate_order", order),
            ("primitive_types", primitives),
            ("rotation_groups", groups),
            ("times_s", times),
            ("native_reference", reference),
            ("states", tuple(states)),
        ]:
            object.__setattr__(self, name, value)


def export_native_motion(
    adapter: NativeJointStateAdapter,
    times_s: ArrayLike,
    coordinates: ArrayLike,
    rates: ArrayLike,
    accelerations: ArrayLike,
    primitive_efforts: ArrayLike,
) -> NativeMotionSequence:
    """Convert (samples, coordinate_order) SI matrices through the shared adapter.

    Retain original q per sample, including branch and winding. A singular sample
    fails the whole conversion with index/time context; no partial result returns.
    """
    times = _times(times_s)
    order = adapter.coordinate_order
    fields = tuple(
        _matrix(value, len(times), len(order))
        for value in (coordinates, rates, accelerations, primitive_efforts)
    )
    states = []
    for index, time in enumerate(times):
        named = [dict(zip(order, field[index], strict=True)) for field in fields]
        try:
            states.append(adapter.export(*named))
        except ValueError as error:
            raise ValueError(
                f"Cannot export sample {index} at {time:g} s: {error}"
            ) from error
    return NativeMotionSequence(
        adapter.specification_sha256,
        order,
        tuple((n, adapter.primitive_types[n]) for n in order),
        adapter.groups,
        times,
        fields[0],
        tuple(states),
    )


def restore_native_motion(
    adapter: NativeJointStateAdapter, motion: NativeMotionSequence
) -> NativeFields:
    """Return immutable q/qd/qdd/effort matrices in motion.coordinate_order.

    Returned rows correspond exactly to motion.times_s; the envelope retains the
    clock, units and model metadata. Native actuator branch preservation is always
    enabled. This performs representation conversion, not dynamical replay.
    """
    if (
        motion.specification_sha256 != adapter.specification_sha256
        or motion.coordinate_order != adapter.coordinate_order
        or motion.rotation_groups != adapter.groups
        or motion.primitive_types
        != tuple((n, adapter.primitive_types[n]) for n in adapter.coordinate_order)
    ):
        raise ValueError("Native motion identity/inventory mismatch")
    fields: list[list[Row]] = [[], [], [], []]
    for index, (time, state, reference) in enumerate(
        zip(motion.times_s, motion.states, motion.native_reference, strict=True)
    ):
        try:
            restored = adapter.restore(
                state,
                dict(zip(motion.coordinate_order, reference, strict=True)),
                preserve_middle_branch=True,
            )
        except ValueError as error:
            raise ValueError(
                f"Cannot restore sample {index} at {time:g} s: {error}"
            ) from error
        for field, values in zip(fields, restored, strict=True):
            field.append(tuple(values[n] for n in motion.coordinate_order))
    return tuple(fields[0]), tuple(fields[1]), tuple(fields[2]), tuple(fields[3])
