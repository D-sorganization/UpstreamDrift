"""Exact-source profiles for OpenSim-native joint and path-function evaluation.

Profiles identify only native mechanics that the constrained-muscle provider's
legacy analytic subset cannot inspect. They do not change source mechanics or
qualify physical behavior; the SDK remains responsible for evaluating them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

_SCHEMA = "native-sdk-mechanics-v1"
_PROFILE_CLASSES = {"CustomJoint", "MovingPathPoint"}


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash_component(component: Any) -> str:
    return _sha(component.dump().encode("utf-8"))


@dataclass(frozen=True)
class NativeMechanicsComponentIdentity:
    """Serialized identity of one SDK-delegated joint or moving path point."""

    absolute_path: str
    concrete_class: str
    serialized_sha256: str

    def __post_init__(self) -> None:
        if not self.absolute_path.startswith("/"):
            raise ValueError("native mechanics component path must be absolute")
        if self.concrete_class not in _PROFILE_CLASSES:
            raise ValueError("native mechanics profile component class is unsupported")
        if len(self.serialized_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in self.serialized_sha256
        ):
            raise ValueError("native mechanics component needs a SHA-256")


@dataclass(frozen=True)
class NativeMechanicsExecutionProfile:
    """Source-closed list of components whose laws are evaluated by OpenSim."""

    schema_version: str
    source_sha256: str
    components: tuple[NativeMechanicsComponentIdentity, ...]

    def __post_init__(self) -> None:
        if self.schema_version != _SCHEMA:
            raise ValueError("unsupported native mechanics profile schema")
        if len(self.source_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in self.source_sha256
        ):
            raise ValueError("native mechanics profile needs an exact source SHA-256")
        if not isinstance(self.components, tuple):
            raise TypeError("native mechanics components must be an immutable tuple")
        paths = tuple(item.absolute_path for item in self.components)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ValueError("native mechanics components must be unique and sorted")

    @property
    def profile_sha256(self) -> str:
        payload = {
            "schema_version": self.schema_version,
            "source_sha256": self.source_sha256,
            "components": [asdict(item) for item in self.components],
        }
        return _sha(json.dumps(payload, sort_keys=True, allow_nan=False).encode())


def _simple_custom_joint(component: Any, osim: Any) -> bool:
    custom = osim.CustomJoint.safeDownCast(component)
    if custom is None or custom.numCoordinates() != 1:
        return False
    coordinate = custom.getCoordinate(0).getName()
    transform = custom.getSpatialTransform()
    for axis_index in range(6):
        axis = transform.getTransformAxis(axis_index)
        names = axis.getCoordinateNames()
        function = axis.getFunction()
        if axis_index == 0:
            linear = osim.LinearFunction.safeDownCast(function)
            if (
                names.size() != 1
                or names.getValue(0) != coordinate
                or linear is None
                or linear.getCoefficients().get(0) != 1.0
                or linear.getCoefficients().get(1) != 0.0
            ):
                return False
        else:
            constant = osim.Constant.safeDownCast(function)
            if names.size() != 0 or constant is None or constant.getValue() != 0.0:
                return False
    return True


def _native_profile_required(component: Any, osim: Any) -> bool:
    concrete = str(component.getConcreteClassName())
    if concrete == "CustomJoint":
        return not _simple_custom_joint(component, osim)
    if concrete == "MovingPathPoint":
        point = osim.MovingPathPoint.safeDownCast(component)
        if point is None:
            raise ValueError("MovingPathPoint has no native concrete binding")
        for function in (
            point.get_x_location(),
            point.get_y_location(),
            point.get_z_location(),
        ):
            if (
                osim.LinearFunction.safeDownCast(function) is None
                and osim.Constant.safeDownCast(function) is None
            ):
                return True
    return False


def _identities(model: Any, osim: Any) -> tuple[NativeMechanicsComponentIdentity, ...]:
    result = []
    for component in tuple(model.getComponentsList()):
        concrete = str(component.getConcreteClassName())
        if concrete not in _PROFILE_CLASSES or not _native_profile_required(
            component, osim
        ):
            continue
        result.append(
            NativeMechanicsComponentIdentity(
                str(component.getAbsolutePathString()),
                concrete,
                _hash_component(component),
            )
        )
    return tuple(sorted(result, key=lambda item: item.absolute_path))


def declare_native_mechanics_execution_profile(
    model_path: Path, source_sha256: str
) -> NativeMechanicsExecutionProfile:
    """Capture only exact native joint/path mechanics outside the narrow subset."""
    import opensim as osim

    path = Path(model_path)
    raw = path.read_bytes()
    if _sha(raw) != source_sha256:
        raise ValueError("native mechanics profile source SHA-256 mismatch")
    model = osim.Model(str(path))
    model.finalizeConnections()
    components = _identities(model, osim)
    if path.read_bytes() != raw:
        raise ValueError("native mechanics source changed during declaration")
    return NativeMechanicsExecutionProfile(_SCHEMA, source_sha256, components)


def validate_native_mechanics_execution_profile(
    model: Any,
    profile: NativeMechanicsExecutionProfile,
    expected_source_sha256: str,
) -> None:
    """Require complete exact path/class/dump closure before SDK delegation."""
    import opensim as osim

    if not isinstance(profile, NativeMechanicsExecutionProfile):
        raise TypeError("native mechanics execution profile must be typed")
    if profile.source_sha256 != expected_source_sha256:
        raise ValueError("native mechanics profile source identity differs")
    actual = _identities(model, osim)
    if actual != profile.components:
        raise ValueError("native mechanics profile component closure differs")


__all__ = [
    "NativeMechanicsComponentIdentity",
    "NativeMechanicsExecutionProfile",
    "declare_native_mechanics_execution_profile",
    "validate_native_mechanics_execution_profile",
]
