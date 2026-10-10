"""Explicit native scalar-muscle and mechanical-assistance admission.

This separate profile does not widen muscle-only replay or qualify a source
model for matching. Controls are dimensionless; native mechanical actuation
is control times the source CoordinateActuator optimal force.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
from typing import Any

from .moco_initial_bindings import _finite_number
from .native_muscle_bundle import _COMPONENTS, _registered_options


class ActuationRole(str, Enum):
    """Declared physical assistance roles, separate from actuator names."""

    MUSCLE = "muscle"
    ROOT_RESIDUAL = "root-residual"
    UPPER_ASSISTANCE = "upper-assistance"
    LEG_RESERVE = "leg-reserve"


@dataclass(frozen=True)
class MixedChannel:
    """One absolute native scalar input and finite experiment bounds."""

    path: str
    role: ActuationRole
    control_bounds: tuple[float, float]

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path.startswith("/"):
            raise ValueError("mixed channel requires an absolute native path")
        if not isinstance(self.role, ActuationRole):
            raise TypeError("mixed channel requires an explicit actuation role")
        if len(self.control_bounds) != 2:
            raise ValueError("mixed control bounds require two endpoints")
        lower, upper = map(_finite_number, self.control_bounds)
        if lower > upper:
            raise ValueError("mixed control bounds require a nonempty interval")
        object.__setattr__(self, "control_bounds", (lower, upper))


@dataclass(frozen=True)
class MixedActuationProfile:
    """Ordered native coverage; declarations do not construct native actuators."""

    channels: tuple[MixedChannel, ...]

    def __post_init__(self) -> None:
        channels = tuple(self.channels)
        if not channels or not all(isinstance(c, MixedChannel) for c in channels):
            raise TypeError("mixed profile requires typed channel declarations")
        if len({c.path for c in channels}) != len(channels):
            raise ValueError("mixed profile channel paths must be unique")
        if not any(c.role == ActuationRole.MUSCLE for c in channels) or all(
            c.role == ActuationRole.MUSCLE for c in channels
        ):
            raise ValueError("mixed profile requires muscles and mechanical assistance")
        object.__setattr__(self, "channels", channels)


@dataclass(frozen=True)
class NativeMixedChannel:
    """Native-readback identity; muscle force is not linear in excitation."""

    path: str
    role: ActuationRole
    control_bounds: tuple[float, float]
    concrete_class: str
    coordinate_path: str | None
    output_unit: str
    optimal_force: float
    native_control_bounds: tuple[float, float]


@dataclass(frozen=True)
class NativeMixedProfile:
    """Admission digest over native model, channel semantics and state options."""

    channels: tuple[NativeMixedChannel, ...]
    sha256: str


def _native_channel(
    actuator: Any, declaration: MixedChannel, state: Any, osim: Any
) -> NativeMixedChannel:
    if actuator.numControls() != 1:
        raise ValueError("mixed native policy requires scalar actuation")
    actuator = osim.ScalarActuator.safeDownCast(actuator)
    if actuator is None:
        raise ValueError("mixed native policy requires ScalarActuator semantics")
    if actuator.isActuationOverridden(state) or not actuator.appliesForce(state):
        raise ValueError("mixed native policy forbids overridden or disabled force")
    law = actuator.getConcreteClassName()
    muscle = osim.Muscle.safeDownCast(actuator)
    if muscle is not None:
        if declaration.role != ActuationRole.MUSCLE:
            raise ValueError("native muscle channel requires muscle role")
        if muscle.getIgnoreActivationDynamics(
            state
        ) or muscle.getIgnoreTendonCompliance(state):
            raise ValueError(
                "ignored muscle dynamics need a separately reviewed profile"
            )
        coordinate_path, unit, gain = (
            None,
            "N",
            _finite_number(muscle.getMaxIsometricForce()),
        )
        admissible = (0.0, 1.0)
    else:
        mechanical = osim.CoordinateActuator.safeDownCast(actuator)
        if mechanical is None or law != "CoordinateActuator":
            raise ValueError("mechanical law needs a separately reviewed profile")
        if declaration.role == ActuationRole.MUSCLE:
            raise ValueError("mechanical assistance cannot claim a muscle role")
        coordinate = mechanical.getCoordinate()
        coordinate_path = coordinate.getAbsolutePathString()
        root = (
            osim.Ground.safeDownCast(
                coordinate.getJoint().getParentFrame().findBaseFrame()
            )
            is not None
        )
        if root != (declaration.role == ActuationRole.ROOT_RESIDUAL):
            raise ValueError("root assistance role must follow the native coordinate")
        motion = coordinate.getMotionType()
        if motion == osim.Coordinate.Rotational:
            unit = "N*m"
        elif motion == osim.Coordinate.Translational:
            unit = "N"
        else:
            raise ValueError(
                "mechanical output units require reviewed coordinate motion"
            )
        gain = _finite_number(mechanical.getOptimalForce())
        admissible = (
            float(mechanical.getMinControl()),
            float(mechanical.getMaxControl()),
        )
    if not gain > 0:
        raise ValueError("native actuation force scale must be positive")
    lower, upper = declaration.control_bounds
    native_lower, native_upper = (
        float(actuator.getMinControl()),
        float(actuator.getMaxControl()),
    )
    if (
        not native_lower <= lower <= upper <= native_upper
        or not admissible[0] <= lower <= upper <= admissible[1]
    ):
        raise ValueError("experiment controls exceed native actuation bounds")
    return NativeMixedChannel(
        declaration.path,
        declaration.role,
        declaration.control_bounds,
        law,
        coordinate_path,
        unit,
        gain,
        (native_lower, native_upper),
    )


def admit_native_mixed_profile(
    model: Any, state: Any, profile: MixedActuationProfile
) -> NativeMixedProfile:
    """Read back actual native law/units/options before admitting any input."""
    import opensim as osim

    if not isinstance(profile, MixedActuationProfile):
        raise TypeError("mixed admission requires a typed profile")
    allowed = _COMPONENTS | {"CoordinateActuator"}
    recursive_actuators = set()
    for component in tuple(model.getComponentsList()):
        if component.getConcreteClassName() not in allowed:
            raise ValueError(
                "component needs a separately reviewed mixed native policy"
            )
        if osim.Actuator.safeDownCast(component) is not None:
            recursive_actuators.add(component.getAbsolutePathString())
    coordinates = model.getCoordinateSet()
    for i in range(coordinates.getSize()):
        coordinate = coordinates.get(i)
        if (
            coordinate.getDefaultLocked()
            or coordinate.getDefaultIsPrescribed()
            or coordinate.getDefaultClamped()
            or coordinate.getLocked(state)
            or coordinate.getClamped(state)
            or coordinate.isPrescribed(state)
        ):
            raise ValueError(
                "coordinate restriction requires a separate initialization policy"
            )
    actuators = model.getActuators()
    paths = tuple(
        actuators.get(i).getAbsolutePathString() for i in range(actuators.getSize())
    )
    if set(paths) != recursive_actuators or paths != tuple(
        c.path for c in profile.channels
    ):
        raise ValueError(
            "mixed profile must cover every registered native actuator in order"
        )
    channels = tuple(
        _native_channel(actuators.get(i), declaration, state, osim)
        for i, declaration in enumerate(profile.channels)
    )
    options = _registered_options(model, state, model.getMuscles())
    channel_payload = [asdict(channel) for channel in channels]
    for channel in channel_payload:
        channel["native_control_bounds"] = tuple(
            str(v) for v in channel["native_control_bounds"]
        )
    payload = {
        "policy": "opensim-exact-muscle-coordinate-assistance/1.0.0",
        "loaded_model_sha256": hashlib.sha256(model.dump().encode()).hexdigest(),
        "channels": channel_payload,
        "registered_options": options,
    }
    # Infinite native bounds are allowed only alongside finite experiment bounds.
    # Encode their representations explicitly, never emit nonstandard JSON NaN.
    digest = hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    return NativeMixedProfile(channels, digest)
