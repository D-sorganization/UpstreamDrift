"""Name-safe degree-six polynomial actuation for full-body motion matching."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from .piecewise_polynomial import PolynomialSegment
from .polynomial_torque import COEFFS_PER_JOINT
from .prefix_fit import bernstein_to_simscape

Array = NDArray[np.float64]

ROOT_COORDINATES = (
    "TranslationInputX",
    "TranslationInputY",
    "TranslationInputZ",
    "HipInputX",
    "HipInputY",
    "HipInputZ",
)
_ROOT_PRIMITIVES = dict(
    zip(ROOT_COORDINATES, ("Px", "Py", "Pz", "Rx", "Ry", "Rz"), strict=True)
)


def _names(values: Sequence[str], label: str) -> tuple[str, ...]:
    result = tuple(values)
    if not result or any(not isinstance(name, str) or not name for name in result):
        raise ValueError(f"{label} must contain nonempty names")
    if len(set(result)) != len(result):
        raise ValueError(f"{label} contains duplicate names")
    return result


def _readonly(values: Any) -> Array:
    result = np.array(values, dtype=float, copy=True)
    if not np.isfinite(result).all():
        raise ValueError("polynomial output must be finite")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class FullBodyPolynomialControl:
    """One global Bernstein effort curve with no world-root actuators."""

    coordinate_names: tuple[str, ...]
    actuated_names: tuple[str, ...]
    duration_s: float

    coefficients_per_actuator: ClassVar[int] = COEFFS_PER_JOINT

    def __post_init__(self) -> None:
        coordinates = _names(self.coordinate_names, "coordinate_names")
        actuated = _names(self.actuated_names, "actuated_names")
        if set(ROOT_COORDINATES) - set(coordinates):
            raise ValueError("coordinate_names must contain the six world-root names")
        unknown = set(actuated) - set(coordinates)
        if unknown:
            raise ValueError(
                f"actuated_names contains unknown names: {sorted(unknown)}"
            )
        if set(actuated).intersection(ROOT_COORDINATES):
            raise ValueError("world-root coordinates cannot be actuators")
        non_root = tuple(name for name in coordinates if name not in ROOT_COORDINATES)
        if set(actuated) != set(non_root) or len(actuated) != len(non_root):
            raise ValueError("actuated_names must contain all non-root coordinates")
        if (
            isinstance(self.duration_s, bool)
            or not isinstance(self.duration_s, Real)
            or not np.isfinite(self.duration_s)
            or self.duration_s <= 0.0
        ):
            raise ValueError("duration_s must be finite and positive")
        object.__setattr__(self, "coordinate_names", coordinates)
        object.__setattr__(self, "actuated_names", actuated)
        object.__setattr__(self, "duration_s", float(self.duration_s))

    @classmethod
    def from_spec(
        cls, spec: Mapping[str, Any], *, duration_s: float
    ) -> FullBodyPolynomialControl:
        """Derive actuator names from a full-body-v1 scalar primitive inventory."""
        if spec.get("schema_version") != "full-body-v1":
            raise ValueError("spec must use the full-body-v1 schema")
        coordinates = _names(spec.get("coordinate_order", ()), "coordinate_order")
        joints = spec.get("joints")
        if (
            not isinstance(joints, Sequence)
            or isinstance(joints, (str, bytes))
            or any(not isinstance(joint, Mapping) for joint in joints)
        ):
            raise ValueError("spec must contain joints")
        roots = [joint for joint in joints if joint.get("parent") == "world"]
        if len(roots) != 1:
            raise ValueError("spec must contain exactly one world root joint")
        primitives = roots[0].get("primitives", ())
        root_map = {
            primitive.get("coordinate"): primitive.get("primitive")
            for primitive in primitives
            if isinstance(primitive, Mapping)
        }
        if len(primitives) != 6 or root_map != _ROOT_PRIMITIVES:
            raise ValueError(
                "world root must preserve the full-body-v1 six-coordinate inventory"
            )
        if any(
            primitive.get("primitive") not in {"Px", "Py", "Pz", "Rx", "Ry", "Rz"}
            for joint in joints
            for primitive in joint.get("primitives", ())
            if isinstance(primitive, Mapping)
        ):
            raise ValueError("full-body control requires scalar joint primitives")
        declared = [
            primitive.get("coordinate")
            for joint in joints
            for primitive in joint.get("primitives", ())
            if isinstance(primitive, Mapping)
        ]
        if len(declared) != len(set(declared)):
            raise ValueError("joint primitives contain duplicate coordinate names")
        if set(declared) != set(coordinates):
            raise ValueError("coordinate order and joint primitives are inconsistent")
        actuated = tuple(name for name in coordinates if name not in ROOT_COORDINATES)
        return cls(coordinates, actuated, duration_s)

    @property
    def n_coordinates(self) -> int:
        return len(self.coordinate_names)

    @property
    def n_parameters(self) -> int:
        return len(self.actuated_names) * self.coefficients_per_actuator

    def _parameters(self, parameters: Any) -> Array:
        values = np.asarray(parameters, dtype=float)
        if values.shape != (self.n_parameters,):
            raise ValueError(f"parameters must have shape ({self.n_parameters},)")
        if not np.isfinite(values).all():
            raise ValueError("parameters must be finite")
        return values

    def _time(self, time_s: float) -> float:
        if isinstance(time_s, bool) or not isinstance(time_s, Real):
            raise ValueError("time_s must be finite and within the control horizon")
        value = float(time_s)
        tolerance = 8.0 * np.finfo(float).eps * max(1.0, self.duration_s)
        if not np.isfinite(value) or value < 0.0 or value > self.duration_s + tolerance:
            raise ValueError("time_s must be finite and within the control horizon")
        return min(value, self.duration_s)

    def _basis(self, time_s: float) -> Array:
        time = self._time(time_s)
        identity = np.eye(self.coefficients_per_actuator)
        return PolynomialSegment(0.0, self.duration_s, identity, True).evaluate(time)

    def efforts(self, parameters: Any, time_s: float) -> dict[str, float]:
        """Evaluate primitive efforts in canonical coordinate-name order."""
        coefficients = self._parameters(parameters).reshape(-1, COEFFS_PER_JOINT)
        values = PolynomialSegment(0.0, self.duration_s, coefficients, True).evaluate(
            self._time(time_s)
        )
        actuated = dict(zip(self.actuated_names, values, strict=True))
        return {name: float(actuated.get(name, 0.0)) for name in self.coordinate_names}

    def effort_jacobian(self, time_s: float) -> Array:
        """Return all-coordinate effort derivatives by flat row-major coefficients."""
        basis = self._basis(time_s)
        result = np.zeros((self.n_coordinates, self.n_parameters))
        rows = {name: index for index, name in enumerate(self.coordinate_names)}
        for actuator, name in enumerate(self.actuated_names):
            start = actuator * COEFFS_PER_JOINT
            result[rows[name], start : start + COEFFS_PER_JOINT] = basis
        return _readonly(result)

    def export_power_coefficients(self, parameters: Any) -> Array:
        """Export all-coordinate ascending power coefficients in physical seconds."""
        coefficients = self._parameters(parameters).reshape(-1, COEFFS_PER_JOINT)
        ascending = bernstein_to_simscape(coefficients, duration_s=self.duration_s)[
            :, ::-1
        ]
        result = np.zeros((self.n_coordinates, COEFFS_PER_JOINT))
        rows = {name: index for index, name in enumerate(self.coordinate_names)}
        for actuator, name in enumerate(self.actuated_names):
            result[rows[name]] = ascending[actuator]
        return _readonly(result)
