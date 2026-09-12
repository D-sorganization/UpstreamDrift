"""Native cylinder/sphere properties in SI, without fallback physical defaults.

Reference: https://www.mathworks.com/help/sm/ref/cylindricalsolid.html
Custom frames: https://www.mathworks.com/help/sm/ug/create-solid-frames.html
"""

from collections.abc import Mapping
from dataclasses import dataclass
from math import pi
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np
from numpy.typing import NDArray

from src.shared.python.model_generation.inertia import cylinder_inertia, sphere_inertia

_UNIT_FACTORS = {
    "length": {"m": 1.0, "in": 0.0254, "mm": 0.001, "cm": 0.01},
    "mass": {"kg": 1.0, "g": 0.001, "lbm": 0.45359237},
    "density": {"kg/m^3": 1.0},
    "angle": {"rad": 1.0, "deg": pi / 180.0},
}


class NativeParameters:
    """Separate literal enums from resolved physical scalars."""

    def __init__(self, block: Mapping[str, Any]) -> None:
        records = block["parameters"]
        if isinstance(records, dict):
            records = [records]
        self._values = {item["name"]: item for item in records}
        if len(self._values) != len(records):
            raise ValueError("Duplicate native parameter")

    def text(self, name: str) -> str:
        value = self._values[name]["expression"]
        if not isinstance(value, str):
            raise ValueError(f"Expected literal text for {name}")
        return value

    def scalar(self, name: str, quantity: str) -> float:
        parameter = self._values[name]
        value = parameter.get("numeric_value")
        if (
            not parameter.get("resolved_numeric")
            or isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not np.isfinite(value)
        ):
            raise ValueError(f"Unresolved or invalid native scalar {name}")
        unit = self.text(name + "Units")
        if unit not in _UNIT_FACTORS[quantity]:
            raise ValueError(f"Unsupported {quantity} unit {unit} for {name}")
        return float(value) * _UNIT_FACTORS[quantity][unit]

    def vector(self, name: str, quantity: str, size: int) -> NDArray[np.float64]:
        parameter = self._values[name]
        if not parameter.get("resolved_numeric"):
            raise ValueError(f"Unresolved native vector {name}")
        value = np.asarray(parameter["numeric_value"], dtype=float)
        if value.shape != (size,) or not np.all(np.isfinite(value)):
            raise ValueError(f"Invalid native vector {name}")
        unit = self.text(name + "Units")
        if unit not in _UNIT_FACTORS[quantity]:
            raise ValueError(f"Unsupported {quantity} unit {unit} for {name}")
        return value * _UNIT_FACTORS[quantity][unit]


@dataclass(frozen=True)
class SolidProperties:
    """Inertia is about the COM in solid-reference axes; frames map to reference."""

    mass_kg: float
    com_m: NDArray[np.float64]
    inertia_com_kg_m2: NDArray[np.float64]
    frames: Mapping[str, NDArray[np.float64]]


def _axis(text: str | None) -> tuple[int, float]:
    if text not in ("+X", "-X", "+Y", "-Y", "+Z", "-Z"):
        raise ValueError(f"Unsupported frame axis {text}")
    return "XYZ".index(text[1]), 1.0 if text[0] == "+" else -1.0


def _frame_rotation(value: ET.Element) -> NDArray[np.float64]:
    rotation = np.zeros((3, 3))
    assigned = set()
    for tag in ("PrimaryAxis", "SecondaryAxis"):
        axis = value.find(tag)
        if axis is None or axis.findtext("Source") != "ReferenceFrame":
            raise ValueError(
                "Only explicit reference-axis frame alignment is supported"
            )
        column, sign = _axis(axis.findtext("DefinedDirection"))
        source, direction = _axis(axis.findtext("SourceDirection"))
        if column in assigned:
            raise ValueError("Custom frame axes must be independent")
        rotation[source, column] = sign * direction
        assigned.add(column)
    missing = (set(range(3)) - assigned).pop()
    rotation[:, missing] = np.cross(
        rotation[:, (missing + 1) % 3], rotation[:, (missing + 2) % 3]
    )
    if not np.allclose(rotation.T @ rotation, np.eye(3)) or not np.isclose(
        np.linalg.det(rotation), 1
    ):
        raise ValueError("Degenerate or improper native frame orientation")
    return rotation


def _solid_frames(
    serialized: str, cylinder_length: float | None
) -> dict[str, NDArray[np.float64]]:
    frames = {"R": np.eye(4)}
    if not serialized:
        return frames
    for frame in ET.fromstring(serialized).findall("Frame"):
        identifier = frame.findtext("Id")
        value = frame.find("Value")
        if not identifier or identifier in frames or value is None:
            raise ValueError("Invalid or duplicate native frame identifier")
        origin = value.find("Origin")
        if (
            origin is None
            or origin.findtext("Source") != "GeometricFeature"
            or cylinder_length is None
        ):
            raise ValueError("Unsupported native solid frame origin")
        feature = origin.findtext("FeatureName")
        signs = {
            "top curve": 1,
            "top surface": 1,
            "bottom curve": -1,
            "bottom surface": -1,
        }
        if feature not in signs:
            raise ValueError(f"Unsupported cylindrical feature {feature}")
        transform = np.eye(4)
        transform[:3, :3] = _frame_rotation(value)
        transform[2, 3] = signs[feature] * cylinder_length / 2
        frames[identifier] = transform
    return frames


def solid_properties(block: Mapping[str, Any]) -> SolidProperties:
    """Convert the native golf model's calculated cylinder and sphere inertias."""
    p = NativeParameters(block)
    if p.text("InertiaType") != "CalculateFromGeometry":
        raise ValueError("Unsupported native inertia mode")
    reference = block["library_reference"]
    length = None
    if reference == "sm_lib/Body Elements/Cylindrical Solid":
        radius = p.scalar("CylinderRadius", "length")
        length = p.scalar("CylinderLength", "length")
        volume = pi * radius**2 * length
    elif reference == "sm_lib/Body Elements/Spherical Solid":
        radius = p.scalar("SphereRadius", "length")
        volume = 4 * pi * radius**3 / 3
    else:
        raise ValueError(f"Unsupported native solid {reference}")
    if radius <= 0 or (length is not None and length <= 0):
        raise ValueError("Solid dimensions must be positive")
    basis = p.text("BasedOnType")
    if basis == "Mass":
        mass = p.scalar("Mass", "mass")
    elif basis == "Density":
        mass = p.scalar("Density", "density") * volume
    else:
        raise ValueError(f"Unsupported native mass basis {basis}")
    if mass < 0:
        raise ValueError("Solid mass must be nonnegative")
    moments = (
        sphere_inertia(mass, radius)
        if length is None
        else cylinder_inertia(mass, radius, length)
    )
    inertia = np.diag([moments[axis] for axis in ("ixx", "iyy", "izz")])
    return SolidProperties(
        mass, np.zeros(3), inertia, _solid_frames(p.text("SerializedFrames"), length)
    )
