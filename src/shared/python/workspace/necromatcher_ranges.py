"""Authored degree ranges bound to exact native geometry, never compiled limits."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real
import re
from types import MappingProxyType
from typing import Any, Literal

from defusedxml import ElementTree

_SOURCE: Literal["bound_native_definition.coordinate_ranges_deg"] = (
    "bound_native_definition.coordinate_ranges_deg"
)


def _pair(value: Any) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("Authored ranges require lower/upper pairs")
    if any(isinstance(item, bool) or not isinstance(item, Real) for item in value):
        raise ValueError("Authored range endpoints must be finite numbers")
    lower, upper = map(float, value)
    if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
        raise ValueError("Authored range endpoints must be finite and ordered")
    return lower, upper


def _radian_pair(value: Any) -> tuple[float, float]:
    lower, upper = _pair(value)
    return math.radians(lower), math.radians(upper)


@dataclass(frozen=True)
class AuthoredCoordinateBounds:
    """Detached immutable authored hypotheses in native scalar radian units."""

    named_bounds: Mapping[str, tuple[float, float]]
    unbounded_names: tuple[str, ...]
    definition_sha256: str
    xml_sha256: str
    range_source: Literal["bound_native_definition.coordinate_ranges_deg"] = _SOURCE
    compiled_limits_enforced: Literal[False] = False

    def __post_init__(self) -> None:
        if not isinstance(self.named_bounds, Mapping) or not isinstance(
            self.unbounded_names, tuple
        ):
            raise ValueError(
                "Authored ranges require named bounds and ordered unbounded names"
            )
        bounds = {name: _pair(value) for name, value in self.named_bounds.items()}
        names = (*bounds, *self.unbounded_names)
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("Authored range identities must be named coordinates")
        if len(set(names)) != len(names):
            raise ValueError(
                "Bounded and unbounded coordinate identities must be unique"
            )
        for identity in (self.definition_sha256, self.xml_sha256):
            if not isinstance(identity, str) or not re.fullmatch(
                r"sha256:[a-f0-9]{64}", identity
            ):
                raise ValueError(
                    "Authored ranges require definition and XML SHA256 identities"
                )
        if self.range_source != _SOURCE or self.compiled_limits_enforced is not False:
            raise ValueError("Authored ranges are not compiled limits")
        object.__setattr__(self, "named_bounds", MappingProxyType(bounds))
        object.__setattr__(self, "unbounded_names", tuple(self.unbounded_names))

    def to_record(self) -> dict[str, Any]:
        """Return fresh JSON-safe lists/dictionaries without mutable aliasing."""
        return {
            "named_bounds": {
                name: list(pair) for name, pair in self.named_bounds.items()
            },
            "unbounded_names": list(self.unbounded_names),
            "definition_sha256": self.definition_sha256,
            "xml_sha256": self.xml_sha256,
            "range_source": self.range_source,
            "compiled_limits_enforced": self.compiled_limits_enforced,
        }


def _xml_coordinates(xml: str, order: tuple[str, ...], units: tuple[str, ...]) -> None:
    joints = ElementTree.fromstring(xml).findall("worldbody//joint")
    inventory = {joint.get("name"): joint for joint in joints}
    if len(inventory) != len(joints) or set(inventory) != set(order):
        raise ValueError("Exported XML scalar coordinates differ from native order")
    for name, unit in zip(order, units, strict=True):
        joint = inventory[name]
        expected = {"hinge": "rad", "slide": "m"}.get(joint.get("type"))
        if expected is None or unit != expected:
            raise ValueError(
                "Exported XML joint types differ from compiled scalar units"
            )
        if joint.get("limited") != "false":
            raise ValueError("Authored range boundary requires XML limited=false")


def extract_authored_bounds(
    definition_bytes: bytes,
    model_hash: str,
    coordinate_order: tuple[str, ...],
    coordinate_units: tuple[str, ...],
) -> AuthoredCoordinateBounds:
    """Validate the captured definition against exact exported XML and native units."""
    from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
        export_full_body_mjcf,
    )

    if not isinstance(definition_bytes, bytes) or not definition_bytes:
        raise ValueError(
            "Authored ranges require captured bound native definition bytes"
        )
    definition = json.loads(definition_bytes)
    if (
        not isinstance(definition, dict)
        or tuple(definition.get("coordinate_order", ())) != coordinate_order
    ):
        raise ValueError(
            "Authored definition coordinate order differs from compiled model"
        )
    if not coordinate_order or len(set(coordinate_order)) != len(coordinate_order):
        raise ValueError("Native scalar coordinate order must be nonempty and unique")
    xml, _ = export_full_body_mjcf(definition_bytes)
    xml_hash = "sha256:" + hashlib.sha256(xml.encode("utf-8")).hexdigest()
    if xml_hash != model_hash:
        raise ValueError("Authored definition does not reproduce bound native XML")
    _xml_coordinates(xml, coordinate_order, coordinate_units)
    ranges = definition.get("coordinate_ranges_deg", {})
    if not isinstance(ranges, dict):
        raise ValueError("Authored coordinate_ranges_deg must be a named mapping")
    unit_map = dict(zip(coordinate_order, coordinate_units, strict=True))
    if any(name not in unit_map or unit_map[name] != "rad" for name in ranges):
        raise ValueError(
            "Authored degree ranges require known compiled scalar hinge names"
        )
    bounds = {
        name: _radian_pair(ranges[name]) for name in coordinate_order if name in ranges
    }
    return AuthoredCoordinateBounds(
        bounds,
        tuple(name for name in coordinate_order if name not in ranges),
        "sha256:" + hashlib.sha256(definition_bytes).hexdigest(),
        xml_hash,
    )
