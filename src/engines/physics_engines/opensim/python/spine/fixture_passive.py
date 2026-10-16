"""Invented bushing forces for the synthetic spine fixture (SPN-8c, #12755).

Stand-in for the per-level intervertebral bushings of Senteler-style models.
EVERY stiffness, damping and expression here is INVENTED test data; nothing is
transcribed from a licensed model.  A bushing is attached across the two joint
offset frames of a level (parent offset frame on the inferior body, child
offset frame on the superior body), which coincide at zero coordinates, so
the OpenSim ``BushingForce`` frame pair identifies the level by the parent
bodies of its frames.  ``frames_reversed`` swaps the frame order to exercise
the orientation flag of the extractor.

Units: N/m and N s/m (translational), N m/rad and N m s/rad (rotational).
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET  # noqa: S405  # nosemgrep: python.lang.security.use-defused-xml.use-defused-xml  # build-only; never parses untrusted XML
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from src.engines.physics_engines.opensim.python.spine.fixture_model import (
    FixtureSpec,
    fixture_level_names,
)
from src.engines.physics_engines.opensim.python.spine.inventory import (
    LevelMapEntry,
    ModelLevelMap,
)
from src.shared.python.spine.levels import SpinalLevel

__all__ = [
    "BUSHING_KINDS",
    "EXPRESSION_KEYS",
    "FixtureBushing",
    "add_fixture_bushings",
    "default_fixture_bushings",
    "fixture_level_map",
]

BUSHING_KINDS = ("BushingForce", "ExpressionBasedBushingForce")
#: Force and moment expression names of an ``ExpressionBasedBushingForce``.
EXPRESSION_KEYS = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")
_Vec3 = tuple[float, float, float]


def _vec3(value: Iterable[float], name: str) -> _Vec3:
    items = tuple(value)
    if len(items) != 3:
        raise ValueError(f"{name} must have 3 entries, got {len(items)}")
    out = tuple(float(v) for v in items)
    if not all(math.isfinite(v) and v >= 0.0 for v in out):
        raise ValueError(f"{name} must be finite and non-negative stiffness/damping")
    return out  # type: ignore[return-value]


@dataclass(frozen=True)
class FixtureBushing:
    """One invented bushing (diagonal stiffness/damping in the frame-1 axes).

    Preconditions: ``kind`` in :data:`BUSHING_KINDS`; each vector has three
    finite non-negative entries; an expression-based bushing carries all six
    :data:`EXPRESSION_KEYS` expressions (its stiffness vectors are unused).
    """

    kind: str = "BushingForce"
    translational_stiffness: _Vec3 = (1000.0, 2000.0, 3000.0)
    rotational_stiffness: _Vec3 = (10.0, 20.0, 30.0)
    translational_damping: _Vec3 = (1.0, 2.0, 3.0)
    rotational_damping: _Vec3 = (0.1, 0.2, 0.3)
    expressions: Mapping[str, str] | None = None
    frames_reversed: bool = False

    def __post_init__(self) -> None:
        if self.kind not in BUSHING_KINDS:
            raise ValueError(f"kind must be one of {BUSHING_KINDS}, got {self.kind!r}")
        for name in (
            "translational_stiffness",
            "rotational_stiffness",
            "translational_damping",
            "rotational_damping",
        ):
            object.__setattr__(self, name, _vec3(getattr(self, name), name))
        if self.kind == "ExpressionBasedBushingForce":
            if self.expressions is None or set(self.expressions) != set(
                EXPRESSION_KEYS
            ):
                raise ValueError(
                    f"expressions must define exactly {EXPRESSION_KEYS}, "
                    f"got {None if self.expressions is None else sorted(self.expressions)}"
                )
            object.__setattr__(
                self, "expressions", MappingProxyType(dict(self.expressions))
            )
        elif self.expressions is not None:
            raise ValueError(
                "expressions are only valid for ExpressionBasedBushingForce"
            )


def default_fixture_bushings(spec: FixtureSpec) -> dict[SpinalLevel, FixtureBushing]:
    """Distinct invented ``BushingForce`` per level of ``spec`` (caudal first).

    Level ``k`` (1-based from the caudal end) scales the base stiffness and
    damping by ``k``, so every level differs and a mixed-up assignment is
    detectable.  Postcondition: one entry per ``spec.levels``.
    """
    out = {}
    for k, level in enumerate(spec.levels, start=1):
        out[level] = FixtureBushing(
            translational_stiffness=(1000.0 * k, 2000.0 * k, 3000.0 * k),
            rotational_stiffness=(10.0 * k, 20.0 * k, 30.0 * k),
            translational_damping=(1.0 * k, 2.0 * k, 3.0 * k),
            rotational_damping=(0.1 * k, 0.2 * k, 0.3 * k),
        )
    return out


def fixture_level_map(spec: FixtureSpec) -> ModelLevelMap:
    """Verified level map of the fixture, with translations when present.

    Postcondition: ``levels`` has one entry per ``spec.levels``; each entry's
    ``translations`` is set iff ``spec.translational_dofs``.
    """
    levels = {
        n.level: LevelMapEntry(n.joint, n.coordinates, (1, 1, 1), n.translations)
        for n in fixture_level_names(spec)
    }
    return ModelLevelMap(
        verified=True,
        pelvis_body="pelvis",
        terminal_body="thorax_proxy",
        levels=levels,
        names_status="synthetic fixture (invented, see fixture_model.py)",
    )


def _vec_text(vec: _Vec3) -> str:
    return " ".join(format(v, ".17g") for v in vec)


def _forceset(model: ET.Element) -> ET.Element:
    forceset = model.find("ForceSet")
    if forceset is None:
        forceset = ET.SubElement(model, "ForceSet", attrib={"name": "forceset"})
    objects = forceset.find("objects")
    return objects if objects is not None else ET.SubElement(forceset, "objects")


def _add_bushing(
    objects: ET.Element, joint: str, level: SpinalLevel, spec: FixtureBushing
) -> None:
    node = ET.SubElement(objects, spec.kind, attrib={"name": f"{level.name}_bushing"})
    frames = [f"/jointset/{joint}/{joint}_parent_offset"]
    frames.append(f"/jointset/{joint}/{joint}_child_offset")
    if spec.frames_reversed:
        frames.reverse()
    ET.SubElement(node, "socket_frame1").text = frames[0]
    ET.SubElement(node, "socket_frame2").text = frames[1]
    if spec.expressions is not None:
        for key in EXPRESSION_KEYS:
            ET.SubElement(node, f"{key}_expression").text = spec.expressions[key]
    else:
        ET.SubElement(node, "rotational_stiffness").text = _vec_text(
            spec.rotational_stiffness
        )
        ET.SubElement(node, "translational_stiffness").text = _vec_text(
            spec.translational_stiffness
        )
    ET.SubElement(node, "rotational_damping").text = _vec_text(spec.rotational_damping)
    ET.SubElement(node, "translational_damping").text = _vec_text(
        spec.translational_damping
    )


def add_fixture_bushings(
    root: ET.Element,
    spec: FixtureSpec,
    bushings: Mapping[SpinalLevel, FixtureBushing],
) -> None:
    """Add one bushing force per entry of ``bushings`` to a fixture tree.

    Preconditions: ``bushings`` maps levels of ``spec`` to
    :class:`FixtureBushing`.  Postconditions: the model gains a ``ForceSet``
    entry ``<LEVEL>_bushing`` per level (appended to an existing ``ForceSet``);
    nothing else changes.

    Raises:
        TypeError: a value is not a :class:`FixtureBushing`.
        ValueError: a key is not a level of ``spec``.
    """
    joints = {n.level: n.joint for n in fixture_level_names(spec)}
    for level, item in bushings.items():
        if level not in joints:
            raise ValueError(
                f"bushing level {getattr(level, 'name', level)!r} is not a level "
                f"of the fixture ({[lv.name for lv in joints]})"
            )
        if not isinstance(item, FixtureBushing):
            raise TypeError(f"bushings[{level.name}] must be a FixtureBushing")
    model = root.find("Model")
    if model is None:
        raise ValueError("fixture tree has no Model element")
    objects = _forceset(model)
    for level in spec.levels:
        if level in bushings:
            _add_bushing(objects, joints[level], level, bushings[level])
