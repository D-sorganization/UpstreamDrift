"""Read which coordinates of an OpenSim model are rotations or translations.

OpenSim motion files store rotations in degrees (``inDegrees=yes``) next to
translations in metres, and the file itself does not say which column is
which. The model does. Without this distinction, a pelvis translation of
0.9 m would be "converted" to 0.0157 (#11403).

Rules, per joint type (OpenSim 4 ``.osim``):

- ``CustomJoint``: a coordinate is translational when it drives one of the
  ``translation1..3`` axes of its own joint and none of its ``rotation1..3``
  axes. A knee whose translations are splines of the knee angle stays
  rotational.
- ``SliderJoint``: translational.
- ``FreeJoint``: the last three of its six coordinates are translational.
- ``PlanarJoint``: the last two of its three coordinates are translational.
- Every other joint type (Pin, Ball, Gimbal, Universal, Weld, ...) is
  rotational.
"""

from __future__ import annotations

import enum
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

from defusedxml import ElementTree


class Element(Protocol):
    """The parts of an XML element this module reads.

    A structural type instead of the stdlib class, so this module never
    imports ``xml``; parsing always goes through ``defusedxml``.
    """

    tag: str

    def get(self, key: str, default: str = ...) -> str: ...

    def iter(self, tag: str | None = ...) -> Iterator[Element]: ...

    def findtext(self, path: str) -> str | None: ...


class CoordinateKind(enum.Enum):
    """Whether a model coordinate is an angle (rad) or a distance (m)."""

    ROTATIONAL = "rotational"
    TRANSLATIONAL = "translational"


# Joint types whose trailing coordinates are translations: type -> count.
_TRAILING_TRANSLATIONS = {"FreeJoint": 3, "PlanarJoint": 2}


def _axis_coordinates(joint: Element, prefix: str) -> set[str]:
    names: set[str] = set()
    for axis in joint.iter("TransformAxis"):
        if not axis.get("name", "").startswith(prefix):
            continue
        text = axis.findtext("coordinates") or ""
        names.update(text.split())
    return names


def _joint_kinds(joint: Element) -> dict[str, CoordinateKind]:
    coordinates = [c.get("name", "") for c in joint.iter("Coordinate")]
    if joint.tag == "SliderJoint":
        return dict.fromkeys(coordinates, CoordinateKind.TRANSLATIONAL)
    if joint.tag in _TRAILING_TRANSLATIONS:
        first_translation = len(coordinates) - _TRAILING_TRANSLATIONS[joint.tag]
        return {
            name: (
                CoordinateKind.TRANSLATIONAL
                if index >= first_translation
                else CoordinateKind.ROTATIONAL
            )
            for index, name in enumerate(coordinates)
        }
    translations = _axis_coordinates(joint, "translation")
    rotations = _axis_coordinates(joint, "rotation")
    return {
        name: (
            CoordinateKind.TRANSLATIONAL
            if name in translations and name not in rotations
            else CoordinateKind.ROTATIONAL
        )
        for name in coordinates
    }


def read_osim_coordinates(model_path: Path) -> dict[str, CoordinateKind]:
    """Return every coordinate of the model, in file order, with its kind.

    Raises:
        FileNotFoundError: ``model_path`` does not exist.
        ValueError: the model defines no coordinates, a coordinate has no
            name, or two joints define the same coordinate name.
    """
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"OpenSim model not found: {path}")
    root = ElementTree.parse(path).getroot()
    kinds: dict[str, CoordinateKind] = {}
    for joint in root.iter():
        # ``<Joint>`` is the OpenSim 3 wrapper around the typed joint element.
        if not joint.tag.endswith("Joint") or joint.tag == "Joint":
            continue
        for name, kind in _joint_kinds(joint).items():
            if not name:
                raise ValueError(f"OpenSim model {path} has an unnamed coordinate")
            if name in kinds:
                raise ValueError(
                    f"OpenSim model {path} defines coordinate {name!r} twice"
                )
            kinds[name] = kind
    if not kinds:
        raise ValueError(f"OpenSim model {path} has no coordinates")
    return kinds
