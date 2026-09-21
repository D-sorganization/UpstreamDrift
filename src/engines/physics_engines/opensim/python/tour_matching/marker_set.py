"""Author an OpenSim ``MarkerSet`` on a parsed .osim document.

Markers are body-fixed points: each placement names a body that must exist in
the document and a finite offset in that body's frame, in metres. Attaching is
idempotent (an existing MarkerSet is replaced). Parsing uses defusedxml;
element construction uses the standard library, which cannot be attacked by
external entities. Offsets come from OS-3 calibration; this module does not
guess them.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import xml.etree.ElementTree as ET  # nosec B405 # nosemgrep: python.lang.security.use-defused-xml.use-defused-xml
from defusedxml import ElementTree as SafeET


@dataclass(frozen=True)
class MarkerPlacement:
    """A marker fixed to ``body`` at ``offset_m`` expressed in that body frame."""

    body: str
    offset_m: tuple[float, float, float]

    def __post_init__(self) -> None:
        offset = tuple(float(v) for v in self.offset_m)
        if not self.body or not self.body.strip():
            raise ValueError("Marker placement requires a body name")
        if len(offset) != 3 or not all(math.isfinite(v) for v in offset):
            raise ValueError("Marker offset must be three finite metres")
        object.__setattr__(self, "offset_m", offset)


def parse_model(osim_path: Path) -> ET.ElementTree:
    """Parse an .osim document safely."""
    return SafeET.parse(str(osim_path))


def _model(tree: ET.ElementTree) -> ET.Element:
    root = tree.getroot()
    if root is None:
        raise ValueError("Document has no root element")
    model = root.find("Model")
    if model is None:
        raise ValueError("Document has no Model element")
    return model


def _body_names(model: ET.Element) -> set[str]:
    return {b.get("name", "") for b in model.findall("BodySet/objects/Body")}


def attach_marker_set(
    tree: ET.ElementTree, placements: Mapping[str, MarkerPlacement]
) -> ET.Element:
    """Replace the model's MarkerSet with ``placements``; returns the new element.

    Preconditions: at least one placement; every body exists in the model;
    labels are nonempty. Postcondition: ``Model/MarkerSet/objects`` holds
    exactly one ``Marker`` per placement, in mapping order.
    """
    if not placements:
        raise ValueError("At least one marker placement is required")
    model = _model(tree)
    bodies = _body_names(model)
    missing = sorted({p.body for p in placements.values()} - bodies)
    if missing:
        raise ValueError(f"Marker bodies absent from model: {missing}")
    for label in placements:
        if not label or not label.strip():
            raise ValueError("Marker labels must be nonempty")
    for old in model.findall("MarkerSet"):
        model.remove(old)
    marker_set = ET.SubElement(model, "MarkerSet", {"name": "markerset"})
    objects = ET.SubElement(marker_set, "objects")
    for label, placement in placements.items():
        marker = ET.SubElement(objects, "Marker", {"name": label})
        ET.SubElement(marker, "socket_parent_frame").text = f"/bodyset/{placement.body}"
        ET.SubElement(marker, "location").text = " ".join(
            f"{v:.6g}" for v in placement.offset_m
        )
        ET.SubElement(marker, "fixed").text = "false"
    ET.SubElement(marker_set, "groups")
    return marker_set


def write_model(tree: ET.ElementTree, path: Path) -> Path:
    """Serialize the document with the upstream XML declaration; returns path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    root = tree.getroot()
    if root is None:
        raise ValueError("Document has no root element")
    ET.indent(root, space="\t")
    body = ET.tostring(root, encoding="utf-8", xml_declaration=False)
    path.write_bytes(b'<?xml version="1.0" encoding="UTF-8" ?>\n' + body + b"\n")
    return path


def _coordinates(model: ET.Element) -> dict[str, ET.Element]:
    found = {c.get("name", ""): c for c in model.iter("Coordinate")}
    if "" in found:
        raise ValueError("Model declares an unnamed coordinate")
    return found


def locked_coordinates(tree: ET.ElementTree) -> tuple[str, ...]:
    """Return the names of coordinates whose ``locked`` flag is true."""
    return tuple(
        name
        for name, c in _coordinates(_model(tree)).items()
        if (c.findtext("locked") or "").strip() == "true"
    )


def unlock_coordinates(tree: ET.ElementTree, names: Sequence[str]) -> tuple[str, ...]:
    """Set ``locked`` false for the named coordinates; return those changed.

    Precondition: every name exists in the model. Coordinates already unlocked
    are left untouched and not reported. The packaged golf humanoid inherits
    locked arm, lumbar, subtalar and toe coordinates from its gait-oriented
    base; a golf tracking variant must unlock them explicitly and record it.
    """
    coordinates = _coordinates(_model(tree))
    missing = [n for n in names if n not in coordinates]
    if missing:
        raise ValueError(f"Unknown coordinates: {missing}")
    changed = []
    for name in names:
        element = coordinates[name]
        flag = element.find("locked")
        if flag is None:
            flag = ET.SubElement(element, "locked")
        if (flag.text or "").strip() == "true":
            flag.text = "false"
            changed.append(name)
    return tuple(changed)
