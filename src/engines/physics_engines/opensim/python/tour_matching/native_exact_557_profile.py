"""Exact derived 557-muscle source profile for diagnostic native replay.

The 106 absent VTP files are attached rendering geometry only. This profile
does not authorize any other external resource or infer physiology validity.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from defusedxml import ElementTree as ET

EXACT_SOURCE_SHA256 = "453e09c4e42dcc3f4b74a3b2efeff063e96c3820b3eee1efe95eef249579010d"
_COMPONENT_SHA256 = "c81fcb306692cfbf34b49466119943f6d1686dc8067d08eb0fe5f18a32d9db0b"
_VISUAL_REFERENCES_SHA256 = (
    "617127d1cb80a2b19c5e9fa106cd51080d8729b53e137ae30c2fe03c615d9f87"
)
PROFILE_VERSION = "3.0.0"
VARIANT_ID = "exact-buet-hamner-557-muscle-diagnostic"
COUPLED_ROTATION_PATHS = frozenset(
    {
        "/jointset/L5_S1_IVDjnt/flex_extension",
        "/jointset/L5_S1_IVDjnt/lat_bending",
        "/jointset/L5_S1_IVDjnt/axial_rotation",
        "/jointset/L1_L2_IVD_jnt/L1_L2_IVDjnt_r3",
    }
)


def _visual_reference_records(raw: bytes) -> tuple[tuple[Any, ...], ...]:
    root = ET.fromstring(raw)
    if root.tag != "OpenSimDocument" or len(root.findall("Model")) != 1:
        raise ValueError("exact source needs one OpenSim model")
    parents = {child: node for node in root.iter() for child in node}
    records = []
    for node in root.iter():
        tag = node.tag.lower()
        external = tag in {"file", "filename", "file_name"} or tag.endswith("_file")
        if any(key.lower() in {"file", "filename", "href"} for key in node.attrib):
            raise ValueError("unreviewed external source resource")
        if not external:
            continue
        chain = []
        cursor = node
        while cursor in parents:
            cursor = parents[cursor]
            chain.append(cursor.tag)
        if (
            tag != "mesh_file"
            or chain[:2] != ["Mesh", "attached_geometry"]
            or len(chain) < 3
            or chain[2] not in {"Body", "PhysicalOffsetFrame"}
            or any(kind in {"ForceSet", "ContactGeometrySet"} for kind in chain)
        ):
            raise ValueError("only attached visual mesh resources are reviewed")
        records.append((node.tag, node.text, chain[:5]))
    return tuple(records)


def inspect_visual_references(raw: bytes) -> tuple[str, ...]:
    """Return declared rendering resources after rejecting dynamic references."""
    return tuple(str(record[1]) for record in _visual_reference_records(raw))


def validate_exact_source_bytes(raw: bytes) -> None:
    """Require the reviewed source bytes and its visual-only file declarations."""
    if hashlib.sha256(raw).hexdigest() != EXACT_SOURCE_SHA256:
        raise ValueError("exact source SHA-256 differs")
    records = _visual_reference_records(raw)
    digest = hashlib.sha256(
        json.dumps(records, separators=(",", ":")).encode()
    ).hexdigest()
    if len(records) != 106 or digest != _VISUAL_REFERENCES_SHA256:
        raise ValueError("exact visual resource manifest differs")


def validate_exact_loaded_model(model: Any, declaration: Any) -> None:
    """Bind native component topology and all source clamped-coordinate ranges."""
    import opensim as osim

    records = sorted(
        (item.getAbsolutePathString(), item.getConcreteClassName())
        for item in model.getComponentsList()
    )
    digest = hashlib.sha256(
        json.dumps(records, separators=(",", ":")).encode()
    ).hexdigest()
    if len(records) != 6691 or digest != _COMPONENT_SHA256:
        raise ValueError("exact native component/path inventory differs")
    if model.getMuscles().getSize() != 557 or model.getConstraintSet().getSize() != 17:
        raise ValueError("exact native muscle or coupler count differs")
    if model.getContactGeometrySet().getSize() != 0:
        raise ValueError("exact native source unexpectedly contains contact geometry")
    coordinates = model.getCoordinateSet()
    coordinate_names = tuple(
        coordinates.get(index).getName() for index in range(coordinates.getSize())
    )
    if len(set(coordinate_names)) != len(coordinate_names):
        raise ValueError("exact coordinate names must be unambiguous")
    observed_coupled: set[str] = set()
    joints = model.getJointSet()
    for index in range(joints.getSize()):
        custom = osim.CustomJoint.safeDownCast(joints.get(index))
        if custom is None:
            continue
        transform = custom.getSpatialTransform()
        for axis_index in range(6):
            names = transform.getTransformAxis(axis_index).getCoordinateNames()
            for name_index in range(names.size()):
                coordinate = coordinates.get(names.getValue(name_index))
                path = coordinate.getAbsolutePathString()
                if path in COUPLED_ROTATION_PATHS:
                    if axis_index >= 3:
                        raise ValueError(
                            "coupled rotation path drives a translation axis"
                        )
                    observed_coupled.add(path)
    if observed_coupled != COUPLED_ROTATION_PATHS:
        raise ValueError("coupled rotation chart differs from exact native source")
    for index in range(coordinates.getSize()):
        coordinate = coordinates.get(index)
        if not coordinate.getDefaultClamped():
            continue
        path = coordinate.getAbsolutePathString()
        bounds = (coordinate.getRangeMin(), coordinate.getRangeMax())
        if declaration.chart_bounds.get(path) != bounds:
            raise ValueError("clamped coordinate needs exact source range declaration")
    for item in model.getComponentsList():
        if osim.Controller.safeDownCast(item) is not None:
            raise ValueError("exact source controller is forbidden")
        if osim.PositionMotion.safeDownCast(item) is not None:
            raise ValueError("exact source prescribed motion is forbidden")
        if (
            osim.Force.safeDownCast(item) is not None
            and osim.Muscle.safeDownCast(item) is None
        ):
            raise ValueError("exact source nonmuscle force is forbidden")


def profile_source_sha256() -> str:
    """Bind this admission implementation independently of the native model."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
