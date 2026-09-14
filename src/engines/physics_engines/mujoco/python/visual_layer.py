"""MuJoCo translation of the shared visual skeleton.

Adds capsule/sphere geoms in visual group 1 with collisions disabled and no
mass, a ground plane opposite gravity, lights and a default camera. The
physics model (bodies, joints, inertias, contact spheres, closure sites) is
untouched; the exporter's plain output remains the qualified representation.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET  # nosec B405 - construction only
from collections.abc import Mapping
from typing import Any

import numpy as np

from src.shared.python.motion_matching.visual_skeleton import (
    VisualSkeleton,
    derive_visual_skeleton,
)

_VISUAL_CLASS = "visual"
_CAPSULE_RGBA = "0.75 0.78 0.85 1"
_COM_RGBA = "0.9 0.35 0.2 1"
_FRAME_RGBA = "0.2 0.6 0.95 1"
_FLOOR_RGBA = "0.35 0.45 0.3 1"


def _numbers(values: Any) -> str:
    return " ".join(format(float(v), ".9g") for v in np.asarray(values).ravel())


def _to_mjcf_frame(offset: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Map a point from the spec body frame into the MJCF body frame."""
    return offset[:3, :3] @ point + offset[:3, 3]


def _plane_quat(normal: np.ndarray) -> str:
    """Quaternion rotating MuJoCo's +z plane normal onto ``normal`` (wxyz)."""
    z = np.array([0.0, 0.0, 1.0])
    n = normal / np.linalg.norm(normal)
    axis = np.cross(z, n)
    s = float(np.linalg.norm(axis))
    c = float(np.dot(z, n))
    if s < 1e-12:
        return "1 0 0 0" if c > 0 else "0 1 0 0"
    axis /= s
    half = float(np.arctan2(s, c)) / 2.0
    return _numbers([np.cos(half), *(np.sin(half) * axis)])


def attach_visual_layer(
    root: ET.Element,
    elements: Mapping[str, ET.Element],
    offsets: Mapping[str, np.ndarray],
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Attach the shared skeleton to an MJCF document; returns a summary.

    ``elements`` and ``offsets`` are the exporter's per-body MJCF elements and
    spec-body-to-MJCF-body transforms. Every added geom is class ``visual``:
    group 1, zero contype/conaffinity, zero mass.
    """
    skeleton: VisualSkeleton = derive_visual_skeleton(spec)
    default_root = root.find("default")
    if default_root is None:
        default_root = ET.SubElement(root, "default")
    visual_default = ET.SubElement(
        default_root, "default", attrib={"class": _VISUAL_CLASS}
    )
    ET.SubElement(
        visual_default, "geom", contype="0", conaffinity="0", group="1", mass="0"
    )
    for index, capsule in enumerate(skeleton.capsules):
        offset = offsets[capsule.body]
        start = _to_mjcf_frame(offset, np.asarray(capsule.start_m))
        end = _to_mjcf_frame(offset, np.asarray(capsule.end_m))
        ET.SubElement(
            elements[capsule.body],
            "geom",
            name=f"visual_capsule_{index}_{capsule.body}",
            type="capsule",
            fromto=_numbers(np.concatenate((start, end))),
            size=_numbers([capsule.radius_m]),
            rgba=_CAPSULE_RGBA,
            attrib={"class": _VISUAL_CLASS},
        )
    for i, sphere in enumerate(skeleton.spheres):
        ET.SubElement(
            elements[sphere.body],
            "geom",
            name=f"visual_{sphere.kind}_{i}",
            type="sphere",
            pos=_numbers(
                _to_mjcf_frame(offsets[sphere.body], np.asarray(sphere.center_m))
            ),
            size=_numbers([sphere.radius_m]),
            rgba=_COM_RGBA if sphere.kind == "com" else _FRAME_RGBA,
            attrib={"class": _VISUAL_CLASS},
        )
    world = elements["world"]
    normal = np.asarray(skeleton.ground.normal)
    ET.SubElement(
        world,
        "geom",
        name="visual_floor",
        type="plane",
        size="3 3 0.05",
        pos=_numbers(normal * skeleton.ground.height_m),
        quat=_plane_quat(normal),
        rgba=_FLOOR_RGBA,
        attrib={"class": _VISUAL_CLASS},
    )
    up = normal * 3.0
    ET.SubElement(
        world,
        "light",
        name="visual_key",
        pos=_numbers(up + np.array([1.5, -1.5, 0.0])),
        dir=_numbers(-(up + np.array([1.5, -1.5, 0.0]))),
        diffuse="0.8 0.8 0.8",
    )
    ET.SubElement(
        world,
        "light",
        name="visual_fill",
        pos=_numbers(up + np.array([-2.0, 1.0, 0.0])),
        dir=_numbers(-(up + np.array([-2.0, 1.0, 0.0]))),
        diffuse="0.4 0.4 0.4",
    )
    ET.SubElement(
        world,
        "camera",
        name="visual_default",
        pos=_numbers(normal * 1.2 + np.array([2.8, -2.8, 0.0])),
        mode="targetbody",
        target=next(e.get("name", "") for k, e in elements.items() if k != "world"),
    )
    return {
        "capsules": len(skeleton.capsules),
        "spheres": len(skeleton.spheres),
        "floor": True,
        "ground_calibrated": skeleton.ground.calibrated,
        "lights": 2,
    }
