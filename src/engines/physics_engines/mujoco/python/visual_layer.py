"""MuJoCo translation of the shared visual skeleton.

Adds capsule/sphere geoms in visual group 1 with collisions disabled and no
mass, a ground plane opposite gravity, lights and a default camera. The
physics model (bodies, joints, inertias, contact spheres, closure sites) is
untouched; the exporter's plain output remains the qualified representation.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET  # nosec B405 # nosemgrep: python.lang.security.use-defused-xml.use-defused-xml - construction only; parsing is defused
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from src.shared.python.motion_matching.visual_skeleton import (
    VisualSkeleton,
    derive_visual_skeleton,
)

_VISUAL_CLASS = "visual"
_CAPSULE_RGBA = "0.75 0.78 0.85 1"
_SHAPE_RGBA = "0.7 0.72 0.8 1"
# Centre-of-mass and frame spheres sit in a group the renderer hides by
# default (groups 0 to 2 are shown); enable group 3 to inspect them.
_MARKER_SPHERE_GROUP = "3"
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
    for index, shape in enumerate(skeleton.shapes):
        ET.SubElement(
            elements[shape.body],
            "geom",
            name=f"visual_{shape.kind}_{index}_{shape.body}",
            type=shape.kind,
            pos=_numbers(
                _to_mjcf_frame(offsets[shape.body], np.asarray(shape.center_m))
            ),
            size=_numbers(shape.half_size_m),
            rgba=_SHAPE_RGBA,
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
            group=_MARKER_SPHERE_GROUP,
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


def whole_body_com(model: Any, data: Any) -> np.ndarray:
    """Centre of mass of every body in the tree (the body-plus-club system) at
    the current ``data`` state, from MuJoCo's subtree centre of mass of the
    first body under the world. Postcondition: a finite 3-vector."""
    import mujoco

    if model.nbody < 2:
        raise ValueError("Model has no bodies below the world")
    mujoco.mj_comPos(model, data)
    com = np.asarray(data.subtree_com[1], dtype=float).copy()
    if not np.isfinite(com).all():
        raise ValueError("Centre of mass is not finite")
    return com


def add_scene_marker(
    scene: Any,
    position: np.ndarray,
    radius_m: float,
    rgba: tuple[float, float, float, float],
) -> None:
    """Add a sphere marker to a rendered scene (an overlay, not a model geom).
    Precondition: the scene has a free geom slot and the radius is positive."""
    import mujoco

    if radius_m <= 0:
        raise ValueError("Marker radius must be positive")
    if scene.ngeom >= scene.maxgeom:
        raise ValueError("Scene has no free geom slot for a marker")
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius_m, 0.0, 0.0]),
        np.asarray(position, dtype=float),
        np.eye(3).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


COM_MARKER_RGBA = (0.95, 0.15, 0.15, 1.0)
COM_GROUND_RGBA = (0.95, 0.85, 0.1, 1.0)


def add_com_markers(
    scene: Any, model: Any, data: Any, ground_height_m: float
) -> np.ndarray:
    """Overlay the whole-body centre of mass (red) and its vertical projection
    onto the ground plane (yellow); returns the centre of mass."""
    com = whole_body_com(model, data)
    add_scene_marker(scene, com, 0.03, COM_MARKER_RGBA)
    add_scene_marker(
        scene,
        np.array([com[0], com[1], ground_height_m + 0.005]),
        0.025,
        COM_GROUND_RGBA,
    )
    return com


def render_playback(
    spec_bytes: bytes,
    names: Sequence[str],
    q: np.ndarray,
    lookat: np.ndarray,
    path: Path,
    show_com: bool = True,
    playback_stride: int = 4,
    rate_hz: float = 120.0,
) -> None:
    """Render animated GIF of motion from spec and joint trajectory."""
    import json

    import imageio
    import mujoco

    from src.engines.physics_engines.mujoco.python import full_body_mjcf as exporter

    xml, _ = exporter.export_full_body_mjcf(spec_bytes, visual=True)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    addresses = [model.joint(n).qposadr[0] for n in names]
    ground_height = float(
        json.loads(spec_bytes)["contact"].get("ground_height_m") or 0.0
    )
    renderer = mujoco.Renderer(model, 240, 320)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = lookat
    cam.distance, cam.azimuth, cam.elevation = 3.2, 135.0, -12.0
    frames_out = []
    for k in range(0, q.shape[0], playback_stride):
        data.qpos[addresses] = q[k]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=cam)
        if show_com:
            add_com_markers(renderer.scene, model, data, ground_height)
        frames_out.append(renderer.render().copy())
    imageio.mimsave(path, frames_out, duration=1000 * playback_stride / rate_hz, loop=0)
