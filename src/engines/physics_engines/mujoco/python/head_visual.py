"""Visual head, face and neck for the MuJoCo visual layers (visual only).

Attaches the parametric head of :mod:`src.shared.python.model_appearance.head`
as massless, non-colliding mesh geoms (class ``visual``). Anthropometric
full-body specs carry a ``Head`` body on a three-axis neck at the cervicale;
the head geoms ride on that body, so they follow the fitted neck motion. A spec
without a head body (the native Simscape layout) gets the head on the torso
body that holds the shoulders, at the anthropometric neck point above the
shoulder centre, and it then follows the trunk only.

When the appearance document declares an ``orientation_override`` the head
instead lives on a mocap body (``visual_head``; no degree of freedom, no mass)
and :func:`drive_visual_head` places it each frame from a yaw/pitch/roll
sample, so a gaze controller can show the same stabilised head even where the
physics model has no neck. Nothing here changes a mass, inertia or degree of
freedom.
"""

from __future__ import annotations

import dataclasses
import xml.etree.ElementTree as ET  # nosec B405 # nosemgrep: python.lang.security.use-defused-xml.use-defused-xml - construction only; parsing is defused
from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from src.engines.physics_engines.mujoco.python.appearance_layer import _Assets
from src.engines.physics_engines.mujoco.python.native_mjcf import _pose
from src.shared.python.model_appearance import head as head_model
from src.shared.python.model_appearance.schema import AppearanceDocument

HEAD_MOCAP_BODY = "visual_head"
HEAD_ANCHOR_SITE = "visual_head_anchor"
_CLASS = "visual"
_SHOULDER_BELOW_NECK_M = 0.04  # shoulder centre to cervicale (anthropometric_geometry)


def _quat_wxyz(matrix: np.ndarray) -> np.ndarray:
    xyzw = Rotation.from_matrix(matrix).as_quat()
    return xyzw[[3, 0, 1, 2]]


def _local_pose(element: ET.Element) -> np.ndarray:
    pose = np.eye(4)
    if element.get("quat"):
        w, x, y, z = (float(v) for v in element.get("quat", "").split())
        pose[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
    if element.get("pos"):
        pose[:3, 3] = [float(v) for v in element.get("pos", "").split()]
    return pose


def _zero_pose_worlds(
    world: ET.Element,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """World pose of every body at zero coordinates, and each body's parent."""
    poses: dict[str, np.ndarray] = {"world": np.eye(4)}
    parents: dict[str, str] = {}
    stack = [(world, "world")]
    while stack:
        element, name = stack.pop()
        for child in element.findall("body"):
            child_name = child.get("name", "")
            poses[child_name] = poses[name] @ _local_pose(child)
            parents[child_name] = name
            stack.append((child, child_name))
    return poses, parents


def _torso_frame(
    elements: Mapping[str, ET.Element], up_hint: np.ndarray
) -> tuple[str, np.ndarray, float] | None:
    """Torso body, canonical head pose in its MJCF frame, and head length.

    Used only when the spec has no head body: the neck point is
    ``_SHOULDER_BELOW_NECK_M`` above the shoulder centre, the head looks the
    way the feet point.
    """
    poses, parents = _zero_pose_worlds(elements["world"])
    shoulders = [n for n in poses if "hubto" in n.lower()]
    if not shoulders:
        return None
    torso = parents[shoulders[0]]
    centre = np.mean([poses[n][:3, 3] for n in shoulders], axis=0)
    up = up_hint / np.linalg.norm(up_hint)
    heel = [
        poses[n][:3, 3] for n in poses if "calcn" in n.lower() or "talus" in n.lower()
    ]
    toe = [poses[n][:3, 3] for n in poses if "toes" in n.lower()]
    fwd = np.array([1.0, 0.0, 0.0])
    if heel and toe:
        d = np.mean(toe, axis=0) - np.mean(heel, axis=0)
        d = d - (d @ up) * up
        if np.linalg.norm(d) > 1e-6:
            fwd = d / np.linalg.norm(d)
    fwd = fwd - (fwd @ up) * up
    fwd /= np.linalg.norm(fwd)
    world_pose = np.eye(4)
    world_pose[:3, :3] = np.column_stack([fwd, np.cross(up, fwd), up])
    world_pose[:3, 3] = centre + up * _SHOULDER_BELOW_NECK_M
    return (
        torso,
        np.linalg.inv(poses[torso]) @ world_pose,
        head_model.DEFAULT_HEAD_LENGTH_M,
    )


def _registry(root: ET.Element, doc: AppearanceDocument) -> _Assets:
    """Asset builder that reuses materials the appearance layer already made."""
    assets = _Assets(root, doc)
    for material in assets.asset.findall("material"):
        name = material.get("name", "")
        if name.startswith("mat_") and not name.endswith("_plane"):
            assets._made.add(name[4:])  # noqa: SLF001 - same package, shared dedupe set
    return assets


def _place(
    spec: Mapping[str, Any],
    elements: Mapping[str, ET.Element],
    offsets: Mapping[str, np.ndarray],
    doc: AppearanceDocument,
) -> tuple[str, np.ndarray, float, str] | None:
    """Anchor body, canonical head pose in its MJCF frame, length, source."""
    anchor = head_model.resolve_head_anchor(spec)
    if anchor is not None:
        pose = offsets[anchor.body] @ head_model.head_frame_in_body(anchor, doc.head)
        return anchor.body, pose, anchor.length_m, anchor.source
    gravity = np.asarray(spec.get("gravity_m_s2", [0.0, 0.0, -9.81]), dtype=float)
    found = _torso_frame(elements, -gravity)
    if found is None:
        return None
    return found[0], found[1], found[2], "torso"


def attach_head_visual(
    root: ET.Element,
    elements: Mapping[str, ET.Element],
    offsets: Mapping[str, np.ndarray],
    spec: Mapping[str, Any],
    doc: AppearanceDocument,
) -> dict[str, Any]:
    """Add the head, face and neck geoms; returns a summary for the metadata.

    Postconditions: only ``visual`` class geoms (and, with an orientation
    override, one massless mocap body and one site) are added; physics arrays
    of the existing bodies are untouched.
    """
    cfg = doc.head
    if not cfg.enabled:
        return {"enabled": False, "reason": "disabled in the appearance document"}
    placed = _place(spec, elements, offsets, doc)
    if placed is None:
        return {"enabled": False, "reason": "spec has no head body or shoulder bodies"}
    body, pose, length, source = placed
    canonical = dataclasses.replace(cfg, forward_axis="+x", up_axis="+z")
    parts = head_model.build_head_parts(length, canonical)
    assets = _registry(root, doc)
    override = cfg.orientation_override
    if override is None:
        target = elements[body]
        parts = head_model.place_parts(parts, pose)
    else:
        target = _mocap_body(elements, body, pose)
        ET.SubElement(
            elements[body], "site", name=HEAD_ANCHOR_SITE, size=".002", **_pose(pose)
        )
    for part in parts:
        mesh_name = f"vmesh_head_{part.name}"
        assets.mesh(mesh_name, part.mesh)
        material = head_model.part_material_name(part, doc)
        ET.SubElement(
            target,
            "geom",
            name=f"visual_head_{part.name}",
            type="mesh",
            mesh=mesh_name,
            material=assets.material(material),
            attrib={"class": _CLASS},
        )
    return {
        "enabled": True,
        "body": body,
        "source": source,
        "head_length_m": length,
        "headwear": cfg.headwear,
        "parts": [p.name for p in parts],
        "mode": "body" if override is None else "mocap",
        "mocap_body": None if override is None else HEAD_MOCAP_BODY,
        "channel": None if override is None else override.channel,
    }


def _mocap_body(
    elements: Mapping[str, ET.Element], anchor_body: str, pose_in_body: np.ndarray
) -> ET.Element:
    """Mocap body at the head's zero-pose world pose (no DOF, no mass)."""
    world = elements["world"]
    poses, _ = _zero_pose_worlds(world)
    start = poses[anchor_body] @ pose_in_body
    return ET.SubElement(
        world,
        "body",
        {"name": HEAD_MOCAP_BODY, "mocap": "true", **_pose(start)},
    )


def drive_visual_head(
    model: Any,
    data: Any,
    yaw_rad: float = 0.0,
    pitch_rad: float = 0.0,
    roll_rad: float = 0.0,
    frame: str = "parent",
) -> None:
    """Place the mocap head from a yaw/pitch/roll sample (call after forward).

    ``parent`` applies the angles in the physics head's own frame; ``world``
    applies them in the upright rest frame (a gaze-stabilised head that ignores
    trunk and neck motion). Position always follows the neck anchor. Raises
    ``ValueError`` for an unknown frame or a model without a visual head.
    """
    import mujoco

    if frame not in ("parent", "world"):
        raise ValueError("frame must be 'parent' or 'world'")
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, HEAD_MOCAP_BODY)
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, HEAD_ANCHOR_SITE)
    if body < 0 or site < 0 or model.body_mocapid[body] < 0:
        raise ValueError("Model has no orientation-override visual head")
    rotation = head_model.head_override_rotation(yaw_rad, pitch_rad, roll_rad)
    base = (
        np.asarray(data.site_xmat[site]).reshape(3, 3)
        if frame == "parent"
        else Rotation.from_quat(
            np.asarray(model.body_quat[body])[[1, 2, 3, 0]]
        ).as_matrix()
    )
    mocap = int(model.body_mocapid[body])
    data.mocap_pos[mocap] = np.asarray(data.site_xpos[site])
    data.mocap_quat[mocap] = _quat_wxyz(base @ rotation)


__all__ = [
    "HEAD_ANCHOR_SITE",
    "HEAD_MOCAP_BODY",
    "attach_head_visual",
    "drive_visual_head",
]
