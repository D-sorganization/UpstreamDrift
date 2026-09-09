"""Adapt URDF kinematic trees to the existing continuous model fitter."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from src.motion_capture.reconstruct.model import Joint, ModelSpec
from src.motion_capture.reconstruct.model.registry import RegisteredModel
from src.motion_capture.reconstruct.model.session import LandmarkMap
from src.shared.python.model_generation.converters.urdf_parser import URDFParser
from src.shared.python.model_generation.core.types import Joint as URDFJoint


def _vector3(values: np.ndarray) -> tuple[float, float, float]:
    """Keep native vectors explicit at the typed model-contract boundary."""
    return float(values[0]), float(values[1]), float(values[2])


def _joint(source: URDFJoint, observed: bool) -> Joint:
    kind = source.joint_type.value
    if kind not in {"fixed", "revolute", "continuous"}:
        raise ValueError(f"Unsupported URDF joint {source.name}: {kind}")
    origin = source.origin
    offset = np.asarray(origin.xyz, dtype=float)
    length = float(np.linalg.norm(offset))
    rotation = Rotation.from_euler("xyz", origin.rpy)
    axis_rotation = Rotation.identity()
    limits: tuple[tuple[float, float], ...] = ()
    if kind != "fixed":
        axis = np.asarray(source.axis, dtype=float)
        if not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-12:
            raise ValueError(f"Invalid axis for {source.name}")
        axis_rotation, _ = Rotation.align_vectors(
            [axis / np.linalg.norm(axis)], [[1, 0, 0]]
        )
        if kind == "revolute" and source.limits is not None:
            limits = ((source.limits.lower, source.limits.upper),)
    return Joint(
        name=source.child,
        parent=source.parent,
        direction=_vector3(offset / length) if length else (0.0, 0.0, 0.0),
        length=source.name if length else None,
        axes="" if kind == "fixed" else "x",
        limits_rad=limits,
        landmark=observed,
        pre_rotvec=_vector3((rotation * axis_rotation).as_rotvec()),
        post_rotvec=_vector3(axis_rotation.inv().as_rotvec()),
    )


def load_urdf_model(
    path: Path,
    *,
    name: str,
    landmark_map: Mapping[str, str | tuple[str, ...]],
) -> RegisteredModel:
    """Load native revolute/fixed URDF geometry with explicitly mapped link origins.

    Preconditions: one connected tree, valid mapping and no unsupported joints.
    Postcondition: offsets, joint axes, fixed rotations and limits are preserved;
    the added root translation/rotation permits scene registration. No mesh or
    engine fallback is substituted. URDF XML is parsed by the shared parser.
    """
    if not name.strip():
        raise ValueError("Model name must be nonempty")
    parser = URDFParser(resolve_meshes=False)
    parsed = parser.parse(path)
    if parsed.warnings:
        raise ValueError(f"URDF parser warnings: {parsed.warnings}")
    # The shared parser does not represent mimic constraints; refuse them rather
    # than treating coupled coordinates as independent fitting variables.
    if parsed.original_xml and "<mimic" in parsed.original_xml:
        raise ValueError("URDF mimic joints require a coupled-coordinate adapter")
    links = {link.name for link in parsed.links}
    external = [j for j in parsed.joints if j.parent not in links]
    if external and (len(external) != 1 or external[0].joint_type.value != "floating"):
        raise ValueError("Only one external floating world joint is supported")
    # Root q is absolute world position/orientation, absorbing the free joint's
    # origin. Internal floating joints remain unsupported and fail in _joint.
    internal = [j for j in parsed.joints if j not in external]
    children = [joint.child for joint in internal]
    roots = links - set(children)
    if len(roots) != 1 or len(set(children)) != len(children):
        raise ValueError("URDF must be a single tree with one parent per link")
    if not landmark_map or set(landmark_map) - links:
        raise ValueError("Model landmark mapping must name existing links")
    root = next(iter(roots))
    joints = [Joint(root, None, axes="xyz", landmark=root in landmark_map)]
    lengths = {}
    pending = list(internal)
    seen = {root}
    while pending:
        ready = [j for j in pending if j.parent in seen]
        if not ready:
            raise ValueError("URDF contains disconnected or cyclic joints")
        for source in ready:
            converted = _joint(source, source.child in landmark_map)
            joints.append(converted)
            if converted.length:
                lengths[converted.length] = float(np.linalg.norm(source.origin.xyz))
            seen.add(source.child)
            pending.remove(source)
    spec = ModelSpec(name=name, joints=tuple(joints), lengths_m=lengths)
    return RegisteredModel(
        name,
        spec,
        LandmarkMap(dict(landmark_map)),
        (),
        f"URDF link-origin kinematics from {path.name}; fixed geometry.",
    )
