"""Native fixed transforms; preserve base/follower rotation-axis convention."""

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from src.shared.python.motion_matching.native_solids import NativeParameters


def rigid_transform(block: Mapping[str, Any]) -> NDArray[np.float64]:
    """Map follower-frame vectors to base; reject unsupported native modes.

    The golf inventory uses rotation sequences with no translation. Support
    Cartesian translation explicitly for reusable fixtures and future exports.
    """
    parameters = NativeParameters(block)
    transform = np.eye(4)
    method = parameters.text("RotationMethod")
    if method == "RotationSequence":
        sequence = parameters.text("RotationSequence")
        axes = parameters.text("RotationSequenceAxes")
        if axes not in ("BaseAxes", "FollowerAxes"):
            raise ValueError(f"Unknown native rotation axes {axes}")
        if len(sequence) != 3 or any(axis not in "XYZ" for axis in sequence):
            raise ValueError(f"Invalid native rotation sequence {sequence}")
        sequence = sequence if axes == "FollowerAxes" else sequence.lower()
        angles = parameters.vector("RotationSequenceAngles", "angle", 3)
        transform[:3, :3] = Rotation.from_euler(sequence, angles).as_matrix()
    elif method != "None":
        raise ValueError(f"Unsupported native rotation method {method}")
    translation = parameters.text("TranslationMethod")
    if translation == "Cartesian":
        transform[:3, 3] = parameters.vector("TranslationCartesianOffset", "length", 3)
    elif translation != "None":
        raise ValueError(f"Unsupported native translation method {translation}")
    return transform


def _measured_pose(record: Mapping[str, Any]) -> NDArray[np.float64]:
    xyz = np.asarray(record["translation_m"], dtype=float)
    angles = np.asarray(record["rotation_intrinsic_xyz_rad"], dtype=float)
    if (
        xyz.shape != (3,)
        or angles.shape != (3,)
        or not np.all(np.isfinite([xyz, angles]))
    ):
        raise ValueError("Invalid native frame measurement")
    pose = np.eye(4)
    pose[:3, 3] = xyz
    pose[:3, :3] = Rotation.from_euler("XYZ", angles).as_matrix()
    return pose


def bind_solid_ports(probe: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    """Bind physical endpoints to all coincident named frames from native poses.

    Keep aliases when two frames are physically identical; never infer mapping
    from diagram left/right placement. This validates a pose probe, not dynamics.
    """
    if (
        probe.get("matlab_release") != "2025b"
        or probe.get("original_port_layout") is not True
    ):
        raise ValueError("Port binding requires original-layout R2025b evidence")
    groups: dict[str, dict[str, NDArray[np.float64]]] = {
        "named_frame": {},
        "physical_port": {},
    }
    for record in probe["measurements"]:
        kind, identity = record["kind"], record["identity"]
        if kind not in groups or identity in groups[kind]:
            raise ValueError("Unknown or duplicate native frame measurement")
        groups[kind][identity] = _measured_pose(record)
    if not groups["named_frame"] or not groups["physical_port"]:
        raise ValueError("Native port/frame measurements must be nonempty")
    bindings = {}
    for port, pose in groups["physical_port"].items():
        matches = tuple(
            sorted(
                name
                for name, frame in groups["named_frame"].items()
                if np.allclose(pose, frame, atol=1e-9, rtol=0)
            )
        )
        if not matches:
            raise ValueError(f"No measured frame matches native port {port}")
        bindings[port] = matches
    return bindings
