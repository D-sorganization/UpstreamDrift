"""Preserve native solids and scalar joint sequences in an MJCF tree.

The exported weld is compliant in stock MuJoCo. NativeMujocoModel instead
enforces the same six-dimensional closure with an explicit rigid solve.
"""

import hashlib
import json
import xml.etree.ElementTree as ET
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from src.shared.python.motion_matching.native_spec import order_native_tree


def transform(value: Any) -> np.ndarray:
    """Validate a proper finite homogeneous rigid transform."""
    matrix = np.asarray(value, dtype=float)
    if (
        matrix.shape != (4, 4)
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-12, rtol=0)
        or not np.allclose(
            matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-12, rtol=0
        )
        or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-12, rtol=0)
    ):
        raise ValueError("Invalid native rigid transform")
    return matrix


def _numbers(values: Any) -> str:
    return " ".join(format(float(x), ".17g") for x in np.asarray(values).ravel())


def _pose(matrix: np.ndarray) -> dict[str, str]:
    xyzw = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    return {"pos": _numbers(matrix[:3, 3]), "quat": _numbers(xyzw[[3, 0, 1, 2]])}


def _inertia(body: dict, follower: np.ndarray) -> dict[str, str]:
    masses, centers, tensors = [], [], []
    for solid in body["solids"]:
        pose = follower @ transform(solid["placement"])
        mass = float(solid["mass_kg"])
        com = np.asarray(solid["com_m"], dtype=float)
        inertia = np.asarray(solid["inertia_com_kg_m2"], dtype=float)
        if (
            mass < 0
            or not np.isfinite(mass)
            or com.shape != (3,)
            or inertia.shape != (3, 3)
            or not np.isfinite(com).all()
            or not np.isfinite(inertia).all()
            or not np.allclose(inertia, inertia.T, atol=1e-14, rtol=0)
            or np.linalg.eigvalsh(inertia).min() < 0
            or (mass == 0 and np.any(inertia))
        ):
            raise ValueError("Invalid native solid inertia")
        if mass == 0:
            continue
        masses.append(mass)
        centers.append(pose[:3, :3] @ com + pose[:3, 3])
        tensors.append(pose[:3, :3] @ inertia @ pose[:3, :3].T)
    total = sum(masses)
    if total <= 0:
        raise ValueError("Moving native body needs physical inertia")
    center = np.asarray(masses) @ np.asarray(centers) / total
    tensor = np.zeros((3, 3))
    for mass, point, inertia in zip(masses, centers, tensors, strict=True):
        d = point - center
        tensor += inertia + mass * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
    return {
        "mass": _numbers([total]),
        "pos": _numbers(center),
        "fullinertia": _numbers(tensor[[0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]]),
    }


def export_native_mjcf(model_bytes: bytes) -> tuple[str, dict[str, Any]]:
    """Export source-bound native geometry; never add helper mass or limits."""
    spec = json.loads(model_bytes)
    if spec.get("schema_version") != 1 or not isinstance(spec.get("closure"), dict):
        raise ValueError("Unsupported native schema or missing closure")
    root = ET.Element("mujoco", model="native_golf")
    ET.SubElement(
        root, "compiler", angle="radian", autolimits="false", inertiafromgeom="false"
    )
    gravity = np.asarray(spec["gravity_m_s2"], dtype=float)
    if gravity.shape != (3,) or not np.isfinite(gravity).all():
        raise ValueError("Invalid native gravity")
    option = ET.SubElement(root, "option", gravity=_numbers(gravity), jacobian="dense")
    ET.SubElement(option, "flag", contact="disable")
    world = ET.SubElement(root, "worldbody")
    bodies = {body["name"]: body for body in spec["bodies"]}
    if len(bodies) != len(spec["bodies"]) or "world" not in bodies:
        raise ValueError("Invalid native body inventory")
    if bodies["world"]["solids"]:
        raise ValueError("World solids require explicit static inertia support")
    elements, offsets = {"world": world}, {"world": np.eye(4)}
    coordinates = []
    for joint in order_native_tree(spec["joints"]):
        parent, child = joint["parent"], joint["child"]
        offset = np.linalg.inv(transform(joint["child_to_follower"]))
        element = ET.SubElement(
            elements[parent],
            "body",
            {
                "name": child,
                **_pose(offsets[parent] @ transform(joint["parent_to_base"])),
            },
        )
        for primitive in joint["primitives"]:
            kind, name = primitive["primitive"], primitive["coordinate"]
            if kind not in ("Px", "Py", "Pz", "Rx", "Ry", "Rz") or name in coordinates:
                raise ValueError("Duplicate or unsupported native coordinate")
            coordinates.append(name)
            ET.SubElement(
                element,
                "joint",
                name=name,
                type="slide" if kind[0] == "P" else "hinge",
                axis=_numbers(np.eye(3)["xyz".index(kind[1])]),
                limited="false",
                damping="0",
                armature="0",
                frictionloss="0",
                stiffness="0",
            )
        ET.SubElement(element, "inertial", _inertia(bodies[child], offset))
        elements[child], offsets[child] = element, offset
    if (
        set(elements) != set(bodies)
        or len(coordinates) != len(spec["coordinate_order"])
        or set(coordinates) != set(spec["coordinate_order"])
    ):
        raise ValueError("Native body or coordinate inventory was not preserved")
    sites = {}
    for i, frame in enumerate(spec["frames"]):
        if frame["name"] in sites:
            raise ValueError("Duplicate native frame")
        site = f"native_frame_{i}"
        sites[frame["name"]] = site
        ET.SubElement(
            elements[frame["body"]],
            "site",
            {
                "name": site,
                "size": ".001",
                **_pose(offsets[frame["body"]] @ transform(frame["placement"])),
            },
        )
    for suffix in ("a", "b"):
        body = spec["closure"][f"body_{suffix}"]
        ET.SubElement(
            elements[body],
            "site",
            {
                "name": f"native_closure_{suffix}",
                "size": ".001",
                **_pose(
                    offsets[body] @ transform(spec["closure"][f"placement_{suffix}"])
                ),
            },
        )
    ET.SubElement(
        ET.SubElement(root, "equality"),
        "weld",
        name="native_grip",
        site1="native_closure_a",
        site2="native_closure_b",
    )
    xml = ET.tostring(root, encoding="unicode")
    return xml, {
        "model_sha256": hashlib.sha256(model_bytes).hexdigest(),
        "mjcf_sha256": hashlib.sha256(xml.encode()).hexdigest(),
        "frame_sites": sites,
        "coordinate_order": spec["coordinate_order"],
        "execution": "explicit-rigid-closure; stock mj_step unqualified",
    }
