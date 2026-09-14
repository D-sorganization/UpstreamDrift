"""Preserve native solids and scalar joint sequences in an MJCF tree.

The exported weld is compliant in stock MuJoCo. NativeMujocoModel instead
enforces the same six-dimensional closure with an explicit rigid solve.
"""

import hashlib
import json
import xml.etree.ElementTree as ET  # nosec B405 # nosemgrep: python.lang.security.use-defused-xml.use-defused-xml - construction only; parsing is defused
from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from src.shared.python.motion_matching.full_body_spec import (
    order_directed_tree as order_native_tree,
)


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


def _add_weld_equality(
    root: ET.Element,
    name: str = "native_grip",
    site1: str = "native_closure_a",
    site2: str = "native_closure_b",
) -> ET.Element:
    """Attach weld equality element between two sites."""
    return ET.SubElement(
        ET.SubElement(root, "equality"),
        "weld",
        name=name,
        site1=site1,
        site2=site2,
    )


def _attach_joint_element(
    parent_element: ET.Element,
    child_body: dict[str, Any],
    joint: Mapping[str, Any],
    parent_offset: np.ndarray,
    coordinates: list[str],
) -> tuple[ET.Element, np.ndarray]:
    """Attach a rigid body child element, its scalar 1-DOF joints, and inertial data."""
    offset = np.linalg.inv(transform(joint["child_to_follower"]))
    element = ET.SubElement(
        parent_element,
        "body",
        {
            "name": joint["child"],
            **_pose(parent_offset @ transform(joint["parent_to_base"])),
        },
    )
    for primitive in joint["primitives"]:
        kind, name = primitive["primitive"], primitive["coordinate"]
        if kind not in ("Px", "Py", "Pz", "Rx", "Ry", "Rz") or name in coordinates:
            raise ValueError(f"Duplicate or unsupported coordinate: {name}")
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
    ET.SubElement(element, "inertial", _inertia(child_body, offset))
    return element, offset


def _attach_frame_site(
    element: ET.Element,
    site_name: str,
    body_offset: np.ndarray,
    placement: Any,
) -> ET.Element:
    """Attach a marker frame site with relative rigid pose."""
    return ET.SubElement(
        element,
        "site",
        {
            "name": site_name,
            "size": ".001",
            **_pose(body_offset @ transform(placement)),
        },
    )


def _attach_native_sites_and_weld(
    root: ET.Element,
    elements: Mapping[str, ET.Element],
    offsets: Mapping[str, np.ndarray],
    spec: Mapping[str, Any],
) -> dict[str, str]:
    """Attach marker frame sites, closure sites, and weld constraint."""
    sites: dict[str, str] = {}
    for i, frame in enumerate(spec["frames"]):
        if frame["name"] in sites:
            raise ValueError("Duplicate native frame")
        site = f"native_frame_{i}"
        sites[frame["name"]] = site
        _attach_frame_site(
            elements[frame["body"]], site, offsets[frame["body"]], frame["placement"]
        )
    for suffix in ("a", "b"):
        body = spec["closure"][f"body_{suffix}"]
        _attach_frame_site(
            elements[body],
            f"native_closure_{suffix}",
            offsets[body],
            spec["closure"][f"placement_{suffix}"],
        )
    _add_weld_equality(root)
    return sites


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
    coordinates: list[str] = []
    for joint in order_native_tree(spec["joints"]):
        parent, child = joint["parent"], joint["child"]
        element, offset = _attach_joint_element(
            elements[parent], bodies[child], joint, offsets[parent], coordinates
        )
        elements[child], offsets[child] = element, offset
    if (
        set(elements) != set(bodies)
        or len(coordinates) != len(spec["coordinate_order"])
        or set(coordinates) != set(spec["coordinate_order"])
    ):
        raise ValueError("Native body or coordinate inventory was not preserved")

    sites = _attach_native_sites_and_weld(root, elements, offsets, spec)
    xml = ET.tostring(root, encoding="unicode")
    return xml, {
        "model_sha256": hashlib.sha256(model_bytes).hexdigest(),
        "mjcf_sha256": hashlib.sha256(xml.encode()).hexdigest(),
        "frame_sites": sites,
        "coordinate_order": spec["coordinate_order"],
        "execution": "explicit-rigid-closure; stock mj_step unqualified",
    }
