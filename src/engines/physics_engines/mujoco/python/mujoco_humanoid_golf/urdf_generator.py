"""URDF geometry helpers used by URDFExporter.

Stand-alone functions for converting MuJoCo geometry and math primitives
to URDF XML elements.  Imported by ``urdf_io.URDFExporter``.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from src.shared.python.core.constants import PI_HALF
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)


def quat_to_rpy(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion (w, x, y, z) to roll-pitch-yaw.

    Args:
        quat: Array of shape (4,) with components [w, x, y, z].

    Returns:
        Array [roll, pitch, yaw] in radians.
    """
    if quat is None:
        raise ValueError("quat must be provided")
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]

    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    pitch = np.copysign(PI_HALF, sinp) if abs(sinp) >= 1 else np.arcsin(sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw])


def create_geometry_element(  # noqa: PLR0911
    model: mujoco.MjModel,
    geom_id: int,
    parent: ET.Element,
) -> ET.Element | None:
    """Create a URDF geometry child element inside *parent*.

    Supports box, sphere, cylinder, capsule (mapped to cylinder), and mesh.
    Returns ``None`` for unsupported geometry types.

    Args:
        model: MuJoCo model containing the geometry data.
        geom_id: Index of the geom in *model*.
        parent: ``<geometry>`` XML element to attach the new child to.
    """
    if geom_id is None:
        raise ValueError("geom_id must be provided")
    geom_type = model.geom_type[geom_id]
    geom_size = model.geom_size[geom_id]

    if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
        box = ET.SubElement(parent, "box")
        box.set("size", f"{2 * geom_size[0]} {2 * geom_size[1]} {2 * geom_size[2]}")
        return box

    if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
        sphere = ET.SubElement(parent, "sphere")
        sphere.set("radius", str(geom_size[0]))
        return sphere

    if geom_type in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE):
        # URDF doesn't have capsule, approximate as cylinder
        cylinder = ET.SubElement(parent, "cylinder")
        cylinder.set("radius", str(geom_size[0]))
        cylinder.set("length", str(2 * geom_size[1]))
        return cylinder

    if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
        mesh_id = model.geom_dataid[geom_id]
        if mesh_id >= 0:
            mesh = ET.SubElement(parent, "mesh")
            mesh.set("filename", f"mesh_{mesh_id}.stl")
            if geom_size[0] != 1.0 or geom_size[1] != 1.0 or geom_size[2] != 1.0:
                mesh.set("scale", f"{geom_size[0]} {geom_size[1]} {geom_size[2]}")
            return mesh

    logger.warning("Unsupported geom type %s for URDF export", geom_type)
    return None


def create_material_element(model: mujoco.MjModel, mat_id: int) -> ET.Element | None:
    """Create a URDF ``<material>`` element from a MuJoCo material index.

    Args:
        model: MuJoCo model containing material RGBA data.
        mat_id: Index of the material in *model*.
    """
    if mat_id is None:
        raise ValueError("mat_id must be provided")
    mat_rgba = model.mat_rgba[mat_id]
    material = ET.Element("material", name=f"material_{mat_id}")
    color = ET.SubElement(material, "color")
    color.set("rgba", f"{mat_rgba[0]} {mat_rgba[1]} {mat_rgba[2]} {mat_rgba[3]}")
    return material


def create_inertial_element(model: mujoco.MjModel, body_id: int) -> ET.Element | None:
    """Create a URDF ``<inertial>`` element from MuJoCo body inertial data.

    Returns ``None`` if the body has zero or negative mass.

    Args:
        model: MuJoCo model.
        body_id: Index of the body in *model*.
    """
    if body_id is None:
        raise ValueError("body_id must be provided")
    mass = model.body_mass[body_id]
    if mass <= 0:
        return None

    inertia = np.zeros((3, 3))
    inertia[0, 0] = model.body_inertia[body_id, 0]
    inertia[1, 1] = model.body_inertia[body_id, 1]
    inertia[2, 2] = model.body_inertia[body_id, 2]
    com = model.body_ipos[body_id]

    inertial = ET.Element("inertial")
    origin = ET.SubElement(inertial, "origin")
    origin.set("xyz", f"{com[0]} {com[1]} {com[2]}")
    origin.set("rpy", "0 0 0")
    mass_elem = ET.SubElement(inertial, "mass")
    mass_elem.set("value", str(mass))
    inertia_elem = ET.SubElement(inertial, "inertia")
    inertia_elem.set("ixx", str(inertia[0, 0]))
    inertia_elem.set("ixy", str(inertia[0, 1]))
    inertia_elem.set("ixz", str(inertia[0, 2]))
    inertia_elem.set("iyy", str(inertia[1, 1]))
    inertia_elem.set("iyz", str(inertia[1, 2]))
    inertia_elem.set("izz", str(inertia[2, 2]))
    return inertial


def build_geom_elements(
    model: mujoco.MjModel, body_id: int, elem_tag: str
) -> list[ET.Element]:
    """Build a list of ``<visual>`` or ``<collision>`` elements for *body_id*.

    Args:
        model: MuJoCo model.
        body_id: Index of the body whose geoms should be converted.
        elem_tag: Either ``"visual"`` or ``"collision"``.
    """
    if body_id is None:
        raise ValueError("body_id must be provided")
    results = []
    for geom_id in range(model.ngeom):
        if model.geom_bodyid[geom_id] != body_id:
            continue
        geom_pos = model.geom_pos[geom_id]
        geom_quat = model.geom_quat[geom_id]
        elem = ET.Element(elem_tag)
        origin = ET.SubElement(elem, "origin")
        origin.set("xyz", f"{geom_pos[0]} {geom_pos[1]} {geom_pos[2]}")
        rpy = quat_to_rpy(geom_quat)
        origin.set("rpy", f"{rpy[0]} {rpy[1]} {rpy[2]}")
        geometry = ET.SubElement(elem, "geometry")
        if create_geometry_element(model, geom_id, geometry) is None:
            continue
        if elem_tag == "visual":
            geom_matid = model.geom_matid[geom_id]
            if geom_matid >= 0:
                material = create_material_element(model, geom_matid)
                if material is not None:
                    elem.append(material)
        results.append(elem)
    return results
