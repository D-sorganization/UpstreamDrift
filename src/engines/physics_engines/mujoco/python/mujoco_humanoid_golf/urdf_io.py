"""URDF import and export functionality for MuJoCo models.

This module provides utilities to convert between MuJoCo MJCF and URDF formats,
enabling model sharing with other robotics frameworks like ROS, Pinocchio, and Drake.

Features:
- Export MuJoCo models to URDF format
- Import URDF models into MuJoCo
- Handle joint type conversions
- Preserve inertial properties
- Convert visual and collision geometries
"""

from __future__ import annotations

import contextlib
from src.shared.python.logging_config import get_logger
import xml.etree.ElementTree as ET
from pathlib import Path

import defusedxml.ElementTree as DefusedET
import mujoco
import numpy as np

from src.shared.python.constants import GRAVITY_M_S2 as GRAVITY_STANDARD_M_S2
from src.shared.python.constants import PI, PI_HALF

logger = get_logger(__name__)

# Joint type mappings between MuJoCo and URDF
MJCF_TO_URDF_JOINT_TYPES = {
    mujoco.mjtJoint.mjJNT_HINGE: "revolute",
    mujoco.mjtJoint.mjJNT_SLIDE: "prismatic",
    mujoco.mjtJoint.mjJNT_FREE: "floating",  # Not standard URDF, handled specially
    mujoco.mjtJoint.mjJNT_BALL: "spherical",  # Not standard URDF, approximated
}

URDF_TO_MJCF_JOINT_TYPES = {
    "revolute": mujoco.mjtJoint.mjJNT_HINGE,
    "prismatic": mujoco.mjtJoint.mjJNT_SLIDE,
    "continuous": mujoco.mjtJoint.mjJNT_HINGE,  # Continuous = unlimited revolute
    "fixed": None,  # Fixed joints are handled differently in MuJoCo
    "floating": mujoco.mjtJoint.mjJNT_FREE,
}


class URDFExporter:
    """Exports MuJoCo models to URDF format."""

    def __init__(self, model: mujoco.MjModel) -> None:
        """Initialize exporter with MuJoCo model.

        Args:
            model: MuJoCo model to export
        """
        self.model = model
        self.data = mujoco.MjData(model)

    def export_to_urdf(
        self,
        output_path: str | Path,
        model_name: str | None = None,
        *,
        include_visual: bool = True,
        include_collision: bool = True,
    ) -> str:
        """Export MuJoCo model to URDF format.

        Args:
            output_path: Path to save URDF file
            model_name: Name for the robot model (defaults to model name)
            include_visual: Include visual geometries
            include_collision: Include collision geometries

        Returns:
            URDF XML string
        """
        output_path = Path(output_path)
        if model_name is None:
            with contextlib.suppress(AttributeError):
                # mjOBJ_MODEL might not be available in older MuJoCo versions
                model_name = mujoco.mj_id2name(
                    self.model,
                    mujoco.mjtObj.mjOBJ_MODEL,
                    0,
                )

        model_name = model_name or "robot"

        # Create root element
        robot = ET.Element("robot", name=model_name)

        # Build link tree from MuJoCo bodies
        self._build_urdf_tree(
            robot,
            include_visual=include_visual,
            include_collision=include_collision,
        )

        # Convert to string
        ET.indent(robot, space="  ")
        urdf_string = str(ET.tostring(robot, encoding="unicode", xml_declaration=True))

        # Save to file
        output_path.write_text(urdf_string, encoding="utf-8")
        logger.info("Exported URDF to %s", output_path)

        return urdf_string

    def _build_urdf_tree(
        self,
        robot: ET.Element,
        *,
        include_visual: bool,
        include_collision: bool,
    ) -> None:
        """Build URDF tree from MuJoCo model structure."""
        # Find root body (worldbody's first child or free joint body)
        root_body_id = self._find_root_body()

        if root_body_id is None:
            logger.warning("No root body found, creating default")
            return

        # Build link for root body
        root_link = self._create_link(
            root_body_id,
            include_visual=include_visual,
            include_collision=include_collision,
        )
        robot.append(root_link)

        # Recursively build child links and joints
        self._build_children(
            robot,
            root_body_id,
            include_visual=include_visual,
            include_collision=include_collision,
        )

    def _find_root_body(self) -> int | None:
        """Find the root body (first non-world body)."""
        # Look for bodies with free joints or first child of worldbody
        for i in range(1, self.model.nbody):  # Skip worldbody (id=0)
            # Check if this body has a free joint
            body_jntadr = self.model.body_jntadr[i]
            if body_jntadr >= 0:
                jnt_type = self.model.jnt_type[body_jntadr]
                if jnt_type == mujoco.mjtJoint.mjJNT_FREE:
                    return i

            # Check if parent is worldbody
            parent_id = self.model.body_parentid[i]
            if parent_id == 0:  # worldbody
                return i

        return None

    def _build_children(
        self,
        parent: ET.Element,
        body_id: int,
        *,
        include_visual: bool,
        include_collision: bool,
    ) -> None:
        """Recursively build child links and joints."""
        # Find children of this body
        for child_id in range(1, self.model.nbody):
            if self.model.body_parentid[child_id] == body_id:
                # Create joint between parent and child
                joint = self._create_joint(body_id, child_id)
                if joint is not None:
                    parent.append(joint)

                # Create child link
                child_link = self._create_link(
                    child_id,
                    include_visual=include_visual,
                    include_collision=include_collision,
                )
                parent.append(child_link)

                # Recursively build children
                self._build_children(
                    parent,
                    child_id,
                    include_visual=include_visual,
                    include_collision=include_collision,
                )

    def _create_link(
        self,
        body_id: int,
        *,
        include_visual: bool,
        include_collision: bool,
    ) -> ET.Element:
        """Create URDF link element from MuJoCo body."""
        body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if not body_name:
            body_name = f"link_{body_id}"

        link = ET.Element("link", name=body_name)

        # Inertial properties
        inertial = self._create_inertial(body_id)
        if inertial is not None:
            link.append(inertial)

        # Visual geometries
        if include_visual:
            visuals = self._create_visuals(body_id)
            link.extend(visuals)

        # Collision geometries
        if include_collision:
            collisions = self._create_collisions(body_id)
            link.extend(collisions)

        return link

    def _create_inertial(self, body_id: int) -> ET.Element | None:
        """Create inertial element from MuJoCo body."""
        # Get body mass and inertia
        mass = self.model.body_mass[body_id]
        if mass <= 0:
            return None

        # Get inertia matrix (in body frame)
        inertia = np.zeros((3, 3))
        inertia[0, 0] = self.model.body_inertia[body_id, 0]
        inertia[1, 1] = self.model.body_inertia[body_id, 1]
        inertia[2, 2] = self.model.body_inertia[body_id, 2]

        # Get body position (center of mass offset)
        com = self.model.body_ipos[body_id]

        inertial = ET.Element("inertial")

        # Origin (center of mass)
        origin = ET.SubElement(inertial, "origin")
        origin.set("xyz", f"{com[0]} {com[1]} {com[2]}")
        origin.set("rpy", "0 0 0")  # MuJoCo uses quaternions, URDF uses RPY

        # Mass
        mass_elem = ET.SubElement(inertial, "mass")
        mass_elem.set("value", str(mass))

        # Inertia matrix (URDF uses Ixx, Ixy, Ixz, Iyy, Iyz, Izz)
        inertia_elem = ET.SubElement(inertial, "inertia")
        inertia_elem.set("ixx", str(inertia[0, 0]))
        inertia_elem.set("ixy", str(inertia[0, 1]))
        inertia_elem.set("ixz", str(inertia[0, 2]))
        inertia_elem.set("iyy", str(inertia[1, 1]))
        inertia_elem.set("iyz", str(inertia[1, 2]))
        inertia_elem.set("izz", str(inertia[2, 2]))

        return inertial

    def _create_visuals(self, body_id: int) -> list[ET.Element]:
        """Create visual geometry elements for a body."""
        visuals = []

        # Find all geoms attached to this body
        for geom_id in range(self.model.ngeom):
            if self.model.geom_bodyid[geom_id] == body_id:
                visual = self._geom_to_visual(geom_id)
                if visual is not None:
                    visuals.append(visual)

        return visuals

    def _create_collisions(self, body_id: int) -> list[ET.Element]:
        """Create collision geometry elements for a body."""
        collisions = []

        # Find all geoms attached to this body
        for geom_id in range(self.model.ngeom):
            if self.model.geom_bodyid[geom_id] == body_id:
                collision = self._geom_to_collision(geom_id)
                if collision is not None:
                    collisions.append(collision)

        return collisions

    def _geom_to_visual(self, geom_id: int) -> ET.Element | None:
        """Convert MuJoCo geom to URDF visual element."""
        _geom_type = self.model.geom_type[geom_id]
        geom_pos = self.model.geom_pos[geom_id]
        geom_quat = self.model.geom_quat[geom_id]

        visual = ET.Element("visual")

        # Origin
        origin = ET.SubElement(visual, "origin")
        origin.set("xyz", f"{geom_pos[0]} {geom_pos[1]} {geom_pos[2]}")

        # Convert quaternion to RPY (simplified)
        rpy = self._quat_to_rpy(geom_quat)
        origin.set("rpy", f"{rpy[0]} {rpy[1]} {rpy[2]}")

        # Geometry
        geometry = ET.SubElement(visual, "geometry")
        geom_elem = self._create_geometry_element(geom_id, geometry)
        if geom_elem is None:
            return None

        # Material (if available)
        geom_matid = self.model.geom_matid[geom_id]
        if geom_matid >= 0:
            material = self._create_material(geom_matid)
            if material is not None:
                visual.append(material)

        return visual

    def _geom_to_collision(self, geom_id: int) -> ET.Element | None:
        """Convert MuJoCo geom to URDF collision element."""
        _geom_type = self.model.geom_type[geom_id]
        geom_pos = self.model.geom_pos[geom_id]
        geom_quat = self.model.geom_quat[geom_id]

        collision = ET.Element("collision")

        # Origin
        origin = ET.SubElement(collision, "origin")
        origin.set("xyz", f"{geom_pos[0]} {geom_pos[1]} {geom_pos[2]}")

        # Convert quaternion to RPY
        rpy = self._quat_to_rpy(geom_quat)
        origin.set("rpy", f"{rpy[0]} {rpy[1]} {rpy[2]}")

        # Geometry
        geometry = ET.SubElement(collision, "geometry")
        geom_elem = self._create_geometry_element(geom_id, geometry)
        if geom_elem is None:
            return None

        return collision

    def _create_geometry_element(  # noqa: PLR0911
        self,
        geom_id: int,
        parent: ET.Element,
    ) -> ET.Element | None:
        """Create geometry element (box, sphere, cylinder, mesh) from MuJoCo geom."""
        geom_type = self.model.geom_type[geom_id]
        geom_size = self.model.geom_size[geom_id]

        if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            box = ET.SubElement(parent, "box")
            box.set("size", f"{2 * geom_size[0]} {2 * geom_size[1]} {2 * geom_size[2]}")
            return box

        if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
            sphere = ET.SubElement(parent, "sphere")
            sphere.set("radius", str(geom_size[0]))
            return sphere

        if geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
            cylinder = ET.SubElement(parent, "cylinder")
            cylinder.set("radius", str(geom_size[0]))
            cylinder.set("length", str(2 * geom_size[1]))
            return cylinder

        if geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
            # URDF doesn't have capsule, approximate as cylinder
            cylinder = ET.SubElement(parent, "cylinder")
            cylinder.set("radius", str(geom_size[0]))
            cylinder.set("length", str(2 * geom_size[1]))
            return cylinder

        if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
            # Mesh geometry
            mesh_id = self.model.geom_dataid[geom_id]
            if mesh_id >= 0:
                mesh = ET.SubElement(parent, "mesh")
                # Note: URDF mesh paths are relative to package
                # This is a simplified implementation
                mesh.set("filename", f"mesh_{mesh_id}.stl")
                if geom_size[0] != 1.0 or geom_size[1] != 1.0 or geom_size[2] != 1.0:
                    mesh.set("scale", f"{geom_size[0]} {geom_size[1]} {geom_size[2]}")
                return mesh

        logger.warning("Unsupported geom type %s for URDF export", geom_type)
        return None

    def _create_material(self, mat_id: int) -> ET.Element | None:
        """Create material element from MuJoCo material."""
        mat_rgba = self.model.mat_rgba[mat_id]

        material = ET.Element("material", name=f"material_{mat_id}")
        color = ET.SubElement(material, "color")
        color.set("rgba", f"{mat_rgba[0]} {mat_rgba[1]} {mat_rgba[2]} {mat_rgba[3]}")

        return material

    def _create_joint(  # noqa: PLR0915
        self, parent_body_id: int, child_body_id: int
    ) -> ET.Element | None:
        """Create URDF joint element between two bodies."""
        # Find joint connecting parent to child
        child_jntadr = self.model.body_jntadr[child_body_id]
        if child_jntadr < 0:
            # No joint means welded body - create fixed joint for URDF
            # URDF requires every non-root link to have a joint
            parent_name = (
                mujoco.mj_id2name(
                    self.model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    parent_body_id,
                )
                or f"link_{parent_body_id}"
            )
            child_name = (
                mujoco.mj_id2name(
                    self.model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    child_body_id,
                )
                or f"link_{child_body_id}"
            )
            joint = ET.Element("joint", name=f"{parent_name}_to_{child_name}_fixed")
            joint.set("type", "fixed")
            ET.SubElement(joint, "parent", link=parent_name)
            ET.SubElement(joint, "child", link=child_name)
            return joint

        jnt_type = self.model.jnt_type[child_jntadr]
        urdf_jnt_type = MJCF_TO_URDF_JOINT_TYPES.get(jnt_type)

        if urdf_jnt_type is None:
            logger.warning("Unsupported joint type %s for URDF export", jnt_type)
            return None

        parent_name = (
            mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                parent_body_id,
            )
            or f"link_{parent_body_id}"
        )

        child_name = (
            mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                child_body_id,
            )
            or f"link_{child_body_id}"
        )

        joint_name = (
            mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                child_jntadr,
            )
            or f"joint_{child_jntadr}"
        )

        joint = ET.Element("joint", name=joint_name, type=urdf_jnt_type)

        # Parent and child links
        ET.SubElement(joint, "parent", link=parent_name)
        ET.SubElement(joint, "child", link=child_name)

        # Origin (joint position/orientation)
        origin = ET.SubElement(joint, "origin")
        joint_pos = self.model.jnt_pos[child_jntadr]
        origin.set("xyz", f"{joint_pos[0]} {joint_pos[1]} {joint_pos[2]}")

        # Joint axis
        axis = ET.SubElement(joint, "axis")
        joint_axis = self.model.jnt_axis[child_jntadr]
        axis.set("xyz", f"{joint_axis[0]} {joint_axis[1]} {joint_axis[2]}")

        # Joint limits
        if self.model.jnt_limited[child_jntadr]:
            limit = ET.SubElement(joint, "limit")
            limit.set("lower", str(self.model.jnt_range[child_jntadr, 0]))
            limit.set("upper", str(self.model.jnt_range[child_jntadr, 1]))
            limit.set("effort", "1000")  # Default effort limit
            limit.set("velocity", "10")  # Default velocity limit

        return joint

    def _quat_to_rpy(self, quat: np.ndarray) -> np.ndarray:
        """Convert quaternion (w, x, y, z) to roll-pitch-yaw."""
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        pitch = np.copysign(PI_HALF, sinp) if abs(sinp) >= 1 else np.arcsin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return np.array([roll, pitch, yaw])


class URDFImporter:
    """Imports URDF models into MuJoCo format."""

    def __init__(self) -> None:
        """Initialize URDF importer."""

    def import_from_urdf(  # noqa: PLR0915
        self,
        urdf_path: str | Path,
        model_name: str | None = None,
    ) -> str:
        """Import URDF model and convert to MuJoCo MJCF XML.

        Args:
            urdf_path: Path to URDF file
            model_name: Name for the MuJoCo model

        Returns:
            MuJoCo MJCF XML string

        Note:
            This is a basic implementation. Complex URDF features like
            transmission, gazebo plugins, etc. are not supported.
        """
        urdf_path = Path(urdf_path)
        if not urdf_path.exists():
            msg = f"URDF file not found: {urdf_path}"
            raise FileNotFoundError(msg)

        tree = DefusedET.parse(urdf_path)
        root = tree.getroot()

        model_name = str(
            model_name or root.get("name", "imported_robot") or "imported_robot"
        )

        # Build MuJoCo XML
        mujoco_root = ET.Element("mujoco", model=model_name)

        # Compiler options
        compiler = ET.SubElement(mujoco_root, "compiler")
        compiler.set("angle", "radian")
        compiler.set("coordinate", "local")
        compiler.set("inertiafromgeom", "false")

        # Options
        option = ET.SubElement(mujoco_root, "option")
        option.set("timestep", "0.001")
        # Use standard gravity constant (NIST reference: 9.80665 m/s²)
        option.set("gravity", f"0 0 -{GRAVITY_STANDARD_M_S2}")
        option.set("integrator", "RK4")

        # Default settings
        default = ET.SubElement(mujoco_root, "default")
        geom_default = ET.SubElement(default, "geom")
        geom_default.set("friction", "0.9 0.005 0.0001")
        joint_default = ET.SubElement(default, "joint")
        joint_default.set("damping", "0.5")
        joint_default.set("armature", "0.01")

        # Worldbody
        worldbody = ET.SubElement(mujoco_root, "worldbody")
        floor = ET.SubElement(worldbody, "geom", name="floor", type="plane")
        floor.set("size", "10 10 0.1")
        floor.set("rgba", "0.8 0.8 0.8 1")

        # Parse URDF links and joints
        links: dict[str, ET.Element] = {
            str(link.get("name")): link
            for link in root.findall("link")
            if link.get("name") is not None
        }
        joints = list(root.findall("joint"))

        # Find root link (link not referenced as child in any joint)
        child_links: set[str] = set()
        for joint in joints:
            child_elem = joint.find("child")
            if child_elem is not None:
                child_link_name = child_elem.get("link")
                if child_link_name is not None:
                    child_links.add(str(child_link_name))
        root_link_name = next(
            (name for name in links if name not in child_links),
            None,
        )

        if root_link_name:
            self._build_mujoco_body(
                worldbody,
                links[root_link_name],
                links,
                joints,
                root_link_name,
            )

        # Convert to string
        ET.indent(mujoco_root, space="  ")
        mujoco_xml = str(
            ET.tostring(mujoco_root, encoding="unicode", xml_declaration=True)
        )

        logger.info("Imported URDF from %s", urdf_path)

        return mujoco_xml

    def _build_mujoco_body(  # noqa: C901,PLR0913,PLR0912,PLR0915
        self,
        parent: ET.Element,
        link: ET.Element,
        links: dict[str, ET.Element],
        joints: list[ET.Element],
        link_name: str,
        visited: set[str] | None = None,
    ) -> None:
        """Recursively build MuJoCo body structure from URDF."""
        if visited is None:
            visited = set()

        if link_name in visited:
            return  # Avoid cycles
        visited.add(link_name)

        # Create body element
        body = ET.SubElement(parent, "body", name=link_name)

        # Add inertial properties
        inertial = link.find("inertial")
        if inertial is not None:
            self._add_inertial(body, inertial)

        # Add visual geometries
        for visual in link.findall("visual"):
            self._add_visual_geom(body, visual)

        # Add collision geometries
        for collision in link.findall("collision"):
            self._add_collision_geom(body, collision)

        # Find child joints
        for joint in joints:
            parent_elem = joint.find("parent")
            if parent_elem is None:
                continue
            parent_link_name = parent_elem.get("link")
            if parent_link_name != link_name:
                continue

            child_elem = joint.find("child")
            if child_elem is None:
                continue
            child_link_name_raw = child_elem.get("link")
            if child_link_name_raw is None:
                continue
            child_link_name = str(child_link_name_raw)
            child_link = links.get(child_link_name)

            if child_link is not None:
                # Create child body first
                child_body = ET.SubElement(body, "body", name=child_link_name)
                # Set body position from joint origin
                # (URDF joint origin specifies child body position relative to parent)
                origin = joint.find("origin")
                if origin is not None:
                    xyz = origin.get("xyz", "0 0 0").split()
                    child_body.set("pos", f"{xyz[0]} {xyz[1]} {xyz[2]}")
                # Add joint to child body (MuJoCo requires joints in child body)
                self._add_joint(child_body, joint)
                # Add inertial properties to child
                inertial = child_link.find("inertial")
                if inertial is not None:
                    self._add_inertial(child_body, inertial)
                # Add visual geometries to child
                for visual in child_link.findall("visual"):
                    self._add_visual_geom(child_body, visual)
                # Add collision geometries to child
                for collision in child_link.findall("collision"):
                    self._add_collision_geom(child_body, collision)
                # Recursively build grandchildren
                for grandchild_joint in joints:
                    grandchild_parent_elem = grandchild_joint.find("parent")
                    if grandchild_parent_elem is None:
                        continue
                    grandchild_parent_link_name = grandchild_parent_elem.get("link")
                    if grandchild_parent_link_name != child_link_name:
                        continue

                    grandchild_child_elem = grandchild_joint.find("child")
                    if grandchild_child_elem is None:
                        continue
                    grandchild_link_name_raw = grandchild_child_elem.get("link")
                    if grandchild_link_name_raw is None:
                        continue
                    grandchild_link_name = str(grandchild_link_name_raw)
                    grandchild_link = links.get(grandchild_link_name)
                    if grandchild_link is not None:
                        self._build_mujoco_body(
                            child_body,
                            grandchild_link,
                            links,
                            joints,
                            grandchild_link_name,
                            visited,
                        )

    def _add_inertial(self, body: ET.Element, inertial: ET.Element) -> None:
        """Add inertial properties to MuJoCo body."""
        inertial_elem = ET.SubElement(body, "inertial")

        # Origin (center of mass position)
        origin = inertial.find("origin")
        if origin is not None:
            xyz = origin.get("xyz", "0 0 0").split()
            inertial_elem.set("pos", f"{xyz[0]} {xyz[1]} {xyz[2]}")
        else:
            # Default to zero if origin not specified
            inertial_elem.set("pos", "0 0 0")

        # Mass
        mass_elem = inertial.find("mass")
        if mass_elem is not None:
            inertial_elem.set("mass", mass_elem.get("value", "1.0"))
        else:
            inertial_elem.set("mass", "1.0")

        # Inertia matrix
        inertia_elem = inertial.find("inertia")
        if inertia_elem is not None:
            ixx = inertia_elem.get("ixx", "0.001")
            iyy = inertia_elem.get("iyy", "0.001")
            izz = inertia_elem.get("izz", "0.001")
            # Check for off-diagonal elements
            ixy = inertia_elem.get("ixy", "0.0")
            ixz = inertia_elem.get("ixz", "0.0")
            iyz = inertia_elem.get("iyz", "0.0")

            # Use fullinertia if off-diagonal terms present, else use diaginertia
            has_off_diagonal = (
                float(ixy) != 0.0 or float(ixz) != 0.0 or float(iyz) != 0.0
            )
            if has_off_diagonal:
                # MuJoCo fullinertia format: "ixx iyy izz ixy ixz iyz"
                inertial_elem.set(
                    "fullinertia",
                    f"{ixx} {iyy} {izz} {ixy} {ixz} {iyz}",
                )
            else:
                # Use diaginertia for diagonal-only inertia (more efficient)
                inertial_elem.set("diaginertia", f"{ixx} {iyy} {izz}")
        else:
            # Default inertia if not specified
            inertial_elem.set("diaginertia", "0.001 0.001 0.001")

    def _add_visual_geom(self, body: ET.Element, visual: ET.Element) -> None:
        """Add visual geometry to MuJoCo body."""
        geom = ET.SubElement(body, "geom", type="box")  # Default type

        origin = visual.find("origin")
        if origin is not None:
            xyz = origin.get("xyz", "0 0 0").split()
            geom.set("pos", f"{xyz[0]} {xyz[1]} {xyz[2]}")

        geometry = visual.find("geometry")
        if geometry is not None:
            self._parse_geometry(geom, geometry)

        material = visual.find("material")
        if material is not None:
            color = material.find("color")
            if color is not None:
                rgba = color.get("rgba", "0.5 0.5 0.5 1").split()
                geom.set("rgba", f"{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}")

    def _add_collision_geom(self, body: ET.Element, collision: ET.Element) -> None:
        """Add collision geometry to MuJoCo body."""
        geom = ET.SubElement(body, "geom", type="box")  # Default type
        geom.set("contype", "1")
        geom.set("conaffinity", "1")

        origin = collision.find("origin")
        if origin is not None:
            xyz = origin.get("xyz", "0 0 0").split()
            geom.set("pos", f"{xyz[0]} {xyz[1]} {xyz[2]}")

        geometry = collision.find("geometry")
        if geometry is not None:
            self._parse_geometry(geom, geometry)

    def _parse_geometry(self, geom: ET.Element, geometry: ET.Element) -> None:
        """Parse URDF geometry element and set MuJoCo geom properties."""
        box = geometry.find("box")
        if box is not None:
            size = box.get("size", "0.1 0.1 0.1").split()
            geom.set("type", "box")
            geom.set(
                "size",
                f"{float(size[0]) / 2} {float(size[1]) / 2} {float(size[2]) / 2}",
            )
            return

        sphere = geometry.find("sphere")
        if sphere is not None:
            radius = sphere.get("radius", "0.05")
            geom.set("type", "sphere")
            geom.set("size", radius)
            return

        cylinder = geometry.find("cylinder")
        if cylinder is not None:
            radius = cylinder.get("radius", "0.05")
            length = cylinder.get("length", "0.1")
            geom.set("type", "cylinder")
            geom.set("size", f"{radius} {float(length) / 2}")
            return

        mesh = geometry.find("mesh")
        if mesh is not None:
            filename = mesh.get("filename", "")
            geom.set("type", "mesh")
            geom.set("mesh", filename)
            scale = mesh.get("scale")
            if scale:
                geom.set("size", scale)

    def _add_joint(self, body: ET.Element, joint: ET.Element) -> None:
        """Add joint to MuJoCo body."""
        joint_type = joint.get("type", "revolute")
        mjcf_type = URDF_TO_MJCF_JOINT_TYPES.get(joint_type)

        if mjcf_type is None:
            logger.warning("Unsupported joint type %s, skipping", joint_type)
            return

        joint_elem = ET.SubElement(body, "joint")
        joint_elem.set("name", joint.get("name", "joint"))
        # Map MJCF joint types to URDF joint types
        if mjcf_type == mujoco.mjtJoint.mjJNT_HINGE:
            joint_elem.set("type", "hinge")
        elif mjcf_type == mujoco.mjtJoint.mjJNT_SLIDE:
            joint_elem.set("type", "slide")
        elif mjcf_type == mujoco.mjtJoint.mjJNT_FREE:
            joint_elem.set("type", "free")
        else:
            # Default to hinge for unknown types
            joint_elem.set("type", "hinge")

        # Note: Joint origin is handled in _build_mujoco_body (sets body position)
        # Joint pos attribute not used here as URDF joint origin specifies body pos

        axis = joint.find("axis")
        if axis is not None:
            xyz = axis.get("xyz", "0 0 1").split()
            joint_elem.set("axis", f"{xyz[0]} {xyz[1]} {xyz[2]}")

        limit = joint.find("limit")
        # Continuous joints are unlimited revolute joints - skip position limits
        # The <limit> element for continuous joints only specifies effort/velocity
        if limit is not None and joint_type != "continuous":
            # Use PI constant for default joint limits
            lower = limit.get("lower", str(-PI))
            upper = limit.get("upper", str(PI))
            joint_elem.set("range", f"{lower} {upper}")


def export_model_to_urdf(
    model: mujoco.MjModel,
    output_path: str | Path,
    model_name: str | None = None,
    *,
    include_visual: bool = True,
    include_collision: bool = True,
) -> str:
    """Convenience function to export MuJoCo model to URDF.

    Args:
        model: MuJoCo model to export
        output_path: Path to save URDF file
        model_name: Name for the robot model
        include_visual: Include visual geometries
        include_collision: Include collision geometries

    Returns:
        URDF XML string

    Example:
        >>> import mujoco
        >>> from mujoco_humanoid_golf.urdf_io import export_model_to_urdf
        >>> model = mujoco.MjModel.from_xml_string(xml_string)
        >>> urdf_xml = export_model_to_urdf(model, "robot.urdf")
    """
    exporter = URDFExporter(model)
    return exporter.export_to_urdf(
        output_path,
        model_name,
        include_visual=include_visual,
        include_collision=include_collision,
    )


def import_urdf_to_mujoco(
    urdf_path: str | Path,
    model_name: str | None = None,
) -> str:
    """Convenience function to import URDF model to MuJoCo MJCF.

    Args:
        urdf_path: Path to URDF file
        model_name: Name for the MuJoCo model

    Returns:
        MuJoCo MJCF XML string

    Example:
        >>> from mujoco_humanoid_golf.urdf_io import import_urdf_to_mujoco
        >>> import mujoco
        >>> mujoco_xml = import_urdf_to_mujoco("robot.urdf")
        >>> model = mujoco.MjModel.from_xml_string(mujoco_xml)
    """
    importer = URDFImporter()
    return importer.import_from_urdf(urdf_path, model_name)
