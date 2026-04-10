"""URDF parsing: import URDF models into MuJoCo MJCF format."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import defusedxml.ElementTree as DefusedET

from src.shared.python.core.constants import GRAVITY_M_S2 as GRAVITY_STANDARD_M_S2
from src.shared.python.core.constants import PI
from src.shared.python.logging_pkg.logging_config import get_logger

from .urdf_constants import URDF_TO_MJCF_JOINT_TYPES

logger = get_logger(__name__)


class URDFImporter:
    """Imports URDF models into MuJoCo format."""

    def __init__(self) -> None:
        """Initialize URDF importer."""

    @staticmethod
    def _create_mjcf_skeleton(model_name: str) -> tuple[ET.Element, ET.Element]:
        """Create the base MuJoCo MJCF XML structure with defaults.

        Args:
            model_name: Name for the MuJoCo model.

        Returns:
            Tuple of (mujoco_root, worldbody) elements.
        """
        mujoco_root = ET.Element("mujoco", model=model_name)

        compiler = ET.SubElement(mujoco_root, "compiler")
        compiler.set("angle", "radian")
        compiler.set("coordinate", "local")
        compiler.set("inertiafromgeom", "false")

        option = ET.SubElement(mujoco_root, "option")
        option.set("timestep", "0.001")
        option.set("gravity", f"0 0 -{GRAVITY_STANDARD_M_S2}")
        option.set("integrator", "RK4")

        default = ET.SubElement(mujoco_root, "default")
        geom_default = ET.SubElement(default, "geom")
        geom_default.set("friction", "0.9 0.005 0.0001")
        joint_default = ET.SubElement(default, "joint")
        joint_default.set("damping", "0.5")
        joint_default.set("armature", "0.01")

        worldbody = ET.SubElement(mujoco_root, "worldbody")
        floor = ET.SubElement(worldbody, "geom", name="floor", type="plane")
        floor.set("size", "10 10 0.1")
        floor.set("rgba", "0.8 0.8 0.8 1")

        return mujoco_root, worldbody

    def import_from_urdf(
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

        mujoco_root, worldbody = self._create_mjcf_skeleton(model_name)

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

        ET.indent(mujoco_root, space="  ")
        mujoco_xml = str(
            ET.tostring(mujoco_root, encoding="unicode", xml_declaration=True)
        )

        logger.info("Imported URDF from %s", urdf_path)
        return mujoco_xml

    def _populate_body_geometry(self, body: ET.Element, link: ET.Element) -> None:
        """Add inertial, visual, and collision geometry to a MuJoCo body element.

        Args:
            body: MuJoCo XML body element to populate.
            link: URDF link element with geometry definitions.
        """
        if not (body is not None):
            raise ValueError("body must be provided")
        inertial = link.find("inertial")
        if inertial is not None:
            self._add_inertial(body, inertial)
        for visual in link.findall("visual"):
            self._add_visual_geom(body, visual)
        for collision in link.findall("collision"):
            self._add_collision_geom(body, collision)

    def _find_child_links(
        self, joints: list[ET.Element], parent_link_name: str
    ) -> list[tuple[ET.Element, str]]:
        """Find all joints whose parent is the given link name.

        Args:
            joints: All URDF joint elements.
            parent_link_name: Name of the parent link.

        Returns:
            List of (joint_element, child_link_name) pairs.
        """
        if not (joints is not None):
            raise ValueError("joints must be provided")
        children = []
        for joint in joints:
            parent_elem = joint.find("parent")
            if parent_elem is None or parent_elem.get("link") != parent_link_name:
                continue
            child_elem = joint.find("child")
            if child_elem is None:
                continue
            child_link_name_raw = child_elem.get("link")
            if child_link_name_raw is not None:
                children.append((joint, str(child_link_name_raw)))
        return children

    def _build_mujoco_body(  # noqa: PLR0913
        self,
        parent: ET.Element,
        link: ET.Element,
        links: dict[str, ET.Element],
        joints: list[ET.Element],
        link_name: str,
        visited: set[str] | None = None,
    ) -> None:
        """Recursively build MuJoCo body structure from URDF."""
        if not (parent is not None):
            raise ValueError("parent must be provided")
        if visited is None:
            visited = set()

        if link_name in visited:
            return  # Avoid cycles
        visited.add(link_name)

        # Create body element and populate its geometry
        body = ET.SubElement(parent, "body", name=link_name)
        self._populate_body_geometry(body, link)

        # Process child joints
        for joint, child_link_name in self._find_child_links(joints, link_name):
            child_link = links.get(child_link_name)
            if child_link is None:
                continue

            child_body = ET.SubElement(body, "body", name=child_link_name)
            origin = joint.find("origin")
            if origin is not None:
                xyz = origin.get("xyz", "0 0 0").split()
                child_body.set("pos", f"{xyz[0]} {xyz[1]} {xyz[2]}")

            self._add_joint(child_body, joint)
            self._populate_body_geometry(child_body, child_link)

            # Recursively build grandchildren
            for _grandchild_joint, gc_link_name in self._find_child_links(
                joints, child_link_name
            ):
                grandchild_link = links.get(gc_link_name)
                if grandchild_link is not None:
                    self._build_mujoco_body(
                        child_body,
                        grandchild_link,
                        links,
                        joints,
                        gc_link_name,
                        visited,
                    )

    def _add_inertial(self, body: ET.Element, inertial: ET.Element) -> None:
        """Add inertial properties to MuJoCo body."""
        if not (body is not None):
            raise ValueError("body must be provided")
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
        if not (body is not None):
            raise ValueError("body must be provided")
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
        if not (body is not None):
            raise ValueError("body must be provided")
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
        if not (geom is not None):
            raise ValueError("geom must be provided")
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
        if not (body is not None):
            raise ValueError("body must be provided")
        # Import mujoco here to avoid circular dependency at module level
        import mujoco  # noqa: PLC0415

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
