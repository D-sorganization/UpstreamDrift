"""URDF import and export functionality for MuJoCo models.

This module provides utilities to convert between MuJoCo MJCF and URDF formats,
enabling model sharing with other robotics frameworks like ROS, Pinocchio, and Drake.

Features:
- Export MuJoCo models to URDF format
- Import URDF models into MuJoCo
- Handle joint type conversions
- Preserve inertial properties
- Convert visual and collision geometries

Implementation split:
- urdf_constants.py  — shared joint-type mapping tables
- urdf_generator.py  — geometry/math helper functions for URDFExporter
- urdf_parser.py     — URDFImporter: URDF → MuJoCo MJCF
"""

from __future__ import annotations

import contextlib
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from src.shared.python.logging_pkg.logging_config import get_logger

from .urdf_constants import MJCF_TO_URDF_JOINT_TYPES, URDF_TO_MJCF_JOINT_TYPES
from .urdf_generator import build_geom_elements, create_inertial_element, quat_to_rpy
from .urdf_parser import URDFImporter

logger = get_logger(__name__)

__all__ = [
    "MJCF_TO_URDF_JOINT_TYPES",
    "URDF_TO_MJCF_JOINT_TYPES",
    "URDFExporter",
    "URDFImporter",
    "export_model_to_urdf",
    "import_urdf_to_mujoco",
]


class URDFExporter:
    """Exports MuJoCo models to URDF format."""

    def __init__(self, model: mujoco.MjModel) -> None:
        """Initialize exporter with MuJoCo model.

        Args:
            model: MuJoCo model to export
        """
        if model is None:
            raise ValueError("model must be provided")
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
        if output_path is None:
            raise ValueError("output_path must be provided")
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
        robot = ET.Element("robot", name=model_name)
        self._build_urdf_tree(
            robot,
            include_visual=include_visual,
            include_collision=include_collision,
        )
        ET.indent(robot, space="  ")
        urdf_string = str(ET.tostring(robot, encoding="unicode", xml_declaration=True))
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
        if robot is None:
            raise ValueError("robot must be provided")
        root_body_id = self._find_root_body()
        if root_body_id is None:
            logger.warning("No root body found, creating default")
            return
        root_link = self._create_link(
            root_body_id,
            include_visual=include_visual,
            include_collision=include_collision,
        )
        robot.append(root_link)
        self._build_children(
            robot,
            root_body_id,
            include_visual=include_visual,
            include_collision=include_collision,
        )

    def _find_root_body(self) -> int | None:
        """Find the root body (first non-world body)."""
        for i in range(1, self.model.nbody):  # Skip worldbody (id=0)
            body_jntadr = self.model.body_jntadr[i]
            if body_jntadr >= 0:
                jnt_type = self.model.jnt_type[body_jntadr]
                if jnt_type == mujoco.mjtJoint.mjJNT_FREE:
                    return i
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
        for child_id in range(1, self.model.nbody):
            if self.model.body_parentid[child_id] == body_id:
                joint = self._create_joint(body_id, child_id)
                if joint is not None:
                    parent.append(joint)
                child_link = self._create_link(
                    child_id,
                    include_visual=include_visual,
                    include_collision=include_collision,
                )
                parent.append(child_link)
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
        if body_id is None:
            raise ValueError("body_id must be provided")
        body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if not body_name:
            body_name = f"link_{body_id}"

        link = ET.Element("link", name=body_name)
        inertial = create_inertial_element(self.model, body_id)
        if inertial is not None:
            link.append(inertial)
        if include_visual:
            link.extend(build_geom_elements(self.model, body_id, "visual"))
        if include_collision:
            link.extend(build_geom_elements(self.model, body_id, "collision"))
        return link

    def _create_joint(  # noqa: PLR0912
        self, parent_body_id: int, child_body_id: int
    ) -> ET.Element | None:
        """Create URDF joint element between two bodies."""
        if parent_body_id is None:
            raise ValueError("parent_body_id must be provided")
        child_jntadr = self.model.body_jntadr[child_body_id]
        if child_jntadr < 0:
            # No joint means welded body — create fixed joint for URDF
            parent_name = (
                mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, parent_body_id)
                or f"link_{parent_body_id}"
            )
            child_name = (
                mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, child_body_id)
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
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, parent_body_id)
            or f"link_{parent_body_id}"
        )
        child_name = (
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, child_body_id)
            or f"link_{child_body_id}"
        )
        joint_name = (
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, child_jntadr)
            or f"joint_{child_jntadr}"
        )

        joint = ET.Element("joint", name=joint_name, type=urdf_jnt_type)
        ET.SubElement(joint, "parent", link=parent_name)
        ET.SubElement(joint, "child", link=child_name)

        origin = ET.SubElement(joint, "origin")
        joint_pos = self.model.jnt_pos[child_jntadr]
        origin.set("xyz", f"{joint_pos[0]} {joint_pos[1]} {joint_pos[2]}")

        axis = ET.SubElement(joint, "axis")
        joint_axis = self.model.jnt_axis[child_jntadr]
        axis.set("xyz", f"{joint_axis[0]} {joint_axis[1]} {joint_axis[2]}")

        if self.model.jnt_limited[child_jntadr]:
            limit = ET.SubElement(joint, "limit")
            limit.set("lower", str(self.model.jnt_range[child_jntadr, 0]))
            limit.set("upper", str(self.model.jnt_range[child_jntadr, 1]))
            limit.set("effort", "1000")
            limit.set("velocity", "10")

        return joint

    # Keep as method for backward compatibility with any code that calls it directly
    def _quat_to_rpy(self, quat: np.ndarray) -> np.ndarray:
        """Convert quaternion (w, x, y, z) to roll-pitch-yaw."""
        return quat_to_rpy(quat)


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
    if model is None:
        raise ValueError("model must be provided")
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
    if urdf_path is None:
        raise ValueError("urdf_path must be provided")
    importer = URDFImporter()
    return importer.import_from_urdf(urdf_path, model_name)
