"""
SimScape MDL/SLX file parser.

Parses MATLAB Simulink/SimScape model files to extract
multibody system definitions.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class SimscapeBlockType(Enum):
    """SimScape Multibody block types."""

    # Bodies
    RIGID_BODY = "RigidBody"
    SOLID = "Solid"
    INERTIA = "Inertia"
    BRICK_SOLID = "BrickSolid"
    CYLINDER_SOLID = "CylinderSolid"
    SPHERE_SOLID = "SphereSolid"

    # Joints
    REVOLUTE_JOINT = "RevoluteJoint"
    PRISMATIC_JOINT = "PrismaticJoint"
    WELD_JOINT = "WeldJoint"
    SPHERICAL_JOINT = "SphericalJoint"
    UNIVERSAL_JOINT = "UniversalJoint"
    CYLINDRICAL_JOINT = "CylindricalJoint"
    PLANAR_JOINT = "PlanarJoint"
    GIMBAL_JOINT = "GimbalJoint"
    BUSHING_JOINT = "BushingJoint"
    SIX_DOF_JOINT = "6DOFJoint"

    # Frames
    RIGID_TRANSFORM = "RigidTransform"
    WORLD_FRAME = "WorldFrame"
    REFERENCE_FRAME = "ReferenceFrame"

    # Constraints
    POINT_ON_CURVE = "PointOnCurve"
    GEAR_CONSTRAINT = "GearConstraint"

    # Other
    MECHANISM_CONFIG = "MechanismConfiguration"
    SOLVER_CONFIG = "SolverConfiguration"
    SUBSYSTEM = "Subsystem"

    UNKNOWN = "Unknown"


@dataclass
class SimscapeParameter:
    """A parameter in a SimScape block."""

    name: str
    value: str
    unit: str | None = None
    evaluated_value: float | None = None

    def as_float(self, default: float = 0.0) -> float:
        """Convert to float value."""
        if self.evaluated_value is not None:
            return self.evaluated_value

        try:
            # Try to parse numeric value
            # Handle MATLAB expressions like [1 2 3]
            val = self.value.strip()
            if val.startswith("[") and val.endswith("]"):
                # Vector - return first element or magnitude
                nums = [float(x) for x in val[1:-1].split()]
                return nums[0] if len(nums) == 1 else (sum(x**2 for x in nums) ** 0.5)
            return float(val)
        except (ValueError, TypeError):
            return default

    def as_vector(
        self, default: tuple[float, ...] = (0.0, 0.0, 0.0)
    ) -> tuple[float, ...]:
        """Convert to vector value."""
        try:
            val = self.value.strip()
            if val.startswith("[") and val.endswith("]"):
                return tuple(float(x) for x in val[1:-1].split())
            return (float(val),)
        except (ValueError, TypeError):
            return default


@dataclass
class SimscapePort:
    """A connection port on a SimScape block."""

    name: str
    port_type: str  # 'frame', 'signal', 'conserving'
    index: int = 0


@dataclass
class SimscapeConnection:
    """A connection between SimScape blocks."""

    source_block: str
    source_port: str
    dest_block: str
    dest_port: str


@dataclass
class SimscapeBlock:
    """A block in a SimScape model."""

    name: str
    block_type: SimscapeBlockType
    full_path: str
    parameters: dict[str, SimscapeParameter] = field(default_factory=dict)
    ports: list[SimscapePort] = field(default_factory=list)
    position: tuple[int, int, int, int] | None = None  # [x, y, width, height]
    parent: str | None = None

    def get_param(
        self,
        name: str,
        default: str = "",
    ) -> str:
        """Get parameter value as string."""
        param = self.parameters.get(name)
        return param.value if param else default

    def get_param_float(self, name: str, default: float = 0.0) -> float:
        """Get parameter value as float."""
        param = self.parameters.get(name)
        return param.as_float(default) if param else default

    def get_param_vector(
        self,
        name: str,
        default: tuple[float, ...] = (0.0, 0.0, 0.0),
    ) -> tuple[float, ...]:
        """Get parameter value as vector."""
        param = self.parameters.get(name)
        return param.as_vector(default) if param else default


@dataclass
class SimscapeModel:
    """Parsed SimScape model."""

    name: str
    source_path: Path | None
    blocks: dict[str, SimscapeBlock] = field(default_factory=dict)
    connections: list[SimscapeConnection] = field(default_factory=list)
    solver_config: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def get_blocks_by_type(self, block_type: SimscapeBlockType) -> list[SimscapeBlock]:
        """Get all blocks of a specific type."""
        return [b for b in self.blocks.values() if b.block_type == block_type]

    def get_body_blocks(self) -> list[SimscapeBlock]:
        """Get all body/solid blocks."""
        body_types = {
            SimscapeBlockType.RIGID_BODY,
            SimscapeBlockType.SOLID,
            SimscapeBlockType.BRICK_SOLID,
            SimscapeBlockType.CYLINDER_SOLID,
            SimscapeBlockType.SPHERE_SOLID,
            SimscapeBlockType.INERTIA,
        }
        return [b for b in self.blocks.values() if b.block_type in body_types]

    def get_joint_blocks(self) -> list[SimscapeBlock]:
        """Get all joint blocks."""
        joint_types = {
            SimscapeBlockType.REVOLUTE_JOINT,
            SimscapeBlockType.PRISMATIC_JOINT,
            SimscapeBlockType.WELD_JOINT,
            SimscapeBlockType.SPHERICAL_JOINT,
            SimscapeBlockType.UNIVERSAL_JOINT,
            SimscapeBlockType.CYLINDRICAL_JOINT,
            SimscapeBlockType.PLANAR_JOINT,
            SimscapeBlockType.GIMBAL_JOINT,
            SimscapeBlockType.BUSHING_JOINT,
            SimscapeBlockType.SIX_DOF_JOINT,
        }
        return [b for b in self.blocks.values() if b.block_type in joint_types]

    def get_transform_blocks(self) -> list[SimscapeBlock]:
        """Get all rigid transform blocks."""
        return [
            b
            for b in self.blocks.values()
            if b.block_type == SimscapeBlockType.RIGID_TRANSFORM
        ]

    def get_connections_to(self, block_name: str) -> list[SimscapeConnection]:
        """Get all connections to a block."""
        return [c for c in self.connections if c.dest_block == block_name]

    def get_connections_from(self, block_name: str) -> list[SimscapeConnection]:
        """Get all connections from a block."""
        return [c for c in self.connections if c.source_block == block_name]


class MDLParser:
    """
    Parser for SimScape MDL/SLX files.

    Supports both:
    - MDL: Legacy text-based format
    - SLX: Modern XML-based format (ZIP archive)

    Example:
        parser = MDLParser()
        model = parser.parse("/path/to/robot.slx")

        for body in model.get_body_blocks():
            print(f"Body: {body.name}")
    """

    # Block type mapping from SimScape names
    BLOCK_TYPE_MAP = {
        "sm_lib/Body Elements/Brick Solid": SimscapeBlockType.BRICK_SOLID,
        "sm_lib/Body Elements/Cylindrical Solid": SimscapeBlockType.CYLINDER_SOLID,
        "sm_lib/Body Elements/Sphere Solid": SimscapeBlockType.SPHERE_SOLID,
        "sm_lib/Body Elements/Solid": SimscapeBlockType.SOLID,
        "sm_lib/Body Elements/Inertia": SimscapeBlockType.INERTIA,
        "sm_lib/Joints/Revolute Joint": SimscapeBlockType.REVOLUTE_JOINT,
        "sm_lib/Joints/Prismatic Joint": SimscapeBlockType.PRISMATIC_JOINT,
        "sm_lib/Joints/Weld Joint": SimscapeBlockType.WELD_JOINT,
        "sm_lib/Joints/Spherical Joint": SimscapeBlockType.SPHERICAL_JOINT,
        "sm_lib/Joints/Universal Joint": SimscapeBlockType.UNIVERSAL_JOINT,
        "sm_lib/Joints/Cylindrical Joint": SimscapeBlockType.CYLINDRICAL_JOINT,
        "sm_lib/Joints/Planar Joint": SimscapeBlockType.PLANAR_JOINT,
        "sm_lib/Joints/Gimbal Joint": SimscapeBlockType.GIMBAL_JOINT,
        "sm_lib/Joints/Bushing Joint": SimscapeBlockType.BUSHING_JOINT,
        "sm_lib/Joints/6-DOF Joint": SimscapeBlockType.SIX_DOF_JOINT,
        "sm_lib/Frames and Transforms/Rigid Transform": SimscapeBlockType.RIGID_TRANSFORM,
        "sm_lib/Frames and Transforms/World Frame": SimscapeBlockType.WORLD_FRAME,
        "sm_lib/Frames and Transforms/Reference Frame": SimscapeBlockType.REFERENCE_FRAME,
        "sm_lib/Utilities/Mechanism Configuration": SimscapeBlockType.MECHANISM_CONFIG,
        "sm_lib/Utilities/Solver Configuration": SimscapeBlockType.SOLVER_CONFIG,
        "simulink/Ports & Subsystems/Subsystem": SimscapeBlockType.SUBSYSTEM,
    }

    def __init__(self):
        """Initialize parser."""
        self._current_path: list[str] = []

    def parse(self, source: str | Path) -> SimscapeModel:
        """
        Parse a SimScape model file.

        Args:
            source: Path to MDL or SLX file

        Returns:
            Parsed SimscapeModel
        """
        source_path = Path(source)
        if not source_path.exists():
            raise FileNotFoundError(f"File not found: {source_path}")

        suffix = source_path.suffix.lower()

        if suffix == ".slx":
            return self._parse_slx(source_path)
        elif suffix == ".mdl":
            return self._parse_mdl(source_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    def _parse_slx(self, path: Path) -> SimscapeModel:
        """Parse SLX (ZIP/XML) format."""
        logger.info(f"Parsing SLX file: {path}")

        model = SimscapeModel(
            name=path.stem,
            source_path=path,
        )

        try:
            with zipfile.ZipFile(path, "r") as zf:
                # Find the model XML file
                model_file = None
                for name in zf.namelist():
                    if (
                        name.endswith("blockdiagram.xml")
                        or name == "simulink/blockdiagram.xml"
                    ):
                        model_file = name
                        break

                if not model_file:
                    # Try to find any XML file
                    xml_files = [n for n in zf.namelist() if n.endswith(".xml")]
                    if xml_files:
                        model_file = xml_files[0]

                if not model_file:
                    model.warnings.append("Could not find model XML in SLX archive")
                    return model

                # Parse the model XML
                with zf.open(model_file) as f:
                    self._parse_slx_xml(f, model)

        except zipfile.BadZipFile:
            raise ValueError(f"Invalid SLX file: {path}")

        return model

    def _parse_slx_xml(self, file, model: SimscapeModel) -> None:
        """Parse SLX model XML content."""
        try:
            tree = ET.parse(file)
            root = tree.getroot()
        except ET.ParseError as e:
            model.warnings.append(f"XML parse error: {e}")
            return

        # Find model name
        model_info = root.find(".//Model")
        if model_info is not None:
            name = model_info.get("Name")
            if name:
                model.name = name

        # Parse blocks
        self._parse_slx_blocks(root, model)

        # Parse connections (lines)
        self._parse_slx_connections(root, model)

    def _parse_slx_blocks(
        self,
        root: ET.Element,
        model: SimscapeModel,
        parent_path: str = "",
    ) -> None:
        """Parse blocks from SLX XML."""
        for block_elem in root.findall(".//Block"):
            block_name = block_elem.get("Name", "")
            block_type_str = block_elem.get("BlockType", "")
            source_block = block_elem.get("SourceBlock", "")

            # Determine block type
            block_type = self._get_block_type(block_type_str, source_block)

            full_path = f"{parent_path}/{block_name}" if parent_path else block_name

            block = SimscapeBlock(
                name=block_name,
                block_type=block_type,
                full_path=full_path,
                parent=parent_path if parent_path else None,
            )

            # Parse parameters
            for param_elem in block_elem.findall(".//P"):
                param_name = param_elem.get("Name", "")
                param_value = param_elem.text or ""
                block.parameters[param_name] = SimscapeParameter(
                    name=param_name,
                    value=param_value,
                )

            # Parse ports
            for port_elem in block_elem.findall(".//Port"):
                port = SimscapePort(
                    name=port_elem.get("Name", ""),
                    port_type=port_elem.get("Type", "signal"),
                    index=int(port_elem.get("Index", 0)),
                )
                block.ports.append(port)

            model.blocks[full_path] = block

            # Parse subsystems recursively
            if block_type == SimscapeBlockType.SUBSYSTEM:
                system_elem = block_elem.find("System")
                if system_elem is not None:
                    self._parse_slx_blocks(system_elem, model, full_path)

    def _parse_slx_connections(self, root: ET.Element, model: SimscapeModel) -> None:
        """Parse connections (lines) from SLX XML."""
        for line_elem in root.findall(".//Line"):
            src_block = ""
            src_port = ""
            dst_block = ""
            dst_port = ""

            # Get source
            src_elem = line_elem.find("P[@Name='Src']")
            if src_elem is not None and src_elem.text:
                parts = src_elem.text.split("#")
                if len(parts) >= 2:
                    src_block = parts[0]
                    src_port = parts[1]

            # Get destination
            dst_elem = line_elem.find("P[@Name='Dst']")
            if dst_elem is not None and dst_elem.text:
                parts = dst_elem.text.split("#")
                if len(parts) >= 2:
                    dst_block = parts[0]
                    dst_port = parts[1]

            if src_block and dst_block:
                model.connections.append(
                    SimscapeConnection(
                        source_block=src_block,
                        source_port=src_port,
                        dest_block=dst_block,
                        dest_port=dst_port,
                    )
                )

    def _parse_mdl(self, path: Path) -> SimscapeModel:
        """Parse MDL (text) format."""
        logger.info(f"Parsing MDL file: {path}")

        model = SimscapeModel(
            name=path.stem,
            source_path=path,
        )

        content = path.read_text(errors="ignore")
        self._parse_mdl_content(content, model)

        return model

    def _parse_mdl_content(self, content: str, model: SimscapeModel) -> None:
        """Parse MDL text content."""
        # Find Model name
        match = re.search(r'Name\s+"([^"]+)"', content)
        if match:
            model.name = match.group(1)

        # Parse blocks using regex (simplified parsing)
        block_pattern = re.compile(
            r"Block\s*\{\s*" r'BlockType\s+"?(\w+)"?\s*' r'Name\s+"([^"]+)"',
            re.MULTILINE | re.DOTALL,
        )

        for match in block_pattern.finditer(content):
            block_type_str = match.group(1)
            block_name = match.group(2)

            # Get block content for parameter parsing
            start = match.end()
            brace_count = 1
            end = start
            while end < len(content) and brace_count > 0:
                if content[end] == "{":
                    brace_count += 1
                elif content[end] == "}":
                    brace_count -= 1
                end += 1

            block_content = content[start:end]

            block_type = self._get_block_type(block_type_str, "")

            block = SimscapeBlock(
                name=block_name,
                block_type=block_type,
                full_path=block_name,
            )

            # Parse parameters from block content
            param_pattern = re.compile(r'(\w+)\s+"([^"]*)"')
            for pmatch in param_pattern.finditer(block_content):
                param_name = pmatch.group(1)
                param_value = pmatch.group(2)
                block.parameters[param_name] = SimscapeParameter(
                    name=param_name,
                    value=param_value,
                )

            model.blocks[block_name] = block

        # Parse lines (connections) - simplified
        line_pattern = re.compile(
            r"Line\s*\{\s*"
            r'SrcBlock\s+"([^"]+)"\s*'
            r'SrcPort\s+"?(\d+)"?\s*'
            r'DstBlock\s+"([^"]+)"\s*'
            r'DstPort\s+"?(\d+)"?',
            re.MULTILINE,
        )

        for match in line_pattern.finditer(content):
            model.connections.append(
                SimscapeConnection(
                    source_block=match.group(1),
                    source_port=match.group(2),
                    dest_block=match.group(3),
                    dest_port=match.group(4),
                )
            )

    def _get_block_type(
        self, block_type_str: str, source_block: str
    ) -> SimscapeBlockType:
        """Determine SimscapeBlockType from strings."""
        # Check source block mapping first
        if source_block:
            for pattern, block_type in self.BLOCK_TYPE_MAP.items():
                if pattern in source_block:
                    return block_type

        # Check block type string
        type_lower = block_type_str.lower()

        type_mapping = {
            "revolute": SimscapeBlockType.REVOLUTE_JOINT,
            "prismatic": SimscapeBlockType.PRISMATIC_JOINT,
            "weld": SimscapeBlockType.WELD_JOINT,
            "spherical": SimscapeBlockType.SPHERICAL_JOINT,
            "universal": SimscapeBlockType.UNIVERSAL_JOINT,
            "cylindrical": SimscapeBlockType.CYLINDRICAL_JOINT,
            "planar": SimscapeBlockType.PLANAR_JOINT,
            "gimbal": SimscapeBlockType.GIMBAL_JOINT,
            "bushing": SimscapeBlockType.BUSHING_JOINT,
            "6dof": SimscapeBlockType.SIX_DOF_JOINT,
            "sixdof": SimscapeBlockType.SIX_DOF_JOINT,
            "brick": SimscapeBlockType.BRICK_SOLID,
            "cylinder": SimscapeBlockType.CYLINDER_SOLID,
            "sphere": SimscapeBlockType.SPHERE_SOLID,
            "solid": SimscapeBlockType.SOLID,
            "inertia": SimscapeBlockType.INERTIA,
            "rigidbody": SimscapeBlockType.RIGID_BODY,
            "rigidtransform": SimscapeBlockType.RIGID_TRANSFORM,
            "transform": SimscapeBlockType.RIGID_TRANSFORM,
            "worldframe": SimscapeBlockType.WORLD_FRAME,
            "world": SimscapeBlockType.WORLD_FRAME,
            "reference": SimscapeBlockType.REFERENCE_FRAME,
            "subsystem": SimscapeBlockType.SUBSYSTEM,
            "mechanism": SimscapeBlockType.MECHANISM_CONFIG,
            "solver": SimscapeBlockType.SOLVER_CONFIG,
        }

        for pattern, block_type in type_mapping.items():
            if pattern in type_lower:
                return block_type

        return SimscapeBlockType.UNKNOWN

    def parse_string(self, content: str, format: str = "mdl") -> SimscapeModel:
        """
        Parse model from string content.

        Args:
            content: Model content as string
            format: Format ('mdl' or 'xml')

        Returns:
            Parsed model
        """
        model = SimscapeModel(name="unnamed", source_path=None)

        if format.lower() == "xml":
            # Parse as SLX XML
            try:
                root = ET.fromstring(content)
                self._parse_slx_blocks(root, model)
                self._parse_slx_connections(root, model)
            except ET.ParseError as e:
                model.warnings.append(f"XML parse error: {e}")
        else:
            # Parse as MDL text
            self._parse_mdl_content(content, model)

        return model
