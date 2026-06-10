from __future__ import annotations

import copy
import xml.etree.ElementTree as ET  # stdlib retained for Element/SubElement
from dataclasses import dataclass, field
from pathlib import Path

import defusedxml.ElementTree as DefusedET  # noqa: S314  # Security: defusedxml prevents XML attacks

from src.tools.model_explorer.attachment_metadata import (
    AttachmentManifest,
    attachment_warnings_for_link,
    load_attachment_manifest,
)
from src.tools.model_explorer.composition_validator import (
    CompositionFinding,
    CompositionValidationResult,
    CompositionValidator,
)


@dataclass
class URDFModel:
    """Represents a loaded URDF model."""

    file_path: Path | None
    robot_name: str
    links: dict[str, ET.Element]
    joints: dict[str, ET.Element]
    materials: dict[str, ET.Element]
    other_elements: list[ET.Element]
    is_modified: bool = False
    attachment_manifest: AttachmentManifest = field(default_factory=AttachmentManifest)

    @classmethod
    def from_file(cls, file_path: Path) -> URDFModel:
        """Load a URDF model from file."""
        if file_path is None:
            raise ValueError("file_path must be provided")
        tree = DefusedET.parse(file_path)
        root = tree.getroot()
        model = cls.from_element(root, file_path)
        model.attachment_manifest = load_attachment_manifest(
            file_path,
            known_links=set(model.links),
        )
        return model

    @classmethod
    def from_element(cls, root: ET.Element, file_path: Path | None = None) -> URDFModel:
        """Create model from XML element."""
        if root is None:
            raise ValueError("root must be provided")
        robot_name = root.get("name", "unnamed_robot")

        links = {}
        joints = {}
        materials = {}
        other_elements = []

        for child in root:
            name = child.get("name", "")
            if child.tag == "link":
                links[name] = child
            elif child.tag == "joint":
                joints[name] = child
            elif child.tag == "material":
                materials[name] = child
            else:
                other_elements.append(child)

        return cls(
            file_path=file_path,
            robot_name=robot_name,
            links=links,
            joints=joints,
            materials=materials,
            other_elements=other_elements,
        )

    @classmethod
    def create_empty(cls, name: str = "new_robot") -> URDFModel:
        """Create an empty URDF model."""
        return cls(
            file_path=None,
            robot_name=name,
            links={},
            joints={},
            materials={},
            other_elements=[],
        )

    def validate_composition(self) -> CompositionValidationResult:
        """Validate this composed model before export."""
        return CompositionValidator().validate_model(self)

    def to_xml(self, *, force: bool = False) -> str:
        """Convert model to XML string."""
        if force is None:
            raise ValueError("force must be provided")
        validation = self.validate_composition()
        if not force:
            validation.raise_for_errors()

        root = ET.Element("robot", name=self.robot_name)

        # Add materials first
        root.extend([copy.deepcopy(material) for material in self.materials.values()])

        # Add links
        root.extend([copy.deepcopy(link) for link in self.links.values()])

        # Add joints
        root.extend([copy.deepcopy(joint) for joint in self.joints.values()])

        # Add other elements
        root.extend([copy.deepcopy(elem) for elem in self.other_elements])

        ET.indent(root, space="  ")
        return ET.tostring(root, encoding="unicode", xml_declaration=True)

    def add_link(self, link: ET.Element, new_name: str | None = None) -> str:
        """Add a link to the model.

        Args:
            link: Link element to add
            new_name: Optional new name for the link

        Returns:
            The name used for the link
        """
        if link is None:
            raise ValueError("link must be provided")
        link_copy = copy.deepcopy(link)
        name = new_name or link_copy.get("name") or "unnamed_link"

        # Ensure unique name
        original_name = name
        counter = 1
        while name in self.links:
            name = f"{original_name}_{counter}"
            counter += 1

        link_copy.set("name", name)
        self.links[name] = link_copy
        self.is_modified = True
        return name

    def add_joint(
        self,
        joint: ET.Element,
        new_name: str | None = None,
        parent_mapping: dict[str, str] | None = None,
    ) -> str:
        """Add a joint to the model.

        Args:
            joint: Joint element to add
            new_name: Optional new name for the joint
            parent_mapping: Optional mapping from old link names to new ones

        Returns:
            The name used for the joint
        """
        if joint is None:
            raise ValueError("joint must be provided")
        joint_copy = copy.deepcopy(joint)
        name = new_name or joint_copy.get("name") or "unnamed_joint"

        # Ensure unique name
        original_name = name
        counter = 1
        while name in self.joints:
            name = f"{original_name}_{counter}"
            counter += 1

        joint_copy.set("name", name)

        # Update parent/child references if mapping provided
        if parent_mapping:
            parent = joint_copy.find("parent")
            if parent is not None:
                old_link = parent.get("link", "")
                if old_link in parent_mapping:
                    parent.set("link", parent_mapping[old_link])

            child = joint_copy.find("child")
            if child is not None:
                old_link = child.get("link", "")
                if old_link in parent_mapping:
                    child.set("link", parent_mapping[old_link])

        self.joints[name] = joint_copy
        self.is_modified = True
        return name

    def add_attachment_joint(
        self,
        *,
        parent_link: str,
        child_link: str,
        joint_name: str = "attachment_joint",
        payload_kg: float | None = None,
    ) -> tuple[str, tuple[CompositionFinding, ...]]:
        """Add a fixed attachment joint using declared interface-frame metadata."""
        if parent_link is None:
            raise ValueError("parent_link must be provided")
        if child_link is None:
            raise ValueError("child_link must be provided")

        point = self.attachment_manifest.get(parent_link)
        joint = ET.Element("joint", {"name": joint_name, "type": "fixed"})
        ET.SubElement(joint, "parent", {"link": parent_link})
        ET.SubElement(joint, "child", {"link": child_link})
        if point is not None:
            ET.SubElement(
                joint,
                "origin",
                {
                    "xyz": _format_vector(point.interface_frame.xyz),
                    "rpy": _format_vector(point.interface_frame.rpy),
                },
            )

        added_name = self.add_joint(joint)
        warnings = self.attachment_warnings(parent_link, payload_kg=payload_kg)
        return added_name, warnings

    def attachment_warnings(
        self,
        link_name: str,
        *,
        payload_kg: float | None = None,
    ) -> tuple[CompositionFinding, ...]:
        """Return non-blocking editor warnings for an attachment target."""
        messages = attachment_warnings_for_link(
            self.attachment_manifest,
            link_name,
            payload_kg=payload_kg,
        )
        return tuple(
            CompositionFinding(
                code="attachment_metadata_warning",
                severity="warning",
                message=message,
                elements=(link_name,),
                category="attachment",
            )
            for message in messages
        )

    def add_material(self, material: ET.Element) -> str:
        """Add a material to the model."""
        if material is None:
            raise ValueError("material must be provided")
        material_copy = copy.deepcopy(material)
        name = material_copy.get("name", "unnamed_material")

        if name not in self.materials:
            self.materials[name] = material_copy
            self.is_modified = True

        return name

    def remove_link(self, name: str) -> bool:
        """Remove a link and its connected joints."""
        if name is None:
            raise ValueError("name must be provided")
        if name not in self.links:
            return False

        del self.links[name]

        # Remove joints connected to this link
        joints_to_remove = []
        for joint_name, joint in self.joints.items():
            parent = joint.find("parent")
            child = joint.find("child")
            if (parent is not None and parent.get("link") == name) or (
                child is not None and child.get("link") == name
            ):
                joints_to_remove.append(joint_name)

        for joint_name in joints_to_remove:
            del self.joints[joint_name]

        self.is_modified = True
        return True

    def remove_joint(self, name: str) -> bool:
        """Remove a joint."""
        if name is None:
            raise ValueError("name must be provided")
        if name not in self.joints:
            return False

        del self.joints[name]
        self.is_modified = True
        return True

    def get_link_names(self) -> list[str]:
        """Get list of link names."""
        return list(self.links.keys())

    def get_joint_names(self) -> list[str]:
        """Get list of joint names."""
        return list(self.joints.keys())


def _format_vector(values: tuple[float, float, float]) -> str:
    return " ".join(f"{value:g}" for value in values)
