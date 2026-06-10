"""Headless composition flow for model explorer UI controllers."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET  # stdlib retained for Element/SubElement
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from src.tools.model_explorer._mujoco_viewer_backend import URDFToMJCFConverter
from src.tools.model_explorer.composition_validator import CompositionValidationResult

if TYPE_CHECKING:
    from src.tools.model_explorer.frankenstein_editor.model import URDFModel

ExportFormat = Literal["urdf", "mjcf"]


class CompositionFlowError(ValueError):
    """Raised when a composition flow cannot be completed."""


@dataclass(frozen=True)
class AttachmentSelection:
    """A target mount point selected by the composition UI."""

    target_link: str
    attachment_name: str | None = None
    interface_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    interface_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)
    source_prefix: str = ""
    joint_type: str = "fixed"


@dataclass(frozen=True)
class CompositionResult:
    """Result of attaching one source model into a target model."""

    source_root_link: str
    mapped_root_link: str
    attachment_joint: str
    validation: CompositionValidationResult


@dataclass(frozen=True)
class ExportedComposition:
    """Serialized composed model content."""

    format: ExportFormat
    content: str
    validation: CompositionValidationResult


class CompositionFlowController:
    """Controller for library-to-Frankenstein model composition workflows."""

    def attach_source_model(
        self,
        *,
        target_model: URDFModel,
        source_model: URDFModel,
        selection: AttachmentSelection,
    ) -> CompositionResult:
        """Attach a complete source model under a target attachment point."""
        if target_model is None:
            raise ValueError("target_model must be provided")
        if source_model is None:
            raise ValueError("source_model must be provided")
        if selection is None:
            raise ValueError("selection must be provided")
        if selection.target_link not in target_model.links:
            raise CompositionFlowError(f"unknown target link: {selection.target_link}")
        if selection.joint_type != "fixed":
            raise CompositionFlowError(
                "model composition currently supports fixed joints"
            )

        source_root = _single_root_link(source_model)
        name_mapping = self._copy_source_into_target(
            target_model=target_model,
            source_model=source_model,
            source_prefix=selection.source_prefix,
        )
        mapped_root = name_mapping[source_root]
        joint_name = self._add_attachment_joint(
            target_model=target_model,
            target_link=selection.target_link,
            mapped_root=mapped_root,
            selection=selection,
        )
        validation = target_model.validate_composition()
        return CompositionResult(
            source_root_link=source_root,
            mapped_root_link=mapped_root,
            attachment_joint=joint_name,
            validation=validation,
        )

    def export_model(
        self,
        model: URDFModel,
        *,
        export_format: ExportFormat,
        force: bool = False,
    ) -> ExportedComposition:
        """Serialize a composed model, refusing validation errors unless forced."""
        if model is None:
            raise ValueError("model must be provided")
        if force is None:
            raise ValueError("force must be provided")
        if export_format not in ("urdf", "mjcf"):
            raise CompositionFlowError(f"unsupported export format: {export_format}")

        validation = model.validate_composition()
        if validation.errors and not force:
            messages = "; ".join(finding.message for finding in validation.errors)
            raise CompositionFlowError(f"composition has validation errors: {messages}")

        urdf = model.to_xml(force=force)
        if export_format == "urdf":
            return ExportedComposition("urdf", urdf, validation)
        return ExportedComposition(
            "mjcf", URDFToMJCFConverter.convert(urdf), validation
        )

    def _copy_source_into_target(
        self,
        *,
        target_model: URDFModel,
        source_model: URDFModel,
        source_prefix: str,
    ) -> dict[str, str]:
        for material in source_model.materials.values():
            target_model.add_material(material)

        name_mapping: dict[str, str] = {}
        for name, link in source_model.links.items():
            requested_name = f"{source_prefix}{name}" if source_prefix else None
            name_mapping[name] = target_model.add_link(link, requested_name)

        for name, joint in source_model.joints.items():
            requested_name = f"{source_prefix}{name}" if source_prefix else None
            target_model.add_joint(joint, requested_name, name_mapping)

        return name_mapping

    def _add_attachment_joint(
        self,
        *,
        target_model: URDFModel,
        target_link: str,
        mapped_root: str,
        selection: AttachmentSelection,
    ) -> str:
        joint = ET.Element(
            "joint",
            name=f"attach_{_safe_name(target_link)}_{_safe_name(mapped_root)}",
            type=selection.joint_type,
        )
        ET.SubElement(joint, "parent", link=target_link)
        ET.SubElement(joint, "child", link=mapped_root)
        ET.SubElement(
            joint,
            "origin",
            xyz=_format_triplet(selection.interface_xyz),
            rpy=_format_triplet(selection.interface_rpy),
        )
        return target_model.add_joint(joint)


def selection_from_attachment_point(
    attachment_point: dict[str, object],
    *,
    source_prefix: str = "",
) -> AttachmentSelection:
    """Build an attachment selection from model-library sidecar metadata."""
    if attachment_point is None:
        raise ValueError("attachment_point must be provided")
    target_link = str(attachment_point.get("link_name", "")).strip()
    if not target_link:
        raise CompositionFlowError("attachment point is missing link_name")
    frame = attachment_point.get("interface_frame")
    frame_dict = frame if isinstance(frame, dict) else {}
    return AttachmentSelection(
        target_link=target_link,
        attachment_name=str(attachment_point.get("name", "")).strip() or None,
        interface_xyz=_triplet(frame_dict.get("xyz")),
        interface_rpy=_triplet(frame_dict.get("rpy")),
        source_prefix=source_prefix,
    )


def _single_root_link(model: URDFModel) -> str:
    child_links = set()
    for joint in model.joints.values():
        child = joint.find("child")
        if child is not None and child.get("link"):
            child_links.add(child.get("link"))
    roots = [name for name in model.links if name not in child_links]
    if len(roots) != 1:
        raise CompositionFlowError(
            f"source model must have exactly one root link; found {len(roots)}"
        )
    return roots[0]


def _triplet(raw: object) -> tuple[float, float, float]:
    if not isinstance(raw, list | tuple) or len(raw) != 3:
        return (0.0, 0.0, 0.0)
    try:
        return (float(raw[0]), float(raw[1]), float(raw[2]))
    except (TypeError, ValueError):
        return (0.0, 0.0, 0.0)


def _format_triplet(values: tuple[float, float, float]) -> str:
    return f"{values[0]} {values[1]} {values[2]}"


def _safe_name(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_]+", "_", value.strip())
    return normalized.strip("_") or "link"
