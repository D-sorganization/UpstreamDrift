"""Headless UI orchestration for model-explorer composition workflows."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from src.tools.model_explorer.composition_flow import (
    AttachmentSelection,
    CompositionFlowController,
    CompositionFlowError,
    CompositionResult,
    ExportFormat,
)
from src.tools.model_explorer.composition_validator import (
    CompositionValidationResult,
)

if TYPE_CHECKING:
    from src.tools.model_explorer.frankenstein_editor.model import URDFModel
    from src.tools.model_explorer.library_panel_model import LibraryModelEntry

DropState = Literal["ready", "blocked"]


@dataclass(frozen=True)
class CompositionDragPayload:
    """Model-library row data carried by a composition drag operation."""

    category: str
    key: str
    name: str
    format_badge: str
    source_prefix: str


@dataclass(frozen=True)
class CompositionDropPreview:
    """Non-mutating preview of a pending source-to-target attachment."""

    state: DropState
    payload: CompositionDragPayload
    selection: AttachmentSelection
    source_root_link: str | None
    mapped_root_link: str | None
    attachment_joint: str | None
    validation: CompositionValidationResult | None
    message: str
    highlighted_links: tuple[str, ...]


@dataclass(frozen=True)
class CompositionDropCommit:
    """Committed result of a drag/drop attachment."""

    payload: CompositionDragPayload
    selection: AttachmentSelection
    result: CompositionResult


@dataclass(frozen=True)
class ExportChoice:
    """One selectable export option for the composed model."""

    format: str
    label: str
    enabled: bool
    reason: str = ""


class CompositionUxController:
    """Coordinate drag/drop composition while keeping widgets thin."""

    _SUPPORTED_EXPORTS: tuple[ExportFormat, ...] = ("urdf", "mjcf")
    _PENDING_EXPORTS: tuple[str, ...] = ("sdf", "osim")

    def __init__(self, flow: CompositionFlowController | None = None) -> None:
        self._flow = flow or CompositionFlowController()

    def payload_from_library_entry(
        self,
        entry: LibraryModelEntry,
    ) -> CompositionDragPayload:
        """Build a deterministic drag payload from a library-panel row."""
        if entry is None:
            raise ValueError("entry must be provided")
        return CompositionDragPayload(
            category=entry.category,
            key=entry.key,
            name=entry.name,
            format_badge=entry.format_badge,
            source_prefix=_safe_prefix(entry.name or entry.key),
        )

    def preview_drop(
        self,
        *,
        payload: CompositionDragPayload,
        target_model: URDFModel,
        source_model: URDFModel,
        selection: AttachmentSelection,
    ) -> CompositionDropPreview:
        """Return a ghost-preview result without mutating either input model."""
        self._validate_drop_inputs(payload, target_model, source_model, selection)
        preview_target = _clone_model(target_model)
        preview_source = _clone_model(source_model)
        try:
            result = self._flow.attach_source_model(
                target_model=preview_target,
                source_model=preview_source,
                selection=selection,
            )
        except (CompositionFlowError, ValueError) as exc:
            return CompositionDropPreview(
                state="blocked",
                payload=payload,
                selection=selection,
                source_root_link=None,
                mapped_root_link=None,
                attachment_joint=None,
                validation=None,
                message=f"Cannot attach {payload.name}: {exc}",
                highlighted_links=(selection.target_link,),
            )

        state: DropState = "ready" if result.validation.ok else "blocked"
        status = "ready" if result.validation.ok else "blocked by validation"
        return CompositionDropPreview(
            state=state,
            payload=payload,
            selection=selection,
            source_root_link=result.source_root_link,
            mapped_root_link=result.mapped_root_link,
            attachment_joint=result.attachment_joint,
            validation=result.validation,
            message=(
                f"{payload.name} -> {selection.target_link}: {status}; "
                f"ghost root {result.mapped_root_link}"
            ),
            highlighted_links=(selection.target_link, result.mapped_root_link),
        )

    def commit_drop(
        self,
        *,
        payload: CompositionDragPayload,
        target_model: URDFModel,
        source_model: URDFModel,
        selection: AttachmentSelection,
        force: bool = False,
    ) -> CompositionDropCommit:
        """Attach the dragged source model to the working target model."""
        if force is None:
            raise ValueError("force must be provided")
        preview = self.preview_drop(
            payload=payload,
            target_model=target_model,
            source_model=source_model,
            selection=selection,
        )
        if preview.state == "blocked" and not force:
            raise CompositionFlowError(preview.message)
        result = self._flow.attach_source_model(
            target_model=target_model,
            source_model=source_model,
            selection=selection,
        )
        return CompositionDropCommit(
            payload=payload,
            selection=selection,
            result=result,
        )

    def export_choices(self, model: URDFModel) -> tuple[ExportChoice, ...]:
        """Return format-chooser state for the current composed model."""
        if model is None:
            raise ValueError("model must be provided")
        validation = model.validate_composition()
        blocker = _validation_blocker(validation)
        choices: list[ExportChoice] = [
            ExportChoice(
                format=export_format,
                label=export_format.upper(),
                enabled=blocker == "",
                reason=blocker,
            )
            for export_format in self._SUPPORTED_EXPORTS
        ]
        choices.extend(
            ExportChoice(
                format=export_format,
                label=export_format.upper(),
                enabled=False,
                reason="writer not available",
            )
            for export_format in self._PENDING_EXPORTS
        )
        return tuple(choices)

    @staticmethod
    def validation_summary(result: CompositionValidationResult) -> str:
        """Return concise live-validation text for status bars and logs."""
        if result is None:
            raise ValueError("result must be provided")
        if not result.findings:
            return "validation passed"
        return f"{len(result.errors)} error(s), {len(result.warnings)} warning(s)"

    @staticmethod
    def _validate_drop_inputs(
        payload: CompositionDragPayload,
        target_model: URDFModel,
        source_model: URDFModel,
        selection: AttachmentSelection,
    ) -> None:
        if payload is None:
            raise ValueError("payload must be provided")
        if target_model is None:
            raise ValueError("target_model must be provided")
        if source_model is None:
            raise ValueError("source_model must be provided")
        if selection is None:
            raise ValueError("selection must be provided")


def _clone_model(model: URDFModel) -> URDFModel:
    from src.tools.model_explorer.frankenstein_editor.model import URDFModel

    return URDFModel(
        file_path=model.file_path,
        robot_name=model.robot_name,
        links={name: copy.deepcopy(link) for name, link in model.links.items()},
        joints={name: copy.deepcopy(joint) for name, joint in model.joints.items()},
        materials={
            name: copy.deepcopy(material) for name, material in model.materials.items()
        },
        other_elements=[copy.deepcopy(element) for element in model.other_elements],
        attachment_points=tuple(
            copy.deepcopy(point) for point in model.attachment_points
        ),
        is_modified=model.is_modified,
    )


def _validation_blocker(result: CompositionValidationResult) -> str:
    if not result.errors:
        return ""
    return "; ".join(finding.message for finding in result.errors)


def _safe_prefix(value: str) -> str:
    normalized = "".join(
        char.lower() if char.isalnum() else "_" for char in value.strip()
    ).strip("_")
    return f"{normalized or 'model'}_"
