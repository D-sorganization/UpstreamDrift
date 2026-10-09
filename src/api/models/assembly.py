"""Request and response models for the Frankenstein assembly API (CMB-10, #11661).

The web Model Explorer keeps an ordered assembly *plan* (a base part plus the
drops made so far) and the server replays it through the headless
``AssemblySession``. The models below are a JSON view of that session; they add
no rules of their own.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from .responses import ModelExplorerResponse


class AssemblyStep(BaseModel):
    """One drop: put ``part_id`` on the host socket ``host_port``."""

    part_id: str = Field(..., min_length=1, description="Catalog part id")
    host_port: str = Field(
        ..., min_length=1, description="Model-level name of the host socket"
    )


class AssemblyPlanRequest(BaseModel):
    """An assembly to rebuild, plus an optional drop to evaluate (drag hover)."""

    base_part_id: str = Field(..., min_length=1, description="Base part id")
    steps: list[AssemblyStep] = Field(
        default_factory=list, description="Drops applied in order"
    )
    candidate: AssemblyStep | None = Field(
        None, description="Drop to evaluate without applying it"
    )


class AssemblyExportRequest(BaseModel):
    """An assembly to rebuild and serialize."""

    base_part_id: str = Field(..., min_length=1, description="Base part id")
    steps: list[AssemblyStep] = Field(
        default_factory=list, description="Drops applied in order"
    )
    format: Literal["urdf", "mjcf"] = Field("urdf", description="Export format")
    force: bool = Field(False, description="Export despite validation errors")


class AssemblyPort(BaseModel):
    """A typed attachment port (``AttachmentPoint.to_dict`` shape)."""

    name: str
    link_name: str
    role: str
    interface_frame: dict[str, list[float]] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    max_payload_kg: float | None = None
    port_type: str | None = None
    polarity: str | None = None


class AssemblyPart(BaseModel):
    """A catalog part that can be dragged into an assembly."""

    part_id: str
    name: str
    category: str
    description: str
    ports: list[AssemblyPort]


class AssemblyCategory(BaseModel):
    """A catalog category id and its display label."""

    id: str
    label: str


class AssemblyCatalogResponse(BaseModel):
    """The browsable part library."""

    categories: list[AssemblyCategory]
    parts: list[AssemblyPart]


class AssemblyFinding(BaseModel):
    """One composition-validation finding."""

    code: str
    severity: str
    message: str
    elements: list[str] = Field(default_factory=list)
    category: str


class AssemblyValidation(BaseModel):
    """Composition validation for the assembled model."""

    ok: bool
    findings: list[AssemblyFinding]


class AssemblyPlacedPart(BaseModel):
    """A part instance inside the assembly (base first)."""

    instance_id: str
    part_id: str
    host_port: str | None
    host_instance: str | None
    links: list[str]
    joints: list[str]


class AssemblyDropDecision(BaseModel):
    """Whether the candidate drop is allowed, and why."""

    accepted: bool
    reason: str
    part_id: str
    host_port: str
    findings: list[AssemblyFinding]


class AssemblyStateResponse(BaseModel):
    """The rebuilt assembly: tree, placed parts, free sockets, validation."""

    model: ModelExplorerResponse
    placed: list[AssemblyPlacedPart]
    free_sockets: list[AssemblyPort]
    validation: AssemblyValidation
    drop: AssemblyDropDecision | None = None


class AssemblyExportResponse(BaseModel):
    """Serialized assembly content."""

    format: Literal["urdf", "mjcf"]
    content: str
    validation: AssemblyValidation


def port_payload(port: Any) -> AssemblyPort:
    """Convert an ``AttachmentPoint`` to its API model."""
    if port is None:
        raise ValueError("port must be provided")
    return AssemblyPort.model_validate(port.to_dict())
