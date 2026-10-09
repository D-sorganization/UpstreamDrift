"""Frankenstein assembly routes for the web Model Explorer (CMB-10, #11661).

Exposes the headless :class:`AssemblySession` (the engine behind the PyQt6
assembly panel, CMB-8/9) over REST so the web canvas uses the same port typing,
occupancy and composition-validation rules:

- ``GET  /tools/model-explorer/parts``            browse the part catalog
- ``POST /tools/model-explorer/assembly``         rebuild a plan, evaluate a drop
- ``POST /tools/model-explorer/assembly/export``  serialize a plan to URDF/MJCF

The routes are stateless: the client owns the ordered plan and the server
replays it on every call. Instance ids are deterministic (``<part>_<n>``), so a
client can name sockets on placed parts (``leg_left_1__ankle``). No
module-level mutable state.
"""

from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache

import anyio.to_thread
from fastapi import APIRouter, HTTPException

from src.tools.model_explorer.assembly_session import (
    AssemblyError,
    AssemblySession,
    DropDecision,
)
from src.tools.model_explorer.composition_flow import (
    CompositionFlowController,
    CompositionFlowError,
)
from src.tools.model_explorer.composition_validator import (
    CompositionFinding,
    CompositionValidationResult,
)
from src.tools.model_explorer.part_catalog import PartCatalog, PartSpec

from ..models.assembly import (
    AssemblyCatalogResponse,
    AssemblyCategory,
    AssemblyDropDecision,
    AssemblyExportRequest,
    AssemblyExportResponse,
    AssemblyFinding,
    AssemblyPart,
    AssemblyPlacedPart,
    AssemblyPlanRequest,
    AssemblyStateResponse,
    AssemblyStep,
    AssemblyValidation,
    port_payload,
)
from .model_explorer import _parse_urdf_tree

router = APIRouter()


@lru_cache(maxsize=1)
def _catalog() -> PartCatalog:
    """The bundled part catalog; built once, never mutated by these routes."""
    return PartCatalog.bundled()


def _part_payload(part: PartSpec) -> AssemblyPart:
    return AssemblyPart(
        part_id=part.part_id,
        name=part.name,
        category=part.category,
        description=part.description,
        ports=[port_payload(port) for port in part.ports],
    )


def _findings(findings: tuple[CompositionFinding, ...]) -> list[AssemblyFinding]:
    return [AssemblyFinding.model_validate(asdict(f)) for f in findings]


def _validation(result: CompositionValidationResult) -> AssemblyValidation:
    return AssemblyValidation(ok=result.ok, findings=_findings(result.findings))


def _drop_payload(decision: DropDecision) -> AssemblyDropDecision:
    return AssemblyDropDecision(
        accepted=decision.accepted,
        reason=decision.reason,
        part_id=decision.part_id,
        host_port=decision.host_port,
        findings=_findings(decision.findings),
    )


def _replay(base_part_id: str, steps: list[AssemblyStep]) -> AssemblySession:
    """Rebuild a session from a plan.

    Raises:
        HTTPException: 404 for an unknown base part, 422 naming the first
            rejected step (1-based) and the session's reason.
    """
    try:
        session = AssemblySession(_catalog(), base_part_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail=f"unknown base part {base_part_id!r}"
        ) from exc
    for index, step in enumerate(steps, start=1):
        try:
            session.attach(step.part_id, step.host_port)
        except AssemblyError as exc:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"step {index} ({step.part_id} -> {step.host_port}) rejected: {exc}"
                ),
            ) from exc
    return session


def _state(request: AssemblyPlanRequest) -> AssemblyStateResponse:
    session = _replay(request.base_part_id, request.steps)
    drop = None
    if request.candidate is not None:
        candidate = request.candidate
        drop = _drop_payload(
            session.evaluate_drop(candidate.part_id, candidate.host_port)
        )
    return AssemblyStateResponse(
        model=_parse_urdf_tree(session.to_urdf(force=True), "assembly.urdf"),
        placed=[
            AssemblyPlacedPart(
                instance_id=p.instance_id,
                part_id=p.part_id,
                host_port=p.host_port,
                host_instance=p.host_instance,
                links=list(p.links),
                joints=list(p.joints),
            )
            for p in session.placed_parts
        ],
        free_sockets=[port_payload(port) for port in session.free_sockets()],
        validation=_validation(session.validate()),
        drop=drop,
    )


def _export(request: AssemblyExportRequest) -> AssemblyExportResponse:
    session = _replay(request.base_part_id, request.steps)
    try:
        exported = CompositionFlowController().export_model(
            session.model, export_format=request.format, force=request.force
        )
    except CompositionFlowError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return AssemblyExportResponse(
        format=exported.format,
        content=exported.content,
        validation=_validation(exported.validation),
    )


@router.get("/tools/model-explorer/parts", response_model=AssemblyCatalogResponse)
async def list_parts(
    category: str | None = None, query: str = ""
) -> AssemblyCatalogResponse:
    """Browse the part library, optionally by category and search text."""
    catalog = _catalog()
    parts = catalog.list_parts(category=category, query=query)
    return AssemblyCatalogResponse(
        categories=[
            AssemblyCategory(id=cid, label=label) for cid, label in catalog.categories()
        ],
        parts=[_part_payload(part) for part in parts],
    )


@router.post("/tools/model-explorer/assembly", response_model=AssemblyStateResponse)
async def build_assembly(request: AssemblyPlanRequest) -> AssemblyStateResponse:
    """Rebuild an assembly plan and optionally evaluate a candidate drop.

    Postcondition: ``placed[0]`` is the base part and ``drop`` is set exactly
    when ``candidate`` was given; the candidate is never applied.
    """
    return await anyio.to_thread.run_sync(_state, request)


@router.post(
    "/tools/model-explorer/assembly/export", response_model=AssemblyExportResponse
)
async def export_assembly(request: AssemblyExportRequest) -> AssemblyExportResponse:
    """Serialize an assembly plan; validation errors are a 422 unless forced."""
    return await anyio.to_thread.run_sync(_export, request)
