"""HTTP routes for the canonical-core estimation and comparison workspaces.

Endpoints
---------
``GET /tools/canonical-core/status``
    Availability summary for every canonical-core workspace.

``GET /tools/canonical-core/{mode}/status``
    Availability report for one workspace (``estimation`` or ``comparison``).

Why this exists (#8081)
-----------------------
``/tools/canonical-core/estimation`` and ``/tools/canonical-core/comparison``
rendered as static text: no input, no dataset or engine selector, no execute
action, no result, and — critically — no way for a user to tell whether the
workspace was broken, still loading, or simply not built yet.

``src.tools.canonical_core`` currently ships **descriptors only**
(:mod:`src.tools.canonical_core.registry`) plus PyQt6 shell entry points. There
is no estimation or comparison compute service behind them, so inventing an
execute endpoint here would be fiction. Instead this module reports the honest
truth in a machine-readable form: which workspaces exist, that their services
are not yet implemented, why, and what the user should do next. The React page
renders that instead of a silent shell, and gains a real service-status error
path when the API is unreachable.

When the estimation/comparison services land, flip ``available`` to ``True``
and add the execute routes here; the frontend already branches on the flag.

This route is auto-discovered by ``src.api.route_registry`` — no explicit
registration in ``server.py`` is needed.
"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from src.tools.canonical_core.registry import (
    CanonicalCoreTool,
    canonical_core_tools,
)

router = APIRouter(prefix="/tools/canonical-core", tags=["canonical-core"])

CanonicalCoreMode = Literal["estimation", "comparison"]

# Reason and next step surfaced to the user, keyed by workspace mode. Kept
# beside the registry so the copy stays with the capability it describes.
_UNAVAILABLE_REASON = (
    "The canonical-core {mode} service is not implemented yet. "
    "src/tools/canonical_core currently provides workspace descriptors and the "
    "PyQt6 shell only — there is no {mode} compute backend to call."
)

_NEXT_STEP = {
    "estimation": (
        "Track CC-19 for the estimation service. In the meantime, run a "
        "canonical-state fit through the Simulation workspace and export the "
        "result from Data Explorer."
    ),
    "comparison": (
        "Track CC-27 for the comparison service. In the meantime, use the "
        "Cross-Engine Analysis tools to compare canonical states across "
        "engines."
    ),
}


class CanonicalCoreStatus(BaseModel):
    """Availability report for one canonical-core workspace."""

    tool_id: str = Field(description="Registry id, e.g. canonical_core_estimation")
    mode: str = Field(description="Workspace mode: estimation or comparison")
    name: str = Field(description="Human-readable workspace name")
    description: str = Field(description="Workspace description from the registry")
    web_route: str = Field(description="React route that renders this workspace")
    capabilities: list[str] = Field(
        default_factory=list,
        description="Capability tags declared by the registry descriptor",
    )
    available: bool = Field(
        description="True once a compute service backs this workspace"
    )
    reason: str = Field(
        description="Why the workspace is unavailable; empty when available"
    )
    next_step: str = Field(
        description="Actionable guidance for the user; empty when available"
    )


class CanonicalCoreStatusList(BaseModel):
    """Availability report for every canonical-core workspace."""

    workspaces: list[CanonicalCoreStatus] = Field(default_factory=list)


def _to_status(tool: CanonicalCoreTool) -> CanonicalCoreStatus:
    """Build the status payload for one registry descriptor.

    Args:
        tool: Registry descriptor to describe.

    Returns:
        The workspace status, currently always ``available=False``.

    Postcondition:
        When ``available`` is False, both ``reason`` and ``next_step`` are
        non-empty so the UI always has something actionable to render.
    """
    if tool is None:
        raise ValueError("tool must be provided")

    status_model = CanonicalCoreStatus(
        tool_id=tool.tool_id,
        mode=tool.mode,
        name=tool.name,
        description=tool.description,
        web_route=tool.web_route,
        capabilities=list(tool.capabilities),
        available=False,
        reason=_UNAVAILABLE_REASON.format(mode=tool.mode),
        next_step=_NEXT_STEP[tool.mode],
    )
    if not status_model.available and not (
        status_model.reason and status_model.next_step
    ):
        raise ValueError("unavailable workspaces must carry a reason and next step")
    return status_model


@router.get("/status", response_model=CanonicalCoreStatusList)
async def get_canonical_core_status() -> CanonicalCoreStatusList:
    """Return the availability report for every canonical-core workspace."""
    return CanonicalCoreStatusList(
        workspaces=[_to_status(tool) for tool in canonical_core_tools()]
    )


@router.get("/{mode}/status", response_model=CanonicalCoreStatus)
async def get_canonical_core_mode_status(mode: str) -> CanonicalCoreStatus:
    """Return the availability report for one canonical-core workspace.

    Args:
        mode: ``estimation`` or ``comparison``.

    Raises:
        HTTPException: 404 when ``mode`` is not a known workspace.
    """
    if not isinstance(mode, str) or not mode.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="mode must be a non-empty string",
        )

    for tool in canonical_core_tools():
        if tool.mode == mode:
            return _to_status(tool)

    known = ", ".join(sorted({tool.mode for tool in canonical_core_tools()}))
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Unknown canonical-core workspace {mode!r}. Known modes: {known}",
    )
