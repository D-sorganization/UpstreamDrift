"""Static analysis plot-data routes (issue #7449, epic #7462).

Exposes the headless :class:`AnalysisOrchestrator` plot catalogue over HTTP
so the web frontend reaches parity with the PyQt6 dashboard's static
post-run plots:

- ``GET /analysis/plot-types`` enumerates the orchestrator's registered
  plot types (data-driven — new types added to the orchestrator registry
  appear here automatically, no per-type web PR required).
- ``GET /analysis/plot-data/{plot_type}`` computes structured
  :class:`~src.shared.python.analysis.plot_data.PlotData` from the most
  recently completed simulation session's recorder (retained by
  ``SimulationService``) — never rendered images.

Data source: the in-memory active simulation session only. Persisted
recording ids are tracked separately by the export/recording parity work
in epic #7462.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from src.shared.python.analysis.orchestrator import AnalysisOrchestrator

from ..dependencies import get_simulation_service

if TYPE_CHECKING:
    from ..services.simulation_service import SimulationService

router = APIRouter()

#: Reverse map of the dashboard combo-box labels keyed by plot-type id,
#: derived from the orchestrator's single-source-of-truth registries.
_PLOT_TYPE_LABELS: dict[str, str] = {
    plot_type: label
    for label, plot_type in AnalysisOrchestrator.DASHBOARD_LABEL_TO_PLOT_TYPE.items()
}


class PlotTypeInfo(BaseModel):
    """One enumerable plot type served by the analysis API."""

    id: str = Field(description="Registry id, e.g. 'joint_angles'")
    label: str = Field(description="Human-readable dashboard label")


class PlotTypesResponse(BaseModel):
    """Catalogue of plot types available from the orchestrator registry."""

    plot_types: list[PlotTypeInfo]


def _label_for(plot_type: str) -> str:
    """Dashboard label for a plot type (fallback: title-cased id)."""
    return _PLOT_TYPE_LABELS.get(plot_type, plot_type.replace("_", " ").title())


@router.get("/analysis/plot-types", response_model=PlotTypesResponse)
async def list_plot_types() -> PlotTypesResponse:
    """Enumerate the static analysis plot types served by this API.

    The list is derived from ``AnalysisOrchestrator.PLOT_TYPES`` so the
    web UI is data-driven: adding a plot type to the orchestrator
    automatically surfaces it in both frontends.
    """
    return PlotTypesResponse(
        plot_types=[
            PlotTypeInfo(id=plot_type, label=_label_for(plot_type))
            for plot_type in AnalysisOrchestrator.available_plot_types()
        ]
    )


@router.get("/analysis/plot-data/{plot_type}")
async def get_plot_data(
    plot_type: str,
    service: SimulationService = Depends(get_simulation_service),
) -> dict[str, Any]:
    """Compute structured plot data for one registered plot type.

    Args:
        plot_type: One of the ids from ``GET /analysis/plot-types``.
        service: Injected simulation service holding the active recorder.

    Returns:
        JSON-serialized ``PlotData`` (series with x/y[/z], units, labels).

    Raises:
        HTTPException: 404 for an unknown plot type; 409 when no
            simulation has completed yet (no recorder to analyze).
    """
    if plot_type not in AnalysisOrchestrator.PLOT_TYPES:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Unknown plot type '{plot_type}'. "
                f"Valid types: {AnalysisOrchestrator.available_plot_types()}"
            ),
        )

    recorder = service.active_recorder
    if recorder is None:
        raise HTTPException(
            status_code=409,
            detail=(
                "No simulation data available. Run a simulation first "
                "(POST /simulate), then request plot data."
            ),
        )

    orchestrator = AnalysisOrchestrator(recorder, service.active_joint_names)
    return orchestrator.get_plot_data(plot_type).to_dict()
