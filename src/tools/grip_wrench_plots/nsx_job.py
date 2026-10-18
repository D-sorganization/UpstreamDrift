"""The NSX analysis job of the desktop tabs (NSX-43, #12575).

No PyQt here: a job is a plain callable that the Qt worker runs off the GUI
thread.  It asks the NSX-40 service (the same facade the API uses) for a
report, asks ``display_decision`` once per quantity
(:func:`~.plot_model_decisions.report_decisions`) and builds the
:class:`NSXPlotModel` the tabs render.  It restates no display rule.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from src.api.services import loop_nullspace_service as service
from src.shared.python.loop_nullspace import registry
from src.shared.python.loop_nullspace.plot_model import NSXPlotModel, build_plot_model
from src.shared.python.loop_nullspace.plot_model_decisions import report_decisions

__all__ = [
    "NSXJob",
    "NSXRequest",
    "PresetChoice",
    "preset_choices",
    "request_job",
    "run_report",
]

StageHook = Callable[[str, float, "Mapping[str, Any] | None"], None]
CancelCheck = Callable[[], bool]
NSXJob = Callable[[StageHook, CancelCheck], NSXPlotModel]
SURFACE = "research_view"


@dataclass(frozen=True)
class PresetChoice:
    """One registry preset as the selector lists it."""

    preset_id: str
    label: str


@dataclass(frozen=True)
class NSXRequest:
    """What the user chose: source, preset, frames, allocation, sensitivity (NSX-100)."""

    source_id: str
    preset: str
    frame_indices: tuple[int, ...]
    engine: str = "pinocchio"
    include_allocation: bool = False
    sensitivity: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.source_id:
            raise ValueError("a source is required")
        if not self.preset:
            raise ValueError("a preset is required")
        if not self.frame_indices:
            raise ValueError("at least one frame index is required")


def preset_choices() -> tuple[PresetChoice, ...]:
    """The NSX-11 registry presets, in registry order."""
    reg = registry.load_registry()
    return tuple(PresetChoice(p.id, p.label) for p in reg.presets)


def run_report(report: Mapping[str, Any], surface: str = SURFACE) -> NSXPlotModel:
    """``display_decision`` for every quantity of ``report``, then the model."""
    return build_plot_model(report, report_decisions(report, surface))


def request_job(request: NSXRequest) -> NSXJob:
    """A job that runs ``request`` through the service and builds the model.

    Raises (when run): ``service.RequestRefusedError`` for a request the service
    refuses, ``AnalysisCancelledError`` once the cancel check returns ``True``.
    """
    api_request = service.LoopNullspaceRequest(
        source_id=request.source_id,
        engine=request.engine,
        preset=request.preset,
        frame_indices=list(request.frame_indices),
        options=service.LoopNullspaceOptions(
            include_allocation=request.include_allocation,
            sensitivity=(
                None
                if request.sensitivity is None
                else service.SensitivityOptions.model_validate(request.sensitivity)
            ),
        ),
    )

    def job(on_stage: StageHook, should_cancel: CancelCheck) -> NSXPlotModel:
        report = service.run(
            api_request, on_stage=on_stage, should_cancel=should_cancel
        )
        return run_report(report)

    return job
