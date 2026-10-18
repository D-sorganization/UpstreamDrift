"""NSX-43: the job asks the service and display_decision; presets are the registry's."""

from __future__ import annotations

import pytest

pytest.importorskip("highspy")

from src.api.services import loop_nullspace_service as service  # noqa: E402
from src.shared.python.loop_nullspace.facade import AnalysisCancelledError  # noqa: E402
from src.shared.python.loop_nullspace.plot_model import NSXPlotModel  # noqa: E402
from src.shared.python.loop_nullspace.registry import load_registry  # noqa: E402
from src.tools.grip_wrench_plots import nsx_job  # noqa: E402
from tests.unit.loop_nullspace.report_support import (  # noqa: E402
    planar_engine,
    planar_trajectory,
)

pytestmark = pytest.mark.unit

SOURCE = "nsx43-test-planar"


class _Planar:
    def describe(self, request):
        return service.SourceDescription({"id": SOURCE}, {"id": "planar"})

    def load(self, request):
        return planar_trajectory(tuple(request.frame_indices)), planar_engine()


@pytest.fixture(autouse=True)
def _source():
    service.register_source(SOURCE, _Planar())
    service.RESULT_CACHE.clear()
    yield
    service.unregister_source(SOURCE)
    service.RESULT_CACHE.clear()


def _request(**kwargs) -> nsx_job.NSXRequest:
    base = {
        "source_id": SOURCE,
        "preset": "envelope_only",
        "frame_indices": (0, 3, 6, 9),
        "engine": "planar-numpy",
    }
    return nsx_job.NSXRequest(**{**base, **kwargs})


def test_presets_are_the_registry_presets() -> None:
    reg = load_registry()
    choices = nsx_job.preset_choices()
    assert [c.preset_id for c in choices] == [p.id for p in reg.presets]
    assert [c.label for c in choices] == [p.label for p in reg.presets]
    assert "envelope_only" in {c.preset_id for c in choices}


def test_job_returns_a_plot_model_and_reports_stages() -> None:
    stages: list[str] = []
    model = nsx_job.request_job(_request(include_allocation=True))(
        lambda stage, pct, partial: stages.append(stage), lambda: False
    )
    assert isinstance(model, NSXPlotModel)
    assert model.preset == "envelope_only"
    assert stages.index("invariants") < stages.index("envelopes")
    assert model.allocation.overlays


def test_job_cancels_between_stages() -> None:
    seen: list[str] = []

    def stage(name: str, pct: float, partial) -> None:
        seen.append(name)

    with pytest.raises(AnalysisCancelledError):
        nsx_job.request_job(_request())(stage, lambda: len(seen) >= 1)
    assert seen == ["decomposition"]


def test_request_is_validated() -> None:
    with pytest.raises(ValueError, match="source"):
        nsx_job.NSXRequest("", "envelope_only", (0,))
    with pytest.raises(ValueError, match="frame"):
        nsx_job.NSXRequest(SOURCE, "envelope_only", ())
    with pytest.raises(ValueError, match="preset"):
        nsx_job.NSXRequest(SOURCE, "", (0,))
