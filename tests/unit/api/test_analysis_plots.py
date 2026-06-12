"""Tests for the static analysis plot-data API (issue #7449).

Parity contract: the set of plot types enumerated by the API must equal
the orchestrator's registry keys (the single source of truth shared with
the PyQt6 dashboard), and every enumerated type must return valid,
JSON-serializable ``PlotData`` from a recorded session.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes.analysis_plots import router
from src.api.services.simulation_service import SimulationService
from src.shared.python.analysis.orchestrator import AnalysisOrchestrator
from tests.unit.shared_python.test_analysis_orchestrator import StubRecorder

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


class _ServiceStub:
    """Minimal SimulationService stand-in exposing the recorder surface."""

    def __init__(self, recorder: Any, joint_names: list[str] | None = None) -> None:
        self._recorder = recorder
        self._joint_names = joint_names or []

    @property
    def active_recorder(self) -> Any:
        return self._recorder

    @property
    def active_joint_names(self) -> list[str]:
        return list(self._joint_names)


def _client(service: Any | None) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    if service is not None:
        app.state.simulation_service = service
    return TestClient(app)


@pytest.fixture
def client_with_data() -> TestClient:
    return _client(_ServiceStub(StubRecorder(), ["Hip", "Shoulder", "Wrist"]))


# ----------------------------------------------------------------------
# Enumeration / parity
# ----------------------------------------------------------------------


def test_plot_types_equal_orchestrator_registry(client_with_data: TestClient) -> None:
    """API-enumerated plot types == orchestrator registry keys (parity pin)."""
    resp = client_with_data.get("/analysis/plot-types")
    assert resp.status_code == 200
    ids = [entry["id"] for entry in resp.json()["plot_types"]]
    assert ids == AnalysisOrchestrator.available_plot_types()
    assert set(ids) == set(AnalysisOrchestrator.PLOT_TYPES)


def test_plot_type_labels_match_dashboard(client_with_data: TestClient) -> None:
    """Every registered type carries the PyQt6 dashboard label when one exists."""
    resp = client_with_data.get("/analysis/plot-types")
    label_by_id = {e["id"]: e["label"] for e in resp.json()["plot_types"]}
    for label, plot_type in AnalysisOrchestrator.DASHBOARD_LABEL_TO_PLOT_TYPE.items():
        assert label_by_id[plot_type] == label
    # Every entry has a non-empty label even without a dashboard mapping.
    assert all(label_by_id.values())


# ----------------------------------------------------------------------
# Plot data
# ----------------------------------------------------------------------


def test_every_plot_type_returns_valid_plot_data(
    client_with_data: TestClient,
) -> None:
    """Each enumerated type serves PlotData with consistent series shapes."""
    for plot_type in AnalysisOrchestrator.available_plot_types():
        resp = client_with_data.get(f"/analysis/plot-data/{plot_type}")
        assert resp.status_code == 200, plot_type
        body = resp.json()
        assert body["plot_type"] == plot_type
        assert isinstance(body["title"], str) and body["title"]
        assert isinstance(body["series"], list)
        for series in body["series"]:
            assert isinstance(series["name"], str)
            assert len(series["x"]) == len(series["y"])
            if series.get("z") is not None:
                assert len(series["z"]) == len(series["x"])


def test_plot_data_uses_joint_names(client_with_data: TestClient) -> None:
    resp = client_with_data.get("/analysis/plot-data/joint_angles")
    names = [s["name"] for s in resp.json()["series"]]
    assert names == ["Hip", "Shoulder", "Wrist"]


def test_unknown_plot_type_404(client_with_data: TestClient) -> None:
    resp = client_with_data.get("/analysis/plot-data/not_a_plot")
    assert resp.status_code == 404
    assert "Unknown plot type" in resp.json()["detail"]
    assert "joint_angles" in resp.json()["detail"]


def test_no_active_session_409() -> None:
    client = _client(_ServiceStub(None))
    resp = client.get("/analysis/plot-data/joint_angles")
    assert resp.status_code == 409
    assert "No simulation data available" in resp.json()["detail"]


def test_service_not_initialized_503() -> None:
    client = _client(None)
    resp = client.get("/analysis/plot-data/joint_angles")
    assert resp.status_code == 503


def test_empty_recorder_returns_empty_plot_data_not_error() -> None:
    """A session with no recorded frames yields empty PlotData (GUI parity)."""
    client = _client(_ServiceStub(StubRecorder(with_data=False)))
    resp = client.get("/analysis/plot-data/joint_angles")
    assert resp.status_code == 200
    body = resp.json()
    assert body["series"] == []
    assert "message" in body["metadata"]


# ----------------------------------------------------------------------
# SimulationService recorder retention
# ----------------------------------------------------------------------


def test_simulation_service_retains_recorder() -> None:
    class _Engine:
        def get_joint_names(self) -> list[str]:
            return ["j0", "j1"]

    service = SimulationService(engine_manager=object())  # type: ignore[arg-type]
    assert service.active_recorder is None
    recorder = StubRecorder()
    service._retain_active_session(_Engine(), recorder)
    assert service.active_recorder is recorder
    assert service.active_joint_names == ["j0", "j1"]


def test_simulation_service_handles_engine_without_joint_names() -> None:
    service = SimulationService(engine_manager=object())  # type: ignore[arg-type]
    recorder = StubRecorder()
    service._retain_active_session(object(), recorder)
    assert service.active_recorder is recorder
    assert service.active_joint_names == []
