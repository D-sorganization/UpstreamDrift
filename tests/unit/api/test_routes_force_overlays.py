"""Unit tests for the force overlays API route (#1199, #11307, FTO-22)."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from src.api.dependencies import get_engine_manager
from src.api.routes.force_overlays import router
from src.shared.python.force_overlay import (
    ForceTorqueFrame,
    OverlayWrench,
    WrenchKind,
)

pytestmark = pytest.mark.unit


def _make_fixture_frame(time_s: float = 0.5) -> ForceTorqueFrame:
    """Deterministic frame with distinct non-fabricated coordinates."""
    wrenches = (
        OverlayWrench(
            kind=WrenchKind.JOINT_ACTUATOR,
            label="joint_actuator:elbow",
            body="arm",
            point_m=(0.12, 0.34, 0.56),
            torque_nm=(0.0, 0.0, 10.0),
            source="mock_provider",
        ),
        OverlayWrench(
            kind=WrenchKind.CONTACT,
            label="contact:lead_foot",
            body="foot",
            point_m=(0.25, -0.15, 0.0),
            force_n=(0.0, 0.0, 450.0),
            source="mock_provider",
        ),
    )
    return ForceTorqueFrame(time_s=time_s, engine="mock_engine", wrenches=wrenches)


class MockProviderEngine:
    """Mock engine satisfying ForceTorqueProvider."""

    def __init__(self, frame: ForceTorqueFrame | None = None) -> None:
        self.time = 0.5
        self._frame = frame or _make_fixture_frame()

    def get_force_torque_frame(self) -> ForceTorqueFrame | None:
        return self._frame

    def get_state(self) -> dict[str, Any]:
        return {"time": self.time}


class MockNonProviderEngine:
    """Mock engine without ForceTorqueProvider."""

    def __init__(self) -> None:
        self.time = 0.5

    def get_state(self) -> dict[str, Any]:
        return {"time": self.time}


class MockEngineManager:
    """Mock engine manager."""

    def __init__(self, engine: Any = None) -> None:
        self._engine = engine

    def get_active_engine(self) -> Any:
        return self._engine


@pytest.fixture
def provider_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    manager = MockEngineManager(MockProviderEngine())
    app.dependency_overrides[get_engine_manager] = lambda: manager
    return app


@pytest.fixture
def provider_client(provider_app: FastAPI) -> TestClient:
    return TestClient(provider_app)


def test_get_force_overlays_with_provider(provider_client: TestClient) -> None:
    """Real provider produces schema-valid glyphs and matching legacy vectors."""
    response = provider_client.get("/simulation/forces?force_types=all")
    assert response.status_code == 200
    data = response.json()

    assert data["sim_time"] == 0.5
    assert data["unavailable_reason"] is None

    # Glyphs drawing payload
    glyphs = data["glyphs"]
    assert glyphs is not None
    assert glyphs["schema_version"] == "glyph-set-v1"
    assert len(glyphs["arrows"]) == 1
    assert len(glyphs["torque_arcs"]) == 1

    # Verify glyph tail matches provider point (not fabricated 0.5 + i * 0.3)
    contact_arrow = glyphs["arrows"][0]
    assert contact_arrow["tail_m"] == [0.25, -0.15, 0.0]

    torque_arc = glyphs["torque_arcs"][0]
    assert torque_arc["center_m"] == [0.12, 0.34, 0.56]

    # The deprecated legacy ``vectors`` list is removed (#11362)
    assert "vectors" not in data


def test_get_force_overlays_without_engine() -> None:
    """Without engine loaded, response contains null payloads and an unavailable reason."""
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_engine_manager] = lambda: MockEngineManager(None)
    client = TestClient(app)

    response = client.get("/simulation/forces?force_types=all")
    assert response.status_code == 200
    data = response.json()

    assert data["glyphs"] is None
    assert data["frame"] is None
    assert "vectors" not in data
    assert data["unavailable_reason"] is not None
    # No demo vectors!
    assert data["total_force_magnitude"] == 0.0
    assert data["total_torque_magnitude"] == 0.0


def test_get_force_overlays_non_provider_engine() -> None:
    """With non-provider engine, response gracefully reports unavailable."""
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_engine_manager] = lambda: MockEngineManager(
        MockNonProviderEngine()
    )
    client = TestClient(app)

    response = client.get("/simulation/forces?force_types=all")
    assert response.status_code == 200
    data = response.json()

    assert data["glyphs"] is None
    assert data["frame"] is None
    assert "vectors" not in data
    assert data["unavailable_reason"] is not None


def test_filters_map_to_kinds(provider_client: TestClient) -> None:
    """Query parameter force_types correctly filters the emitted glyphs."""
    # Only contact
    response = provider_client.get("/simulation/forces?force_types=contact")
    assert response.status_code == 200
    data = response.json()
    assert len(data["glyphs"]["arrows"]) == 1
    assert len(data["glyphs"]["torque_arcs"]) == 0
    assert data["glyphs"]["arrows"][0]["label"] == "contact:lead_foot"

    # Only applied
    response_applied = provider_client.get("/simulation/forces?force_types=applied")
    assert response_applied.status_code == 200
    data_applied = response_applied.json()
    assert len(data_applied["glyphs"]["arrows"]) == 0
    assert len(data_applied["glyphs"]["torque_arcs"]) == 1
    assert data_applied["glyphs"]["torque_arcs"][0]["label"] == "joint_actuator:elbow"


def test_body_filter_parameter(provider_client: TestClient) -> None:
    """Query parameter body_filter restricts wrenches to selected bodies."""
    response = provider_client.get(
        "/simulation/forces?body_filter=foot&force_types=all"
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["glyphs"]["arrows"]) == 1
    assert len(data["glyphs"]["torque_arcs"]) == 0
    assert data["glyphs"]["arrows"][0]["label"] == "contact:lead_foot"


def test_update_force_overlay_config(provider_client: TestClient) -> None:
    """POST /simulation/forces/config applies config and returns updated data."""
    payload = {
        "enabled": True,
        "force_types": ["contact"],
        "color_by_magnitude": True,
        "show_labels": True,
        "scale_factor": 0.05,
    }
    response = provider_client.post("/simulation/forces/config", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["overlay_config"]["scale_factor"] == 0.05
    assert data["overlay_config"]["show_labels"] is True
    assert len(data["glyphs"]["arrows"]) == 1
    assert len(data["glyphs"]["torque_arcs"]) == 0
