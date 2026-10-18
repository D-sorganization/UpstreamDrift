"""Unit tests for the dataset API route."""

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes.dataset import router
from src.api.dependencies import get_engine_manager
from src.shared.python.engine_core.mock_engine import MockPhysicsEngine

pytestmark = pytest.mark.unit


class MockEngineManager:
    def __init__(self, has_engine=True):
        self.has_engine = has_engine
        self.engine = MockPhysicsEngine() if has_engine else None

    def get_active_physics_engine(self):
        return self.engine


@pytest.fixture
def mock_engine_manager():
    return MockEngineManager()


@pytest.fixture
def app(mock_engine_manager) -> FastAPI:
    """Create a FastAPI app with the dataset router."""
    test_app = FastAPI()
    test_app.include_router(router)
    test_app.dependency_overrides[get_engine_manager] = lambda: mock_engine_manager
    return test_app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """Create a test client."""
    return TestClient(app)


def test_get_plot_types(client: TestClient) -> None:
    """Test getting plot types.

    The whole body used to sit in `try: ... except Exception: pass`, so this
    test could not fail (#8035). The endpoint is served by the app fixture in
    this module with its dependencies overridden, so there is no optional-import
    hazard to tolerate -- it either responds correctly or the test fails.
    """
    response = client.get("/dataset/plots/types")

    assert response.status_code == 200, response.text
    payload = response.json()
    assert isinstance(payload, list)


def test_get_export_formats(client: TestClient) -> None:
    """Test getting export formats."""
    response = client.get("/dataset/export/formats")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert any(fmt["format"] == "hdf5" for fmt in data)


def test_generate_no_engine(app: FastAPI) -> None:
    """Test generate dataset fails without active engine."""
    app.dependency_overrides[get_engine_manager] = lambda: MockEngineManager(
        has_engine=False
    )
    client = TestClient(app)
    response = client.post(
        "/dataset/generate", json={"num_samples": 10, "duration": 2.0}
    )
    assert response.status_code == 409


def test_get_dataset_controls_exists(client: TestClient) -> None:
    """GET /dataset/control returns the generation control catalog (#7981).

    The Dataset Generator page fetches this on mount; before #7981 it did not
    exist and every page load 404'd.
    """
    response = client.get("/dataset/control")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["controls"], list)
    assert data["controls"], "control catalog must not be empty"


def test_dataset_control_ids_match_generation_request(client: TestClient) -> None:
    """Control ids are exactly the POST /dataset/generate field names."""
    from src.api.routes.dataset import DatasetGenerationRequest

    ids = {ctrl["id"] for ctrl in client.get("/dataset/control").json()["controls"]}
    assert ids == set(DatasetGenerationRequest.model_fields)


def test_dataset_control_descriptors_are_well_formed(client: TestClient) -> None:
    """Every descriptor carries the fields the UI widget switch reads."""
    controls = client.get("/dataset/control").json()["controls"]
    for ctrl in controls:
        assert ctrl["type"] in {"select", "range", "text"}
        assert ctrl["name"]
        if ctrl["type"] == "select":
            assert ctrl["options"], f"{ctrl['id']} select needs options"
        if ctrl["type"] == "range":
            assert ctrl["min"] is not None and ctrl["max"] is not None
            assert ctrl["min"] < ctrl["max"]


def test_dataset_control_defaults_round_trip_through_generate_model() -> None:
    """The advertised defaults validate against DatasetGenerationRequest."""
    from fastapi.testclient import TestClient as _TestClient

    from src.api.routes.dataset import DatasetGenerationRequest, router as _router

    app = FastAPI()
    app.include_router(_router)
    controls = _TestClient(app).get("/dataset/control").json()["controls"]
    payload = {ctrl["id"]: ctrl["value"] for ctrl in controls}
    # Raises ValidationError if a default is not a legal request value.
    DatasetGenerationRequest(**payload)


def test_dataset_control_does_not_shadow_control_subroutes(client: TestClient) -> None:
    """/dataset/control must not swallow /dataset/control/state etc."""
    routes = {
        (method, route.path)
        for route in app_routes(client)
        for method in (route.methods or set())
    }
    assert ("GET", "/dataset/control") in routes
    assert ("GET", "/dataset/control/state") in routes
    assert ("POST", "/dataset/control/configure") in routes
    assert ("GET", "/dataset/control/strategies") in routes


def app_routes(client: TestClient):
    """Yield the APIRoute objects registered on the client's app."""
    from fastapi.routing import APIRoute

    routes = []
    app = getattr(client, "app", None)
    app_routes_list = getattr(app, "routes", [])
    for r in app_routes_list:
        if isinstance(r, APIRoute):
            routes.append(r)
        elif hasattr(r, "original_router") and hasattr(r.original_router, "routes"):
            routes.extend(
                [sub for sub in r.original_router.routes if isinstance(sub, APIRoute)]
            )
    return routes


def test_dataset_generate_and_export_contract(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """POST /dataset/generate returns UI GenerateResult shape and export streams the file (#12967)."""
    output_root = tmp_path / "output"
    output_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)
    from src.api.routes import dataset as dataset_module

    monkeypatch.setattr(dataset_module, "_dataset_output_roots", lambda: [output_root])

    payload = {
        "num_samples": 2,
        "duration": 0.02,
        "timestep": 0.01,
        "export_format": "csv",
        "output_path": "output/test_dataset",
    }
    response = client.post("/dataset/generate", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()

    # Required by UI GenerateResult contract (#12967)
    assert "dataset_id" in data
    assert isinstance(data["dataset_id"], str) and data["dataset_id"].startswith("ds_")
    assert "name" in data and data["name"] == "test_dataset"
    assert "rows" in data and isinstance(data["rows"], int) and data["rows"] > 0
    assert (
        "columns" in data
        and isinstance(data["columns"], list)
        and len(data["columns"]) > 0
    )
    assert "created_at" in data and isinstance(data["created_at"], str)

    # Legacy fields preserved
    assert data["status"] == "success"
    assert data["num_samples"] == 2
    assert data["total_frames"] == data["rows"]
    assert "export_path" in data
    assert data["export_format"] == "csv"

    # Export route GET /dataset/export/{dataset_id}
    export_resp = client.get(f"/dataset/export/{data['dataset_id']}?format=csv")
    assert export_resp.status_code == 200, export_resp.text
    assert export_resp.headers.get("content-type", "").startswith("text/csv")
    assert f'filename="{data["dataset_id"]}.csv"' in export_resp.headers.get(
        "content-disposition", ""
    )
    assert len(export_resp.content) > 0


def test_dataset_export_rejects_path_traversal(client: TestClient) -> None:
    """GET /dataset/export/{dataset_id} rejects path-traversal tokens."""
    # Parent directory escape
    resp = client.get("/dataset/export/../../etc/passwd")
    assert resp.status_code in {400, 404}

    # URL-encoded traversal
    resp = client.get("/dataset/export/..%2F..%2Fetc%2Fpasswd")
    assert resp.status_code in {400, 404}

    # Non-opaque characters
    resp = client.get("/dataset/export/bad;id")
    assert resp.status_code == 400

    # Unmapped dataset ID
    resp = client.get("/dataset/export/ds_nonexistent123")
    assert resp.status_code == 404
