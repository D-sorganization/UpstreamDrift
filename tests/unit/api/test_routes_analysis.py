"""Unit tests for the analysis API route."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes.analysis import router
from src.api.dependencies import get_analysis_service


class MockAnalysisService:
    async def analyze_biomechanics(self, request):
        return {
            "analysis_type": request.analysis_type,
            "success": True,
            "results": {"overall_score": 85.0, "feedback": ["Good swing"]},
        }


@pytest.fixture
def mock_analysis_service():
    return MockAnalysisService()


@pytest.fixture
def app(mock_analysis_service) -> FastAPI:
    test_app = FastAPI()
    test_app.include_router(router)
    test_app.dependency_overrides[get_analysis_service] = lambda: mock_analysis_service
    return test_app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


def test_analyze_biomechanics(client: TestClient) -> None:
    payload = {
        "analysis_type": "kinematics",
        "data_source": "simulation",
        "parameters": {},
    }
    response = client.post("/analyze/biomechanics", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["results"]["overall_score"] == 85.0, data
    assert "feedback" in data["results"], data


def test_analyze_biomechanics_route_uses_parameters_payload() -> None:
    from src.api.services.analysis_service import AnalysisService

    class NoEngineManager:
        def get_active_physics_engine(self):
            return None

    test_app = FastAPI()
    test_app.include_router(router)
    test_app.dependency_overrides[get_analysis_service] = lambda: AnalysisService(
        NoEngineManager()
    )
    route_client = TestClient(test_app)

    response = route_client.post(
        "/analyze/biomechanics",
        json={
            "analysis_type": "kinematics",
            "data_source": "simulation",
            "parameters": {"joint_angles": [1.0, 2.0, 3.0]},
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["success"] is True
    assert body["results"]["joint_angles"] == [1.0, 2.0, 3.0]
    assert body["results"]["metadata"]["data_source"] == "request"


def test_analyze_biomechanics_route_returns_engine_extraction_failure() -> None:
    from src.api.services.analysis_service import AnalysisService

    class FailingEngine:
        def get_joint_positions(self):
            raise RuntimeError("route extractor failed")

    class EngineManager:
        def get_active_physics_engine(self):
            return FailingEngine()

    test_app = FastAPI()
    test_app.include_router(router)
    test_app.dependency_overrides[get_analysis_service] = lambda: AnalysisService(
        EngineManager()
    )
    route_client = TestClient(test_app)

    response = route_client.post(
        "/analyze/biomechanics",
        json={"analysis_type": "kinematics", "data_source": "simulation"},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["success"] is False
    assert "route extractor failed" in body["results"]["error"]
