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
