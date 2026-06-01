"""Unit tests for the export API route."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes.export import router
from src.api.dependencies import get_task_manager


class MockTaskManager:
    def exists(self, task_id):
        return task_id == "valid_task"

    def get(self, task_id):
        if task_id == "valid_task":
            return {"status": "completed", "result": {"data": "test_data"}}
        return None


@pytest.fixture
def mock_task_manager():
    return MockTaskManager()


@pytest.fixture
def app(mock_task_manager) -> FastAPI:
    """Create a FastAPI app with the export router."""
    test_app = FastAPI()
    test_app.include_router(router)
    test_app.dependency_overrides[get_task_manager] = lambda: mock_task_manager
    return test_app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """Create a test client."""
    return TestClient(app)


def test_export_results_success(client: TestClient) -> None:
    """Test successful export."""
    response = client.get("/export/valid_task?format=json")
    assert response.status_code == 200
    assert response.json() == {"data": "test_data"}


def test_export_results_invalid_format(client: TestClient) -> None:
    """Test export with invalid format."""
    response = client.get("/export/valid_task?format=invalid")
    assert response.status_code == 400


def test_export_results_not_found(client: TestClient) -> None:
    """Test export for non-existent task."""
    response = client.get("/export/invalid_task?format=json")
    assert response.status_code == 404
