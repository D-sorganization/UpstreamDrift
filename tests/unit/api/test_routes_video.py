"""Unit tests for the video API route."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes.video import router
from src.api.dependencies import get_task_manager, get_video_pipeline


class MockVideoPipeline:
    def __init__(self):
        pass

    async def run_pipeline(self, request, task_manager=None, task_id=None):
        pass


class MockTaskManager:
    def __init__(self):
        self.tasks = {"existing_video": {"status": "completed"}}

    async def exists(self, task_id):
        return task_id in self.tasks

    async def get(self, task_id):
        return self.tasks.get(task_id)

    async def set(self, task_id, value):
        self.tasks[task_id] = value


@pytest.fixture
def mock_video_pipeline():
    return MockVideoPipeline()


@pytest.fixture
def mock_task_manager():
    return MockTaskManager()


@pytest.fixture
def app(mock_video_pipeline, mock_task_manager) -> FastAPI:
    test_app = FastAPI()
    from src.api.rate_limit import limiter

    test_app.state.limiter = limiter
    test_app.include_router(router)
    test_app.dependency_overrides[get_video_pipeline] = lambda: mock_video_pipeline
    test_app.dependency_overrides[get_task_manager] = lambda: mock_task_manager
    return test_app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


def test_analyze_video(client: TestClient) -> None:
    # A placeholder test because the route requires a multipart form with a file.
    # We will simulate a failure since we're not passing a file.
    response = client.post("/analyze/video")
    assert response.status_code == 422  # Missing required field 'file'
