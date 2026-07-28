"""Unit tests for the video API route."""

from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.dependencies import (
    get_task_manager,
    get_video_pipeline,
    get_video_pipeline_factory,
)
from src.api.routes.video import _looks_like_supported_video_bytes, router


MP4_HEADER = b"\x00\x00\x00\x20ftypisommore-bytes"


class MockVideoPipeline:
    def __init__(self) -> None:
        self.process_video_calls: list[Path] = []

    def process_video(self, video_path: Path) -> SimpleNamespace:
        self.process_video_calls.append(video_path)
        assert video_path.exists()
        return SimpleNamespace(
            total_frames=3,
            valid_frames=1,
            average_confidence=0.875,
            quality_metrics={"valid_frame_ratio": 1 / 3},
            pose_results=[
                SimpleNamespace(
                    timestamp=0.0,
                    confidence=0.875,
                    joint_angles={"lead_elbow": 12.5},
                    raw_keypoints={"lead_wrist": [1.0, 2.0, 0.875]},
                )
            ],
        )


class MockTaskManager:
    def __init__(self) -> None:
        self.tasks = {"existing_video": {"status": "completed"}}

    def exists(self, task_id: str) -> bool:
        return task_id in self.tasks

    def get(self, task_id: str) -> dict[str, Any] | None:
        return self.tasks.get(task_id)

    def set(self, task_id: str, value: dict[str, Any]) -> None:
        self.tasks[task_id] = value


@pytest.fixture
def mock_video_pipeline():
    return MockVideoPipeline()


@pytest.fixture
def mock_task_manager():
    return MockTaskManager()


@pytest.fixture
def mock_video_pipeline_factory(mock_video_pipeline):
    calls = []

    def factory(
        estimator_type: str,
        min_confidence: float,
        enable_smoothing: bool,
    ) -> MockVideoPipeline:
        calls.append((estimator_type, min_confidence, enable_smoothing))
        return mock_video_pipeline

    factory.calls = calls
    return factory


@pytest.fixture
def app(mock_video_pipeline, mock_task_manager, mock_video_pipeline_factory) -> FastAPI:
    test_app = FastAPI()
    from src.api.rate_limit import limiter

    test_app.state.limiter = limiter
    test_app.include_router(router)
    test_app.dependency_overrides[get_video_pipeline] = lambda: mock_video_pipeline
    test_app.dependency_overrides[get_video_pipeline_factory] = lambda: (
        mock_video_pipeline_factory
    )
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


def test_analyze_video_success_uses_injected_pipeline(
    client: TestClient,
    mock_video_pipeline: MockVideoPipeline,
) -> None:
    response = client.post(
        "/analyze/video",
        files={
            "file": (
                "swing.mp4",
                io.BytesIO(MP4_HEADER + b"payload"),
                "video/mp4",
            )
        },
    )

    assert response.status_code == 200
    assert len(mock_video_pipeline.process_video_calls) == 1
    body = response.json()
    assert body["filename"] == "swing.mp4"
    assert body["total_frames"] == 3
    assert body["valid_frames"] == 1
    assert body["average_confidence"] == 0.875
    assert body["pose_data"] == [
        {
            "timestamp": 0.0,
            "confidence": 0.875,
            "joint_angles": {"lead_elbow": 12.5},
            "keypoints": {"lead_wrist": [1.0, 2.0, 0.875]},
        }
    ]


def test_analyze_video_async_success_uses_injected_pipeline_factory(
    client: TestClient,
    mock_video_pipeline: MockVideoPipeline,
    mock_video_pipeline_factory,
    mock_task_manager: MockTaskManager,
) -> None:
    response = client.post(
        "/analyze/video/async",
        params={"estimator_type": "mediapipe", "min_confidence": "0.75"},
        files={
            "file": (
                "swing.mp4",
                io.BytesIO(MP4_HEADER + b"payload"),
                "video/mp4",
            )
        },
    )

    assert response.status_code == 200
    assert mock_video_pipeline_factory.calls == [("mediapipe", 0.75, True)]
    assert len(mock_video_pipeline.process_video_calls) == 1
    task_id = response.json()["task_id"]
    task = mock_task_manager.get(task_id)
    assert task is not None
    assert task["status"] == "completed"
    assert task["result"]["filename"] == "swing.mp4"
    assert task["result"]["total_frames"] == 3


def test_video_signature_detection_accepts_mp4_header() -> None:
    assert _looks_like_supported_video_bytes(b"\x00\x00\x00\x20ftypisommore-bytes")


def test_video_signature_detection_rejects_non_video_header() -> None:
    assert not _looks_like_supported_video_bytes(b"PK\x03\x04not-a-video")


def test_analyze_video_rejects_fake_video_payload(client: TestClient) -> None:
    response = client.post(
        "/analyze/video",
        files={
            "file": (
                "fake.mp4",
                io.BytesIO(b"PK\x03\x04this-is-a-zip"),
                "video/mp4",
            )
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Uploaded file content does not match a supported video format."
    )


def test_analyze_video_async_rejects_fake_video_payload(client: TestClient) -> None:
    response = client.post(
        "/analyze/video/async",
        files={
            "file": (
                "fake.mp4",
                io.BytesIO(b"PK\x03\x04this-is-a-zip"),
                "video/mp4",
            )
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Uploaded file content does not match a supported video format."
    )
