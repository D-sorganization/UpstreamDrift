"""Unit tests for the video API route."""

import asyncio
import io

import pytest
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.testclient import TestClient
from starlette.datastructures import Headers

from src.api.routes.video import (
    _looks_like_supported_video_bytes,
    _validate_video_upload,
    router,
)
from src.api.dependencies import get_task_manager, get_video_pipeline


class MockVideoPipeline:
    def __init__(self):
        pass

    async def run_pipeline(self, request, task_manager=None, task_id=None):
        pass


class MockTaskManager:
    def __init__(self):
        self.tasks = {"existing_video": {"status": "completed"}}

    def exists(self, task_id):
        return task_id in self.tasks

    def get(self, task_id):
        return self.tasks.get(task_id)

    def set(self, task_id, value):
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


def _make_upload(
    filename: str | None,
    content: bytes,
    content_type: str = "video/mp4",
) -> UploadFile:
    """Build an ``UploadFile`` with the given metadata and payload."""
    headers = Headers({"content-type": content_type})
    return UploadFile(
        file=io.BytesIO(content),
        size=len(content),
        filename=filename,
        headers=headers,
    )


_MP4_FTYP_BYTES = b"\x00\x00\x00\x20ftypisommore-bytes"
_MKV_EBML_BYTES = b"\x1a\x45\xdf\xa3" + b"\x00" * 8


def test_analyze_video(client: TestClient) -> None:
    # A placeholder test because the route requires a multipart form with a file.
    # We will simulate a failure since we're not passing a file.
    response = client.post("/analyze/video")
    assert response.status_code == 422  # Missing required field 'file'


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


@pytest.mark.unit
@pytest.mark.parametrize("suffix", [".mp4", ".mov", ".mkv", ".avi", ".webm"])
def test_validate_video_upload_derives_supported_suffix(suffix: str) -> None:
    upload = _make_upload(f"clip{suffix}", _MKV_EBML_BYTES, "video/x-matroska")
    assert asyncio.run(_validate_video_upload(upload)) == suffix


@pytest.mark.unit
def test_validate_video_upload_rejects_unsupported_extension() -> None:
    upload = _make_upload("clip.txt", _MP4_FTYP_BYTES)
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(_validate_video_upload(upload))
    assert exc_info.value.status_code == 400
    assert "extension" in exc_info.value.detail


@pytest.mark.unit
def test_validate_video_upload_rejects_missing_filename() -> None:
    upload = _make_upload(None, _MP4_FTYP_BYTES)
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(_validate_video_upload(upload))
    assert exc_info.value.status_code == 400


@pytest.mark.unit
def test_analyze_video_rejects_unsupported_extension(client: TestClient) -> None:
    response = client.post(
        "/analyze/video",
        files={"file": ("clip.txt", io.BytesIO(_MP4_FTYP_BYTES), "video/mp4")},
    )

    assert response.status_code == 400
    assert "extension" in response.json()["detail"]


@pytest.mark.unit
def test_analyze_video_async_rejects_unsupported_extension(client: TestClient) -> None:
    response = client.post(
        "/analyze/video/async",
        files={"file": ("clip.txt", io.BytesIO(_MP4_FTYP_BYTES), "video/mp4")},
    )

    assert response.status_code == 400
    assert "extension" in response.json()["detail"]
