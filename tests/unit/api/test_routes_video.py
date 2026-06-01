"""Unit tests for the video API route."""

import io
import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.testclient import TestClient
from starlette.datastructures import Headers

from src.api.routes.video import (
    _looks_like_supported_video_bytes,
    _validate_video_upload,
    _process_video_background,
    router,
)
from src.api.dependencies import get_task_manager, get_video_pipeline


@pytest.fixture(autouse=True)
def disable_contracts():
    from src.shared.python._contracts_level import ContractLevel

    # Mock both level retrieval methods to completely bypass DBC precondition enforcement in tests
    with (
        patch(
            "src.shared.python.core.contracts.level.get_contract_level",
            return_value=ContractLevel.OFF,
        ),
        patch(
            "src.shared.python._contracts_level.get_contract_level",
            return_value=ContractLevel.OFF,
        ),
    ):
        yield


# ── Mocking pipeline classes and outputs ──


class MockPoseResult:
    def __init__(
        self,
        timestamp: float,
        confidence: float,
        joint_angles: dict,
        raw_keypoints: dict,
    ):
        self.timestamp = timestamp
        self.confidence = confidence
        self.joint_angles = joint_angles
        self.raw_keypoints = raw_keypoints


class MockPipelineResult:
    def __init__(self):
        self.total_frames = 10
        self.valid_frames = 8
        self.average_confidence = 0.85
        self.quality_metrics = {"blurriness": 0.12}
        self.pose_results = [
            MockPoseResult(0.0, 0.9, {"knee": 1.2}, {"kp1": [0.1, 0.2]}),
            MockPoseResult(0.1, 0.8, {"knee": 1.3}, {"kp1": [0.1, 0.3]}),
        ]


class MockVideoProcessingConfig:
    def __init__(
        self,
        estimator_type: str,
        min_confidence: float,
        enable_temporal_smoothing: bool = True,
    ):
        self.estimator_type = estimator_type
        self.min_confidence = min_confidence
        self.enable_temporal_smoothing = enable_temporal_smoothing


class MockVideoPosePipeline:
    def __init__(self, config):
        self.config = config

    def process_video(self, video_path: Path):
        # Trigger mock failures if test demands it
        if "fail_os" in str(video_path):
            raise OSError("Mock disk read failure")
        if "fail_runtime" in str(video_path):
            raise RuntimeError("Mock pipeline logic failure")
        return MockPipelineResult()


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


# ── Signature Detection Tests ──


def test_video_signature_detection_accepts_mp4_header() -> None:
    assert _looks_like_supported_video_bytes(b"\x00\x00\x00\x20ftypisommore-bytes")


def test_video_signature_detection_webm() -> None:
    assert _looks_like_supported_video_bytes(b"\x1a\x45\xdf\xa3mkv-or-webm")


def test_video_signature_detection_avi() -> None:
    assert _looks_like_supported_video_bytes(b"RIFF\x00\x00\x00\x00AVI ")


def test_video_signature_detection_rejects_non_video_header() -> None:
    assert not _looks_like_supported_video_bytes(b"PK\x03\x04not-a-video")
    assert not _looks_like_supported_video_bytes(b"short")


# ── Validation Tests ──


@pytest.mark.asyncio
async def test_validate_video_upload_content_type() -> None:
    file = UploadFile(
        filename="test.txt",
        file=io.BytesIO(b"abc"),
        headers=Headers({"content-type": "text/plain"}),
    )
    with pytest.raises(HTTPException) as exc_info:
        await _validate_video_upload(file)
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "File must be a video"


@pytest.mark.asyncio
async def test_validate_video_upload_size_limit() -> None:
    file = UploadFile(
        filename="large.mp4",
        file=io.BytesIO(b"abc"),
        headers=Headers({"content-type": "video/mp4"}),
    )
    # Artificially set size above limit
    file.size = 200 * 1024 * 1024
    with pytest.raises(HTTPException) as exc_info:
        await _validate_video_upload(file)
    assert exc_info.value.status_code == 413
    assert "too large" in exc_info.value.detail


# ── Post Request Validation Errors ──


def test_analyze_video_missing_file(client: TestClient) -> None:
    response = client.post("/analyze/video")
    assert response.status_code == 422  # Missing required field 'file'


@patch("src.api.routes.video._load_video_pipeline_classes")
def test_analyze_video_invalid_estimator(
    mock_load: MagicMock, client: TestClient
) -> None:
    mock_load.return_value = (MockVideoPosePipeline, MockVideoProcessingConfig)
    response = client.post(
        "/analyze/video",
        params={"estimator_type": "invalid_backend"},
        files={
            "file": (
                "test.mp4",
                io.BytesIO(b"\x00\x00\x00\x20ftypisommorebytes"),
                "video/mp4",
            )
        },
    )
    assert response.status_code == 400
    assert "Invalid estimator_type" in response.json()["detail"]


@patch("src.api.routes.video._load_video_pipeline_classes")
def test_analyze_video_invalid_confidence(
    mock_load: MagicMock, client: TestClient
) -> None:
    mock_load.return_value = (MockVideoPosePipeline, MockVideoProcessingConfig)
    response = client.post(
        "/analyze/video",
        params={"min_confidence": 1.5},
        files={
            "file": (
                "test.mp4",
                io.BytesIO(b"\x00\x00\x00\x20ftypisommorebytes"),
                "video/mp4",
            )
        },
    )
    assert response.status_code == 400
    assert "min_confidence must be between" in response.json()["detail"]


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


# ── Sync Analysis Happy Path and Exception Handling ──


@patch("src.api.routes.video._load_video_pipeline_classes")
def test_analyze_video_success(mock_load: MagicMock, client: TestClient) -> None:
    mock_load.return_value = (MockVideoPosePipeline, MockVideoProcessingConfig)

    response = client.post(
        "/analyze/video",
        params={"estimator_type": "mediapipe", "min_confidence": 0.6},
        files={
            "file": ("test.mp4", io.BytesIO(b"\x00\x00\x00\x20ftypisom"), "video/mp4")
        },
    )
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["filename"] == "test.mp4"
    assert json_data["total_frames"] == 10
    assert json_data["valid_frames"] == 8
    assert len(json_data["pose_data"]) == 2
    assert json_data["pose_data"][0]["confidence"] == 0.9
    assert json_data["pose_data"][0]["joint_angles"] == {"knee": 1.2}


@patch("src.api.routes.video._load_video_pipeline_classes")
def test_analyze_video_os_error(mock_load: MagicMock, client: TestClient) -> None:
    mock_load.return_value = (MockVideoPosePipeline, MockVideoProcessingConfig)

    # Use patch to make the tempfile suffix unique, so we can trigger OSError
    with patch("tempfile.mkstemp", return_value=(999, "fail_os.mp4")):
        response = client.post(
            "/analyze/video",
            files={
                "file": (
                    "test.mp4",
                    io.BytesIO(b"\x00\x00\x00\x20ftypisom"),
                    "video/mp4",
                )
            },
        )
        assert response.status_code == 500
        assert "Video analysis failed" in response.json()["detail"]


# ── Async Route and Background Worker ──


@patch("src.api.routes.video._load_video_pipeline_classes")
def test_analyze_video_async_validation_errors(
    mock_load: MagicMock, client: TestClient
) -> None:
    mock_load.return_value = (MockVideoPosePipeline, MockVideoProcessingConfig)
    # Invalid estimator
    response = client.post(
        "/analyze/video/async",
        params={"estimator_type": "invalid_backend"},
        files={
            "file": (
                "test.mp4",
                io.BytesIO(b"\x00\x00\x00\x20ftypisommorebytes"),
                "video/mp4",
            )
        },
    )
    assert response.status_code == 400

    # Invalid confidence
    response = client.post(
        "/analyze/video/async",
        params={"min_confidence": -0.5},
        files={
            "file": (
                "test.mp4",
                io.BytesIO(b"\x00\x00\x00\x20ftypisommorebytes"),
                "video/mp4",
            )
        },
    )
    assert response.status_code == 400


@patch("src.api.routes.video._load_video_pipeline_classes")
def test_analyze_video_async_success(
    mock_load: MagicMock, client: TestClient, mock_task_manager: MockTaskManager
) -> None:
    mock_load.return_value = (MockVideoPosePipeline, MockVideoProcessingConfig)

    response = client.post(
        "/analyze/video/async",
        params={"estimator_type": "mediapipe", "min_confidence": 0.5},
        files={
            "file": ("test.mp4", io.BytesIO(b"\x00\x00\x00\x20ftypisom"), "video/mp4")
        },
    )
    assert response.status_code == 200
    res = response.json()
    assert "task_id" in res
    assert res["status"] == "pending"

    # Verify task was set in task manager and ran to completion synchronously in TestClient
    task_id = res["task_id"]
    task_meta = mock_task_manager.get(task_id)
    assert task_meta is not None
    assert task_meta["status"] == "completed"
    assert task_meta["result"]["filename"] == "test.mp4"
    assert task_meta["result"]["estimator_type"] == "mediapipe"


@pytest.mark.asyncio
@patch("src.api.routes.video._load_video_pipeline_classes")
async def test_process_video_background_success(mock_load: MagicMock) -> None:
    mock_load.return_value = (MockVideoPosePipeline, MockVideoProcessingConfig)
    task_manager = MockTaskManager()
    task_id = "test-bg-task"
    task_manager.set(task_id, {"status": "pending", "created_at": "some-date"})

    # Setup dummy temp video file
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(b"dummy content")
        temp_path = Path(f.name)

    try:
        await _process_video_background(
            task_id=task_id,
            video_path=temp_path,
            filename="user_video.mp4",
            estimator_type="mediapipe",
            min_confidence=0.5,
            input_hash="hash123",
            task_manager=task_manager,
        )

        task_state = task_manager.get(task_id)
        assert task_state["status"] == "completed"
        assert task_state["result"]["total_frames"] == 10
        assert task_state["result"]["average_confidence"] == 0.85
        assert not temp_path.exists()  # Cleaned up
    finally:
        if temp_path.exists():
            temp_path.unlink()


@pytest.mark.asyncio
@patch("src.api.routes.video._load_video_pipeline_classes")
async def test_process_video_background_failure(mock_load: MagicMock) -> None:
    mock_load.return_value = (MockVideoPosePipeline, MockVideoProcessingConfig)
    task_manager = MockTaskManager()
    task_id = "test-bg-task-fail"
    task_manager.set(task_id, {"status": "pending"})

    # Create dummy temp file named to trigger failure in mock pipeline
    temp_path = Path("fail_runtime.mp4")
    temp_path.touch()

    try:
        await _process_video_background(
            task_id=task_id,
            video_path=temp_path,
            filename="user_video.mp4",
            estimator_type="mediapipe",
            min_confidence=0.5,
            input_hash="hash123",
            task_manager=task_manager,
        )

        task_state = task_manager.get(task_id)
        assert task_state["status"] == "failed"
        assert "Mock pipeline logic failure" in task_state["error"]
        assert not temp_path.exists()  # Cleaned up
    finally:
        if temp_path.exists():
            temp_path.unlink()
