"""Tests for Phase 5 API: Tool pages - Putting Green, Data Explorer, Motion Capture.

Validates Pydantic contract models and route logic for:
- Putting Green simulation endpoints (#1206)
- Data Explorer dataset browsing endpoints (#1206)
- Motion Capture skeleton/session endpoints (#1206)

See issue #1206
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.api.routes.data_explorer import (
    DatasetFilterRequest,
    DatasetInfo,
    DatasetListResponse,
    DatasetPreviewResponse,
    DatasetStatsResponse,
    ImportResponse,
)
from src.api.routes.motion_capture import (
    CaptureSessionRequest,
    CaptureSessionResponse,
    CaptureSource,
    JointData,
    PlaybackRequest,
    PlaybackResponse,
    RecordingInfo,
    SkeletonFrame,
)
from src.api.routes.putting_green import (
    GreenContourResponse,
    GreenReadingRequest,
    GreenReadingResponse,
    PuttSimulationRequest,
    PuttSimulationResponse,
    ScatterAnalysisRequest,
    ScatterAnalysisResponse,
)

# ──────────────────────────────────────────────────────────────
#  Contract Tests: Putting Green (#1206)
# ──────────────────────────────────────────────────────────────


class TestPuttSimulationRequestContract:
    """Validate PuttSimulationRequest model."""

    def test_default_values(self) -> None:
        """Defaults should be sensible."""
        req = PuttSimulationRequest()
        assert req.ball_x == 5.0
        assert req.ball_y == 10.0
        assert req.speed == 2.0
        assert req.stimp_rating == 10.0
        assert req.green_width == 20.0
        assert req.green_height == 20.0

    def test_valid_speed_range(self) -> None:
        """Speed within range accepted."""
        req = PuttSimulationRequest(speed=0.5)
        assert req.speed == 0.5
        req2 = PuttSimulationRequest(speed=10.0)
        assert req2.speed == 10.0

    def test_invalid_speed_zero(self) -> None:
        """Speed at zero rejected."""
        with pytest.raises(ValidationError):
            PuttSimulationRequest(speed=0)

    def test_invalid_speed_too_high(self) -> None:
        """Speed over max rejected."""
        with pytest.raises(ValidationError):
            PuttSimulationRequest(speed=15.0)

    def test_invalid_stimp_too_low(self) -> None:
        """Stimp below minimum rejected."""
        with pytest.raises(ValidationError):
            PuttSimulationRequest(stimp_rating=3.0)

    def test_invalid_stimp_too_high(self) -> None:
        """Stimp above maximum rejected."""
        with pytest.raises(ValidationError):
            PuttSimulationRequest(stimp_rating=20.0)


class TestPuttSimulationResponseContract:
    """Validate PuttSimulationResponse model."""

    def test_valid_response(self) -> None:
        """Valid response parses correctly."""
        resp = PuttSimulationResponse(
            positions=[[5.0, 10.0], [5.0, 12.0], [5.0, 14.5]],
            velocities=[[0.0, 2.0], [0.0, 1.5], [0.0, 0.2]],
            times=[0.0, 0.5, 1.5],
            holed=False,
            final_position=[5.0, 14.5],
            total_distance=4.5,
            duration=1.5,
        )
        assert len(resp.positions) == 3
        assert resp.holed is False
        assert resp.total_distance == 4.5

    def test_holed_response(self) -> None:
        """Holed putt response."""
        resp = PuttSimulationResponse(
            positions=[[10.0, 5.0], [10.0, 15.0]],
            velocities=[[0.0, 2.0], [0.0, 0.0]],
            times=[0.0, 2.5],
            holed=True,
            final_position=[10.0, 15.0],
            total_distance=10.0,
            duration=2.5,
        )
        assert resp.holed is True


class TestGreenReadingContract:
    """Validate GreenReading request/response."""

    def test_reading_request_defaults(self) -> None:
        """Defaults populate correctly."""
        req = GreenReadingRequest()
        assert req.ball_x == 5.0
        assert req.target_x == 10.0

    def test_reading_response(self) -> None:
        """Valid reading response."""
        resp = GreenReadingResponse(
            distance=11.0,
            total_break=0.05,
            recommended_speed=2.1,
            aim_point=[10.02, 14.95],
            elevations=[0.0, 0.01],
            slopes=[[0.001, 0.002]],
        )
        assert resp.distance == 11.0
        assert resp.recommended_speed > 0


class TestScatterAnalysisContract:
    """Validate ScatterAnalysis request/response."""

    def test_scatter_request_defaults(self) -> None:
        """Defaults populate correctly."""
        req = ScatterAnalysisRequest()
        assert req.n_simulations == 10
        assert req.speed_variance == 0.1

    def test_invalid_n_simulations(self) -> None:
        """Zero simulations rejected."""
        with pytest.raises(ValidationError):
            ScatterAnalysisRequest(n_simulations=0)

    def test_scatter_response(self) -> None:
        """Valid scatter response."""
        resp = ScatterAnalysisResponse(
            final_positions=[[10.0, 14.9], [10.1, 15.0]],
            holed_count=1,
            total_simulations=2,
            average_distance_from_hole=0.08,
            make_percentage=50.0,
        )
        assert resp.holed_count == 1
        assert resp.make_percentage == 50.0


class TestGreenContourContract:
    """Validate GreenContourResponse."""

    def test_contour_response(self) -> None:
        """Valid contour response."""
        resp = GreenContourResponse(
            width=20.0,
            height=20.0,
            grid_x=[[0.0, 10.0], [0.0, 10.0]],
            grid_y=[[0.0, 0.0], [10.0, 10.0]],
            elevations=[[0.0, 0.01], [0.02, 0.03]],
            hole_position=[10.0, 10.0],
        )
        assert resp.width == 20.0
        assert len(resp.elevations) == 2


# ──────────────────────────────────────────────────────────────
#  Contract Tests: Data Explorer (#1206)
# ──────────────────────────────────────────────────────────────


class TestDatasetInfoContract:
    """Validate DatasetInfo model."""

    def test_basic_dataset_info(self) -> None:
        """Basic info parses correctly."""
        info = DatasetInfo(
            name="results.csv",
            path="/output/results.csv",
            format="csv",
            size_bytes=45200,
            columns=["time", "x", "y"],
        )
        assert info.name == "results.csv"
        assert len(info.columns) == 3


class TestDatasetListResponseContract:
    """Validate DatasetListResponse."""

    def test_list_response(self) -> None:
        """Valid list response."""
        resp = DatasetListResponse(
            datasets=[
                DatasetInfo(
                    name="a.csv", path="/a.csv", format="csv",
                    size_bytes=100, columns=["x"],
                ),
            ],
            total=1,
            search_dir="/output",
        )
        assert resp.total == 1
        assert len(resp.datasets) == 1


class TestDatasetPreviewResponseContract:
    """Validate DatasetPreviewResponse."""

    def test_preview_response(self) -> None:
        """Valid preview response."""
        resp = DatasetPreviewResponse(
            name="test.csv",
            columns=["a", "b"],
            rows=[{"a": "1", "b": "2"}, {"a": "3", "b": "4"}],
            total_rows=2,
            format="csv",
        )
        assert len(resp.rows) == 2
        assert resp.total_rows == 2


class TestDatasetFilterRequestContract:
    """Validate DatasetFilterRequest."""

    def test_filter_request(self) -> None:
        """Valid filter request."""
        req = DatasetFilterRequest(
            column="force",
            operator="gt",
            value="50.0",
        )
        assert req.column == "force"
        assert req.operator == "gt"

    def test_filter_default_limit(self) -> None:
        """Default limit is 100."""
        req = DatasetFilterRequest(column="x", value="1")
        assert req.limit == 100


class TestDatasetStatsResponseContract:
    """Validate DatasetStatsResponse."""

    def test_stats_response(self) -> None:
        """Valid stats response."""
        resp = DatasetStatsResponse(
            name="test.csv",
            columns=["time", "force"],
            row_count=100,
            stats={
                "time": {"min": 0.0, "max": 10.0, "mean": 5.0, "count": 100.0},
                "force": {"min": -50.0, "max": 150.0, "mean": 42.5, "count": 100.0},
            },
        )
        assert resp.row_count == 100
        assert resp.stats["time"]["mean"] == 5.0


class TestImportResponseContract:
    """Validate ImportResponse."""

    def test_import_response(self) -> None:
        """Valid import response."""
        resp = ImportResponse(
            name="uploaded.csv",
            format="csv",
            columns=["a", "b", "c"],
            row_count=50,
        )
        assert resp.row_count == 50
        assert len(resp.columns) == 3


# ──────────────────────────────────────────────────────────────
#  Contract Tests: Motion Capture (#1206)
# ──────────────────────────────────────────────────────────────


class TestCaptureSourceContract:
    """Validate CaptureSource model."""

    def test_mediapipe_source(self) -> None:
        """MediaPipe source info."""
        source = CaptureSource(
            id="mediapipe",
            name="MediaPipe Pose",
            type="mediapipe",
            available=True,
            description="Real-time pose estimation",
        )
        assert source.available is True
        assert source.type == "mediapipe"

    def test_unavailable_source(self) -> None:
        """Unavailable source."""
        source = CaptureSource(
            id="openpose",
            name="OpenPose",
            type="openpose",
            available=False,
            description="OpenPose library not installed",
        )
        assert source.available is False


class TestJointDataContract:
    """Validate JointData model."""

    def test_root_joint(self) -> None:
        """Root joint has no parent."""
        joint = JointData(
            name="nose",
            position=[0.0, 0.8, 0.0],
            confidence=0.95,
            parent=None,
        )
        assert joint.parent is None
        assert joint.confidence == 0.95

    def test_child_joint(self) -> None:
        """Child joint references parent."""
        joint = JointData(
            name="left_elbow",
            position=[-0.25, 0.4, 0.0],
            confidence=0.88,
            parent="left_shoulder",
        )
        assert joint.parent == "left_shoulder"

    def test_confidence_bounds(self) -> None:
        """Confidence must be in [0, 1]."""
        with pytest.raises(ValidationError):
            JointData(
                name="test",
                position=[0, 0, 0],
                confidence=1.5,
            )
        with pytest.raises(ValidationError):
            JointData(
                name="test",
                position=[0, 0, 0],
                confidence=-0.1,
            )


class TestCaptureSessionContract:
    """Validate CaptureSession request/response."""

    def test_session_request_default(self) -> None:
        """Default source type is mediapipe."""
        req = CaptureSessionRequest()
        assert req.source_type == "mediapipe"
        assert req.frame_rate == 30.0

    def test_session_response(self) -> None:
        """Valid session response."""
        resp = CaptureSessionResponse(
            session_id="session_1",
            status="recording",
            source_type="mediapipe",
            message="Started",
        )
        assert resp.status == "recording"


class TestRecordingInfoContract:
    """Validate RecordingInfo model."""

    def test_recording_info(self) -> None:
        """Valid recording info."""
        info = RecordingInfo(
            name="rec_1",
            source_type="mediapipe",
            total_frames=300,
            duration_seconds=10.0,
            frame_rate=30.0,
            joint_names=["nose", "left_shoulder"],
        )
        assert info.total_frames == 300
        assert info.frame_rate == 30.0


class TestPlaybackContract:
    """Validate Playback request/response."""

    def test_playback_request(self) -> None:
        """Valid playback request."""
        req = PlaybackRequest(
            recording_name="rec_1",
            action="play",
        )
        assert req.action == "play"
        assert req.seek_frame is None

    def test_playback_seek(self) -> None:
        """Seek request with frame."""
        req = PlaybackRequest(
            recording_name="rec_1",
            action="seek",
            seek_frame=150,
        )
        assert req.seek_frame == 150

    def test_playback_response(self) -> None:
        """Valid playback response."""
        resp = PlaybackResponse(
            recording_name="rec_1",
            status="playing",
            current_frame=0,
            total_frames=300,
        )
        assert resp.status == "playing"


class TestSkeletonFrameContract:
    """Validate SkeletonFrame model."""

    def test_skeleton_frame(self) -> None:
        """Valid skeleton frame."""
        frame = SkeletonFrame(
            frame_index=42,
            timestamp=1.4,
            joints=[
                JointData(
                    name="nose",
                    position=[0.0, 0.8, 0.0],
                    confidence=0.95,
                ),
                JointData(
                    name="left_shoulder",
                    position=[-0.15, 0.6, 0.0],
                    confidence=0.92,
                    parent="nose",
                ),
            ],
        )
        assert frame.frame_index == 42
        assert len(frame.joints) == 2
