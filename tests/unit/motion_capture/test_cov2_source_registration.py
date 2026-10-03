"""Unit tests for capture-O video source registration, timing, swing windows, and grading (COV-2, #11270).

Tests adhere strictly to TDD, Design by Contract (DbC), Law of Demeter (LoD),
and verify fail-closed contracts for video source registration.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import pytest

from src.motion_capture.capture_registry import (
    CaptureDataUnavailable,
    CaptureInfo,
    CaptureIntegrityError,
    capture_info,
    resolve_capture,
)
from src.shared.python.shadow_tracker.source_records import (
    SwingGradeResult,
    SwingWindow,
    VariableFrameRateError,
    VideoTimingEvidence,
    extract_video_timing_evidence,
    grade_swing_window,
    validate_swing_windows,
)
from src.shared.python.workspace.necromatcher_video import (
    source_frame_rate,
)

pytestmark = pytest.mark.unit


def _synthetic_probe_data(
    *,
    r_frame_rate: str = "30/1",
    avg_frame_rate: str = "30/1",
    tags: dict[str, Any] | None = None,
    format_tags: dict[str, Any] | None = None,
    side_data_list: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Synthetic ffprobe JSON result for deterministic unit testing."""
    stream_dict: dict[str, Any] = {
        "codec_type": "video",
        "width": 1920,
        "height": 1080,
        "r_frame_rate": r_frame_rate,
        "avg_frame_rate": avg_frame_rate,
        "tags": tags or {},
    }
    if side_data_list is not None:
        stream_dict["side_data_list"] = side_data_list

    return {
        "streams": [stream_dict],
        "format": {
            "filename": "synthetic_clip.mp4",
            "format_name": "mov,mp4",
            "duration": "10.000000",
            "tags": format_tags or {},
        },
    }


class TestCOV2SourceRegistration:
    """TDD tests for issue #11270 contracts."""

    def test_missing_slow_motion_tags_playback_30fps_physical_clock_is_unknown(
        self,
    ) -> None:
        """Missing slow-motion tags + playback 30 fps -> physical_clock == 'unknown', never 'known'."""
        probe = _synthetic_probe_data(
            r_frame_rate="30/1",
            avg_frame_rate="30/1",
            tags={"creation_time": "2019-05-18T14:22:10.000000Z"},
        )
        evidence = extract_video_timing_evidence(probe)
        assert evidence.physical_clock == "unknown"
        assert evidence.container_fps == 30.0
        assert not evidence.is_vfr

        # DbC: explicitly attempting to construct known physical_clock without slow-motion tags must raise ValueError
        with pytest.raises(
            ValueError,
            match="physical_clock cannot be known or ratio_known without slow-motion tags",
        ):
            VideoTimingEvidence(
                container_fps=30.0,
                r_frame_rate="30/1",
                avg_frame_rate="30/1",
                is_vfr=False,
                physical_clock="known",
                slow_motion_tags={},
            )

    def test_vfr_stream_flagged_and_uniform_pts_consumer_raises_typed_error(
        self,
    ) -> None:
        """VFR stream -> flagged and uniform-PTS consumer raises typed error (VariableFrameRateError)."""
        probe_vfr = _synthetic_probe_data(
            r_frame_rate="60/1",
            avg_frame_rate="29.97/1",
        )
        evidence = extract_video_timing_evidence(probe_vfr)
        assert evidence.is_vfr is True

        # Uniform-PTS consumer source_frame_rate refuses VFR sequence with VariableFrameRateError
        indices = [0, 1, 2]
        # Non-uniform timestamps (e.g. 0/30, 1/30, 3/30)
        frames = [
            {"pts_ticks": 0, "timebase_numerator": 1, "timebase_denominator": 30},
            {"pts_ticks": 1, "timebase_numerator": 1, "timebase_denominator": 30},
            {"pts_ticks": 3, "timebase_numerator": 1, "timebase_denominator": 30},
        ]
        with pytest.raises(VariableFrameRateError) as exc_info:
            source_frame_rate(indices, frames)
        assert isinstance(exc_info.value, ValueError)

    def test_rotation_metadata_90_frames_reported_with_explicit_flag(self) -> None:
        """Rotation metadata 90° -> explicitly flagged so consumers know rotation is required."""
        # 1. Via stream tags rotate = "90"
        probe_tag = _synthetic_probe_data(tags={"rotate": "90"})
        evidence_tag = extract_video_timing_evidence(probe_tag)
        assert evidence_tag.rotation_degrees == 90
        assert evidence_tag.rotation_applied is False

        # 2. Via displaymatrix side data (-90 or 270 -> normalized 270 / 90)
        probe_side = _synthetic_probe_data(
            side_data_list=[{"side_data_type": "Displaymatrix", "rotation": 90}]
        )
        evidence_side = extract_video_timing_evidence(probe_side)
        assert evidence_side.rotation_degrees == 90
        assert evidence_side.rotation_applied is False

    def test_overlapping_swing_windows_in_one_clip_raises_value_error(self) -> None:
        """Overlapping swing windows in one clip -> ValueError."""
        w1 = SwingWindow(
            swing_id="cov-01-s1",
            clip_id="cov-01",
            start_pts_s=1.0,
            end_pts_s=3.0,
            view="face_on",
        )
        w2_overlap = SwingWindow(
            swing_id="cov-01-s2",
            clip_id="cov-01",
            start_pts_s=2.5,
            end_pts_s=4.5,
            view="down_the_line",
        )
        w_non_overlap = SwingWindow(
            swing_id="cov-01-s2",
            clip_id="cov-01",
            start_pts_s=3.0,
            end_pts_s=5.0,
            view="down_the_line",
        )

        with pytest.raises(ValueError, match="Overlapping swing windows"):
            validate_swing_windows([w1, w2_overlap])

        # Non-overlapping must validate cleanly
        validate_swing_windows([w1, w_non_overlap])

    def test_grade_without_at_least_one_reason_raises_value_error(self) -> None:
        """A grade without at least one reason -> ValueError."""
        window = SwingWindow(
            swing_id="cov-01-s1",
            clip_id="cov-01",
            start_pts_s=1.0,
            end_pts_s=3.0,
            view="face_on",
        )
        # Direct construction of SwingGradeResult without reasons must fail
        with pytest.raises(ValueError, match="A grade must have at least one reason"):
            SwingGradeResult(grade="A", reasons=())

        # Calling grade_swing_window with empty reasons must raise ValueError
        with pytest.raises(ValueError, match="A grade must have at least one reason"):
            grade_swing_window(window, reasons=())

        # Calling grade_swing_window without manual reasons produces valid result with reasons
        result = grade_swing_window(window)
        assert result.grade in ("A", "B", "C", "R")
        assert len(result.reasons) >= 1
        # Test tuple unpacking
        grade, reasons = result
        assert grade == result.grade
        assert reasons == result.reasons

    def test_registry_entry_sha256_differs_from_file_raises_hash_mismatch(
        self, tmp_path: Path
    ) -> None:
        """A registry entry whose SHA-256 differs from the file -> hash-mismatch error via resolver."""
        rel_path = Path("capture-O-video/originals/cov-01.mp4")
        clip_file = tmp_path / rel_path
        clip_file.parent.mkdir(parents=True, exist_ok=True)
        clip_file.write_bytes(b"actual file bytes on disk")

        # Resolve capture with mismatched registered SHA-256
        with pytest.raises(CaptureIntegrityError):
            resolve_capture("capture-O-video/cov-01", data_dir=tmp_path)

    def test_public_catalog_entry_for_private_source_contains_no_filename_url_title(
        self,
    ) -> None:
        """Public catalog entry for private source contains no filename, URL, or title."""
        repo_root = Path(__file__).resolve().parents[3]
        catalog_path = (
            repo_root
            / "docs"
            / "development"
            / "historical_capture"
            / "source-catalog.json"
        )
        assert catalog_path.is_file(), f"Catalog not found at {catalog_path}"

        with catalog_path.open("r", encoding="utf-8") as f:
            catalog = json.load(f)

        sources = catalog.get("sources", [])
        private_entries = [s for s in sources if s.get("private") is True]
        assert len(private_entries) >= 1, (
            "Expected at least one private entry in source-catalog.json"
        )

        for entry in private_entries:
            # Privacy invariant: no filename, URL or title in public entry
            assert (
                "original_filename" not in entry
                or entry.get("original_filename") is None
            )
            assert "source_url" not in entry or entry.get("source_url") is None
            assert "title" not in entry or entry.get("title") is None
            assert entry.get("private") is True
            assert "sha256" in entry
