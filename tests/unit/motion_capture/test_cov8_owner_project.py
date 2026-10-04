"""Unit tests for COV-8 Necromatcher owner project and marker-anchored anthropometry (#11276).

Tests adhere strictly to TDD, Design by Contract (DbC), Law of Demeter (LoD),
and DRY, verifying fail-closed privacy guards, provenance tracking, and
immutable versioning.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
import numpy as np
import pytest

from src.motion_capture.reference.owner_project import (
    MarkerAnchoredAnthropometry,
    OwnerPlayerProject,
    SegmentLengthEstimate,
    compute_marker_anchored_anthropometry,
    create_owner_project,
    import_video_swings_to_owner_project,
)

pytestmark = pytest.mark.unit


def _synthetic_joint_centres_swings(
    known_lengths: dict[str, float],
    num_swings: int = 13,
    noise_std: float = 0.001,
) -> dict[str, np.ndarray]:
    """Generate synthetic joint-centre derived segment lengths across swings."""
    rng = np.random.default_rng(seed=42)
    result = {}
    for seg, length in known_lengths.items():
        samples = rng.normal(loc=length, scale=noise_std, size=num_swings)
        result[seg] = samples
    return result


def _synthetic_capture_run_dir(
    directory: Path,
    subject_id: str = "subject-O",
    frame_count: int = 3,
) -> Path:
    """Create a minimal synthetic historical-capture run directory for testing."""
    import cv2
    from src.shared.python.shadow_tracker.ingestion import compute_frame_hash

    directory.mkdir(parents=True, exist_ok=True)
    width, height = 64, 64
    observations = []
    receipt_source = {
        "schema_version": "shadow-tracker/source/1.0.0",
        "asset_id": "source-synthetic",
        "source_uri": "urn:asset:source-synthetic",
        "content_sha256": "0" * 64,
        "width_px": width,
        "height_px": height,
        "rights_status": "permitted",
        "rights_note": "synthetic",
    }

    for i in range(frame_count):
        img: np.ndarray = np.zeros((height, width, 3), dtype=np.uint8)
        img[10:20, 10:20] = 255
        frame_id = f"frame-{i}"
        img_name = f"{frame_id}.png"
        img_path = directory / img_name
        cv2.imwrite(str(img_path), img, [cv2.IMWRITE_PNG_COMPRESSION, 1])

        frame_hash = compute_frame_hash(img.tobytes(), decoder_name="pyav")
        frame_identity = {
            "schema_version": "shadow-tracker/frame/1.1.0",
            "asset_id": "source-synthetic",
            "shot_id": f"{subject_id}-window",
            "swing_id": f"{subject_id}-swing",
            "camera_id": "source-cam",
            "frame_id": frame_id,
            "pts_ticks": i * 1000,
            "timebase_numerator": 1,
            "timebase_denominator": 30000,
            "physical_time_s": None,
            "physical_time_reason": "Historical playback scale is unverified",
            "frame_sha256": frame_hash,
            "timing_mode": "container_pts",
            "is_timing_exact": True,
            "clock_evidence": "synthetic test",
            "decoder_name": "pyav",
            "decoder_version": "test",
            "pixel_format": "bgr24",
        }
        obs = {
            "status": "detected",
            "coordinate_system": "normalized_image_xy",
            "confidence": 0.95,
            "landmarks": {
                "nose": {"x": 0.5, "y": 0.5, "visibility": 0.99},
                "left_shoulder": {"x": 0.45, "y": 0.6, "visibility": 0.95},
                "right_shoulder": {"x": 0.55, "y": 0.6, "visibility": 0.95},
            },
            "physical_time_s": None,
            "physical_time_reason": "Historical playback scale is unverified",
        }
        observations.append(
            {"frame": frame_identity, "image": img_name, "observation": obs}
        )

    obs_file = directory / "observations.jsonl"
    with obs_file.open("w", encoding="utf-8", newline="\n") as f:
        for row in observations:
            f.write(json.dumps(row) + "\n")

    obs_bytes = obs_file.read_bytes()
    observations_sha256 = hashlib.sha256(obs_bytes).hexdigest()

    receipt = {
        "schema_version": "historical-capture/1.0.0",
        "subject_id": subject_id,
        "source": receipt_source,
        "window_presentation_s": [0.0, float(frame_count) / 30.0],
        "frame_count": frame_count,
        "detected_count": frame_count,
        "observations_sha256": observations_sha256,
        "detector": {"name": "mediapipe", "version": "synthetic"},
        "qualification": "image_observations_only",
        "physical_time_verified": False,
        "shot_continuity_reviewed": False,
        "publication_permitted": False,
    }
    (directory / "receipt.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )
    return directory


class TestMarkerAnchoredAnthropometry:
    """TDD tests for marker-anchored anthropometry computation and validation."""

    def test_synthetic_marker_joint_centres_recovers_known_segment_lengths_and_spread(
        self,
    ) -> None:
        """Synthetic joint centres with known lengths recover lengths within tolerance with spread."""
        known = {
            "thigh_r": 0.422,
            "thigh_l": 0.422,
            "shank_r": 0.434,
            "shank_l": 0.434,
            "upper_arm_r": 0.282,
            "upper_arm_l": 0.282,
            "forearm_r": 0.269,
            "forearm_l": 0.269,
        }
        synthetic_data = _synthetic_joint_centres_swings(
            known, num_swings=13, noise_std=0.002
        )

        anthro = compute_marker_anchored_anthropometry(
            synthetic_data,
            subject_id="subject-O",
            height_m=1.83,
            mass_kg=84.0,
            height_source="owner_reported",
            mass_source="owner_reported",
        )

        assert isinstance(anthro, MarkerAnchoredAnthropometry)
        assert anthro.subject_id == "subject-O"
        assert anthro.height_m == pytest.approx(1.83)
        assert anthro.mass_kg == pytest.approx(84.0)
        assert anthro.height_source == "owner_reported"
        assert anthro.mass_source == "owner_reported"

        # Verify recovered observed lengths match known within 5 mm tolerance
        for seg, expected_len in known.items():
            est = anthro.segments[seg]
            assert isinstance(est, SegmentLengthEstimate)
            assert est.segment == seg
            assert est.length_m == pytest.approx(expected_len, abs=0.005)
            assert est.spread_m > 0.0
            assert est.sample_count == 13
            assert est.provenance == "observed (marker-derived)"

        # Unobserved segments (e.g. head, trunk) fallback to population prior
        assert "head" in anthro.segments
        head_est = anthro.segments["head"]
        assert head_est.provenance == "population-prior"
        assert head_est.spread_m == 0.0

    def test_synthetic_nonfinite_or_negative_lengths_raise_value_error_with_named_segment(
        self,
    ) -> None:
        """Nonfinite or negative lengths raise ValueError with the offending segment named."""
        with pytest.raises(ValueError, match="thigh_r"):
            compute_marker_anchored_anthropometry(
                {"thigh_r": np.array([float("nan"), 0.42])},
                subject_id="subject-O",
            )

        with pytest.raises(ValueError, match="upper_arm_l"):
            compute_marker_anchored_anthropometry(
                {"upper_arm_l": np.array([-0.25, -0.28])},
                subject_id="subject-O",
            )

        with pytest.raises(ValueError, match="shank_r"):
            compute_marker_anchored_anthropometry(
                {"shank_r": np.array([float("inf")])},
                subject_id="subject-O",
            )

    def test_synthetic_bilateral_asymmetry_beyond_declared_bound_raises_value_error(
        self,
    ) -> None:
        """Left/right asymmetry beyond declared bound raises ValueError naming the segment."""
        # Declared max asymmetry 5%, but thigh_r (0.40) and thigh_l (0.48) differ by ~17%
        asymmetric_data = {
            "thigh_r": np.array([0.40] * 5),
            "thigh_l": np.array([0.48] * 5),
        }
        with pytest.raises(ValueError, match="thigh"):
            compute_marker_anchored_anthropometry(
                asymmetric_data,
                subject_id="subject-O",
                max_bilateral_asymmetry=0.05,
            )

    def test_synthetic_missing_provenance_tag_raises_value_error(self) -> None:
        """An anchored subject record segment with missing or invalid provenance raises ValueError."""
        with pytest.raises(ValueError, match="provenance"):
            SegmentLengthEstimate(
                segment="thigh",
                length_m=0.42,
                spread_m=0.001,
                sample_count=5,
                provenance="",
            )

        with pytest.raises(ValueError, match="provenance"):
            SegmentLengthEstimate(
                segment="thigh",
                length_m=0.42,
                spread_m=0.001,
                sample_count=5,
                provenance="arbitrary-unverified",
            )


class TestOwnerPlayerProjectPrivacyAndImport:
    """TDD tests for owner player project privacy guards and immutable-version import."""

    def test_synthetic_player_library_outside_capture_data_dir_refused_by_privacy_guard(
        self, tmp_path: Path
    ) -> None:
        """A player library rooted outside CAPTURE_DATA_DIR with private=True raises ValueError."""
        capture_data_dir = tmp_path / "valid_capture_root"
        capture_data_dir.mkdir(parents=True, exist_ok=True)
        outside_library_dir = tmp_path / "outside_root" / "necromatcher"

        with pytest.raises(ValueError, match="outside CAPTURE_DATA_DIR"):
            create_owner_project(
                outside_library_dir,
                capture_data_dir=capture_data_dir,
                private=True,
            )

        # Inside capture_data_dir succeeds
        inside_library_dir = capture_data_dir / "capture-O-video" / "necromatcher"
        project = create_owner_project(
            inside_library_dir,
            capture_data_dir=capture_data_dir,
            private=True,
        )
        assert isinstance(project, OwnerPlayerProject)
        assert project.player_id == "subject-O"
        assert project.library_root == inside_library_dir.resolve()

    def test_synthetic_reimport_same_capture_archive_immutable_version_behavior(
        self, tmp_path: Path
    ) -> None:
        """Re-importing the same capture archive preserves immutable version without overwrite."""
        capture_data_dir = tmp_path / "capture_dir"
        inside_library_dir = capture_data_dir / "necromatcher"
        project = create_owner_project(
            inside_library_dir,
            capture_data_dir=capture_data_dir,
            private=True,
        )

        capture_run = _synthetic_capture_run_dir(tmp_path / "cov5_run")
        swing_spec = {
            "swing_id": "cov-01-s1",
            "name": "Swing 1",
            "grade": "A",
            "view": "face_on",
            "physical_clock": "unknown",
            "capture_dir": capture_run,
        }

        # First import
        imported_first = import_video_swings_to_owner_project(project, [swing_spec])
        assert "cov-01-s1" in imported_first
        assets_first = project.library.assets("cov-01-s1")
        assert len(assets_first) == 1
        first_hash = assets_first[0].metadata["hash"]

        # Re-import identical capture archive
        imported_second = import_video_swings_to_owner_project(project, [swing_spec])
        assert "cov-01-s1" in imported_second
        assets_second = project.library.assets("cov-01-s1")
        # Invariant: no duplicate asset created, hash is identical
        assert len(assets_second) == 1
        assert assets_second[0].metadata["hash"] == first_hash
