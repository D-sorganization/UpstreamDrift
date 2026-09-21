"""Unit tests for Video Ingestion, Shots, Timing, and Capture Evidence (ST-03).

Tests verify:
- SourceAsset ingestion and cataloging with content checksum and rights metadata.
- Rejection of corrupt, truncated, or missing media files.
- Variable PTS support without constant-FPS assumptions.
- Cuts handling and filtering within shot intervals.
- Rejection of cross-swing fusion and mixed camera/asset identities.
- Preservation of image transform provenance (mirror, crop, rotation, slow motion).
- Unknown physical time preservation for archive clips.
- Slow motion, affine timing, and piecewise timing mappings.
- Telecine duplicate detection with auditable frame hashes.
- Multi-camera synchronization with offset and drift uncertainty.
- Capture Rig adapter integration.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
import hashlib
from pathlib import Path
from typing import Any
import pytest

from shared.python.shadow_tracker._validation import (
    FRAME_SCHEMA_VERSION,
    SHOT_SCHEMA_VERSION,
    SOURCE_SCHEMA_VERSION,
)
from shared.python.shadow_tracker.contracts import (
    FrameObservation,
    Shot,
)
from shared.python.shadow_tracker.ingestion import (
    AffineTimingMapping,
    CameraSynchronization,
    PiecewiseTimingMapping,
    ShotDefinition,
    SourceCatalog,
    create_shot,
    detect_telecine_duplicates,
    filter_shot_frames,
    ingest_source_asset,
    map_frame_to_observation,
    validate_shot_frames,
)
from shared.python.shadow_tracker.source_records import (
    FrameIdentity,
    SourceAsset,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def sample_video_file(tmp_path: Path) -> Path:
    """Create a minimal valid synthetic media file fixture."""
    file_path = tmp_path / "swing_sample.bin"
    # A distinct binary payload simulating valid container content
    content = (
        b"RIFF\x24\x00\x00\x00AVI LIST\x18\x00\x00\x00hdrlavih8\x00\x00\x00"
        + b"\x00" * 64
    )
    file_path.write_bytes(content)
    return file_path


@pytest.fixture
def corrupt_video_file(tmp_path: Path) -> Path:
    """Create a truncated/corrupted media file fixture."""
    file_path = tmp_path / "corrupt_sample.bin"
    file_path.write_bytes(b"CORR")  # Truncated header
    return file_path


# ---------------------------------------------------------------------------
# 1. Ingestion & Catalog Tests
# ---------------------------------------------------------------------------


def test_ingest_source_asset_success(sample_video_file: Path) -> None:
    expected_sha = hashlib.sha256(sample_video_file.read_bytes()).hexdigest()
    asset = ingest_source_asset(
        sample_video_file,
        asset_id="asset-001",
        width_px=1920,
        height_px=1080,
        rights_status="permitted",
        rights_note="License: CC-BY-4.0",
        source_uri="https://assets.upstreamdrift.org/swings/001.mp4",
    )
    assert isinstance(asset, SourceAsset)
    assert asset.asset_id == "asset-001"
    assert asset.content_sha256 == expected_sha
    assert asset.width_px == 1920
    assert asset.height_px == 1080
    assert asset.rights_status == "permitted"
    assert asset.rights_note == "License: CC-BY-4.0"
    assert asset.source_uri == "https://assets.upstreamdrift.org/swings/001.mp4"


def test_ingest_source_asset_corrupt_or_missing(
    tmp_path: Path, corrupt_video_file: Path
) -> None:
    # Missing file
    non_existent = tmp_path / "nonexistent.mp4"
    with pytest.raises((FileNotFoundError, ValueError)):
        ingest_source_asset(
            non_existent,
            asset_id="asset-missing",
            width_px=1280,
            height_px=720,
        )

    # Corrupt/empty file
    empty_file = tmp_path / "empty.mp4"
    empty_file.write_bytes(b"")
    with pytest.raises(ValueError, match="corrupt or empty"):
        ingest_source_asset(
            empty_file,
            asset_id="asset-empty",
            width_px=1280,
            height_px=720,
        )


def test_source_catalog_registration_and_lookup(sample_video_file: Path) -> None:
    catalog = SourceCatalog()
    asset = ingest_source_asset(
        sample_video_file,
        asset_id="asset-001",
        width_px=1920,
        height_px=1080,
        rights_status="unknown",
        rights_note="",
    )
    catalog.register(asset)
    assert catalog.get("asset-001") == asset
    assert len(catalog) == 1

    # Duplicate registration of same asset is idempotent
    catalog.register(asset)
    assert len(catalog) == 1

    # Conflict with same ID but different hash raises ValueError
    conflicting_asset = SourceAsset(
        schema_version=SOURCE_SCHEMA_VERSION,
        asset_id="asset-001",
        source_uri="urn:asset:asset-001",
        content_sha256="0" * 64,
        width_px=1920,
        height_px=1080,
        rights_status="unknown",
        rights_note="",
    )
    with pytest.raises(ValueError, match="already registered with different content"):
        catalog.register(conflicting_asset)


# ---------------------------------------------------------------------------
# 2. Variable PTS & Timing Tests
# ---------------------------------------------------------------------------


def test_variable_pts_handling_without_fixed_fps() -> None:
    # PTS ticks irregularly spaced (e.g. variable frame rate: 0, 33, 71, 105, 142)
    pts_sequence = [0, 33, 71, 105, 142]
    frames = [
        FrameIdentity(
            schema_version=FRAME_SCHEMA_VERSION,
            asset_id="asset-01",
            shot_id="shot-01",
            swing_id="swing-01",
            camera_id="cam-01",
            frame_id=f"f-{i:03d}",
            pts_ticks=pts,
            timebase_numerator=1,
            timebase_denominator=1000,
            physical_time_s=pts / 1000.0,
            physical_time_reason="",
            frame_sha256=hashlib.sha256(f"frame-{i}".encode()).hexdigest(),
        )
        for i, pts in enumerate(pts_sequence)
    ]
    # Check that rational presentation times are exact
    assert frames[2].presentation_time == Fraction(71, 1000)
    assert frames[4].presentation_time == Fraction(142, 1000)

    # Frame sequence validation accepts variable PTS as long as strictly increasing
    from shared.python.shadow_tracker.source_records import validate_frame_sequence

    validate_frame_sequence(frames)


def test_unknown_physical_time_preservation() -> None:
    # Archive clip with unknown physical time
    frame = FrameIdentity(
        schema_version=FRAME_SCHEMA_VERSION,
        asset_id="asset-archive-1930",
        shot_id="shot-archive",
        swing_id="swing-jones-01",
        camera_id="cam-hand-crank",
        frame_id="f-001",
        pts_ticks=42,
        timebase_numerator=1,
        timebase_denominator=24,
        physical_time_s=None,
        physical_time_reason="archive footage with hand-cranked variable speed",
        frame_sha256="a" * 64,
    )
    assert frame.physical_time_s is None
    assert (
        frame.physical_time_reason == "archive footage with hand-cranked variable speed"
    )

    d = frame.to_dict()
    reloaded = FrameIdentity.from_dict(d)
    assert reloaded.physical_time_s is None
    assert reloaded.physical_time_reason == frame.physical_time_reason


# ---------------------------------------------------------------------------
# 3. Shots, Cuts & Fusion Rejection Tests
# ---------------------------------------------------------------------------


def test_shot_creation_and_cut_filtering() -> None:
    asset = SourceAsset(
        schema_version=SOURCE_SCHEMA_VERSION,
        asset_id="asset-01",
        source_uri="urn:asset:asset-01",
        content_sha256="1" * 64,
        width_px=1920,
        height_px=1080,
        rights_status="unknown",
        rights_note="",
    )

    shot = create_shot(
        asset,
        ShotDefinition(
            shot_id="shot-01",
            start_pts=100,
            end_pts=500,
            start_frame_id="f-010",
            end_frame_id="f-050",
            subject_id="sub-01",
            swing_id="swing-01",
            camera_id="cam-01",
            cuts=((200, 250),),
            transforms=("mirror_h", "crop:0,0,1280,720"),
        ),
    )

    assert isinstance(shot, Shot)
    assert shot.duration_pts == 400
    assert shot.cuts == ((200, 250),)
    assert shot.transforms == ("mirror_h", "crop:0,0,1280,720")

    # Generate candidate frames across [100, 500]
    all_frames = [
        FrameIdentity(
            schema_version=FRAME_SCHEMA_VERSION,
            asset_id="asset-01",
            shot_id="shot-01",
            swing_id="swing-01",
            camera_id="cam-01",
            frame_id=f"f-{pts:03d}",
            pts_ticks=pts,
            timebase_numerator=1,
            timebase_denominator=1000,
            physical_time_s=pts / 1000.0,
            physical_time_reason="",
            frame_sha256=hashlib.sha256(f"frame-{pts}".encode()).hexdigest(),
        )
        for pts in (100, 150, 220, 240, 260, 300, 500)
    ]

    filtered = filter_shot_frames(shot, all_frames)
    # Frames at 220 and 240 fall inside cut (200, 250) and should be excluded
    filtered_pts = [f.pts_ticks for f in filtered]
    assert filtered_pts == [100, 150, 260, 300, 500]


def test_rejection_of_different_swing_fusion() -> None:
    shot = Shot(
        schema_version=SHOT_SCHEMA_VERSION,
        asset_id="asset-01",
        shot_id="shot-01",
        start_pts=0,
        end_pts=100,
        start_frame_id="f-00",
        end_frame_id="f-10",
        subject_id="sub-01",
        swing_id="swing-01",
        camera_id="cam-01",
        cuts=(),
        transforms=(),
    )

    # Frame belonging to a different swing
    alien_frame = FrameIdentity(
        schema_version=FRAME_SCHEMA_VERSION,
        asset_id="asset-01",
        shot_id="shot-01",
        swing_id="swing-02",  # Different swing!
        camera_id="cam-01",
        frame_id="f-05",
        pts_ticks=50,
        timebase_numerator=1,
        timebase_denominator=1000,
        physical_time_s=0.05,
        physical_time_reason="",
        frame_sha256="c" * 64,
    )

    with pytest.raises(ValueError, match="swing_id mismatch"):
        validate_shot_frames(shot, [alien_frame])


def test_rejection_of_out_of_bounds_shot_frames() -> None:
    shot = Shot(
        schema_version=SHOT_SCHEMA_VERSION,
        asset_id="asset-01",
        shot_id="shot-01",
        start_pts=100,
        end_pts=200,
        start_frame_id="f-10",
        end_frame_id="f-20",
        subject_id="sub-01",
        swing_id="swing-01",
        camera_id="cam-01",
        cuts=(),
        transforms=(),
    )

    early_frame = FrameIdentity(
        schema_version=FRAME_SCHEMA_VERSION,
        asset_id="asset-01",
        shot_id="shot-01",
        swing_id="swing-01",
        camera_id="cam-01",
        frame_id="f-05",
        pts_ticks=50,  # Precedes start_pts=100
        timebase_numerator=1,
        timebase_denominator=1000,
        physical_time_s=0.05,
        physical_time_reason="",
        frame_sha256="d" * 64,
    )

    with pytest.raises(ValueError, match="precedes shot start_pts"):
        validate_shot_frames(shot, [early_frame])


# ---------------------------------------------------------------------------
# 4. Telecine & Duplication Tests
# ---------------------------------------------------------------------------


def test_telecine_duplicate_detection() -> None:
    # Simulating 3:2 pulldown duplicate frames (identical frame_sha256, strictly increasing pts)
    hash_a = "a" * 64
    hash_b = "b" * 64

    frames = [
        FrameIdentity(
            schema_version=FRAME_SCHEMA_VERSION,
            asset_id="asset-01",
            shot_id="shot-01",
            swing_id="swing-01",
            camera_id="cam-01",
            frame_id="f-001",
            pts_ticks=0,
            timebase_numerator=1,
            timebase_denominator=60,
            physical_time_s=0.0,
            physical_time_reason="",
            frame_sha256=hash_a,
        ),
        FrameIdentity(
            schema_version=FRAME_SCHEMA_VERSION,
            asset_id="asset-01",
            shot_id="shot-01",
            swing_id="swing-01",
            camera_id="cam-01",
            frame_id="f-002",
            pts_ticks=1,
            timebase_numerator=1,
            timebase_denominator=60,
            physical_time_s=1 / 60.0,
            physical_time_reason="",
            frame_sha256=hash_a,  # Telecine repeat of hash_a!
        ),
        FrameIdentity(
            schema_version=FRAME_SCHEMA_VERSION,
            asset_id="asset-01",
            shot_id="shot-01",
            swing_id="swing-01",
            camera_id="cam-01",
            frame_id="f-003",
            pts_ticks=2,
            timebase_numerator=1,
            timebase_denominator=60,
            physical_time_s=2 / 60.0,
            physical_time_reason="",
            frame_sha256=hash_b,
        ),
    ]

    deduped, duplicates = detect_telecine_duplicates(frames)
    assert len(deduped) == 2
    assert [f.frame_id for f in deduped] == ["f-001", "f-003"]
    assert len(duplicates) == 1
    assert duplicates[0].frame_id == "f-002"
    assert duplicates[0].frame_sha256 == hash_a


# ---------------------------------------------------------------------------
# 5. Timing Mappings & Synchronization
# ---------------------------------------------------------------------------


def test_affine_timing_mapping() -> None:
    # 2x slow motion: presentation time progresses at half speed relative to physical time
    mapping = AffineTimingMapping(scale=Fraction(1, 2), offset_seconds=0.1)
    pts_sec = Fraction(2, 1)  # 2.0s in presentation
    phys_t = mapping.to_physical_time(pts_sec)
    assert phys_t == 1.1  # 2.0 * 0.5 + 0.1

    # Invert back to presentation time
    pres_t = mapping.to_presentation_time(phys_t)
    assert float(pres_t) == pytest.approx(2.0)


def test_piecewise_timing_mapping() -> None:
    intervals = [
        (
            Fraction(0, 1),
            Fraction(1, 1),
            AffineTimingMapping(scale=Fraction(1, 1), offset_seconds=0.0),
        ),
        (
            Fraction(1, 1),
            Fraction(3, 1),
            AffineTimingMapping(scale=Fraction(1, 4), offset_seconds=0.75),
        ),
    ]
    piecewise = PiecewiseTimingMapping(intervals=intervals)

    # In first interval [0, 1]
    assert piecewise.to_physical_time(Fraction(1, 2)) == 0.5

    # In second interval [1, 3] (slow motion quarter speed)
    # At t_pres = 2.0: 2.0 * 0.25 + 0.75 = 1.25
    assert piecewise.to_physical_time(Fraction(2, 1)) == 1.25

    # Out of range points raise ValueError
    with pytest.raises(ValueError, match="outside piecewise timing intervals"):
        piecewise.to_physical_time(Fraction(4, 1))


def test_camera_synchronization_offset_and_drift() -> None:
    sync = CameraSynchronization(
        camera_id="cam-aux",
        reference_camera_id="cam-main",
        offset_seconds=0.025,
        drift_rate=0.0001,  # 100 microseconds per second drift
        uncertainty_seconds=0.002,
    )

    t_main = 10.0
    t_aux = sync.reference_to_camera_time(t_main)
    # t_aux = t_main + offset + drift * t_main = 10.0 + 0.025 + 0.0001 * 10 = 10.026
    assert t_aux == pytest.approx(10.026)

    t_roundtrip = sync.camera_to_reference_time(t_aux)
    assert t_roundtrip == pytest.approx(t_main)


# ---------------------------------------------------------------------------
# 6. Map Frame to FrameObservation
# ---------------------------------------------------------------------------


def test_map_frame_to_observation() -> None:
    frame = FrameIdentity(
        schema_version=FRAME_SCHEMA_VERSION,
        asset_id="asset-01",
        shot_id="shot-01",
        swing_id="swing-01",
        camera_id="cam-01",
        frame_id="f-042",
        pts_ticks=42,
        timebase_numerator=1,
        timebase_denominator=1000,
        physical_time_s=0.042,
        physical_time_reason="",
        frame_sha256="e" * 64,
    )

    obs = map_frame_to_observation(
        frame,
        body_mask_ref="masks/body_042.png",
        club_mask_ref="masks/club_042.png",
        valid_mask_ref="masks/valid_042.png",
        confidence_provenance="manual_review_v1",
    )

    assert isinstance(obs, FrameObservation)
    assert obs.frame_id == "f-042"
    assert obs.body_mask_ref == "masks/body_042.png"
    assert obs.physical_time_s == 0.042
    assert obs.confidence_provenance == "manual_review_v1"


# ---------------------------------------------------------------------------
# 7. Synthetic Decoder & Capture Rig Adapter Tests
# ---------------------------------------------------------------------------


def test_synthetic_video_decoder() -> None:
    from shared.python.shadow_tracker.ingestion import SyntheticVideoDecoder

    decoder = SyntheticVideoDecoder(
        frame_count=5,
        width=1920,
        height=1080,
        pts_ticks=(0, 33, 67, 100, 133),
    )
    assert decoder.frame_count == 5
    assert decoder.width == 1920
    assert decoder.height == 1080
    assert decoder.pts_ticks(0) == 0
    assert decoder.pts_ticks(2) == 67
    assert len(decoder.read_frame_hash(0)) == 64

    with pytest.raises(IndexError):
        decoder.pts_ticks(5)


def test_ingest_capture_rig_view(tmp_path: Path) -> None:
    from shared.python.shadow_tracker.ingestion import ingest_capture_rig_view

    media_file = tmp_path / "recording.mkv"
    media_file.write_bytes(b"\x1a\x45\xdf\xa3" + b"\x00" * 32)

    class MockView:
        view = "face_on"
        identity = "cam_01"
        playable = media_file

    view = MockView()
    asset = ingest_capture_rig_view(
        view,
        asset_id="asset-rig-01",
        width_px=1920,
        height_px=1080,
        rights_status="permitted",
        rights_note="Studio Recording",
    )

    assert asset.asset_id == "asset-rig-01"
    assert asset.rights_status == "permitted"
    assert asset.rights_note == "Studio Recording"
    assert asset.source_uri == "urn:capture-rig:view:face_on:asset-rig-01"

    # Missing playable raises ValueError
    class EmptyView:
        view = "down_the_line"
        identity = "cam_02"
        playable = None

    with pytest.raises(ValueError, match="no playable recording"):
        ingest_capture_rig_view(
            EmptyView(),
            asset_id="asset-rig-02",
            width_px=1920,
            height_px=1080,
        )


# ---------------------------------------------------------------------------
# 8. Real Local Video Decoding (ST-03A, #10168)
# ---------------------------------------------------------------------------


@pytest.fixture
def test_clip_file(tmp_path: Path) -> Path:
    """Generate a tiny 5-frame local video clip (64x64) with distinct frame contents."""
    import cv2
    import numpy as np

    clip_path = tmp_path / "test_swing_clip.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(clip_path), fourcc, 25.0, (64, 64))
    assert writer.isOpened(), "Could not open VideoWriter for test clip fixture"
    try:
        for i in range(5):
            # Generate deterministic frame with distinct color
            img = np.full((64, 64, 3), (i + 1) * 40, dtype=np.uint8)
            writer.write(img)
    finally:
        writer.release()
    return clip_path


def test_opencv_video_decoder_properties_and_iteration(test_clip_file: Path) -> None:
    from shared.python.shadow_tracker.ingestion import OpenCvVideoDecoder

    decoder = OpenCvVideoDecoder(test_clip_file)
    assert decoder.frame_count == 5
    assert decoder.width == 64
    assert decoder.height == 64
    assert decoder.timebase_numerator == 1
    assert decoder.timebase_denominator == 25
    assert decoder.pts_ticks(0) == 0
    assert decoder.pts_ticks(4) == 4

    hashes = [decoder.read_frame_hash(i) for i in range(5)]
    assert len(hashes) == 5
    # All frames have distinct contents, so hashes must be unique
    assert len(set(hashes)) == 5
    for h in hashes:
        assert len(h) == 64

    # Out of bounds access raises IndexError
    with pytest.raises(IndexError):
        decoder.pts_ticks(5)
    with pytest.raises(IndexError):
        decoder.read_frame_hash(5)
    with pytest.raises(IndexError):
        decoder.pts_ticks(-1)


def test_opencv_video_decoder_missing_or_corrupt_file(tmp_path: Path) -> None:
    from shared.python.shadow_tracker.ingestion import OpenCvVideoDecoder

    missing = tmp_path / "missing.mp4"
    with pytest.raises(FileNotFoundError):
        OpenCvVideoDecoder(missing)

    corrupt = tmp_path / "corrupt.mp4"
    corrupt.write_bytes(b"NOT_A_VIDEO")
    with pytest.raises(ValueError, match="Could not open video file"):
        OpenCvVideoDecoder(corrupt)


def test_decode_video_frames_produces_validated_frame_identities(
    test_clip_file: Path,
) -> None:
    from shared.python.shadow_tracker.ingestion import (
        OpenCvVideoDecoder,
        decode_video_frames,
    )
    from shared.python.shadow_tracker.source_records import validate_frame_sequence

    decoder = OpenCvVideoDecoder(test_clip_file)
    asset = ingest_source_asset(
        test_clip_file,
        asset_id="asset-test-01",
        width_px=decoder.width,
        height_px=decoder.height,
        rights_status="permitted",
        rights_note="Test dataset",
    )

    frames = list(
        decode_video_frames(
            decoder,
            asset=asset,
            shot_id="shot-01",
            swing_id="swing-01",
            camera_id="cam-face-on",
        )
    )
    assert len(frames) == 5
    validate_frame_sequence(frames)

    for i, frame in enumerate(frames):
        assert frame.asset_id == "asset-test-01"
        assert frame.shot_id == "shot-01"
        assert frame.swing_id == "swing-01"
        assert frame.camera_id == "cam-face-on"
        assert frame.pts_ticks == i
        assert frame.timebase_numerator == 1
        assert frame.timebase_denominator == 25
        assert frame.presentation_time == Fraction(i, 25)
        # Default physical_time_s must be None (unknown) without explicit evidence
        assert frame.physical_time_s is None
        assert (
            frame.physical_time_reason
            == "unknown physical time without evidenced clock mapping"
        )

    # Explicit physical time mapping provides verified physical time in SI seconds
    mapped_frames = list(
        decode_video_frames(
            decoder,
            asset=asset,
            shot_id="shot-01",
            swing_id="swing-01",
            camera_id="cam-face-on",
            physical_time_s_fn=lambda idx, pres: float(pres),
            physical_time_reason="hardware genlock clock",
        )
    )
    assert len(mapped_frames) == 5
    for i, frame in enumerate(mapped_frames):
        assert frame.physical_time_s == pytest.approx(i / 25.0)
        assert frame.physical_time_reason == "hardware genlock clock"


def test_decode_video_frames_max_frames_and_cancellation(
    test_clip_file: Path,
) -> None:
    from shared.python.shadow_tracker.ingestion import (
        DecodeLimits,
        OpenCvVideoDecoder,
        decode_video_frames,
    )

    # Test DecodeLimits invariants
    with pytest.raises(ValueError, match="max_frames cannot be negative"):
        DecodeLimits(max_frames=-1)

    decoder = OpenCvVideoDecoder(test_clip_file)
    asset = ingest_source_asset(
        test_clip_file,
        asset_id="asset-test-01",
        width_px=decoder.width,
        height_px=decoder.height,
    )

    # Bounded decoding with max_frames
    bounded_frames = list(
        decode_video_frames(
            decoder,
            asset=asset,
            shot_id="shot-01",
            swing_id="swing-01",
            camera_id="cam-01",
            limits=DecodeLimits(max_frames=2),
        )
    )
    assert len(bounded_frames) == 2

    # Cancellation via is_cancelled callback
    call_count = 0

    def cancel_after_one() -> bool:
        nonlocal call_count
        call_count += 1
        return call_count > 1

    cancelled_frames = list(
        decode_video_frames(
            decoder,
            asset=asset,
            shot_id="shot-01",
            swing_id="swing-01",
            camera_id="cam-01",
            limits=DecodeLimits(is_cancelled=cancel_after_one),
        )
    )
    assert len(cancelled_frames) == 1


def test_opencv_video_decoder_preserves_authoritative_pts_and_vfr(
    test_clip_file: Path,
) -> None:
    from shared.python.shadow_tracker.ingestion import (
        OpenCvVideoDecoder,
        decode_video_frames,
    )
    from shared.python.shadow_tracker.source_records import validate_frame_sequence

    # Explicit authoritative VFR / negative-start PTS track
    custom_pts = [-10, 0, 15, 30, 60]
    decoder = OpenCvVideoDecoder(
        test_clip_file,
        pts_ticks=custom_pts,
        timebase=(1, 60),
    )
    assert decoder.frame_count == 5
    assert decoder.timebase_numerator == 1
    assert decoder.timebase_denominator == 60
    assert decoder.is_timing_exact is True
    assert decoder.timing_mode == "authoritative"
    for i, expected_pts in enumerate(custom_pts):
        assert decoder.pts_ticks(i) == expected_pts

    asset = ingest_source_asset(
        test_clip_file,
        asset_id="asset-vfr-01",
        width_px=decoder.width,
        height_px=decoder.height,
    )

    frames = list(
        decode_video_frames(
            decoder,
            asset=asset,
            shot_id="shot-vfr",
            swing_id="swing-vfr",
            camera_id="cam-vfr",
        )
    )
    assert len(frames) == 5
    validate_frame_sequence(frames)
    assert frames[0].pts_ticks == -10
    assert frames[0].presentation_time == Fraction(-10, 60)
    assert frames[1].pts_ticks == 0
    assert frames[4].pts_ticks == 60
    assert frames[4].presentation_time == Fraction(1, 1)


def test_repeated_images_with_distinct_pts(tmp_path: Path) -> None:
    """Repeated visual frames (e.g. telecine / freeze) must retain unique PTS and sequence validity."""
    import cv2
    import numpy as np
    from shared.python.shadow_tracker.ingestion import (
        OpenCvVideoDecoder,
        decode_video_frames,
    )
    from shared.python.shadow_tracker.source_records import validate_frame_sequence

    clip_path = tmp_path / "repeated_frames.mp4"
    w = cv2.VideoWriter(str(clip_path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (64, 64))
    assert w.isOpened()
    try:
        # Frames 0 and 1 are identical black frames; frames 2, 3, 4 are identical white frames
        w.write(np.zeros((64, 64, 3), dtype=np.uint8))
        w.write(np.zeros((64, 64, 3), dtype=np.uint8))
        w.write(np.full((64, 64, 3), 255, dtype=np.uint8))
        w.write(np.full((64, 64, 3), 255, dtype=np.uint8))
        w.write(np.full((64, 64, 3), 255, dtype=np.uint8))
    finally:
        w.release()

    decoder = OpenCvVideoDecoder(clip_path)
    asset = ingest_source_asset(
        clip_path,
        asset_id="asset-repeated",
        width_px=64,
        height_px=64,
    )
    frames = list(
        decode_video_frames(
            decoder,
            asset=asset,
            shot_id="shot-rep",
            swing_id="swing-rep",
            camera_id="cam-rep",
        )
    )
    assert len(frames) == 5
    validate_frame_sequence(frames)
    # Repeated frames have identical image hashes
    assert frames[0].frame_sha256 == frames[1].frame_sha256
    assert frames[2].frame_sha256 == frames[3].frame_sha256 == frames[4].frame_sha256
    # But strictly ascending timestamps
    assert frames[0].pts_ticks < frames[1].pts_ticks < frames[2].pts_ticks


def test_unavailable_or_unsupported_timing_mode(test_clip_file: Path) -> None:
    from shared.python.shadow_tracker.ingestion import OpenCvVideoDecoder

    # Default without explicit PTS uses estimated CFR from container FPS
    decoder = OpenCvVideoDecoder(test_clip_file)
    assert decoder.decoder_name == "opencv"
    assert decoder.pixel_format == "bgr24"
    assert decoder.timing_mode == "estimated_cfr"
    assert decoder.is_timing_exact is False


def test_incremental_decoding_and_cancellation_during_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Constructor must not read all frames; decode_video_frames must be genuinely incremental and cancellable."""
    import cv2
    import numpy as np
    from shared.python.shadow_tracker.ingestion import (
        DecodeLimits,
        OpenCvVideoDecoder,
        decode_video_frames,
    )

    clip_path = tmp_path / "long_clip.mp4"
    w = cv2.VideoWriter(str(clip_path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (32, 32))
    assert w.isOpened()
    try:
        for i in range(20):
            w.write(np.full((32, 32, 3), i * 10, dtype=np.uint8))
    finally:
        w.release()

    # Track how many reads are performed
    orig_read = cv2.VideoCapture.read
    read_calls = 0

    def tracked_read(self: cv2.VideoCapture) -> tuple[bool, Any]:
        nonlocal read_calls
        read_calls += 1
        return orig_read(self)

    monkeypatch.setattr(cv2.VideoCapture, "read", tracked_read)

    read_calls = 0
    decoder = OpenCvVideoDecoder(clip_path)
    # Constructor should probe at most 1 frame to verify validity, NOT read all 20 frames
    assert read_calls <= 1, f"Constructor read {read_calls} frames, expected <= 1"

    asset = ingest_source_asset(
        clip_path,
        asset_id="asset-incremental",
        width_px=32,
        height_px=32,
    )

    read_calls = 0
    # Decoding 3 frames with max_frames=3 should only read 3 frames
    frames = list(
        decode_video_frames(
            decoder,
            asset=asset,
            shot_id="shot-01",
            swing_id="swing-01",
            camera_id="cam-01",
            limits=DecodeLimits(max_frames=3),
        )
    )
    assert len(frames) == 3
    assert read_calls == 3, (
        f"Expected exactly 3 reads for max_frames=3, got {read_calls}"
    )

    # Cancellation during reads stops immediately
    read_calls = 0

    def cancel_after_1() -> bool:
        return read_calls >= 1

    cancelled = list(
        decode_video_frames(
            decoder,
            asset=asset,
            shot_id="shot-01",
            swing_id="swing-01",
            camera_id="cam-01",
            limits=DecodeLimits(is_cancelled=cancel_after_1),
        )
    )
    assert len(cancelled) == 1
    assert read_calls == 1


def test_decoder_and_pixel_format_provenance_in_hashes(test_clip_file: Path) -> None:
    from shared.python.shadow_tracker.ingestion import (
        OpenCvVideoDecoder,
        compute_frame_hash,
    )

    decoder = OpenCvVideoDecoder(test_clip_file)
    assert decoder.decoder_name == "opencv"
    assert decoder.pixel_format == "bgr24"

    raw_bytes = b"sample_pixels_12345"
    h_bgr = compute_frame_hash(raw_bytes, decoder_name="opencv", pixel_format="bgr24")
    h_rgb = compute_frame_hash(raw_bytes, decoder_name="opencv", pixel_format="rgb24")
    h_other = compute_frame_hash(raw_bytes, decoder_name="ffmpeg", pixel_format="bgr24")

    # Hashes must differ when decoder or pixel format provenance differs
    assert len(h_bgr) == 64
    assert h_bgr != h_rgb
    assert h_bgr != h_other


def test_videodecoderadapter_protocol_compliance(test_clip_file: Path) -> None:
    from shared.python.shadow_tracker.ingestion import (
        OpenCvVideoDecoder,
        SyntheticVideoDecoder,
        VideoDecoderAdapter,
    )

    synth = SyntheticVideoDecoder(
        5,
        width=100,
        height=100,
        timebase=(1, 30),
    )
    cv = OpenCvVideoDecoder(test_clip_file)

    for d in (synth, cv):
        assert isinstance(d, VideoDecoderAdapter)
        assert hasattr(d, "frame_count")
        assert hasattr(d, "width")
        assert hasattr(d, "height")
        assert hasattr(d, "timebase_numerator")
        assert hasattr(d, "timebase_denominator")
        assert hasattr(d, "is_timing_exact")
        assert hasattr(d, "timing_mode")
        assert hasattr(d, "decoder_name")
        assert hasattr(d, "pixel_format")
        assert hasattr(d, "pts_ticks")
        assert hasattr(d, "read_frame_hash")


def make_vfr_mp4_fixture(
    path: Path, deltas: Sequence[int], timescale: int = 1000
) -> Path:
    """Create a valid ISO-BMFF MP4 media file with nonuniform sample deltas (VFR)."""
    import struct
    import cv2
    import numpy as np

    num_frames = len(deltas)
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 64))
    try:
        for i in range(num_frames):
            w.write(np.full((64, 64, 3), (i + 1) * 35, dtype=np.uint8))
    finally:
        w.release()

    raw = bytearray(path.read_bytes())
    total_duration = sum(deltas)

    # Patch mdhd timescale & duration
    mdhd_idx = raw.find(b"mdhd")
    if mdhd_idx != -1:
        v = raw[mdhd_idx + 4]
        if v == 0:
            struct.pack_into(">II", raw, mdhd_idx + 16, timescale, total_duration)
        else:
            struct.pack_into(">IQ", raw, mdhd_idx + 24, timescale, total_duration)

    # Patch mvhd timescale & duration
    mvhd_idx = raw.find(b"mvhd")
    if mvhd_idx != -1:
        v_mv = raw[mvhd_idx + 4]
        if v_mv == 0:
            struct.pack_into(">II", raw, mvhd_idx + 16, timescale, total_duration)
        else:
            struct.pack_into(">IQ", raw, mvhd_idx + 24, timescale, total_duration)

    # Patch tkhd duration
    tkhd_idx = raw.find(b"tkhd")
    if tkhd_idx != -1:
        v_tk = raw[tkhd_idx + 4]
        if v_tk == 0:
            struct.pack_into(">I", raw, tkhd_idx + 24, total_duration)
        else:
            struct.pack_into(">Q", raw, tkhd_idx + 32, total_duration)

    # Patch elst duration
    elst_idx = raw.find(b"elst")
    if elst_idx != -1:
        v_el = raw[elst_idx + 4]
        if v_el == 0:
            struct.pack_into(">I", raw, elst_idx + 12, total_duration)
        else:
            struct.pack_into(">Q", raw, elst_idx + 12, total_duration)

    # Rewrite stts table
    stts_idx = raw.find(b"stts")
    if stts_idx != -1:
        old_stts_size = struct.unpack_from(">I", raw, stts_idx - 4)[0]
        new_entries = b"".join(struct.pack(">II", 1, int(d)) for d in deltas)
        new_stts_body = (
            struct.pack(">4sB3sI", b"stts", 0, b"\x00\x00\x00", len(deltas))
            + new_entries
        )
        new_stts_size = 4 + len(new_stts_body)
        new_stts = struct.pack(">I", new_stts_size) + new_stts_body
        size_diff = new_stts_size - old_stts_size

        stts_box_start = stts_idx - 4
        raw[stts_box_start : stts_box_start + old_stts_size] = new_stts

        for box_tag in (b"stbl", b"minf", b"mdia", b"trak", b"moov"):
            tag_pos = raw.rfind(box_tag, 0, stts_box_start)
            if tag_pos != -1:
                box_size = struct.unpack_from(">I", raw, tag_pos - 4)[0]
                struct.pack_into(">I", raw, tag_pos - 4, box_size + size_diff)

    path.write_bytes(raw)
    return path


def test_reject_pts_without_explicit_timebase(test_clip_file: Path) -> None:
    """Supplying pts_ticks without an explicit timebase must raise ValueError."""
    from shared.python.shadow_tracker.ingestion import OpenCvVideoDecoder

    with pytest.raises(ValueError, match="timebase must be explicitly provided"):
        OpenCvVideoDecoder(test_clip_file, pts_ticks=[0, 10, 20])


def test_real_vfr_fixture_native_pts_extraction(tmp_path: Path) -> None:
    """Real VFR MP4 container extracts native presentation timestamps and timescale."""
    from shared.python.shadow_tracker.ingestion import OpenCvVideoDecoder

    vfr_clip = tmp_path / "genuine_vfr.mp4"
    deltas = [100, 400, 200, 300]
    make_vfr_mp4_fixture(vfr_clip, deltas, timescale=1000)

    decoder = OpenCvVideoDecoder(vfr_clip)
    assert decoder.frame_count == 4
    assert decoder.is_timing_exact is True
    assert decoder.timing_mode == "container_pts"
    assert decoder.clock_evidence == "container_pts_metadata"
    assert decoder.timebase_numerator == 1
    assert decoder.timebase_denominator == 1000

    expected_pts = [0, 100, 500, 700]
    for i, pts in enumerate(expected_pts):
        assert decoder.pts_ticks(i) == pts

    hashes = [decoder.read_frame_hash(i) for i in range(4)]
    assert len(set(hashes)) == 4


def test_timestamp_authority_and_provenance_preserved_in_records(
    tmp_path: Path,
) -> None:
    """FrameIdentity and FrameObservation must preserve timestamp authority and decoder provenance."""
    from shared.python.shadow_tracker.ingestion import (
        OpenCvVideoDecoder,
        decode_video_frames,
        map_frame_to_observation,
    )

    vfr_clip = tmp_path / "vfr_authority.mp4"
    deltas = [50, 150, 100]
    make_vfr_mp4_fixture(vfr_clip, deltas, timescale=1000)

    decoder = OpenCvVideoDecoder(vfr_clip)
    asset = ingest_source_asset(
        vfr_clip,
        asset_id="asset-vfr-auth",
        width_px=decoder.width,
        height_px=decoder.height,
    )
    frames = list(
        decode_video_frames(
            decoder,
            asset=asset,
            shot_id="shot-01",
            swing_id="swing-01",
            camera_id="cam-01",
        )
    )
    assert len(frames) == 3
    for f in frames:
        assert f.timing_mode == "container_pts"
        assert f.is_timing_exact is True
        assert f.clock_evidence == "container_pts_metadata"
        assert f.decoder_name == "opencv"
        assert f.pixel_format == "bgr24"

    obs = map_frame_to_observation(
        frames[0],
        body_mask_ref="mask-body-0",
        club_mask_ref="mask-club-0",
        valid_mask_ref="mask-valid-0",
        confidence_provenance="reviewed_gold",
    )
    assert obs.timing_mode == "container_pts"
    assert obs.is_timing_exact is True
    assert obs.clock_evidence == "container_pts_metadata"
    assert obs.decoder_name == "opencv"
    assert obs.pixel_format == "bgr24"


def test_frame_identity_and_observation_version_migration() -> None:
    """Legacy 1.0.0 JSON payloads missing timing authority must migrate safely."""
    from shared.python.shadow_tracker.contracts import FrameObservation
    from shared.python.shadow_tracker.source_records import FrameIdentity

    legacy_frame_dict = {
        "schema_version": "shadow-tracker/frame/1.0.0",
        "asset_id": "asset-legacy",
        "shot_id": "shot-01",
        "swing_id": "swing-01",
        "camera_id": "cam-01",
        "frame_id": "frame-000000",
        "pts_ticks": 0,
        "timebase_numerator": 1,
        "timebase_denominator": 30,
        "physical_time_s": None,
        "physical_time_reason": "nominal clock",
        "frame_sha256": "0" * 64,
    }
    migrated_frame = FrameIdentity.from_dict(legacy_frame_dict)
    assert migrated_frame.timing_mode == "estimated_cfr"
    assert migrated_frame.is_timing_exact is False
    assert migrated_frame.clock_evidence == "unverified_legacy_record"

    # Round trip of migrated record preserves attributes
    d = migrated_frame.to_dict()
    reloaded = FrameIdentity.from_dict(d)
    assert reloaded == migrated_frame

    legacy_obs_dict = {
        "schema_version": "shadow-tracker/frame-observation/1.0.0",
        "shot_id": "shot-01",
        "camera_id": "cam-01",
        "frame_id": "frame-000000",
        "pts_ticks": 0,
        "timebase_numerator": 1,
        "timebase_denominator": 30,
        "physical_time_s": None,
        "physical_time_reason": "nominal clock",
        "body_mask_ref": "bref",
        "club_mask_ref": "cref",
        "valid_mask_ref": "vref",
        "confidence_provenance": "cprov",
    }
    migrated_obs = FrameObservation.from_dict(legacy_obs_dict)
    assert migrated_obs.timing_mode == "estimated_cfr"
    assert migrated_obs.is_timing_exact is False
    assert migrated_obs.clock_evidence == "unverified_legacy_record"

    d_obs = migrated_obs.to_dict()
    reloaded_obs = FrameObservation.from_dict(d_obs)
    assert reloaded_obs == migrated_obs
