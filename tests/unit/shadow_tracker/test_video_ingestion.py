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

from fractions import Fraction
import hashlib
from pathlib import Path
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
