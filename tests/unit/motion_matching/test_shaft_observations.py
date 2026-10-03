"""Visible shaft fragments are source-bound bearings, never physical endpoints."""

from dataclasses import FrozenInstanceError, replace
import json

import pytest

from src.shared.python.motion_matching.historical_fit.shaft_observations import (
    ShaftAxisEvidence,
    ShaftAxisSegment,
    SourceBoundShaftFrame,
)
from src.shared.python.shadow_tracker.source_records import FrameIdentity

pytestmark = pytest.mark.unit


def frame_identity(index=0):
    return FrameIdentity(
        schema_version="shadow-tracker/frame/1.1.0",
        asset_id="source-" + "a" * 64,
        shot_id="shot",
        swing_id="unreviewed",
        camera_id="camera",
        frame_id=f"frame-{index}",
        pts_ticks=index + 10,
        timebase_numerator=1,
        timebase_denominator=30,
        physical_time_s=None,
        physical_time_reason="Unknown playback scale",
        frame_sha256="b" * 64,
        timing_mode="container_pts",
        is_timing_exact=True,
    )


def evidence():
    segment = ShaftAxisSegment(
        "observed", ((10, 20), (30, 40)), "operator", "Visible shaft", 0.8, None, 2
    )
    frame = SourceBoundShaftFrame(0, frame_identity(), "sha256:" + "c" * 64, segment)
    return ShaftAxisEvidence(
        "capture", "sha256:" + "d" * 64, "sha256:" + "a" * 64, (100, 80), (frame,)
    )


def test_immutable_copied_roundtrip_and_explicit_unknown_visibility():
    points = [[10, 20], [30, 40]]
    segment = ShaftAxisSegment("observed", points, "reviewer", "Visible", 0.8, None, 2)
    points[0][0] = 99
    assert segment.points_px == ((10.0, 20.0), (30.0, 40.0))
    with pytest.raises(FrozenInstanceError):
        segment.confidence = 0
    original = evidence()
    restored = ShaftAxisEvidence.from_record(
        json.loads(json.dumps(original.to_record()))
    )
    assert restored == original and restored.sha256 == original.sha256
    assert restored.frames[0].segment.visibility is None
    assert restored.frames[0].frame.physical_time_s is None


@pytest.mark.parametrize(
    "updates",
    [
        {"points_px": ((1, 2), (1, 2))},
        {"points_px": ((True, 2), (3, 4))},
        {"points_px": ((float("nan"), 2), (3, 4))},
        {"confidence": True},
        {"confidence": 1.1},
        {"visibility": -1},
        {"sigma_px": 0},
        {"sigma_px": float("inf")},
        {"reviewer": ""},
        {"reason": ""},
        {"status": "inferred"},
        {"points_px": None},
    ],
)
def test_bad_observation_rejected(updates):
    with pytest.raises((ValueError, TypeError)):
        replace(evidence().frames[0].segment, **updates)


def test_abstention_has_no_fabricated_points_or_uncertainty():
    value = ShaftAxisSegment(
        "occluded", None, "reviewer", "Hands obscure shaft", None, None, None
    )
    assert value.points_px is None
    for update in (
        {"points_px": ((1, 2), (3, 4))},
        {"confidence": 0.8},
        {"sigma_px": 2},
    ):
        with pytest.raises(ValueError):
            replace(value, **update)


@pytest.mark.parametrize(
    "updates",
    [
        {"frames": ()},
        {"frames": (evidence().frames[0], evidence().frames[0])},
        {"image_size": (True, 80)},
        {"image_size": (100, 0)},
        {"capture_sha256": "D" * 64},
        {"source_sha256": "sha256:" + "f" * 64},
    ],
)
def test_invalid_evidence_rejected(updates):
    with pytest.raises((ValueError, TypeError)):
        replace(evidence(), **updates)


def test_unknown_record_fields_and_changed_physical_clock_rejected():
    record = evidence().to_record()
    record["fake_anatomy"] = True
    with pytest.raises(ValueError):
        ShaftAxisEvidence.from_record(record)
    with pytest.raises(ValueError):
        replace(
            evidence().frames[0], frame=replace(frame_identity(), physical_time_s=1.0)
        )


def test_observed_pixels_must_lie_in_declared_original_image():
    value = evidence()
    bad = replace(value.frames[0].segment, points_px=((100, 20), (30, 40)))
    with pytest.raises(ValueError, match="image"):
        replace(value, frames=(replace(value.frames[0], segment=bad),))


def test_duplicate_source_frame_id_rejected_even_with_distinct_times():
    value = evidence()
    second = replace(
        value.frames[0],
        frame_index=1,
        frame=replace(frame_identity(1), frame_id=value.frames[0].frame.frame_id),
    )
    with pytest.raises(ValueError, match="identity"):
        replace(value, frames=(value.frames[0], second))
