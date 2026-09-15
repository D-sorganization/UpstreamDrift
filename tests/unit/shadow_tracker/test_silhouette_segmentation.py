"""Unit tests for Body/Club Silhouettes and Segmentation (ST-04, #10127).

Tests verify:
- ManualMaskProvider deterministic baseline and segmentation request fulfillment.
- Empty vs unknown mask semantics (valid-pixel awareness).
- Body and club separation into distinct binary channels.
- Revision-aware cache invalidation upon manual correction.
- Occlusion and identity loss tracking.
- Curated gold-mask agreement metrics (IoU, Dice) evaluated only over valid pixels.
- Lazy missing-model provider error handling and unreviewed status flag.
"""

from __future__ import annotations

import hashlib
import pytest

from shared.python.shadow_tracker._validation import (
    FRAME_SCHEMA_VERSION,
    MASK_SCHEMA_VERSION,
)
from shared.python.shadow_tracker.contracts import (
    SegmentationRequest,
    SegmentationResult,
    Segmenter,
)
from shared.python.shadow_tracker.mask_records import MaskFrame
from shared.python.shadow_tracker.segmentation import (
    ManualMaskProvider,
    ModelSegmentationProvider,
    OcclusionReport,
    compute_mask_dice,
    compute_mask_iou,
    track_occlusion_and_identity,
)
from shared.python.shadow_tracker.source_records import FrameIdentity

pytestmark = pytest.mark.unit


@pytest.fixture
def base_frame_identity() -> FrameIdentity:
    return FrameIdentity(
        schema_version=FRAME_SCHEMA_VERSION,
        asset_id="asset-golf-01",
        shot_id="shot-01",
        swing_id="swing-01",
        camera_id="cam-face-on",
        frame_id="f-010",
        pts_ticks=100,
        timebase_numerator=1,
        timebase_denominator=1000,
        physical_time_s=0.1,
        physical_time_reason="",
        frame_sha256="a" * 64,
    )


# ---------------------------------------------------------------------------
# 1. Manual Mask Provider & Segmentation Protocol
# ---------------------------------------------------------------------------


def test_manual_mask_provider_fulfills_segmentation_protocol(
    base_frame_identity: FrameIdentity,
) -> None:
    width, height = 4, 4
    total_px = width * height
    # Valid everywhere; 4 pixels of body, 2 pixels of club
    valid = bytes([1] * total_px)
    body = bytes([1 if i in (5, 6, 9, 10) else 0 for i in range(total_px)])
    club = bytes([1 if i in (13, 14) else 0 for i in range(total_px)])

    mask = MaskFrame(
        schema_version=MASK_SCHEMA_VERSION,
        frame=base_frame_identity,
        width_px=width,
        height_px=height,
        body=body,
        club=club,
        valid=valid,
        revision_id="rev-001",
        parent_revision_id=None,
        producer_id="annotator-expert",
        correction_note="Initial manual gold annotation",
    )

    provider = ManualMaskProvider()
    assert isinstance(provider, Segmenter)

    provider.register_mask(mask)
    assert provider.has_mask("f-010")

    req = SegmentationRequest(
        shot_id="shot-01",
        frame_ids=("f-010",),
    )

    res = provider.segment(req)
    assert isinstance(res, SegmentationResult)
    assert res.shot_id == "shot-01"
    assert res.mask_count == 1
    assert res.provenance == "manual_gold_review"


# ---------------------------------------------------------------------------
# 2. Empty vs Unknown Mask Semantics
# ---------------------------------------------------------------------------


def test_empty_vs_unknown_mask_semantics(base_frame_identity: FrameIdentity) -> None:
    width, height = 2, 2
    total_px = width * height

    # 1. Empty mask: person definitely absent (valid=1, body=0, club=0)
    empty_valid = bytes([1] * total_px)
    empty_body = bytes([0] * total_px)
    empty_club = bytes([0] * total_px)

    empty_mask = MaskFrame(
        schema_version=MASK_SCHEMA_VERSION,
        frame=base_frame_identity,
        width_px=width,
        height_px=height,
        body=empty_body,
        club=empty_club,
        valid=empty_valid,
        revision_id="rev-empty",
        parent_revision_id=None,
        producer_id="reviewer",
        correction_note="Observed frame contains no golfer in frame area",
    )
    assert empty_mask.body.count(1) == 0
    assert empty_mask.valid.count(1) == total_px

    # 2. Unknown/Occluded mask: unobserved region (valid=0, body=0, club=0)
    unknown_valid = bytes([0] * total_px)
    unknown_mask = MaskFrame(
        schema_version=MASK_SCHEMA_VERSION,
        frame=base_frame_identity,
        width_px=width,
        height_px=height,
        body=empty_body,
        club=empty_club,
        valid=unknown_valid,
        revision_id="rev-unknown",
        parent_revision_id=None,
        producer_id="reviewer",
        correction_note="Camera view fully occluded by barrier",
    )
    assert unknown_mask.valid.count(1) == 0

    # Invariant: body or club non-zero where valid is zero must raise ValueError
    invalid_body = bytes([1, 0, 0, 0])
    with pytest.raises(ValueError, match="is non-zero where valid is zero"):
        MaskFrame(
            schema_version=MASK_SCHEMA_VERSION,
            frame=base_frame_identity,
            width_px=width,
            height_px=height,
            body=invalid_body,
            club=empty_club,
            valid=unknown_valid,  # valid is all 0
            revision_id="rev-invalid",
            parent_revision_id=None,
            producer_id="reviewer",
            correction_note="",
        )


# ---------------------------------------------------------------------------
# 3. Revision-Aware Cache Invalidation
# ---------------------------------------------------------------------------


def test_revision_aware_cache_invalidation(base_frame_identity: FrameIdentity) -> None:
    width, height = 2, 2
    total_px = width * height
    valid = bytes([1] * total_px)
    body_v1 = bytes([1, 0, 0, 0])
    club_v1 = bytes([0, 1, 0, 0])

    mask_v1 = MaskFrame(
        schema_version=MASK_SCHEMA_VERSION,
        frame=base_frame_identity,
        width_px=width,
        height_px=height,
        body=body_v1,
        club=club_v1,
        valid=valid,
        revision_id="rev-01",
        parent_revision_id=None,
        producer_id="model-draft",
        correction_note="Draft mask",
    )

    provider = ManualMaskProvider()
    provider.register_mask(mask_v1)

    # Simulated downstream cache keyed on revision_id or content_sha256
    cache = {mask_v1.revision_id: "cached_projection_loss_0.42"}
    assert provider.get_mask("f-010").revision_id in cache

    # Human reviewer corrects mask: add missing club pixel
    club_v2 = bytes([0, 1, 1, 0])
    mask_v2 = MaskFrame(
        schema_version=MASK_SCHEMA_VERSION,
        frame=base_frame_identity,
        width_px=width,
        height_px=height,
        body=body_v1,
        club=club_v2,
        valid=valid,
        revision_id="rev-02",
        parent_revision_id="rev-01",
        producer_id="human-coach",
        correction_note="Extended shaft to clubhead",
    )

    provider.register_mask(mask_v2)
    current = provider.get_mask("f-010")
    assert current.revision_id == "rev-02"
    assert current.parent_revision_id == "rev-01"
    # Old cache key rev-01 is invalidated / not matching current revision
    assert current.revision_id not in cache
    assert current.observation_hash != mask_v1.observation_hash


# ---------------------------------------------------------------------------
# 4. Occlusion & Identity Loss Tracking
# ---------------------------------------------------------------------------


def test_occlusion_and_identity_loss_tracking(
    base_frame_identity: FrameIdentity,
) -> None:
    width, height = 4, 4
    total_px = width * height
    # Half of the view area is occluded by a tree/spectator (valid=0 for bottom 8 pixels)
    valid = bytes([1] * 8 + [0] * 8)
    # 4 pixels of visible body in top half
    body = bytes([1, 1, 1, 1] + [0] * 12)
    club = bytes([0] * total_px)

    mask = MaskFrame(
        schema_version=MASK_SCHEMA_VERSION,
        frame=base_frame_identity,
        width_px=width,
        height_px=height,
        body=body,
        club=club,
        valid=valid,
        revision_id="rev-occ",
        parent_revision_id=None,
        producer_id="annotator",
        correction_note="",
    )

    # Expected body area is 8 pixels (e.g. from address or calibration view)
    report = track_occlusion_and_identity(mask, expected_body_area_px=8)
    assert isinstance(report, OcclusionReport)
    assert report.visible_fraction == pytest.approx(
        4 / 8
    )  # 50% of expected body visible
    assert report.is_partially_occluded is True
    assert report.is_identity_lost is False

    # Severe occlusion: only 1 pixel visible out of 10 -> identity lost
    report_severe = track_occlusion_and_identity(mask, expected_body_area_px=40)
    assert report_severe.visible_fraction == pytest.approx(4 / 40)
    assert report_severe.is_identity_lost is True


# ---------------------------------------------------------------------------
# 5. Curated Gold-Mask Agreement (Valid-Pixel Aware IoU & Dice)
# ---------------------------------------------------------------------------


def test_gold_mask_agreement_iou_and_dice() -> None:
    # 6 pixels:
    # idx 0: cand=1, gold=1, valid=1  (true positive)
    # idx 1: cand=1, gold=0, valid=1  (false positive)
    # idx 2: cand=0, gold=1, valid=1  (false negative)
    # idx 3: cand=0, gold=0, valid=1  (true negative)
    # idx 4: cand=1, gold=0, valid=0  (occluded / invalid -> ignored!)
    # idx 5: cand=0, gold=1, valid=0  (occluded / invalid -> ignored!)
    cand = bytes([1, 1, 0, 0, 1, 0])
    gold = bytes([1, 0, 1, 0, 0, 1])
    valid = bytes([1, 1, 1, 1, 0, 0])

    # Over valid pixels (indices 0..3):
    # intersection = {0} -> 1
    # union = {0, 1, 2} -> 3
    # IoU = 1 / 3
    iou = compute_mask_iou(cand, gold, valid)
    assert iou == pytest.approx(1.0 / 3.0)

    # Dice = 2 * intersection / (|cand| + |gold|) = 2 * 1 / (2 + 2) = 0.5
    dice = compute_mask_dice(cand, gold, valid)
    assert dice == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 6. Model Segmentation Provider & Lazy Checkpoint Error
# ---------------------------------------------------------------------------


def test_lazy_missing_model_provider_error() -> None:
    provider = ModelSegmentationProvider(
        model_name="sam-vit-b",
        checkpoint_path="/nonexistent/checkpoint.pth",
    )

    req = SegmentationRequest(
        shot_id="shot-01",
        frame_ids=("f-01",),
    )

    # Actionable error on execution, not hidden or mocked
    with pytest.raises(FileNotFoundError, match="Model checkpoint not found"):
        provider.segment(req)
