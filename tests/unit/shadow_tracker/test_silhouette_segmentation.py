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
from pathlib import Path
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


def test_arbitrary_existing_checkpoint_must_not_report_segmentation_success(
    tmp_path: Path,
) -> None:
    """ST-04 / P1 finding: arbitrary file pretending to be checkpoint must not succeed."""
    bogus_ckpt = tmp_path / "not_a_model.pth"
    bogus_ckpt.write_bytes(b"not a model file")

    provider = ModelSegmentationProvider(
        model_name="sam-vit-b",
        checkpoint_path=bogus_ckpt,
    )

    req = SegmentationRequest(
        shot_id="shot-01",
        frame_ids=("f-01",),
    )

    # Must raise RuntimeError or explicit actionable exception indicating automated inference
    # is unsupported/unimplemented rather than returning fake success with mask counts
    with pytest.raises(
        RuntimeError,
        match="Automated inference is not supported|not a valid model checkpoint",
    ):
        provider.segment(req)


def test_manual_mask_provider_shot_isolation(
    base_frame_identity: FrameIdentity,
) -> None:
    """ST-04 / P2 finding: two shots with same frame ID must not collide or exchange masks."""
    width, height = 2, 2
    total_px = width * height
    valid = bytes([1] * total_px)
    body = bytes([1, 0, 0, 0])
    club = bytes([0, 1, 0, 0])

    mask_shot1 = MaskFrame(
        schema_version=MASK_SCHEMA_VERSION,
        frame=base_frame_identity,  # shot_id="shot-01", frame_id="f-010"
        width_px=width,
        height_px=height,
        body=body,
        club=club,
        valid=valid,
        revision_id="rev-shot1",
        parent_revision_id=None,
        producer_id="reviewer",
        correction_note="",
    )

    frame_identity_shot2 = FrameIdentity(
        schema_version=FRAME_SCHEMA_VERSION,
        asset_id="asset-golf-01",
        shot_id="shot-02",  # Different shot!
        swing_id="swing-02",
        camera_id="cam-face-on",
        frame_id="f-010",  # Same local frame ID!
        pts_ticks=200,
        timebase_numerator=1,
        timebase_denominator=1000,
        physical_time_s=0.2,
        physical_time_reason="",
        frame_sha256="b" * 64,
    )

    mask_shot2 = MaskFrame(
        schema_version=MASK_SCHEMA_VERSION,
        frame=frame_identity_shot2,
        width_px=width,
        height_px=height,
        body=body,
        club=club,
        valid=valid,
        revision_id="rev-shot2",
        parent_revision_id=None,
        producer_id="reviewer",
        correction_note="",
    )

    provider = ManualMaskProvider()
    provider.register_mask(mask_shot1)
    provider.register_mask(mask_shot2)

    # Retrieval must support shot disambiguation
    retrieved_shot1 = provider.get_mask("f-010", shot_id="shot-01")
    retrieved_shot2 = provider.get_mask("f-010", shot_id="shot-02")
    assert retrieved_shot1.revision_id == "rev-shot1"
    assert retrieved_shot2.revision_id == "rev-shot2"
    assert retrieved_shot1.frame.shot_id == "shot-01"
    assert retrieved_shot2.frame.shot_id == "shot-02"

    # Requesting segment for shot-01 only yields shot-01 masks
    req_shot1 = SegmentationRequest(
        shot_id="shot-01",
        frame_ids=("f-010",),
    )
    res_shot1 = provider.segment(req_shot1)
    assert res_shot1.mask_count == 1

    # Requesting frame not in shot must raise KeyError
    req_mismatch = SegmentationRequest(
        shot_id="shot-03",
        frame_ids=("f-010",),
    )
    with pytest.raises(KeyError, match="No mask registered for shot 'shot-03'"):
        provider.segment(req_mismatch)


def test_manual_mask_provider_revision_history_addressable(
    base_frame_identity: FrameIdentity,
) -> None:
    """ST-04 / P2 finding: past revisions remain addressable and inspectable."""
    width, height = 2, 2
    total_px = width * height
    valid = bytes([1] * total_px)
    body = bytes([1, 0, 0, 0])
    club_v1 = bytes([0, 1, 0, 0])
    club_v2 = bytes([0, 1, 1, 0])

    m1 = MaskFrame(
        schema_version=MASK_SCHEMA_VERSION,
        frame=base_frame_identity,
        width_px=width,
        height_px=height,
        body=body,
        club=club_v1,
        valid=valid,
        revision_id="rev-01",
        parent_revision_id=None,
        producer_id="annotator-1",
        correction_note="Initial",
    )
    m2 = MaskFrame(
        schema_version=MASK_SCHEMA_VERSION,
        frame=base_frame_identity,
        width_px=width,
        height_px=height,
        body=body,
        club=club_v2,
        valid=valid,
        revision_id="rev-02",
        parent_revision_id="rev-01",
        producer_id="reviewer-1",
        correction_note="Correction",
    )

    provider = ManualMaskProvider()
    provider.register_mask(m1)
    provider.register_mask(m2)

    # Current returns latest revision
    latest = provider.get_mask("f-010", shot_id="shot-01")
    assert latest.revision_id == "rev-02"

    # Specific revision is also addressable
    rev1 = provider.get_revision("rev-01")
    assert rev1.revision_id == "rev-01"
    assert rev1.club == club_v1

    # Revision history list
    history = provider.get_revision_history("f-010", shot_id="shot-01")
    assert len(history) == 2
    assert [h.revision_id for h in history] == ["rev-01", "rev-02"]


def test_segmentation_dto_validation() -> None:
    """ST-02 / ST-04 DTO boundary validation."""
    with pytest.raises((ValueError, TypeError)):
        SegmentationRequest(
            shot_id="",  # invalid empty id
            frame_ids=("f-01",),
        )

    with pytest.raises((ValueError, TypeError)):
        SegmentationRequest(
            shot_id="shot-01",
            frame_ids=(),  # empty frame ids
        )

    with pytest.raises((ValueError, TypeError)):
        SegmentationResult(
            shot_id="",
            mask_count=1,
            provenance="test",
        )

    with pytest.raises((ValueError, TypeError)):
        SegmentationResult(
            shot_id="shot-01",
            mask_count=-1,  # negative mask count
            provenance="test",
        )


# ---------------------------------------------------------------------------
# 10. Issue #10233: Revision Identity Protection & Lineage Persistence
# ---------------------------------------------------------------------------


def test_register_mask_rejects_conflicting_duplicate_revision_id(
    base_frame_identity: FrameIdentity,
) -> None:
    width, height = 2, 2
    total_px = width * height
    valid = bytes([1] * total_px)
    body = bytes([1, 0, 0, 0])
    club = bytes([0, 1, 0, 0])

    mask1 = MaskFrame(
        schema_version=MASK_SCHEMA_VERSION,
        frame=base_frame_identity,
        width_px=width,
        height_px=height,
        body=body,
        club=club,
        valid=valid,
        revision_id="rev-unique-01",
        parent_revision_id=None,
        producer_id="annotator-1",
        correction_note="First registration",
    )

    provider = ManualMaskProvider()
    provider.register_mask(mask1)

    diff_shot_frame = FrameIdentity(
        schema_version=FRAME_SCHEMA_VERSION,
        asset_id=base_frame_identity.asset_id,
        shot_id="shot-02",  # Different shot!
        swing_id=base_frame_identity.swing_id,
        camera_id=base_frame_identity.camera_id,
        frame_id=base_frame_identity.frame_id,
        pts_ticks=base_frame_identity.pts_ticks,
        timebase_numerator=base_frame_identity.timebase_numerator,
        timebase_denominator=base_frame_identity.timebase_denominator,
        physical_time_s=base_frame_identity.physical_time_s,
        physical_time_reason=base_frame_identity.physical_time_reason,
        frame_sha256="b" * 64,
    )
    conflicting_mask_diff_shot = MaskFrame(
        schema_version=MASK_SCHEMA_VERSION,
        frame=diff_shot_frame,
        width_px=width,
        height_px=height,
        body=body,
        club=club,
        valid=valid,
        revision_id="rev-unique-01",  # Duplicate revision ID with differing shot!
        parent_revision_id=None,
        producer_id="annotator-2",
        correction_note="Conflicting revision in shot-02",
    )

    with pytest.raises(ValueError, match="already exists|Conflicting duplicate"):
        provider.register_mask(conflicting_mask_diff_shot)

    assert provider.get_revision("rev-unique-01").frame.shot_id == "shot-01"


def test_register_mask_permits_idempotent_identical_re_registration(
    base_frame_identity: FrameIdentity,
) -> None:
    width, height = 2, 2
    total_px = width * height
    valid = bytes([1] * total_px)
    body = bytes([1, 0, 0, 0])
    club = bytes([0, 1, 0, 0])

    mask = MaskFrame(
        schema_version=MASK_SCHEMA_VERSION,
        frame=base_frame_identity,
        width_px=width,
        height_px=height,
        body=body,
        club=club,
        valid=valid,
        revision_id="rev-idempotent",
        parent_revision_id=None,
        producer_id="annotator-1",
        correction_note="Initial note",
    )

    provider = ManualMaskProvider()
    provider.register_mask(mask)
    assert len(provider.get_revision_history("f-010", shot_id="shot-01")) == 1

    provider.register_mask(mask)
    history = provider.get_revision_history("f-010", shot_id="shot-01")
    assert len(history) == 1
    assert history[0] == mask


def test_register_mask_validates_parent_ownership_and_missing_parent(
    base_frame_identity: FrameIdentity,
) -> None:
    width, height = 2, 2
    total_px = width * height
    valid = bytes([1] * total_px)
    body = bytes([1, 0, 0, 0])
    club = bytes([0, 1, 0, 0])

    provider = ManualMaskProvider()

    orphan_mask = MaskFrame(
        schema_version=MASK_SCHEMA_VERSION,
        frame=base_frame_identity,
        width_px=width,
        height_px=height,
        body=body,
        club=club,
        valid=valid,
        revision_id="rev-child-orphan",
        parent_revision_id="rev-nonexistent-parent",
        producer_id="annotator-1",
        correction_note="Orphan child",
    )
    with pytest.raises(ValueError, match="not found|does not exist"):
        provider.register_mask(orphan_mask)

    parent_mask = MaskFrame(
        schema_version=MASK_SCHEMA_VERSION,
        frame=base_frame_identity,
        width_px=width,
        height_px=height,
        body=body,
        club=club,
        valid=valid,
        revision_id="rev-parent-f010",
        parent_revision_id=None,
        producer_id="annotator-1",
        correction_note="Base f-010 mask",
    )
    provider.register_mask(parent_mask)

    diff_frame_identity = FrameIdentity(
        schema_version=FRAME_SCHEMA_VERSION,
        asset_id=base_frame_identity.asset_id,
        shot_id=base_frame_identity.shot_id,
        swing_id=base_frame_identity.swing_id,
        camera_id=base_frame_identity.camera_id,
        frame_id="f-020",
        pts_ticks=200,
        timebase_numerator=1,
        timebase_denominator=1000,
        physical_time_s=0.2,
        physical_time_reason="",
        frame_sha256="c" * 64,
    )
    cross_frame_child = MaskFrame(
        schema_version=MASK_SCHEMA_VERSION,
        frame=diff_frame_identity,
        width_px=width,
        height_px=height,
        body=body,
        club=club,
        valid=valid,
        revision_id="rev-child-f020",
        parent_revision_id="rev-parent-f010",
        producer_id="annotator-1",
        correction_note="Cross frame child",
    )
    with pytest.raises(
        ValueError, match="belongs to different frame scope|scope mismatch|not found"
    ):
        provider.register_mask(cross_frame_child)


def test_failed_registration_leaves_every_index_unchanged(
    base_frame_identity: FrameIdentity,
) -> None:
    width, height = 2, 2
    total_px = width * height
    valid = bytes([1] * total_px)
    body = bytes([1, 0, 0, 0])
    club = bytes([0, 1, 0, 0])

    m1 = MaskFrame(
        schema_version=MASK_SCHEMA_VERSION,
        frame=base_frame_identity,
        width_px=width,
        height_px=height,
        body=body,
        club=club,
        valid=valid,
        revision_id="rev-good-01",
        parent_revision_id=None,
        producer_id="ann",
        correction_note="Initial",
    )
    provider = ManualMaskProvider()
    provider.register_mask(m1)

    snapshot_rev_count = len(provider.all_revisions())
    snapshot_history = provider.get_revision_history("f-010", shot_id="shot-01")

    m_invalid = MaskFrame(
        schema_version=MASK_SCHEMA_VERSION,
        frame=base_frame_identity,
        width_px=width,
        height_px=height,
        body=body,
        club=club,
        valid=valid,
        revision_id="rev-bad-child",
        parent_revision_id="rev-missing-parent",
        producer_id="ann",
        correction_note="Invalid child",
    )
    with pytest.raises(ValueError):
        provider.register_mask(m_invalid)

    assert len(provider.all_revisions()) == snapshot_rev_count
    assert provider.get_revision_history("f-010", shot_id="shot-01") == snapshot_history
    assert not provider.has_revision("rev-bad-child")


def test_namespaced_full_asset_swing_camera_frame_identity(
    base_frame_identity: FrameIdentity,
) -> None:
    width, height = 2, 2
    total_px = width * height
    valid = bytes([1] * total_px)
    body = bytes([1, 0, 0, 0])
    club = bytes([0, 1, 0, 0])

    cam_dtl_frame = FrameIdentity(
        schema_version=FRAME_SCHEMA_VERSION,
        asset_id=base_frame_identity.asset_id,
        shot_id=base_frame_identity.shot_id,
        swing_id=base_frame_identity.swing_id,
        camera_id="cam-down-the-line",
        frame_id="f-010",
        pts_ticks=100,
        timebase_numerator=1,
        timebase_denominator=1000,
        physical_time_s=0.1,
        physical_time_reason="",
        frame_sha256="d" * 64,
    )

    mask_face_on = MaskFrame(
        schema_version=MASK_SCHEMA_VERSION,
        frame=base_frame_identity,
        width_px=width,
        height_px=height,
        body=body,
        club=club,
        valid=valid,
        revision_id="rev-fo-01",
        parent_revision_id=None,
        producer_id="ann",
        correction_note="Face-on",
    )
    mask_dtl = MaskFrame(
        schema_version=MASK_SCHEMA_VERSION,
        frame=cam_dtl_frame,
        width_px=width,
        height_px=height,
        body=body,
        club=club,
        valid=valid,
        revision_id="rev-dtl-01",
        parent_revision_id=None,
        producer_id="ann",
        correction_note="DTL",
    )

    provider = ManualMaskProvider()
    provider.register_mask(mask_face_on)
    provider.register_mask(mask_dtl)

    with pytest.raises(ValueError, match="Multiple"):
        provider.get_mask("f-010")

    res_fo = provider.get_mask("f-010", camera_id="cam-face-on")
    assert res_fo.revision_id == "rev-fo-01"

    res_dtl = provider.get_mask("f-010", camera_id="cam-down-the-line")
    assert res_dtl.revision_id == "rev-dtl-01"


def test_atomic_save_and_reopen_all_revisions(
    base_frame_identity: FrameIdentity,
    tmp_path: Path,
) -> None:
    width, height = 2, 2
    total_px = width * height
    valid = bytes([1] * total_px)
    body_v1 = bytes([1, 0, 0, 0])
    club_v1 = bytes([0, 1, 0, 0])
    club_v2 = bytes([0, 1, 1, 0])

    m1 = MaskFrame(
        schema_version=MASK_SCHEMA_VERSION,
        frame=base_frame_identity,
        width_px=width,
        height_px=height,
        body=body_v1,
        club=club_v1,
        valid=valid,
        revision_id="rev-save-01",
        parent_revision_id=None,
        producer_id="ann",
        correction_note="Initial",
    )
    m2 = MaskFrame(
        schema_version=MASK_SCHEMA_VERSION,
        frame=base_frame_identity,
        width_px=width,
        height_px=height,
        body=body_v1,
        club=club_v2,
        valid=valid,
        revision_id="rev-save-02",
        parent_revision_id="rev-save-01",
        producer_id="reviewer",
        correction_note="Corrected shaft",
    )

    provider = ManualMaskProvider()
    provider.register_mask(m1)
    provider.register_mask(m2)

    save_path = tmp_path / "masks" / "revisions.json"
    provider.save(save_path)
    assert save_path.is_file()

    restored = ManualMaskProvider.load(save_path)
    assert len(restored.all_revisions()) == 2
    assert restored.get_mask("f-010", shot_id="shot-01").revision_id == "rev-save-02"
    history = restored.get_revision_history("f-010", shot_id="shot-01")
    assert [h.revision_id for h in history] == ["rev-save-01", "rev-save-02"]
