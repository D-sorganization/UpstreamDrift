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

from collections.abc import Sequence
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


# ---------------------------------------------------------------------------
# 11. MMR-13 (#11099): Real / Synthetic Neural Segmentation Contracts
# ---------------------------------------------------------------------------


def test_checkpoint_validation_fails_closed_on_corrupt_or_arbitrary_weights(
    tmp_path: Path,
) -> None:
    """Arbitrary/corrupt checkpoints cannot pass validation."""
    corrupt_ckpt = tmp_path / "corrupt_weights.pth"
    corrupt_ckpt.write_bytes(b"corrupted binary data that does not match pinned hash")

    provider = ModelSegmentationProvider(
        model_name="sam-vit-b-golf",
        checkpoint_path=corrupt_ckpt,
    )

    req = SegmentationRequest(shot_id="shot-01", frame_ids=("f-01",))
    with pytest.raises(
        RuntimeError,
        match="not a valid model checkpoint|hash mismatch",
    ):
        provider.segment(req)


def test_no_hidden_unauthenticated_downloads(tmp_path: Path) -> None:
    """Missing weights must fail closed with actionable error and zero network traffic."""
    nonexistent = tmp_path / "missing_dir" / "weights.pth"
    provider = ModelSegmentationProvider(
        model_name="sam-vit-b-golf",
        checkpoint_path=nonexistent,
    )

    req = SegmentationRequest(shot_id="shot-01", frame_ids=("f-01",))
    with pytest.raises(
        FileNotFoundError, match="Hidden network downloads are disallowed"
    ):
        provider.segment(req)


def test_provider_is_lazy_and_optional(tmp_path: Path) -> None:
    """Instantiating provider does not trigger file I/O or network until execution."""
    missing = tmp_path / "nonexistent.pth"
    # Should not raise on initialization
    provider = ModelSegmentationProvider(
        model_name="sam-vit-b-golf",
        checkpoint_path=missing,
    )
    assert not provider.is_loaded
    assert provider.model_name == "sam-vit-b-golf"
    assert provider.checkpoint_path == missing


def test_inference_adapter_separates_person_and_club_masks(
    tmp_path: Path,
    base_frame_identity: FrameIdentity,
) -> None:
    """Inference adapter separates person and club into distinct binary channels."""
    content = b"valid dummy weights for synthetic/real segmentation harness"
    content_sha = hashlib.sha256(content).hexdigest()
    ckpt_file = tmp_path / "test_model.pth"
    ckpt_file.write_bytes(content)

    provider = ModelSegmentationProvider(
        model_name="sam-vit-b-golf",
        checkpoint_path=ckpt_file,
        expected_sha256=content_sha,
        allow_synthetic=True,
    )

    mask = provider.infer_frame(base_frame_identity, width_px=32, height_px=32)

    assert isinstance(mask, MaskFrame)
    assert mask.width_px == 32
    assert mask.height_px == 32
    assert len(mask.body) == 32 * 32
    assert len(mask.club) == 32 * 32
    assert len(mask.valid) == 32 * 32

    # Separate person and club masks: both are nonempty
    body_count = mask.body.count(1)
    club_count = mask.club.count(1)
    assert body_count > 0, "Person/body mask must contain detected foreground pixels"
    assert club_count > 0, "Club mask must contain detected foreground pixels"

    # Distinct separation: pixels where valid is 0 cannot be foreground
    for idx, (b, c, v) in enumerate(zip(mask.body, mask.club, mask.valid, strict=True)):
        if v == 0:
            assert b == 0 and c == 0, f"Foreground at {idx} where valid=0"


def test_inference_generates_mask_artifacts_with_provenance_hashes(
    tmp_path: Path,
    base_frame_identity: FrameIdentity,
) -> None:
    """Generated mask artifacts record frame and checkpoint provenance SHA-256 hashes."""
    content = b"verified checkpoint binary payload"
    content_sha = hashlib.sha256(content).hexdigest()
    ckpt_file = tmp_path / "verified_model.pth"
    ckpt_file.write_bytes(content)

    provider = ModelSegmentationProvider(
        model_name="sam-vit-b-golf",
        checkpoint_path=ckpt_file,
        expected_sha256=content_sha,
        allow_synthetic=True,
    )

    mask = provider.infer_frame(base_frame_identity, width_px=16, height_px=16)

    # Frame SHA-256 provenance is preserved
    assert mask.frame.frame_sha256 == base_frame_identity.frame_sha256
    # Checkpoint provenance hash is embedded in producer_id and revision_id
    assert content_sha[:8] in mask.revision_id
    assert content_sha[:8] in mask.producer_id
    # Deterministic observation hash
    assert len(mask.observation_hash) == 64

    # Fulfills SegmentationRequest as a Segmenter
    req = SegmentationRequest(
        shot_id=base_frame_identity.shot_id,
        frame_ids=(base_frame_identity.frame_id,),
    )
    res = provider.segment(req)
    assert isinstance(res, SegmentationResult)
    assert res.mask_count == 1
    assert content_sha[:8] in res.provenance


def test_manual_path_works_when_provider_is_absent_and_lineage_preserved(
    tmp_path: Path,
    base_frame_identity: FrameIdentity,
) -> None:
    """Manual path works when provider is unconfigured, and manual touchups branch from model revisions."""
    from shared.python.shadow_tracker.service import DefaultShadowTrackerService
    from shared.python.shadow_tracker.contracts import FrameObservation
    from shared.python.shadow_tracker.source_records import SourceAsset

    # 1. Unconfigured service operates purely manual
    service = DefaultShadowTrackerService()
    assert service.segmenter is None

    # 2. When provider is registered, model masks feed manual provider and support manual correction
    content = b"checkpoint bytes for service test"
    content_sha = hashlib.sha256(content).hexdigest()
    ckpt_file = tmp_path / "service_model.pth"
    ckpt_file.write_bytes(content)

    provider = ModelSegmentationProvider(
        model_name="sam-vit-b-golf",
        checkpoint_path=ckpt_file,
        expected_sha256=content_sha,
        allow_synthetic=True,
    )
    service.register_segmenter(provider)
    assert service.segmenter is not None
    assert isinstance(service.segmenter, ModelSegmentationProvider)
    assert provider is not None

    # Model generates initial mask
    model_mask = provider.infer_frame(base_frame_identity, width_px=4, height_px=4)

    obs = FrameObservation(
        schema_version="shadow-tracker/frame-observation/1.0.0",
        shot_id=base_frame_identity.shot_id,
        camera_id=base_frame_identity.camera_id,
        frame_id=base_frame_identity.frame_id,
        pts_ticks=base_frame_identity.pts_ticks,
        timebase_numerator=base_frame_identity.timebase_numerator,
        timebase_denominator=base_frame_identity.timebase_denominator,
        physical_time_s=base_frame_identity.physical_time_s,
        physical_time_reason="standard_shutter",
        body_mask_ref="mask-body-01",
        club_mask_ref="mask-club-01",
        valid_mask_ref="mask-valid-01",
        confidence_provenance="model_draft",
    )
    asset = SourceAsset(
        schema_version="shadow-tracker/source/1.0.0",
        asset_id=base_frame_identity.asset_id,
        source_uri="urn:asset:golf-01",
        content_sha256="e" * 64,
        width_px=4,
        height_px=4,
        rights_status="permitted",
        rights_note="test",
    )
    service.initialize_session(
        source_asset=asset,
        observations=[obs],
        initial_masks=[model_mask],
    )

    # Human reviewer performs correction
    corrected_club = bytes([0] * 15 + [1])
    revised = service.update_mask(
        frame_id=base_frame_identity.frame_id,
        body=model_mask.body,
        club=corrected_club,
        valid=model_mask.valid,
        parent_revision_id=model_mask.revision_id,
        producer_id="reviewer-human",
        correction_note="Touched up clubhead boundary",
    )

    assert revised.parent_revision_id == model_mask.revision_id
    history = service.get_mask_history(base_frame_identity.frame_id)
    assert len(history) == 2
    assert history[0].revision_id == model_mask.revision_id
    assert history[1].revision_id == revised.revision_id


def test_evaluate_segmentation_benchmark_modern_archive_occluded(
    tmp_path: Path,
) -> None:
    """Benchmark evaluates modern high-speed, archive historical, and occluded clips."""
    from shared.python.shadow_tracker.segmentation import (
        BenchmarkClip,
        evaluate_segmentation_benchmark,
    )

    content = b"pinned weights for benchmark evaluation suite"
    content_sha = hashlib.sha256(content).hexdigest()
    ckpt = tmp_path / "model.pth"
    ckpt.write_bytes(content)

    provider = ModelSegmentationProvider(
        model_name="sam-vit-b-golf",
        checkpoint_path=ckpt,
        expected_sha256=content_sha,
        allow_synthetic=True,
    )

    clips = [
        BenchmarkClip(
            clip_id="modern-120fps",
            clip_type="modern_high_speed",
            description="120 fps modern launch monitor footage with high-speed swing blur",
            width_px=32,
            height_px=32,
            adverse_conditions=("blur_120fps",),
        ),
        BenchmarkClip(
            clip_id="archive-1953",
            clip_type="archive_historical",
            description="1953 archival film scan with shaft dropouts",
            width_px=32,
            height_px=32,
            adverse_conditions=("thin_shaft_loss",),
        ),
        BenchmarkClip(
            clip_id="occluded-gallery",
            clip_type="occluded_adverse",
            description="Tour clip with spectator railing occlusion",
            width_px=32,
            height_px=32,
            adverse_conditions=("partial_occlusion_spectator",),
        ),
    ]

    report = evaluate_segmentation_benchmark(provider, clips)
    assert report["schema_version"] == 1
    assert len(report["clips"]) == 3

    # Check metrics
    modern_m = report["clips"][0]
    archive_m = report["clips"][1]
    occ_m = report["clips"][2]

    assert 0.0 <= modern_m["body_iou"] <= 1.0
    assert 0.0 <= modern_m["club_recall"] <= 1.0
    assert modern_m["latency_ms_per_frame"] >= 0.0
    assert modern_m["peak_memory_mb"] > 0.0

    assert occ_m["occlusion_detected"] is True


# ---------------------------------------------------------------------------
# 11. Issue #11227: Synthetic Fallback Must Not Masquerade as Observed Model Inference
# ---------------------------------------------------------------------------


def test_absent_inference_engine_fails_closed_without_producing_unverified_masks(
    tmp_path: Path,
    base_frame_identity: FrameIdentity,
) -> None:
    """Absent inference engine must fail closed instead of silently returning synthetic masks."""
    content = b"pinned weights bytes"
    content_sha = hashlib.sha256(content).hexdigest()
    ckpt = tmp_path / "model.pth"
    ckpt.write_bytes(content)

    # Default provider without inference_engine and allow_synthetic=False
    provider = ModelSegmentationProvider(
        model_name="sam-vit-b-golf",
        checkpoint_path=ckpt,
        expected_sha256=content_sha,
    )
    assert not provider.allow_synthetic

    # infer_frame must fail closed
    with pytest.raises(
        RuntimeError,
        match="no active inference engine configured|cannot produce observed masks",
    ):
        provider.infer_frame(base_frame_identity, width_px=16, height_px=16)

    # segment must also fail closed
    req = SegmentationRequest(
        shot_id=base_frame_identity.shot_id,
        frame_ids=(base_frame_identity.frame_id,),
    )
    with pytest.raises(
        RuntimeError,
        match="no active inference engine configured|cannot produce observed masks",
    ):
        provider.segment(req)


def test_explicit_synthetic_mode_labels_provenance_and_marks_is_synthetic(
    tmp_path: Path,
    base_frame_identity: FrameIdentity,
) -> None:
    """Explicit allow_synthetic=True marks MaskFrame and SegmentationResult with synthetic provenance."""
    content = b"pinned weights bytes for synthetic fixture"
    content_sha = hashlib.sha256(content).hexdigest()
    ckpt = tmp_path / "model.pth"
    ckpt.write_bytes(content)

    provider = ModelSegmentationProvider(
        model_name="sam-vit-b-golf",
        checkpoint_path=ckpt,
        expected_sha256=content_sha,
        allow_synthetic=True,
    )
    assert provider.allow_synthetic

    mask = provider.infer_frame(base_frame_identity, width_px=16, height_px=16)
    assert mask.is_synthetic is True
    assert mask.producer_id.startswith("synthetic:")
    assert (
        "Automated inference from verified pinned model weights"
        not in mask.correction_note
    )
    assert "Synthetic" in mask.correction_note

    req = SegmentationRequest(
        shot_id=base_frame_identity.shot_id,
        frame_ids=(base_frame_identity.frame_id,),
    )
    res = provider.segment(req)
    assert res.provenance.startswith("synthetic:")


def test_real_injected_inference_produces_model_labeled_masks(
    tmp_path: Path,
    base_frame_identity: FrameIdentity,
) -> None:
    """Injected real inference engine produces legitimate model-labeled masks."""
    content = b"pinned weights bytes for real inference"
    content_sha = hashlib.sha256(content).hexdigest()
    ckpt = tmp_path / "model.pth"
    ckpt.write_bytes(content)

    called = False

    def dummy_inference(
        frame: FrameIdentity,
        width: int,
        height: int,
        adverse: Sequence[str],
    ) -> tuple[bytes, bytes, bytes]:
        nonlocal called
        called = True
        total = width * height
        return bytes([1] * total), bytes([0] * total), bytes([1] * total)

    provider = ModelSegmentationProvider(
        model_name="sam-vit-b-golf",
        checkpoint_path=ckpt,
        expected_sha256=content_sha,
        inference_engine=dummy_inference,
    )
    assert not provider.allow_synthetic

    mask = provider.infer_frame(base_frame_identity, width_px=8, height_px=8)
    assert called is True
    assert mask.is_synthetic is False
    assert mask.producer_id.startswith("model:")
    assert (
        mask.correction_note == "Automated inference from verified pinned model weights"
    )

    req = SegmentationRequest(
        shot_id=base_frame_identity.shot_id,
        frame_ids=(base_frame_identity.frame_id,),
    )
    res = provider.segment(req)
    assert res.provenance.startswith("model_inference:")


def test_synthetic_observations_and_masks_block_release_qualification(
    base_frame_identity: FrameIdentity,
) -> None:
    """Synthetic observations or masks cannot qualify for release (Gate G0 failure and dynamic_candidate demotion)."""
    from shared.python.shadow_tracker.contracts import (
        CandidateResult,
        FrameObservation,
        ReplayAudit,
    )
    from shared.python.shadow_tracker.evaluation import (
        GateProfile,
        audit_gate_profile,
        classify_evidence_quality,
    )

    synth_obs = FrameObservation(
        schema_version="shadow-tracker/frame-observation/1.0.0",
        shot_id=base_frame_identity.shot_id,
        camera_id=base_frame_identity.camera_id,
        frame_id=base_frame_identity.frame_id,
        pts_ticks=10,
        timebase_numerator=1,
        timebase_denominator=1000,
        physical_time_s=0.01,
        physical_time_reason="standard_shutter",
        body_mask_ref="mask-body-synth",
        club_mask_ref="mask-club-synth",
        valid_mask_ref="mask-valid-synth",
        confidence_provenance="synthetic",
    )

    audit = ReplayAudit(
        schema_version="shadow-tracker/replay-audit/1.0.0",
        candidate_id="cand-01",
        reset_count=1,
        integrator_name="mujoco",
        integrator_version="3.3.0",
        coverage_start_s=0.0,
        coverage_end_s=0.1,
        max_grip_translation_error_m=0.001,
        max_grip_rotation_error_rad=0.01,
        is_physically_accepted=True,
    )

    candidate = CandidateResult(
        schema_version="shadow-tracker/candidate-result/1.0.0",
        candidate_id="cand-01",
        request_id="req-01",
        initial_state=(0.0,) * 42,
        trajectory=((0.0,) * 42,),
        diagnostics={"mean_iou": 0.95, "is_synthetic": True},
        uncertainty_method="empirical_holdout",
        replay_audit=audit,
        is_accepted=True,
    )

    # Release profile rejects synthetic observations
    release_profile = GateProfile(allow_synthetic=False)
    all_passed, statuses = audit_gate_profile(candidate, (synth_obs,), release_profile)
    assert not all_passed
    g0 = next(s for s in statuses if s.gate_id == "G0")
    assert not g0.passed
    assert "Synthetic" in g0.reason

    # Classification demotes to dynamic_candidate rather than validated_profile
    from shared.python.shadow_tracker.contracts import FitRequest

    req = FitRequest(
        schema_version="shadow-tracker/fit-request/1.0.0",
        request_id="req-01",
        shot_id="shot-01",
        model_hash="c" * 64,
        candidate_count=1,
        objective_profile="standard",
        time_window_start_pts=0,
        time_window_end_pts=100,
        budget_seconds=10.0,
        engine_capability_requirement=(),
    )
    quality = classify_evidence_quality(
        req, candidate, (audit,), (synth_obs,), release_profile
    )
    assert quality == "dynamic_candidate"
