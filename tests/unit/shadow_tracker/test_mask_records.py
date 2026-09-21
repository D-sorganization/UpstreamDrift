"""Unit tests for Shadow Tracker immutable binary masks (Packet B, #10138)."""

from __future__ import annotations

from typing import Any

import pytest

from shared.python.shadow_tracker.mask_records import MaskFrame
from shared.python.shadow_tracker.source_records import FrameIdentity

pytestmark = pytest.mark.unit

VALID_HASH_A = "a" * 64
VALID_HASH_B = "b" * 64


def _make_frame_identity(**kwargs: object) -> FrameIdentity:
    defaults: dict[str, object] = {
        "schema_version": "shadow-tracker/frame/1.0.0",
        "asset_id": "asset-1",
        "shot_id": "shot-1",
        "swing_id": "swing-1",
        "camera_id": "cam-1",
        "frame_id": "frame-001",
        "pts_ticks": 0,
        "timebase_numerator": 1,
        "timebase_denominator": 24,
        "physical_time_s": 0.0,
        "physical_time_reason": "Synchronized optical timing clock",
        "frame_sha256": VALID_HASH_B,
    }
    defaults.update(kwargs)
    return FrameIdentity(**defaults)  # type: ignore[arg-type]


def _make_mask_frame(**kwargs: object) -> MaskFrame:
    frame = _make_frame_identity()
    defaults: dict[str, object] = {
        "schema_version": "shadow-tracker/mask/1.0.0",
        "frame": frame,
        "width_px": 2,
        "height_px": 2,
        "body": bytes([1, 0, 0, 0]),
        "club": bytes([0, 1, 0, 0]),
        "valid": bytes([1, 1, 0, 0]),
        "revision_id": "rev-1",
        "parent_revision_id": None,
        "producer_id": "manual-annotation-tool-v1",
        "correction_note": "",
    }
    defaults.update(kwargs)
    return MaskFrame(**defaults)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Acceptance Cases from CONTRACT_FREEZE.md                                    #
# --------------------------------------------------------------------------- #


def test_acceptance_case_2x2_foreground() -> None:
    """Acceptance: 2x2: body [1,0,0,0], club [0,1,0,0], valid [1,1,0,0] -> Valid, foreground present."""
    mask = _make_mask_frame(
        width_px=2,
        height_px=2,
        body=bytes([1, 0, 0, 0]),
        club=bytes([0, 1, 0, 0]),
        valid=bytes([1, 1, 0, 0]),
    )
    assert mask.has_valid_pixels is True
    assert mask.has_observed_foreground is True


def test_acceptance_case_hash_changes_on_byte_or_parent_revision() -> None:
    """Acceptance: Change a byte or parent revision -> Different observation hash."""
    base = _make_mask_frame(
        width_px=2,
        height_px=2,
        body=bytes([1, 0, 0, 0]),
        club=bytes([0, 1, 0, 0]),
        valid=bytes([1, 1, 0, 0]),
        revision_id="rev-2",
        parent_revision_id="rev-1",
        correction_note="Initial segmented mask",
    )
    base_hash = base.observation_hash

    # Mutate a body byte
    mut_body = _make_mask_frame(
        width_px=2,
        height_px=2,
        body=bytes([0, 0, 0, 0]),
        club=bytes([0, 1, 0, 0]),
        valid=bytes([1, 1, 0, 0]),
        revision_id="rev-2",
        parent_revision_id="rev-1",
        correction_note="Initial segmented mask",
    )
    assert mut_body.observation_hash != base_hash

    # Mutate parent revision
    mut_parent = _make_mask_frame(
        width_px=2,
        height_px=2,
        body=bytes([1, 0, 0, 0]),
        club=bytes([0, 1, 0, 0]),
        valid=bytes([1, 1, 0, 0]),
        revision_id="rev-2",
        parent_revision_id="rev-0",
        correction_note="Initial segmented mask",
    )
    assert mut_parent.observation_hash != base_hash

    # Mutate club byte
    mut_club = _make_mask_frame(
        width_px=2,
        height_px=2,
        body=bytes([1, 0, 0, 0]),
        club=bytes([0, 0, 0, 0]),
        valid=bytes([1, 1, 0, 0]),
        revision_id="rev-2",
        parent_revision_id="rev-1",
        correction_note="Initial segmented mask",
    )
    assert mut_club.observation_hash != base_hash

    # Mutate valid byte
    mut_valid = _make_mask_frame(
        width_px=2,
        height_px=2,
        body=bytes([1, 0, 0, 0]),
        club=bytes([0, 1, 0, 0]),
        valid=bytes([1, 1, 1, 0]),
        revision_id="rev-2",
        parent_revision_id="rev-1",
        correction_note="Initial segmented mask",
    )
    assert mut_valid.observation_hash != base_hash


def test_acceptance_case_valid_all_zero() -> None:
    """Acceptance: valid all zero, body/club zero -> Valid record; both evidence properties false."""
    mask = _make_mask_frame(
        width_px=2,
        height_px=2,
        body=bytes([0, 0, 0, 0]),
        club=bytes([0, 0, 0, 0]),
        valid=bytes([0, 0, 0, 0]),
    )
    assert mask.has_valid_pixels is False
    assert mask.has_observed_foreground is False


def test_acceptance_case_valid_all_one() -> None:
    """Acceptance: valid all one, body/club zero -> Valid pixels true, foreground false."""
    mask = _make_mask_frame(
        width_px=2,
        height_px=2,
        body=bytes([0, 0, 0, 0]),
        club=bytes([0, 0, 0, 0]),
        valid=bytes([1, 1, 1, 1]),
    )
    assert mask.has_valid_pixels is True
    assert mask.has_observed_foreground is False


def test_acceptance_case_foreground_in_invalid_pixel_rejected() -> None:
    """Acceptance: Foreground in invalid pixel -> ValueError."""
    # Body in invalid pixel
    with pytest.raises(ValueError, match="body.*valid"):
        _make_mask_frame(
            width_px=2,
            height_px=2,
            body=bytes([1, 0, 0, 0]),
            club=bytes([0, 0, 0, 0]),
            valid=bytes([0, 0, 0, 0]),
        )

    # Club in invalid pixel
    with pytest.raises(ValueError, match="club.*valid"):
        _make_mask_frame(
            width_px=2,
            height_px=2,
            body=bytes([0, 0, 0, 0]),
            club=bytes([0, 1, 0, 0]),
            valid=bytes([0, 0, 0, 0]),
        )


def test_acceptance_case_pixel_255_rejected() -> None:
    """Acceptance: pixel 255 -> ValueError."""
    with pytest.raises(ValueError, match="body"):
        _make_mask_frame(
            width_px=2,
            height_px=2,
            body=bytes([255, 0, 0, 0]),
            club=bytes([0, 0, 0, 0]),
            valid=bytes([1, 1, 1, 1]),
        )


def test_acceptance_case_wrong_payload_length_rejected() -> None:
    """Acceptance: wrong payload length -> ValueError."""
    with pytest.raises(ValueError, match="length"):
        _make_mask_frame(
            width_px=2,
            height_px=2,
            body=bytes([1, 0, 0]),  # 3 bytes instead of 4
            club=bytes([0, 0, 0, 0]),
            valid=bytes([1, 1, 1, 1]),
        )


# --------------------------------------------------------------------------- #
# Schema & Type Validation Tests                                              #
# --------------------------------------------------------------------------- #


def test_mask_frame_schema_version_validation() -> None:
    with pytest.raises(ValueError, match="schema_version"):
        _make_mask_frame(schema_version="shadow-tracker/mask/2.0.0")


def test_mask_frame_rejects_non_frame_identity() -> None:
    with pytest.raises(TypeError, match="frame"):
        _make_mask_frame(frame="not-a-frame-identity")


def test_mask_frame_dimension_validation() -> None:
    with pytest.raises(ValueError, match="width_px"):
        _make_mask_frame(width_px=0)
    with pytest.raises(TypeError, match="width_px"):
        _make_mask_frame(width_px=True)  # booleans rejected
    with pytest.raises(ValueError, match="height_px"):
        _make_mask_frame(height_px=-1)
    with pytest.raises(TypeError, match="height_px"):
        _make_mask_frame(height_px="10")


def test_mask_frame_bytearray_rejected() -> None:
    """Bytearray inputs must raise TypeError, not leak mutable ownership."""
    with pytest.raises(TypeError, match="body must be bytes, got bytearray"):
        _make_mask_frame(body=bytearray([1, 0, 0, 0]))
    with pytest.raises(TypeError, match="club must be bytes, got bytearray"):
        _make_mask_frame(club=bytearray([0, 1, 0, 0]))
    with pytest.raises(TypeError, match="valid must be bytes, got bytearray"):
        _make_mask_frame(valid=bytearray([1, 1, 0, 0]))


def test_mask_frame_immutable_ownership() -> None:
    """Ensure record owns immutable bytes."""
    raw_body = bytes([1, 0, 0, 0])
    mask = _make_mask_frame(body=raw_body)
    assert mask.body == raw_body
    assert isinstance(mask.body, bytes)


def test_mask_frame_revision_validation() -> None:
    with pytest.raises(ValueError, match="revision_id"):
        _make_mask_frame(revision_id="")
    with pytest.raises(ValueError, match="revision_id"):
        _make_mask_frame(revision_id=" rev-1 ")  # untrimmed

    # parent_revision_id cannot equal revision_id
    with pytest.raises(ValueError, match="parent_revision_id cannot equal revision_id"):
        _make_mask_frame(revision_id="rev-1", parent_revision_id="rev-1")

    with pytest.raises(ValueError, match="parent_revision_id"):
        _make_mask_frame(parent_revision_id=" rev-0 ")


def test_mask_frame_producer_id_validation() -> None:
    with pytest.raises(ValueError, match="producer_id"):
        _make_mask_frame(producer_id="")
    with pytest.raises(ValueError, match="producer_id"):
        _make_mask_frame(producer_id=" tool ")


def test_mask_frame_correction_note_validation() -> None:
    # Unrevised (parent_revision_id is None): correction_note may be empty
    mask = _make_mask_frame(parent_revision_id=None, correction_note="")
    assert mask.correction_note == ""

    # Revised mask (parent_revision_id is set): correction_note must be non-empty and trimmed
    with pytest.raises(ValueError, match="correction_note"):
        _make_mask_frame(
            revision_id="rev-2",
            parent_revision_id="rev-1",
            correction_note="",
        )
    with pytest.raises(ValueError, match="correction_note"):
        _make_mask_frame(
            revision_id="rev-2",
            parent_revision_id="rev-1",
            correction_note="  fixed hand silhouette  ",
        )

    # Valid revised mask
    revised = _make_mask_frame(
        revision_id="rev-2",
        parent_revision_id="rev-1",
        correction_note="Fixed hand silhouette boundary",
    )
    assert revised.correction_note == "Fixed hand silhouette boundary"


def test_mask_frame_body_club_overlap_allowed() -> None:
    """Body and club may overlap where class ownership is ambiguous."""
    mask = _make_mask_frame(
        width_px=1,
        height_px=1,
        body=bytes([1]),
        club=bytes([1]),
        valid=bytes([1]),
    )
    assert mask.has_observed_foreground is True
    assert mask.body == bytes([1])
    assert mask.club == bytes([1])


# --------------------------------------------------------------------------- #
# Serialization and Deserialization Tests                                     #
# --------------------------------------------------------------------------- #


def test_mask_frame_serialization_round_trip() -> None:
    mask = _make_mask_frame(
        width_px=2,
        height_px=2,
        body=bytes([1, 0, 0, 0]),
        club=bytes([0, 1, 0, 0]),
        valid=bytes([1, 1, 0, 0]),
        revision_id="rev-2",
        parent_revision_id="rev-1",
        producer_id="segmenter-v1",
        correction_note="Refined club shaft",
    )
    data = mask.to_dict()

    # Byte fields serialize as JSON integer lists
    assert data["body"] == [1, 0, 0, 0]
    assert data["club"] == [0, 1, 0, 0]
    assert data["valid"] == [1, 1, 0, 0]
    assert isinstance(data["frame"], dict)
    assert data["schema_version"] == "shadow-tracker/mask/1.0.0"

    # Round trip
    reconstructed = MaskFrame.from_dict(data)
    assert reconstructed == mask
    assert reconstructed.observation_hash == mask.observation_hash


def test_mask_frame_from_dict_rejects_unknown_fields() -> None:
    mask = _make_mask_frame()
    data = mask.to_dict()
    data["extra_field"] = "bad"
    with pytest.raises(ValueError, match="Unknown fields"):
        MaskFrame.from_dict(data)


def test_mask_frame_from_dict_rejects_non_01_integers_in_lists() -> None:
    mask = _make_mask_frame()
    data = mask.to_dict()
    data["body"] = [2, 0, 0, 0]
    with pytest.raises(ValueError, match="body"):
        MaskFrame.from_dict(data)

    data = mask.to_dict()
    data["club"] = [True, 0, 0, 0]  # booleans in list
    with pytest.raises(TypeError, match="club"):
        MaskFrame.from_dict(data)


def test_mask_frame_observation_hash_format_and_determinism() -> None:
    mask = _make_mask_frame()
    h1 = mask.observation_hash
    h2 = mask.observation_hash
    assert h1 == h2
    assert len(h1) == 64
    assert all(c in "0123456789abcdef" for c in h1)


# --------------------------------------------------------------------------- #
#  Issue #10151 Malformed-Input and Pre-Conversion Tests                      #
# --------------------------------------------------------------------------- #


def test_mask_frame_from_dict_wrong_container_type_raises_type_error() -> None:
    """MaskFrame.from_dict([]) must raise TypeError naming payload/container."""
    with pytest.raises(TypeError, match="dict|mapping|payload"):
        MaskFrame.from_dict([])  # type: ignore[arg-type]


def test_mask_frame_from_dict_missing_required_keys_raises_value_error() -> None:
    """MaskFrame.from_dict({}) must raise ValueError naming missing fields, not leak KeyError."""
    with pytest.raises(ValueError, match="Missing required field|frame"):
        MaskFrame.from_dict({})


def test_mask_frame_from_dict_rejects_oversized_sequence_before_consumption() -> None:
    """Validate positive dimensions and all three payload lengths before reading pixels.

    A custom Sequence whose __getitem__ raises should NEVER have __getitem__ called
    if its length does not match width_px * height_px.
    """
    from collections.abc import Sequence

    class OversizeSequence(Sequence):
        def __len__(self) -> int:
            return 5

        def __getitem__(self, index: Any) -> Any:
            raise RuntimeError("pixels consumed before length rejection")

    mask = _make_mask_frame(width_px=2, height_px=2)
    payload = mask.to_dict()
    payload["body"] = OversizeSequence()

    with pytest.raises(ValueError, match="body.*length"):
        MaskFrame.from_dict(payload)


def test_mask_frame_from_dict_rejects_non_dict_frame_with_type_error() -> None:
    """Non-dict / non-FrameIdentity frame field must raise TypeError."""
    mask = _make_mask_frame()
    payload = mask.to_dict()
    payload["frame"] = 123
    with pytest.raises(TypeError, match="frame"):
        MaskFrame.from_dict(payload)
