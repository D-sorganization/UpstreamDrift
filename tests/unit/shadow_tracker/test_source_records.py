"""Unit tests for Shadow Tracker source and frame records (Packet A, #10137)."""

from __future__ import annotations

from fractions import Fraction
import pytest

from shared.python.shadow_tracker.source_records import (
    FrameIdentity,
    SourceAsset,
    validate_frame_sequence,
)

pytestmark = pytest.mark.unit

VALID_HASH_A = "a" * 64
VALID_HASH_B = "b" * 64
VALID_HASH_C = "c" * 64


def _make_source_asset(**kwargs: object) -> SourceAsset:
    defaults: dict[str, object] = {
        "schema_version": "shadow-tracker/source/1.0.0",
        "asset_id": "asset-1",
        "source_uri": "https://example.com/video.mp4",
        "content_sha256": VALID_HASH_A,
        "width_px": 1920,
        "height_px": 1080,
        "rights_status": "permitted",
        "rights_note": "Broadcast rights verified",
    }
    defaults.update(kwargs)
    return SourceAsset(**defaults)  # type: ignore[arg-type]


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


# --------------------------------------------------------------------------- #
#  SourceAsset tests                                                          #
# --------------------------------------------------------------------------- #


def test_source_asset_valid_and_serialization() -> None:
    asset = _make_source_asset()
    assert asset.schema_version == "shadow-tracker/source/1.0.0"
    assert asset.asset_id == "asset-1"
    assert asset.source_uri == "https://example.com/video.mp4"
    assert asset.content_sha256 == VALID_HASH_A
    assert asset.width_px == 1920
    assert asset.height_px == 1080
    assert asset.rights_status == "permitted"
    assert asset.rights_note == "Broadcast rights verified"

    payload = asset.to_dict()
    restored = SourceAsset.from_dict(payload)
    assert restored == asset


def test_source_asset_immutability() -> None:
    asset = _make_source_asset()
    with pytest.raises((AttributeError, TypeError)):
        asset.asset_id = "mutated"  # type: ignore[misc]


def test_source_asset_keyword_only() -> None:
    with pytest.raises(TypeError):
        SourceAsset(  # type: ignore[call-arg]
            "shadow-tracker/source/1.0.0",
            "asset-1",
            "https://example.com/v.mp4",
            VALID_HASH_A,
            1920,
            1080,
            "permitted",
            "note",
        )


def test_source_asset_rights_unknown_allows_empty_note() -> None:
    asset = _make_source_asset(rights_status="unknown", rights_note="")
    assert asset.rights_status == "unknown"
    assert asset.rights_note == ""
    assert SourceAsset.from_dict(asset.to_dict()) == asset


@pytest.mark.parametrize("status", ["permitted", "restricted"])
def test_source_asset_rights_known_requires_nonempty_note(status: str) -> None:
    with pytest.raises(ValueError, match="rights_note"):
        _make_source_asset(rights_status=status, rights_note="")
    with pytest.raises(ValueError, match="rights_note"):
        _make_source_asset(rights_status=status, rights_note="   ")


def test_source_asset_invalid_rights_status() -> None:
    with pytest.raises(ValueError, match="rights_status"):
        _make_source_asset(rights_status="public_domain")


def test_source_asset_schema_version_mismatch() -> None:
    with pytest.raises(ValueError, match="schema_version"):
        _make_source_asset(schema_version="shadow-tracker/source/2.0.0")


@pytest.mark.parametrize(
    "uri",
    [
        "file:///path/to/video.mp4",
        "file://localhost/video.mp4",
        "/local/path/video.mp4",
        "C:\\path\\video.mp4",
        "C:/path/video.mp4",
        "ftp://example.com/video.mp4",
        "relative/path/video.mp4",
        "",
        "   ",
    ],
)
def test_source_asset_rejects_disallowed_uris(uri: str) -> None:
    with pytest.raises(ValueError, match="source_uri"):
        _make_source_asset(source_uri=uri)


@pytest.mark.parametrize(
    "uri",
    [
        "https://example.com/video.mp4",
        "http://example.com/stream.m3u8",
        f"urn:sha256:{VALID_HASH_A}",
        "urn:asset:golf-swing-2026",
    ],
)
def test_source_asset_accepts_valid_uris(uri: str) -> None:
    asset = _make_source_asset(source_uri=uri)
    assert asset.source_uri == uri


@pytest.mark.parametrize(
    "bad_id",
    ["", "   ", " asset", "asset ", "asset\n", "\tasset"],
)
def test_source_asset_rejects_untrimmed_or_empty_id(bad_id: str) -> None:
    with pytest.raises(ValueError, match="asset_id"):
        _make_source_asset(asset_id=bad_id)


def test_source_asset_rejects_non_str_id() -> None:
    with pytest.raises(TypeError, match="asset_id"):
        _make_source_asset(asset_id=123)


@pytest.mark.parametrize(
    "bad_hash",
    [
        "a" * 63,
        "a" * 65,
        "A" * 64,
        "g" * 64,
        "",
    ],
)
def test_source_asset_rejects_invalid_hash(bad_hash: str) -> None:
    with pytest.raises(ValueError, match="content_sha256"):
        _make_source_asset(content_sha256=bad_hash)


@pytest.mark.parametrize("dim_field", ["width_px", "height_px"])
def test_source_asset_rejects_non_positive_or_non_int_dimensions(
    dim_field: str,
) -> None:
    with pytest.raises(ValueError, match=dim_field):
        _make_source_asset(**{dim_field: 0})
    with pytest.raises(ValueError, match=dim_field):
        _make_source_asset(**{dim_field: -100})
    with pytest.raises(TypeError, match=dim_field):
        _make_source_asset(**{dim_field: True})
    with pytest.raises(TypeError, match=dim_field):
        _make_source_asset(**{dim_field: "1920"})
    with pytest.raises(TypeError, match=dim_field):
        _make_source_asset(**{dim_field: 1920.0})


def test_source_asset_from_dict_rejects_unknown_fields() -> None:
    payload = _make_source_asset().to_dict()
    payload["extra_field"] = "illegal"
    with pytest.raises(ValueError, match="extra_field"):
        SourceAsset.from_dict(payload)


# --------------------------------------------------------------------------- #
#  FrameIdentity tests                                                        #
# --------------------------------------------------------------------------- #


def test_frame_identity_valid_and_serialization() -> None:
    frame = _make_frame_identity()
    assert frame.schema_version == "shadow-tracker/frame/1.0.0"
    assert frame.pts_ticks == 0
    assert frame.presentation_time == Fraction(0, 24)
    assert frame.physical_time_s == 0.0

    payload = frame.to_dict()
    restored = FrameIdentity.from_dict(payload)
    assert restored == frame


def test_frame_identity_negative_pts_and_fraction() -> None:
    frame = _make_frame_identity(
        pts_ticks=-1,
        timebase_numerator=1,
        timebase_denominator=24,
        physical_time_s=None,
        physical_time_reason="Unknown physical time; uncalibrated clock",
    )
    assert frame.pts_ticks == -1
    assert frame.presentation_time == Fraction(-1, 24)
    assert frame.physical_time_s is None
    assert frame.physical_time_reason == "Unknown physical time; uncalibrated clock"

    restored = FrameIdentity.from_dict(frame.to_dict())
    assert restored == frame
    assert restored.presentation_time == Fraction(-1, 24)
    assert restored.physical_time_s is None


def test_frame_identity_timebase_must_be_reduced() -> None:
    # 2/48 is unreduced (gcd=2 != 1) -> must raise ValueError
    with pytest.raises(ValueError, match="timebase"):
        _make_frame_identity(timebase_numerator=2, timebase_denominator=48)

    # 3/30 -> unreduced
    with pytest.raises(ValueError, match="timebase"):
        _make_frame_identity(timebase_numerator=3, timebase_denominator=30)

    # 1001/30000 -> reduced (standard 29.97 NTSC drop-frame)
    frame = _make_frame_identity(timebase_numerator=1001, timebase_denominator=30000)
    assert frame.presentation_time == Fraction(0, 1)


@pytest.mark.parametrize("dim_field", ["timebase_numerator", "timebase_denominator"])
def test_frame_identity_timebase_positive_and_strict_type(dim_field: str) -> None:
    with pytest.raises(ValueError, match=dim_field):
        _make_frame_identity(**{dim_field: 0})
    with pytest.raises(ValueError, match=dim_field):
        _make_frame_identity(**{dim_field: -1})
    with pytest.raises(TypeError, match=dim_field):
        _make_frame_identity(**{dim_field: True})
    with pytest.raises(TypeError, match=dim_field):
        _make_frame_identity(**{dim_field: "24"})


def test_frame_identity_pts_ticks_strict_type() -> None:
    with pytest.raises(TypeError, match="pts_ticks"):
        _make_frame_identity(pts_ticks=False)
    with pytest.raises(TypeError, match="pts_ticks"):
        _make_frame_identity(pts_ticks="0")
    with pytest.raises(TypeError, match="pts_ticks"):
        _make_frame_identity(pts_ticks=0.0)


def test_frame_identity_physical_time_validation() -> None:
    # None physical time requires nonempty reason
    with pytest.raises(ValueError, match="physical_time_reason"):
        _make_frame_identity(physical_time_s=None, physical_time_reason="")
    with pytest.raises(ValueError, match="physical_time_reason"):
        _make_frame_identity(physical_time_s=None, physical_time_reason="   ")

    # non-finite floats rejected
    with pytest.raises(ValueError, match="physical_time_s"):
        _make_frame_identity(physical_time_s=float("nan"))
    with pytest.raises(ValueError, match="physical_time_s"):
        _make_frame_identity(physical_time_s=float("inf"))
    with pytest.raises(ValueError, match="physical_time_s"):
        _make_frame_identity(physical_time_s=float("-inf"))

    # bool and string rejected
    with pytest.raises(TypeError, match="physical_time_s"):
        _make_frame_identity(physical_time_s=True)
    with pytest.raises(TypeError, match="physical_time_s"):
        _make_frame_identity(physical_time_s="0.0")


@pytest.mark.parametrize(
    "id_field",
    ["asset_id", "shot_id", "swing_id", "camera_id", "frame_id"],
)
def test_frame_identity_id_fields_validation(id_field: str) -> None:
    with pytest.raises(ValueError, match=id_field):
        _make_frame_identity(**{id_field: ""})
    with pytest.raises(ValueError, match=id_field):
        _make_frame_identity(**{id_field: " "})
    with pytest.raises(ValueError, match=id_field):
        _make_frame_identity(**{id_field: " untrimmed"})
    with pytest.raises(TypeError, match=id_field):
        _make_frame_identity(**{id_field: 42})


def test_frame_identity_from_dict_rejects_unknown_fields() -> None:
    payload = _make_frame_identity().to_dict()
    payload["unexpected"] = 123
    with pytest.raises(ValueError, match="unexpected"):
        FrameIdentity.from_dict(payload)


def test_frame_identity_immutability() -> None:
    frame = _make_frame_identity()
    with pytest.raises((AttributeError, TypeError)):
        frame.frame_id = "f2"  # type: ignore[misc]


# --------------------------------------------------------------------------- #
#  validate_frame_sequence tests                                              #
# --------------------------------------------------------------------------- #


def test_validate_frame_sequence_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        validate_frame_sequence([])


def test_validate_frame_sequence_single_frame() -> None:
    frame = _make_frame_identity()
    validate_frame_sequence([frame])


def test_validate_frame_sequence_strictly_increasing_pts() -> None:
    f1 = _make_frame_identity(frame_id="f1", pts_ticks=0, physical_time_s=0.0)
    f2 = _make_frame_identity(frame_id="f2", pts_ticks=1, physical_time_s=1 / 24)
    f3 = _make_frame_identity(frame_id="f3", pts_ticks=2, physical_time_s=2 / 24)
    validate_frame_sequence([f1, f2, f3])


def test_validate_frame_sequence_allows_repeated_image_hash() -> None:
    f1 = _make_frame_identity(
        frame_id="f1", pts_ticks=0, physical_time_s=0.0, frame_sha256=VALID_HASH_A
    )
    f2 = _make_frame_identity(
        frame_id="f2", pts_ticks=1, physical_time_s=1 / 24, frame_sha256=VALID_HASH_A
    )
    validate_frame_sequence([f1, f2])


def test_validate_frame_sequence_rejects_duplicate_frame_id() -> None:
    f1 = _make_frame_identity(frame_id="f1", pts_ticks=0)
    f2 = _make_frame_identity(frame_id="f1", pts_ticks=1)
    with pytest.raises(ValueError, match="frame_id"):
        validate_frame_sequence([f1, f2])


def test_validate_frame_sequence_rejects_non_increasing_pts() -> None:
    f1 = _make_frame_identity(frame_id="f1", pts_ticks=1)
    f2 = _make_frame_identity(frame_id="f2", pts_ticks=0)
    with pytest.raises(ValueError, match="pts_ticks|presentation_time|order"):
        validate_frame_sequence([f1, f2])

    f3 = _make_frame_identity(frame_id="f3", pts_ticks=1)
    with pytest.raises(ValueError, match="pts_ticks|presentation_time|order"):
        validate_frame_sequence([f1, f3])


@pytest.mark.parametrize(
    ("field", "bad_val"),
    [
        ("asset_id", "asset-other"),
        ("shot_id", "shot-other"),
        ("swing_id", "swing-other"),
        ("camera_id", "cam-other"),
    ],
)
def test_validate_frame_sequence_rejects_mismatched_sequence_identifiers(
    field: str, bad_val: str
) -> None:
    f1 = _make_frame_identity(frame_id="f1", pts_ticks=0)
    f2 = _make_frame_identity(frame_id="f2", pts_ticks=1, **{field: bad_val})
    with pytest.raises(ValueError, match=field):
        validate_frame_sequence([f1, f2])


def test_validate_frame_sequence_physical_times_increase_across_known_subset() -> None:
    f1 = _make_frame_identity(
        frame_id="f1", pts_ticks=0, physical_time_s=0.0, physical_time_reason="clock"
    )
    f2 = _make_frame_identity(
        frame_id="f2",
        pts_ticks=1,
        physical_time_s=None,
        physical_time_reason="gap uncalibrated",
    )
    f3 = _make_frame_identity(
        frame_id="f3", pts_ticks=2, physical_time_s=0.1, physical_time_reason="clock"
    )
    # Known subset 0.0 -> 0.1 is strictly increasing, gap is legal
    validate_frame_sequence([f1, f2, f3])


def test_validate_frame_sequence_rejects_non_increasing_known_physical_time() -> None:
    f1 = _make_frame_identity(
        frame_id="f1", pts_ticks=0, physical_time_s=1.0, physical_time_reason="clock"
    )
    f2 = _make_frame_identity(
        frame_id="f2",
        pts_ticks=1,
        physical_time_s=None,
        physical_time_reason="gap uncalibrated",
    )
    f3 = _make_frame_identity(
        frame_id="f3", pts_ticks=2, physical_time_s=0.9, physical_time_reason="clock"
    )
    with pytest.raises(ValueError, match="physical_time_s"):
        validate_frame_sequence([f1, f2, f3])

    # Equal physical times also rejected (must strictly increase)
    f4 = _make_frame_identity(
        frame_id="f4", pts_ticks=2, physical_time_s=1.0, physical_time_reason="clock"
    )
    with pytest.raises(ValueError, match="physical_time_s"):
        validate_frame_sequence([f1, f4])


# --------------------------------------------------------------------------- #
#  Issue #10151 Malformed-Input and Type Boundary Tests                      #
# --------------------------------------------------------------------------- #


def test_source_asset_from_dict_wrong_container_type_raises_type_error() -> None:
    """SourceAsset.from_dict([]) must raise TypeError naming payload/container."""
    with pytest.raises(TypeError, match="dict|mapping|payload"):
        SourceAsset.from_dict([])  # type: ignore[arg-type]


def test_source_asset_from_dict_missing_required_keys_raises_value_error() -> None:
    """SourceAsset.from_dict({}) must raise ValueError naming missing fields."""
    with pytest.raises(ValueError, match="Missing required field|schema_version"):
        SourceAsset.from_dict({})


def test_frame_identity_from_dict_wrong_container_type_raises_type_error() -> None:
    """FrameIdentity.from_dict([]) must raise TypeError naming payload/container."""
    with pytest.raises(TypeError, match="dict|mapping|payload"):
        FrameIdentity.from_dict([])  # type: ignore[arg-type]


def test_frame_identity_from_dict_missing_required_keys_raises_value_error() -> None:
    """FrameIdentity.from_dict({}) must raise ValueError naming missing fields."""
    with pytest.raises(ValueError, match="Missing required field|schema_version"):
        FrameIdentity.from_dict({})


def test_validate_frame_sequence_wrong_element_type_raises_type_error() -> None:
    """validate_frame_sequence([None]) must raise TypeError naming FrameIdentity."""
    with pytest.raises(TypeError, match="FrameIdentity"):
        validate_frame_sequence([None])  # type: ignore[list-item]

    f1 = _make_frame_identity(frame_id="f1", pts_ticks=0)
    with pytest.raises(TypeError, match="FrameIdentity"):
        validate_frame_sequence([f1, "not-a-frame"])  # type: ignore[list-item]


def test_source_asset_schema_version_wrong_type_raises_type_error() -> None:
    """Wrong-type schema_version must raise TypeError, not ValueError."""
    with pytest.raises(TypeError, match="schema_version"):
        _make_source_asset(schema_version=123)  # type: ignore[arg-type]


def test_frame_identity_schema_version_wrong_type_raises_type_error() -> None:
    """Wrong-type schema_version must raise TypeError, not ValueError."""
    with pytest.raises(TypeError, match="schema_version"):
        _make_frame_identity(schema_version=123)  # type: ignore[arg-type]


def test_source_asset_rights_status_wrong_type_raises_type_error() -> None:
    """Integer or wrong-type rights_status must raise TypeError, not ValueError."""
    with pytest.raises(TypeError, match="rights_status"):
        _make_source_asset(rights_status=123)  # type: ignore[arg-type]


def test_frame_identity_rejects_integer_physical_time_s() -> None:
    """physical_time_s must strictly be float or None; int must raise TypeError."""
    with pytest.raises(TypeError, match="physical_time_s"):
        _make_frame_identity(physical_time_s=1)  # type: ignore[arg-type]
