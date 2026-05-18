"""Unit tests for :class:`CanonicalPose`."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.shared.python.motion_matching.diagnostics.reference_pose import (
    REFERENCE_GOLFER_FIELDS,
    reference_golfer_setup,
)
from src.shared.python.pose_interchange.canonical import (
    CONVENTION_TAG,
    CanonicalPose,
    canonical_from_reference_setup,
    canonical_zero_pose,
)

pytestmark = pytest.mark.unit


# ---- Construction & DbC --------------------------------------------------------


def test_zero_pose_basic() -> None:
    pose = canonical_zero_pose()
    np.testing.assert_array_equal(pose.pelvis_translation_m, np.zeros(3))
    np.testing.assert_array_equal(pose.pelvis_rotation_xyz_deg, np.zeros(3))
    assert pose.joint_angles_deg == {}
    assert pose.convention_tag == CONVENTION_TAG


def test_reference_setup_pose_carries_full_address() -> None:
    pose = canonical_from_reference_setup()
    expected = reference_golfer_setup()
    full = pose.angles_full_dict_deg()
    for key, value in expected.items():
        assert full[key] == pytest.approx(value)


def test_translation_must_be_length_3() -> None:
    with pytest.raises(ValueError, match=r"shape \(3,\)"):
        CanonicalPose(
            pelvis_translation_m=np.array([1.0, 2.0]),
            pelvis_rotation_xyz_deg=np.zeros(3),
        )


def test_rotation_must_be_length_3() -> None:
    with pytest.raises(ValueError, match=r"shape \(3,\)"):
        CanonicalPose(
            pelvis_translation_m=np.zeros(3),
            pelvis_rotation_xyz_deg=np.array([0.0, 0.0]),
        )


def test_translation_must_be_finite() -> None:
    with pytest.raises(ValueError, match="finite"):
        CanonicalPose(
            pelvis_translation_m=np.array([np.nan, 0.0, 0.0]),
            pelvis_rotation_xyz_deg=np.zeros(3),
        )


def test_unknown_joint_field_rejected() -> None:
    with pytest.raises(ValueError, match="unknown field names"):
        CanonicalPose(
            pelvis_translation_m=np.zeros(3),
            pelvis_rotation_xyz_deg=np.zeros(3),
            joint_angles_deg={"NotARealField": 12.0},
        )


def test_non_finite_joint_value_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        CanonicalPose(
            pelvis_translation_m=np.zeros(3),
            pelvis_rotation_xyz_deg=np.zeros(3),
            joint_angles_deg={"HipStartPositionX": float("inf")},
        )


def test_non_numeric_joint_value_rejected() -> None:
    with pytest.raises(TypeError, match="finite float"):
        CanonicalPose(
            pelvis_translation_m=np.zeros(3),
            pelvis_rotation_xyz_deg=np.zeros(3),
            joint_angles_deg={"HipStartPositionX": "twelve"},  # type: ignore[dict-item]
        )


def test_wrong_convention_tag_rejected() -> None:
    with pytest.raises(ValueError, match="convention_tag"):
        CanonicalPose(
            pelvis_translation_m=np.zeros(3),
            pelvis_rotation_xyz_deg=np.zeros(3),
            convention_tag="some-other-tag",
        )


# ---- Immutability --------------------------------------------------------------


def test_pelvis_arrays_are_readonly() -> None:
    pose = canonical_zero_pose()
    with pytest.raises(ValueError, match="read-only"):
        pose.pelvis_translation_m[0] = 99.0
    with pytest.raises(ValueError, match="read-only"):
        pose.pelvis_rotation_xyz_deg[0] = 99.0


def test_joint_dict_snapshot_decouples_from_caller() -> None:
    angles = {"HipStartPositionX": 5.0}
    pose = CanonicalPose(
        pelvis_translation_m=np.zeros(3),
        pelvis_rotation_xyz_deg=np.zeros(3),
        joint_angles_deg=angles,
    )
    angles["HipStartPositionX"] = 999.0
    assert pose.joint_angles_deg["HipStartPositionX"] == pytest.approx(5.0)


def test_frozen_attribute_assignment_blocked() -> None:
    pose = canonical_zero_pose()
    with pytest.raises(
        Exception
    ):  # noqa: B017 — dataclasses raises FrozenInstanceError
        pose.convention_tag = "x"  # type: ignore[misc]


# ---- Accessors -----------------------------------------------------------------


def test_angle_deg_returns_zero_for_absent_field() -> None:
    pose = canonical_zero_pose()
    for field in REFERENCE_GOLFER_FIELDS:
        assert pose.angle_deg(field) == pytest.approx(0.0)


def test_angle_deg_rejects_non_canonical_name() -> None:
    pose = canonical_zero_pose()
    with pytest.raises(KeyError, match="canonical joint-angle field"):
        pose.angle_deg("NopeNotAField")


def test_angles_full_dict_covers_all_fields() -> None:
    pose = canonical_from_reference_setup()
    full = pose.angles_full_dict_deg()
    assert set(full.keys()) == set(REFERENCE_GOLFER_FIELDS)


def test_angles_full_dict_rad_matches_deg_in_radians() -> None:
    pose = canonical_from_reference_setup()
    deg = pose.angles_full_dict_deg()
    rad = pose.angles_full_dict_rad()
    for key in REFERENCE_GOLFER_FIELDS:
        assert rad[key] == pytest.approx(np.radians(deg[key]))


# ---- (de)serialisation ---------------------------------------------------------


def test_json_round_trip_zero_pose() -> None:
    pose = canonical_zero_pose()
    payload = pose.to_json()
    parsed = json.loads(payload)
    assert parsed["convention_tag"] == CONVENTION_TAG
    restored = CanonicalPose.from_json(payload)
    np.testing.assert_array_equal(
        restored.pelvis_translation_m, pose.pelvis_translation_m
    )


def test_json_round_trip_reference_pose() -> None:
    pose = canonical_from_reference_setup()
    restored = CanonicalPose.from_json(pose.to_json())
    np.testing.assert_array_equal(
        restored.pelvis_translation_m, pose.pelvis_translation_m
    )
    np.testing.assert_array_equal(
        restored.pelvis_rotation_xyz_deg, pose.pelvis_rotation_xyz_deg
    )
    assert restored.angles_full_dict_deg() == pose.angles_full_dict_deg()


def test_path_round_trip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    pose = canonical_from_reference_setup()
    target = tmp_path / "pose.json"
    pose.to_path(target)
    assert target.is_file()
    restored = CanonicalPose.from_path(target)
    assert restored.angles_full_dict_deg() == pose.angles_full_dict_deg()


def test_from_json_rejects_non_dict_payload() -> None:
    with pytest.raises(ValueError, match="decode to a dict"):
        CanonicalPose.from_json("[1, 2, 3]")


def test_from_json_rejects_missing_keys() -> None:
    with pytest.raises(ValueError, match="missing required key"):
        CanonicalPose.from_json('{"convention_tag": "canonical-v1"}')
