"""Tests for the thoracolumbar spinal level enumeration (SPN-1a)."""

from __future__ import annotations

import pytest

from src.shared.python.spine import (
    SpinalLevel,
    SpinalRegion,
    levels_between,
    parse_level,
)

pytestmark = pytest.mark.unit

EXPECTED_ORDER = (
    "L5_S1",
    "L4_L5",
    "L3_L4",
    "L2_L3",
    "L1_L2",
    "T12_L1",
    "T11_T12",
    "T10_T11",
    "T9_T10",
    "T8_T9",
    "T7_T8",
    "T6_T7",
    "T5_T6",
    "T4_T5",
    "T3_T4",
    "T2_T3",
    "T1_T2",
)


def test_enum_has_seventeen_thoracolumbar_levels_in_caudal_to_cranial_order():
    assert tuple(member.name for member in SpinalLevel) == EXPECTED_ORDER
    assert len(SpinalLevel) == 17


def test_levels_are_contiguous_superior_matches_next_inferior():
    members = list(SpinalLevel)
    for caudal, cranial in zip(members, members[1:], strict=False):
        assert caudal.superior == cranial.inferior, (caudal, cranial)


def test_inferior_and_superior_vertebrae_for_known_levels():
    assert SpinalLevel.L4_L5.inferior == "L5"
    assert SpinalLevel.L4_L5.superior == "L4"
    assert SpinalLevel.L5_S1.inferior == "S1"
    assert SpinalLevel.L5_S1.superior == "L5"
    assert SpinalLevel.T1_T2.inferior == "T2"
    assert SpinalLevel.T1_T2.superior == "T1"


@pytest.mark.parametrize(
    ("level", "region"),
    [
        (SpinalLevel.L5_S1, SpinalRegion.LUMBAR),
        (SpinalLevel.L1_L2, SpinalRegion.LUMBAR),
        (SpinalLevel.T12_L1, SpinalRegion.THORACIC),
        (SpinalLevel.T1_T2, SpinalRegion.THORACIC),
    ],
)
def test_region_is_region_of_superior_vertebra(level, region):
    assert level.region is region


def test_spinal_region_members():
    assert {member.name for member in SpinalRegion} == {
        "CERVICAL",
        "THORACIC",
        "LUMBAR",
        "SACRAL",
    }


def test_levels_between_l5_s1_and_l1_l2_has_five_members_caudal_to_cranial():
    result = levels_between(SpinalLevel.L5_S1, SpinalLevel.L1_L2)
    assert result == (
        SpinalLevel.L5_S1,
        SpinalLevel.L4_L5,
        SpinalLevel.L3_L4,
        SpinalLevel.L2_L3,
        SpinalLevel.L1_L2,
    )
    assert isinstance(result, tuple)


def test_levels_between_full_range_yields_seventeen_levels():
    result = levels_between(SpinalLevel.L5_S1, SpinalLevel.T1_T2)
    assert tuple(level.name for level in result) == EXPECTED_ORDER


def test_levels_between_same_level_is_singleton():
    assert levels_between(SpinalLevel.T8_T9, SpinalLevel.T8_T9) == (SpinalLevel.T8_T9,)


def test_levels_between_reversed_arguments_raise_value_error():
    with pytest.raises(ValueError, match="caudal"):
        levels_between(SpinalLevel.L1_L2, SpinalLevel.L5_S1)


def test_levels_between_rejects_non_spinal_level():
    with pytest.raises(TypeError):
        levels_between("L5_S1", SpinalLevel.L1_L2)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        levels_between(SpinalLevel.L5_S1, 3)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "spelling", ["L4_L5", "L4-L5", "L4/L5", "l4_l5", "l4-l5", "l4/l5"]
)
def test_parse_level_accepts_three_spellings_case_insensitively(spelling):
    assert parse_level(spelling) is SpinalLevel.L4_L5


def test_parse_level_thoracic_spelling():
    assert parse_level("t12-l1") is SpinalLevel.T12_L1


@pytest.mark.parametrize("bad", ["L6_L5", "", "L1_L3", "garbage", "T1_T2_extra"])
def test_parse_level_rejects_unknown_or_empty_with_value_error(bad):
    with pytest.raises(ValueError):
        parse_level(bad)


@pytest.mark.parametrize("bad", [None, 4, b"L4_L5", ("L4", "L5")])
def test_parse_level_rejects_non_str_with_type_error(bad):
    with pytest.raises(TypeError):
        parse_level(bad)  # type: ignore[arg-type]


def test_parse_level_round_trips_every_member():
    for member in SpinalLevel:
        assert parse_level(member.name) is member
