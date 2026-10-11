"""Foot membership of contact bodies (GCV-2, #11708)."""

from __future__ import annotations

import pytest

from src.shared.python.biomechanics.foot_membership import (
    foot_bodies_from_spec,
    foot_of_body,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


@pytest.mark.parametrize(
    ("name", "side"),
    [
        ("calcn_l", "left"),
        ("calcn_r", "right"),
        ("toes_r", "right"),
        ("left_foot", "left"),
        ("right_toes", "right"),
        ("LeftFoot", "left"),
        ("RightFoot_f1", "right"),
        ("ud_contact_heel_r", "right"),
        ("ud_contact_forefoot_l", "left"),
        ("contact:foot_L", "left"),
    ],
)
def test_foot_bodies_resolve_to_a_side(name: str, side: str) -> None:
    assert foot_of_body(name) == side


@pytest.mark.parametrize(
    "name",
    ["golf_club", "club_head", "pelvis", "tibia_l", "left_hand", "Hip", "world", ""],
)
def test_non_foot_bodies_are_none(name: str) -> None:
    assert foot_of_body(name) is None


def test_ambiguous_or_sideless_foot_names_are_none() -> None:
    assert foot_of_body("foot") is None
    assert foot_of_body("left_right_foot") is None


def test_non_string_is_a_type_error() -> None:
    with pytest.raises(TypeError):
        foot_of_body(3)  # type: ignore[arg-type]


def test_spec_contact_spheres_define_the_foot_bodies() -> None:
    spec = {
        "contact": {
            "spheres": [
                {"name": "heel_r", "body": "calcn_r"},
                {"name": "forefoot_r", "body": "calcn_r"},
                {"name": "heel_l", "body": "calcn_l"},
                {"name": "ball", "body": "golf_ball"},
            ]
        }
    }
    assert foot_bodies_from_spec(spec) == {"calcn_r": "right", "calcn_l": "left"}


def test_spec_without_contact_groups_is_empty() -> None:
    assert foot_bodies_from_spec({}) == {}
    with pytest.raises(TypeError):
        foot_bodies_from_spec([])  # type: ignore[arg-type]
