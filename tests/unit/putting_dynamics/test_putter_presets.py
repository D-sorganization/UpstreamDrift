"""Named putter balance presets over PutterState (#14461)."""

from __future__ import annotations

import pytest

from src.shared.python.putting_dynamics import (
    BALANCE_PRESETS,
    balance_preset,
    strike,
)

pytestmark = pytest.mark.unit


def test_the_four_balance_types_are_offered() -> None:
    assert set(BALANCE_PRESETS) == {
        "toe_hang",
        "face_balanced",
        "zero_torque",
        "counterbalanced",
    }


@pytest.mark.parametrize("key", sorted(BALANCE_PRESETS))
def test_every_preset_builds_a_valid_putter(key: str) -> None:
    preset = balance_preset(key)
    putter = preset.putter(1.8)
    assert putter.speed_mps == 1.8
    assert putter.head_mass_kg == preset.head_mass_kg
    assert putter.hosel_toe_m == preset.hosel_toe_m
    assert strike(putter).ball_speed_mps > 0.0


def test_a_centred_strike_twists_only_a_head_hung_off_the_shaft_line() -> None:
    # The shaft of a face-balanced or zero-torque head lies over the CG's
    # heel-toe position, so a centred strike has no lever about it; a
    # toe-hang blade's shaft sits heel-side, so the same strike opens the face.
    twist = {
        k: strike(balance_preset(k).putter(1.8)).face_twist_rad_s
        for k in BALANCE_PRESETS
    }
    assert twist["face_balanced"] == 0.0
    assert twist["zero_torque"] == 0.0
    assert twist["toe_hang"] > 0.0


def test_a_mallet_keeps_more_speed_on_a_toe_strike_than_a_blade() -> None:
    def kept(key: str) -> float:
        putter = balance_preset(key).putter(1.8)
        return strike(putter, 0.012).ball_speed_mps / strike(putter).ball_speed_mps

    assert kept("face_balanced") > kept("toe_hang")


def test_unknown_presets_are_refused() -> None:
    with pytest.raises(ValueError, match="toe_hang"):
        balance_preset("anser")
