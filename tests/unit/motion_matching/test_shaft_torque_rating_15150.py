"""Torsion from a torque rating, not from the bending flex (#15150 C17).

The chain's torsion springs came from the isotropic ``GJ = EI / (1 + nu)``
of the bending calibration, so a shaft's twist was tied to its flex. Real
shafts are sold with an independent torque rating: degrees of twist under
1 ft lbf. ``torque_rating_deg`` sets the torsion springs from it.
"""

from __future__ import annotations

import argparse
import math

import pytest

from src.shared.python.motion_matching.club_models import DRIVER
from src.shared.python.motion_matching.club_shaft_chain import (
    TORQUE_RATING_TEST_N_M,
    add_shaft_arguments,
    build_shaft_chain,
    checked_shaft_options,
    rated_twist_springs,
    shaft_options,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def span_twist_deg(chain) -> float:
    return math.degrees(
        TORQUE_RATING_TEST_N_M * sum(1.0 / k for k in chain.twist_stiffness)
    )


def test_one_foot_pound_is_the_rating_torque() -> None:
    assert pytest.approx(0.3048 * 4.4482216152605) == TORQUE_RATING_TEST_N_M


@pytest.mark.parametrize("rating", [3.0, 4.5, 6.5])
@pytest.mark.parametrize("lumping", ["midpoint", "tip_slope_exact"])
def test_the_chain_twists_by_its_rating_under_one_foot_pound(
    rating: float, lumping: str
) -> None:
    chain = build_shaft_chain(DRIVER, 4, torque_rating_deg=rating, lumping=lumping)
    assert chain.torsion  # a rating adds the torsion springs
    assert span_twist_deg(chain) == pytest.approx(rating, rel=1e-12)
    assert chain.record["torque_rating"]["degrees"] == rating


def test_the_rating_keeps_the_gj_distribution_and_leaves_bending_alone() -> None:
    base = build_shaft_chain(DRIVER, 6, torsion=True)
    rated = build_shaft_chain(DRIVER, 6, torque_rating_deg=5.0)
    ratios = [
        r / b for r, b in zip(rated.twist_stiffness, base.twist_stiffness, strict=True)
    ]
    assert max(ratios) == pytest.approx(min(ratios), rel=1e-12)
    assert rated.bend_stiffness == base.bend_stiffness
    assert rated.damping_beta_s == base.damping_beta_s


def test_twist_is_independent_of_flex_and_stiffness_scale() -> None:
    soft = build_shaft_chain(DRIVER, 4, flex="Regular", torque_rating_deg=4.0)
    stiff = build_shaft_chain(
        DRIVER, 4, flex="X-Stiff", stiffness_scale=2.0, torque_rating_deg=4.0
    )
    assert span_twist_deg(soft) == pytest.approx(span_twist_deg(stiff), rel=1e-12)
    assert soft.bend_stiffness != stiff.bend_stiffness


def test_without_a_rating_the_chain_and_its_record_are_unchanged() -> None:
    assert "torque_rating" not in build_shaft_chain(DRIVER, 4, torsion=True).record
    assert build_shaft_chain(DRIVER).twist_stiffness == ()


@pytest.mark.parametrize("bad", [0.0, -3.0, 0.5, 20.0, float("nan")])
def test_a_rating_outside_the_window_is_refused(bad: float) -> None:
    with pytest.raises(ValueError, match="torque_rating_deg"):
        build_shaft_chain(DRIVER, 4, torque_rating_deg=bad)


def test_rated_springs_contract() -> None:
    springs = rated_twist_springs((100.0, 200.0), 4.0)
    assert springs[1] / springs[0] == pytest.approx(2.0)
    with pytest.raises(ValueError, match="positive"):
        rated_twist_springs((), 4.0)
    with pytest.raises(ValueError, match="positive"):
        rated_twist_springs((100.0,), 0.0)


def test_the_rating_is_a_shaft_option_on_the_command_line() -> None:
    parser = argparse.ArgumentParser()
    add_shaft_arguments(parser)
    options = shaft_options(parser.parse_args(["--shaft-torque-rating", "3.5"]))
    assert options == {"torque_rating_deg": 3.5}
    assert checked_shaft_options("chain", options) == options
    with pytest.raises(ValueError, match="chain"):
        checked_shaft_options("rigid", options)
    assert shaft_options(parser.parse_args([])) == {}
