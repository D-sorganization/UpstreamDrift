"""C14 (#15150): tip-slope-exact lumping of the shaft chain.

For a cantilever with a unit tip load the continuum tip slope is
``int_0^L (L - s) / EI(s) ds`` and the continuum tip twist under a unit tip
torque is ``int_0^L ds / GJ(s)``. The chain reproduces them when joint ``i``
has ``1/k_i = int_seg (L - s)/EI ds / (L - s_i)`` and ``int_seg ds/GJ``.
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import pytest

from src.shared.python.motion_matching.club_models import DRIVER
from src.shared.python.motion_matching.club_shaft_chain import (
    DEFAULT_LUMPING,
    LUMPING_RULES,
    add_shaft_arguments,
    build_shaft_chain,
    checked_shaft_options,
    shaft_options,
    span_properties,
    station_profile,
)

pytestmark = pytest.mark.unit
LINK_COUNTS = (2, 4, 8)
REFERENCE_POINTS = 400_001


def _reference(kind: str) -> float:
    """Independent dense trapezoid integral of the continuum compliance."""
    profile = station_profile(span_properties(DRIVER), DRIVER)
    pos = np.array([st.position_m for st in profile.stations])
    ei = np.array([st.ei_about_x_n_m2 for st in profile.stations])
    gj = np.array([st.gj_n_m2 for st in profile.stations])
    span = pos[-1]
    s = np.linspace(0.0, span, REFERENCE_POINTS)
    if kind == "bend":
        return float(np.trapezoid((span - s) / np.interp(s, pos, ei), s))
    return float(np.trapezoid(1.0 / np.interp(s, pos, gj), s))


def _slope(chain) -> float:
    return sum(
        (chain.span_m - s) / k
        for s, k in zip(chain.joint_s_m, chain.bend_stiffness, strict=True)
    )


def _relative_error(value: float, reference: float) -> float:
    return abs(value - reference) / reference


@pytest.mark.parametrize("n_links", LINK_COUNTS)
def test_tip_slope_exact(n_links: int) -> None:
    chain = build_shaft_chain(DRIVER, n_links, lumping="tip_slope_exact")
    assert _relative_error(_slope(chain), _reference("bend")) < 1e-9


def test_midpoint_is_not_exact_at_two_links() -> None:
    chain = build_shaft_chain(DRIVER, 2, lumping="midpoint")
    error = _relative_error(_slope(chain), _reference("bend"))
    assert error > 1e-4
    print(f"midpoint tip-slope relative error at N=2: {error:.3e}")


@pytest.mark.parametrize("n_links", LINK_COUNTS)
def test_tip_twist_exact(n_links: int) -> None:
    chain = build_shaft_chain(DRIVER, n_links, torsion=True, lumping="tip_slope_exact")
    twist = sum(1.0 / k for k in chain.twist_stiffness)
    assert _relative_error(twist, _reference("twist")) < 1e-9


def test_default_is_midpoint_and_springs_unchanged() -> None:
    assert DEFAULT_LUMPING == "midpoint"
    default = build_shaft_chain(DRIVER, 4, torsion=True)
    explicit = build_shaft_chain(DRIVER, 4, torsion=True, lumping="midpoint")
    assert default.bend_stiffness == explicit.bend_stiffness
    assert default.twist_stiffness == explicit.twist_stiffness
    profile = station_profile(span_properties(DRIVER), DRIVER)
    ds = default.span_m / 4
    expected = tuple(
        profile.station_at((i + 0.5) * ds).ei_about_x_n_m2 / ds for i in range(4)
    )
    assert default.bend_stiffness == expected
    assert default.joint_s_m == tuple((i + 0.5) * ds for i in range(4))


def test_unknown_lumping_raises() -> None:
    with pytest.raises(ValueError, match="lumping"):
        build_shaft_chain(DRIVER, 4, lumping="trapezoid")
    assert set(LUMPING_RULES) == {"midpoint", "tip_slope_exact"}


def test_record_names_the_rule() -> None:
    mid = build_shaft_chain(DRIVER, 4).record
    exact = build_shaft_chain(DRIVER, 4, lumping="tip_slope_exact").record
    assert "lumping_rule" not in mid  # default record unchanged
    assert exact["lumping_rule"] == "tip_slope_exact"
    assert "midpoint" in mid["lumping"]
    assert "exact tip slope" in exact["lumping"]
    assert exact["lumping"] != mid["lumping"]


def test_joints_stay_at_segment_midpoints() -> None:
    chain = build_shaft_chain(DRIVER, 4, lumping="tip_slope_exact")
    ds = chain.span_m / 4
    assert chain.joint_s_m == tuple((i + 0.5) * ds for i in range(4))
    assert all(math.isfinite(k) and k > 0 for k in chain.bend_stiffness)


def test_cli_flag_round_trips() -> None:
    parser = argparse.ArgumentParser()
    add_shaft_arguments(parser)
    assert shaft_options(parser.parse_args([])) == {}
    args = parser.parse_args(["--shaft-lumping", "tip_slope_exact"])
    options = shaft_options(args)
    assert options == {"lumping": "tip_slope_exact"}
    assert checked_shaft_options("chain", options) == options
    with pytest.raises(SystemExit):
        parser.parse_args(["--shaft-lumping", "bogus"])
    with pytest.raises(ValueError):
        checked_shaft_options("rigid", options)
