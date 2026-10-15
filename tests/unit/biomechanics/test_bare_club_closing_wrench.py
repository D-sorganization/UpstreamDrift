"""Per-hand free-body split on the bare club (NSX-01 review, #12550).

The closure multiplier loads the COMPOSITE body (club plus both welded
hands).  The closing hand's own inertial load must be removed before its
wrench is a hand-on-bare-club quantity.  Numbers are hand calculations on a
static club under gravity only.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.biomechanics.club_inertia_partition import (
    club_solids_partition,
)
from src.shared.python.biomechanics.grip_extraction import (
    closure_and_club_analysis,
)

pytestmark = pytest.mark.unit

G = (0.0, 0.0, -9.81)


def _kw(mass, com, inertia_scale=1e-3):
    return {
        "mass_kg": mass,
        "gravity_m_s2": G,
        "com_m": com,
        "com_acceleration_m_s2": (0.0, 0.0, 0.0),
        "inertia_world_kg_m2": inertia_scale * np.eye(3),
        "angular_velocity_rad_s": (0.0, 0.0, 0.0),
        "angular_acceleration_rad_s2": (0.0, 0.0, 0.0),
    }


def _analysis(**extra):
    return closure_and_club_analysis(
        closing_side="R",
        closing_point_m=(0.0, 0.0, 0.0),
        # weld load on the composite
        closing_wrench=((0.0, 0.0, 12.0), (0.0, 0.0, 0.0)),
        holding_point_m=(1.0, 0.0, 0.0),
        club=_kw(1.0, (0.5, 0.0, 0.0)),
        split_method="constraint_multiplier",
        **extra,
    )


def test_static_club_per_hand_split_matches_hand_calculation() -> None:
    a = _analysis(closing_hand_club=_kw(0.5, (0.1, 0.0, 0.0)))
    # NE(R solids) = 0.5 (0 - g) = (0,0,4.905), torque about the closing point
    # (0.1,0,0) x (0,0,4.905) = (0,-0.4905,0).  H_close = W_weld - NE(R).
    np.testing.assert_allclose(a.right.force_on_club_n, (0, 0, 7.095), atol=1e-12)
    np.testing.assert_allclose(a.right.torque_on_club_nm, (0, 0.4905, 0), atol=1e-12)
    # H_hold = NE(bare) - H_close; NE(bare) = (0,0,9.81) with no torque.
    np.testing.assert_allclose(a.left.force_on_club_n, (0, 0, 2.715), atol=1e-12)
    np.testing.assert_allclose(a.left.torque_on_club_nm, (0, -2.6805, 0), atol=1e-12)
    np.testing.assert_allclose(a.net_force_n, (0, 0, 9.81), atol=1e-12)
    assert a.metadata["closing_hand"] == "R"


def test_bare_club_moment_balance_about_its_com() -> None:
    a = _analysis(closing_hand_club=_kw(0.5, (0.1, 0.0, 0.0)))
    com = np.array([0.5, 0.0, 0.0])
    total = np.zeros(3)
    for hand in (a.left, a.right):
        total += np.array(hand.torque_on_club_nm)
        total += np.cross(np.array(hand.point_m) - com, hand.force_on_club_n)
    np.testing.assert_allclose(total, 0.0, atol=1e-12)


def test_without_closing_hand_inertia_the_legacy_split_is_unchanged() -> None:
    a = _analysis()
    np.testing.assert_allclose(a.right.force_on_club_n, (0, 0, 12.0), atol=1e-12)
    np.testing.assert_allclose(a.left.force_on_club_n, (0, 0, -2.19), atol=1e-12)


def test_closing_hand_keywords_are_validated() -> None:
    bad = _kw(0.5, (0.1, 0.0, 0.0))
    bad.pop("mass_kg")
    with pytest.raises(ValueError, match="closing_hand_club"):
        _analysis(closing_hand_club=bad)


def test_partition_groups_welded_hands_by_side() -> None:
    from tests.unit.biomechanics.test_club_solids_partition import _spec

    sides = club_solids_partition(_spec()).hands_by_side()
    assert set(sides) == {"L", "R"}
    assert sides["R"].mass_kg == pytest.approx(0.6)
    np.testing.assert_allclose(sides["R"].com_m, (0.0, 1.5, 0.0), atol=1e-12)
    # LHand 0.5 kg at y=2, LHandStandoff 0.01 kg at (y=2, z=0.1)
    assert sides["L"].mass_kg == pytest.approx(0.51)
    np.testing.assert_allclose(
        sides["L"].com_m, (0.0, 2.0, 0.01 * 0.1 / 0.51), atol=1e-12
    )
