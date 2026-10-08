"""Shared contact-wrench to ground-reaction adapter (GCV-2, #11708)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.biomechanics.ground_reaction import (
    COP_MIN_FZ_N,
    center_of_pressure,
)
from src.shared.python.biomechanics.ground_reaction_wrenches import (
    foot_contact_sets,
    ground_reaction_overlay,
)
from src.shared.python.force_overlay.contracts import OverlayWrench, WrenchKind

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]
G = 9.80665


def _contact(body, point, force, torque=None, idx=0):
    return OverlayWrench(
        WrenchKind.CONTACT,
        f"contact:{body}:{idx}",
        body,
        tuple(point),
        force_n=tuple(force),
        torque_nm=None if torque is None else tuple(torque),
        source="test",
    )


def _stance(weight=800.0):
    return [
        _contact("calcn_l", (0.0, 0.15, 0.0), (0, 0, weight / 2), idx=0),
        _contact("calcn_r", (0.0, -0.15, 0.0), (0, 0, weight / 2), idx=1),
        # a non-foot contact must be ignored
        _contact("golf_club", (0.5, 0.0, 0.0), (0, 0, 99.0), idx=2),
    ]


def test_center_of_pressure_matches_the_documented_formula() -> None:
    force = np.array([30.0, -20.0, 500.0])
    moment = np.array([40.0, -60.0, 5.0])
    cop = center_of_pressure(force, moment, ground_height_m=0.1)
    assert cop is not None
    np.testing.assert_allclose(
        cop, [(0.1 * 30.0 + 60.0) / 500.0, (40.0 + 0.1 * -20.0) / 500.0, 0.1]
    )


def test_center_of_pressure_below_threshold_is_none_not_zero() -> None:
    assert center_of_pressure([0, 0, COP_MIN_FZ_N - 0.1], [1, 1, 0]) is None
    assert center_of_pressure([0, 0, -50.0], [1, 1, 0]) is None
    assert center_of_pressure([0, 0, 5.0], [1, 1, 0], min_fz=1.0) is not None


def test_center_of_pressure_validates_inputs() -> None:
    with pytest.raises(ValueError):
        center_of_pressure([0, 0], [0, 0, 0])
    with pytest.raises(ValueError):
        center_of_pressure([0, 0, np.nan], [0, 0, 0])
    with pytest.raises(ValueError):
        center_of_pressure([0, 0, 100], [0, 0, 0], min_fz=-1.0)


def test_contact_sets_group_by_foot_and_drop_non_foot_bodies() -> None:
    sets = foot_contact_sets(_stance())
    assert set(sets) == {"left", "right"}
    np.testing.assert_allclose(sets["left"].forces_n.sum(axis=0), (0, 0, 400))
    assert sets["right"].points_m.shape == (1, 3)


def test_missing_foot_is_an_empty_set_not_absent() -> None:
    sets = foot_contact_sets(_stance()[:1])
    assert sets["right"].forces_n.shape == (0, 3)


def test_non_contact_kinds_are_ignored() -> None:
    w = OverlayWrench(
        WrenchKind.EXTERNAL, "external:calcn_l", "calcn_l", (0, 0, 0),
        force_n=(0, 0, 50.0), source="t",
    )
    assert foot_contact_sets([w])["left"].forces_n.shape == (0, 3)


def test_overlay_carries_per_foot_net_free_moment_and_com_moment() -> None:
    wrenches = ground_reaction_overlay(_stance(), (0.0, 0.0, 0.95), source="test")
    labels = {w.label for w in wrenches}
    assert {
        "contact:grf_left",
        "contact:grf_right",
        "contact:grf_net",
        "contact:free_moment_net",
        "contact:moment_com_net",
    } <= labels
    net = next(w for w in wrenches if w.label == "contact:grf_net")
    np.testing.assert_allclose(net.force_n, (0, 0, 800.0))
    np.testing.assert_allclose(net.point_m, (0, 0, 0), atol=1e-12)


def test_unloaded_stance_gives_no_overlay_not_zero_arrows() -> None:
    airborne = [_contact("calcn_l", (0, 0, 0), (0, 0, 0))]
    assert ground_reaction_overlay(airborne, (0, 0, 1.0), source="t") == ()


def test_below_threshold_foot_keeps_force_but_has_no_cop_or_free_moment() -> None:
    light = [_contact("calcn_l", (0.1, 0.0, 0.0), (1.0, 0.0, 4.0))]
    labels = {w.label for w in ground_reaction_overlay(light, (0, 0, 1.0), source="t")}
    assert "contact:grf_left" in labels
    assert "contact:free_moment_left" not in labels


def test_contact_torque_is_included_in_the_free_moment() -> None:
    one = [_contact("calcn_l", (0.0, 0.0, 0.0), (0, 0, 400.0), torque=(0, 0, 3.0))]
    wrenches = ground_reaction_overlay(one, (0, 0, 1.0), source="t")
    free = next(w for w in wrenches if w.label == "contact:free_moment_left")
    np.testing.assert_allclose(free.torque_nm, (0, 0, 3.0), atol=1e-12)


def test_preconditions() -> None:
    with pytest.raises(ValueError):
        ground_reaction_overlay(_stance(), (0, 0), source="t")
    with pytest.raises(TypeError):
        foot_contact_sets([object()])  # type: ignore[list-item]
