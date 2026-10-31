"""Per-hand grip radius in the contact topology (HRG-GD4 PR 1, #13892).

Each pad is placed on ``GripGeometry.radius_at(s)`` at its own axial station
instead of one constant radius, so the lead hand (upper station) and the trail
hand (lower station) see different radii on a tapered grip.
"""

from __future__ import annotations

import math

import pytest

from src.shared.python.grip_contact.contact_topology import (
    DEFAULT_TOPOLOGY_PRELOAD_M,
    ContactSite,
    GripStyle,
    contact_topology,
    pair_surface_gap_m,
    per_hand_required_squeeze_n,
    site_grip_surface_gap_m,
)
from src.shared.python.grip_contact.grip_geometry import (
    GRIP_GEOMETRY_PRESETS,
    STATION_LOWER_M,
    STATION_UPPER_M,
    get_grip_preset,
)
from src.shared.python.motion_matching.grip_contract import GripSpec

pytestmark = pytest.mark.unit

SPECS = tuple(
    GripSpec(style=style, trigger_finger=trigger)
    for style in GripStyle
    for trigger in (False, True)
)
MU = 0.8


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: f"{s.style.value}-{s.trigger_finger}")
def test_legacy_spec_standoff_reproduces_constant_radius_bit_for_bit(
    spec: GripSpec,
) -> None:
    constant = contact_topology(spec)
    legacy = contact_topology(spec, grip_geometry=get_grip_preset())
    assert legacy.sites == constant.sites
    assert legacy.pairs == constant.pairs
    assert legacy.grip_radius_m == constant.grip_radius_m
    for hand in ("lead", "trail"):
        assert legacy.hand_grip_radius_m(hand) == constant.hand_grip_radius_m(hand)


@pytest.mark.parametrize("s_m", [STATION_UPPER_M, STATION_LOWER_M])
@pytest.mark.parametrize("preset", ["mens_standard", "midsize", "ladies"])
def test_surface_gap_is_zero_for_a_point_on_the_surface_at_each_station(
    preset: str, s_m: float
) -> None:
    geom = get_grip_preset(preset)
    radius = 0.008
    site = ContactSite(
        name="probe",
        hand="lead",
        segment="probe",
        primitive="sphere",
        centre_m=(s_m, 0.0, geom.radius_at(s_m) + radius),
        radius_m=radius,
    )
    assert site_grip_surface_gap_m(site, grip_geometry=geom) == pytest.approx(
        0.0, abs=1e-15
    )


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: f"{s.style.value}-{s.trigger_finger}")
@pytest.mark.parametrize("preset", ["mens_standard", "midsize", "undersize", "ladies"])
def test_tapered_topology_keeps_every_pair_at_the_preload(
    spec: GripSpec, preset: str
) -> None:
    topo = contact_topology(spec, grip_geometry=get_grip_preset(preset))
    for pair in topo.pairs:
        assert pair_surface_gap_m(topo, pair) == pytest.approx(
            -DEFAULT_TOPOLOGY_PRELOAD_M, abs=1e-9
        ), pair.name


def test_each_hand_sees_the_radius_at_its_own_station() -> None:
    geom = get_grip_preset("mens_standard")
    spec = GripSpec()
    topo = contact_topology(spec, grip_geometry=geom)
    x_trail = spec.lead_below_butt_m + spec.hand_spacing_m
    assert topo.hand_grip_radius_m("lead") == geom.radius_at(spec.lead_below_butt_m)
    assert topo.hand_grip_radius_m("trail") == geom.radius_at(x_trail)
    assert topo.hand_grip_radius_m("lead") > topo.hand_grip_radius_m("trail")
    pad = topo.site("trail_palm_proximal")
    assert math.hypot(pad.centre_m[1], pad.centre_m[2]) == pytest.approx(
        geom.radius_at(pad.centre_m[0]) + pad.radius_m - DEFAULT_TOPOLOGY_PRELOAD_M,
        abs=1e-15,
    )


@pytest.mark.parametrize("preset", sorted(GRIP_GEOMETRY_PRESETS))
def test_torque_branch_squeeze_ratio_between_hands_is_r_lead_over_r_trail(
    preset: str,
) -> None:
    topo = contact_topology(grip_geometry=get_grip_preset(preset))
    squeeze = per_hand_required_squeeze_n(0.0, 1.5, 0.0, MU, topo)
    expected = topo.hand_grip_radius_m("lead") / topo.hand_grip_radius_m("trail")
    assert squeeze["trail"] / squeeze["lead"] == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("preset", sorted(GRIP_GEOMETRY_PRESETS))
def test_axial_branch_squeeze_is_independent_of_the_preset(preset: str) -> None:
    topo = contact_topology(grip_geometry=get_grip_preset(preset))
    squeeze = per_hand_required_squeeze_n(40.0, 0.0, 0.0, MU, topo)
    assert squeeze == {"lead": 40.0 / MU, "trail": 40.0 / MU}


def test_frustum_normal_tilt_changes_squeeze_by_under_one_percent() -> None:
    """Test-only check: pad normals stay radial in production (YAGNI).

    A frustum normal tilted by ``phi = atan(dr/ds)`` carries ``N cos(phi)``
    radially, so ignoring the tilt changes the squeeze by ``1/cos(phi) - 1``.
    """
    geom = get_grip_preset("mens_standard")
    slope = (geom.radius_at(STATION_LOWER_M) - geom.radius_at(STATION_UPPER_M)) / (
        STATION_LOWER_M - STATION_UPPER_M
    )
    phi = math.atan(abs(slope))
    assert math.degrees(phi) == pytest.approx(0.86, abs=0.01)
    assert 1.0 / math.cos(phi) - 1.0 < 0.01


def test_pad_outside_the_grip_length_is_rejected() -> None:
    short = get_grip_preset("mens_standard")
    spec = GripSpec(lead_below_butt_m=0.20)
    with pytest.raises(ValueError, match="outside the grip"):
        contact_topology(spec, grip_geometry=short)


def test_radius_and_geometry_together_are_rejected() -> None:
    with pytest.raises(ValueError, match="not both"):
        contact_topology(grip_radius_m=0.012, grip_geometry=get_grip_preset())


@pytest.mark.parametrize("hand_length_m", [0.10, 0.30])
def test_hand_length_outside_bounds_is_rejected(hand_length_m: float) -> None:
    with pytest.raises(ValueError):
        contact_topology(hand_length_m, grip_geometry=get_grip_preset("midsize"))
