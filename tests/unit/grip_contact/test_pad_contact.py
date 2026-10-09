"""Pad layout and the shared sphere-on-cylinder contact law (#11739 phase 3)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.grip_contact import GripInterface
from src.shared.python.grip_contact.bushing_law import BushingState
from src.shared.python.grip_contact.pad_contact import (
    build_pad_model,
    grip_cylinder,
    pad_wrench,
)
from src.shared.python.grip_contact.pad_layout import (
    PadLayout,
    matched_pad_parameters,
    required_squeeze_n,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
SPEC = ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
SQUEEZE = 1000.0


@pytest.fixture(scope="module")
def interface() -> GripInterface:
    return GripInterface.from_spec(json.loads(SPEC.read_text()))


def _rest(offset=(0.0, 0.0, 0.0), velocity=(0.0, 0.0, 0.0)) -> tuple:
    hand = BushingState(np.eye(3), np.zeros(3), np.zeros(3), np.zeros(3))
    club = BushingState(
        np.eye(3), np.asarray(offset, float), np.asarray(velocity, float), np.zeros(3)
    )
    return hand, club


def test_ring_is_isotropic_and_symmetric() -> None:
    layout = PadLayout()
    pads = layout.positions_grip_frame("R") - layout.axis_offset_grip_frame("R")
    radial = np.linalg.norm(pads[:, 1:], axis=1)
    np.testing.assert_allclose(radial, layout.centre_radius_m)
    assert pads.shape == (12, 3)
    assert np.sum(pads[:, 1] ** 2) == pytest.approx(np.sum(pads[:, 2] ** 2), rel=1e-12)
    assert abs(np.sum(pads[:, 1] * pads[:, 2])) < 1e-12


@pytest.mark.parametrize(
    "bad", [{"pads_per_ring": 2}, {"rows": 3}, {"pad_radius_m": -1}]
)
def test_layout_rejects_invalid_geometry(bad) -> None:
    with pytest.raises(ValueError):
        PadLayout(**bad)


def test_matched_parameters_reproduce_the_bushing_stiffness(interface) -> None:
    layout = PadLayout()
    m = matched_pad_parameters(interface.bushing, layout, SQUEEZE)
    k_t = interface.bushing.translational_stiffness_n_m[1]
    assert layout.pad_count * m.stiffness_n_m / 2.0 == pytest.approx(k_t, rel=0.03)
    assert (
        layout.pad_count * m.stiffness_n_m * m.preload_penetration_m
        == pytest.approx(SQUEEZE)
    )
    assert m.axial_offset_m == pytest.approx(layout.axial_offset_m)
    with pytest.raises(ValueError):
        matched_pad_parameters(interface.bushing, layout, 0.0)


def test_required_squeeze_takes_the_governing_limit() -> None:
    assert required_squeeze_n(450.0, 0.0, 0.0, 0.9, 0.0127) == pytest.approx(500.0)
    assert required_squeeze_n(0.0, 10.0, 0.0, 0.9, 0.0127) == pytest.approx(
        10.0 / (0.9 * 0.0127)
    )
    assert required_squeeze_n(0.0, 0.0, 500.0, 0.9, 0.0127) == pytest.approx(1000.0)
    with pytest.raises(ValueError):
        required_squeeze_n(-1.0, 0.0, 0.0, 0.9, 0.0127)


def test_cylinder_rejects_grip_points_not_on_a_common_shaft(interface) -> None:
    with pytest.raises(ValueError, match="apart across the grip"):
        grip_cylinder(interface, PadLayout(grip_radius_m=0.02))


def test_closed_grip_carries_no_net_force_and_the_squeeze(interface) -> None:
    model = build_pad_model(interface, SQUEEZE)
    cyl_axis = model.layout.axis_offset_grip_frame("R")
    hand, club = _rest()
    w = pad_wrench(model, "R", hand, club)
    np.testing.assert_allclose(w.force_n, 0.0, atol=1e-6)
    assert w.normal_force_n.sum() == pytest.approx(SQUEEZE, rel=1e-9)
    assert cyl_axis[2] > 0.0


def test_small_offset_gives_the_bushing_stiffness(interface) -> None:
    """A 0.01 mm offset across the grip: force = -K_t delta within 1 %."""
    model = build_pad_model(interface, SQUEEZE)
    k_t = interface.bushing.translational_stiffness_n_m[1]
    for axis in (1, 2):
        delta = np.zeros(3)
        delta[axis] = 1e-5
        hand, club = _rest(offset=delta)
        w = pad_wrench(model, "R", hand, club)
        assert w.force_n[axis] == pytest.approx(-k_t * delta[axis], rel=1e-2)


def test_no_stiffness_along_the_axis_only_friction(interface) -> None:
    model = build_pad_model(interface, SQUEEZE)
    hand, club = _rest(offset=(1e-5, 0.0, 0.0))
    still = pad_wrench(model, "R", hand, club)
    assert abs(still.force_n[0]) < 1e-6  # no relative velocity: no friction force
    hand, club = _rest(velocity=(0.5, 0.0, 0.0))  # club slides along the axis
    slide = pad_wrench(model, "R", hand, club)
    n_total = slide.normal_force_n.sum()
    # the club moves +x relative to the hand; friction on the club opposes it
    assert slide.force_n[0] == pytest.approx(
        -model.law.dynamic_friction * n_total, rel=1e-3
    )


def test_friction_cannot_exceed_the_cone(interface) -> None:
    model = build_pad_model(interface, SQUEEZE)
    hand, club = _rest(velocity=(0.0003, 0.0, 0.0))
    w = pad_wrench(model, "R", hand, club)
    assert np.all(
        w.tangential_force_n <= model.law.static_friction * w.normal_force_n + 1e-9
    )


def test_pads_outside_the_cylinder_carry_nothing(interface) -> None:
    model = build_pad_model(interface, SQUEEZE)
    hand, club = _rest(offset=(0.0, 0.0, 0.0))
    far = BushingState(np.eye(3), np.array([5.0, 0.0, 0.0]), np.zeros(3), np.zeros(3))
    w = pad_wrench(model, "R", far, club)
    assert not w.normal_force_n.any() and not w.force_n.any()


def test_moment_is_taken_about_the_club_grip_origin(interface) -> None:
    """The ring acts at the shaft axis, ``r_g`` from the grip origin: M = r x F."""
    model = build_pad_model(interface, SQUEEZE)
    hand, club = _rest(offset=(0.0, 1e-5, 0.0))
    w = pad_wrench(model, "R", hand, club)
    axis_point = model.layout.axis_offset_grip_frame("R")
    expected = np.cross(axis_point, w.force_n)
    np.testing.assert_allclose(w.moment_nm, expected, rtol=0.05, atol=1e-3)
