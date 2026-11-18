"""The golf grip the hands hold: a taper and a butt cap (epic #15150, #15185)."""

from __future__ import annotations

import json

import math

import numpy as np
import pytest

from src.shared.python.model_appearance.hand_backends.golf_grip import (
    GRIP_PATH,
    GolfGrip,
    load_golf_grip,
)
from src.shared.python.model_appearance.hand_backends.posing import cylinder_distance
from src.shared.python.model_appearance.hand_backends.qa import (
    load_hand_visual_quality,
    max_grip_cylinder_penetration_m,
)
from src.shared.python.model_appearance.hand_backends.skinned_settings import (
    load_settings_mapping,
    settings_from_mapping,
)

pytestmark = pytest.mark.unit

_INCH = 0.0254


def test_the_standard_grip_has_the_published_diameters() -> None:
    grip = load_golf_grip()
    assert 2.0 * grip.radius_at_m(2.0 * _INCH) == pytest.approx(0.900 * _INCH)
    assert 2.0 * grip.radius_at_m(6.0 * _INCH) == pytest.approx(0.780 * _INCH)


def test_the_grip_tapers_evenly_from_cap_to_mouth() -> None:
    grip = load_golf_grip()
    assert grip.taper == pytest.approx(-0.015)
    stations = np.linspace(0.0, grip.length_m, 9)
    radii = [grip.radius_at_m(float(s)) for s in stations]
    assert np.all(np.diff(radii) < 0.0)
    assert grip.radius_at_m(0.0) < grip.cap_radius_m


def test_every_unverified_dimension_is_declared() -> None:
    raw = json.loads(GRIP_PATH.read_text(encoding="utf-8"))
    assert set(raw["unverified"]) == {
        "length_m",
        "cap_diameter_m",
        "cap_height_m",
        "cap_round_m",
        "vent_diameter_m",
    }
    assert all(source.startswith("https://") for source in raw["_sources"])


def test_the_drawing_is_in_the_frame_of_a_hand_below_the_cap() -> None:
    grip = load_golf_grip()
    knots = grip.drawing_knots(0.06)
    assert knots[0] == pytest.approx((-0.06, grip.radius_at_m(0.0)))
    assert knots[-1] == pytest.approx(
        (grip.length_m - 0.06, grip.radius_at_m(grip.length_m))
    )


@pytest.mark.parametrize(
    "stations",
    [
        [[0.05, 0.02]],
        [[0.05, 0.02], [0.05, 0.019]],
        [[0.05, 0.02], [0.15, -0.019]],
        [[0.05, 0.02], [0.15, 0.021]],
    ],
)
def test_a_grip_needs_two_stations_that_narrow_toward_the_head(stations) -> None:
    raw = json.loads(GRIP_PATH.read_text(encoding="utf-8"))
    raw["sizes"][raw["size"]]["diameter_stations_m"] = stations
    with pytest.raises(ValueError):
        GolfGrip.from_mapping(raw)


def test_a_station_off_the_grip_is_refused() -> None:
    grip = load_golf_grip()
    for station in (-0.001, grip.length_m + 0.001, float("nan")):
        with pytest.raises(ValueError):
            grip.radius_at_m(station)


def test_a_tapered_obstacle_narrows_toward_the_head() -> None:
    cone = cylinder_distance(0.011, -0.015)
    points = np.array([[0.02, 0.0, 0.0], [0.02, 0.0, 0.1], [0.0, 0.02, -0.1]])
    slant = math.hypot(1.0, 0.015)  # measured square to the cone's surface
    assert cone(points) * slant == pytest.approx([0.009, 0.0105, 0.0075])
    # A step of that length along the gradient lands on the surface.
    out = np.array([0.02, 0.0, 0.1])
    normal = np.array([1.0, 0.0, 0.015]) / slant
    landed = out - cone(out[None])[0] * normal
    assert cone(landed[None])[0] == pytest.approx(0.0, abs=1e-12)
    assert cylinder_distance(0.011)(points) == pytest.approx([0.009] * 3)
    with pytest.raises(ValueError):
        cylinder_distance(0.011, float("inf"))


def test_penetration_is_measured_against_the_taper() -> None:
    on_the_cone = np.array([[0.011 - 0.015 * 0.1, 0.0, 0.1]])
    assert max_grip_cylinder_penetration_m(on_the_cone, 0.011, -0.015) == 0.0
    assert max_grip_cylinder_penetration_m(on_the_cone, 0.011) == pytest.approx(0.0015)


def test_the_hands_hold_the_tapered_grip_by_default_and_it_can_be_switched_off() -> (
    None
):
    raw = dict(load_settings_mapping())
    assert settings_from_mapping(raw).grip_taper == load_golf_grip().taper
    raw["grip"] = {**raw["grip"], "shape": "cylinder"}
    assert settings_from_mapping(raw).grip_taper == 0.0
    raw["grip"] = {**raw["grip"], "shape": "cone"}
    with pytest.raises(ValueError):
        settings_from_mapping(raw)


def test_the_evidence_radius_is_the_grip_at_the_lead_hand() -> None:
    quality = load_hand_visual_quality()
    grip = load_golf_grip()
    station = float(quality["owner_lead_below_cap_edge_m"])
    assert float(quality["grip_radius_m"]) == pytest.approx(
        grip.radius_at_m(station), abs=1e-6
    )
