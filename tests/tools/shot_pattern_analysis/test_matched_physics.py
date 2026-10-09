"""Physical delivery, impact, and flight contracts for matched shots."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from src.shared.python.physics.rust_kernel import is_rust_available
from src.tools.shot_pattern_analysis.matched_config import matched_preset
from src.tools.shot_pattern_analysis.matched_physics import (
    MatchedShotPhysics,
    UnsupportedShot,
)

pytestmark = [pytest.mark.integration, pytest.mark.scientific]


@pytest.fixture(scope="module")
def engine() -> MatchedShotPhysics:
    if not is_rust_available():
        pytest.skip("upstream_physics Rust wheel is required for flight")
    return MatchedShotPhysics()


def test_central_strike_has_airborne_finite_flight(engine: MatchedShotPhysics) -> None:
    result = engine.simulate_delivery(
        0.0, 0.0, matched_preset("driver"), sample_trajectory=True
    )
    shot = result.outcome
    assert result.delivered_face_deg == pytest.approx(0.0)
    assert result.delivered_loft_deg == pytest.approx(10.0)
    assert shot.ball_speed_mps > 0
    assert shot.spin_rpm >= 0
    assert shot.carry_x_m > 0
    assert shot.trajectory_xy_m[-1] == pytest.approx((shot.carry_x_m, shot.carry_y_m))


def test_toe_strike_and_bulge_change_launch(engine: MatchedShotPhysics) -> None:
    cfg = matched_preset("driver")
    center = engine.simulate_delivery(0.0, 0.0, cfg)
    toe = engine.simulate_delivery(0.0, 0.0, cfg, contact_horizontal_m=0.012)
    flat = engine.simulate_delivery(
        0.0, 0.0, replace(cfg, bulge_radius_m=0.0), contact_horizontal_m=0.012
    )
    assert toe.outcome.launch_azimuth_deg != pytest.approx(
        center.outcome.launch_azimuth_deg
    )
    assert toe.outcome.spin_axis_tilt_deg != pytest.approx(
        center.outcome.spin_axis_tilt_deg
    )
    assert toe.outcome.launch_azimuth_deg != pytest.approx(
        flat.outcome.launch_azimuth_deg
    )
    assert toe.outcome.carry_y_m != pytest.approx(flat.outcome.carry_y_m)
    # Positive native spin tilt is draw/left: landing lies left of launch ray.
    launch_ray_y = flat.outcome.carry_x_m * math.tan(
        math.radians(flat.outcome.launch_azimuth_deg)
    )
    # Isolate gear effect on a flat face: bulge can overpower its net spin.
    assert flat.outcome.spin_axis_tilt_deg > 0
    assert flat.outcome.carry_y_m < launch_ray_y


def test_lean_and_high_loft_are_physical_inputs(engine: MatchedShotPhysics) -> None:
    driver = matched_preset("driver")
    baseline = engine.simulate_delivery(0.0, 0.0, driver)
    leaned = engine.simulate_delivery(0.0, 0.0, replace(driver, shaft_lean_deg=3.0))
    assert leaned.delivered_loft_deg == pytest.approx(7.0)
    assert leaned.outcome.launch_elevation_deg < baseline.outcome.launch_elevation_deg
    wedge = engine.simulate_delivery(0.0, 0.0, matched_preset("pitching_wedge"))
    assert wedge.delivered_loft_deg == pytest.approx(46.0)
    assert wedge.outcome.carry_x_m > 0


def test_out_of_domain_is_explicit(engine: MatchedShotPhysics) -> None:
    cfg = matched_preset("driver")
    with pytest.raises(UnsupportedShot, match="off_face"):
        engine.simulate_delivery(0.0, 0.0, cfg, contact_horizontal_m=0.051)
    with pytest.raises(UnsupportedShot, match="off_face"):
        engine.simulate_delivery(0.0, 0.0, cfg, contact_vertical_m=0.028)
    with pytest.raises(UnsupportedShot, match="delivery"):
        engine.simulate_delivery(35.0, 0.0, cfg)
    with pytest.raises(UnsupportedShot, match="nonairborne_launch"):
        engine.simulate_delivery(0.0, 0.0, replace(cfg, shaft_lean_deg=15.0))


def test_flight_horizon_failure_is_not_sample_rejection(
    engine: MatchedShotPhysics,
) -> None:
    cfg = replace(matched_preset("driver"), max_time_s=0.1)
    with pytest.raises(RuntimeError, match="max_time_s"):
        engine.simulate_delivery(0.0, 0.0, cfg)


def test_simulate_has_same_nominal_geometry(engine: MatchedShotPhysics) -> None:
    cfg = matched_preset("seven_iron")
    direct = engine.simulate_delivery(1.5, 3.0, cfg).outcome
    protocol = engine.simulate(face_deg=1.5, path_deg=3.0, config=cfg)
    assert protocol.carry_x_m == pytest.approx(direct.carry_x_m)
    assert protocol.carry_y_m == pytest.approx(direct.carry_y_m)


def test_curved_contact_normal_is_reported_separately(
    engine: MatchedShotPhysics,
) -> None:
    cfg = matched_preset("driver")
    high = engine.simulate_delivery(0, 0, cfg, contact_vertical_m=0.00635)
    toe = engine.simulate_delivery(0, 0, cfg, contact_horizontal_m=0.00635)
    assert high.delivered_loft_deg == pytest.approx(10)
    assert high.contact_loft_deg == pytest.approx(
        10 + math.degrees(math.atan(0.00635 / 0.3))
    )
    assert toe.delivered_face_deg == pytest.approx(0)
    assert toe.contact_face_deg > 0
