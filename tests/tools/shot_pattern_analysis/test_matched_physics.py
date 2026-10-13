"""Physical delivery, impact, and flight contracts for matched shots."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import numpy as np

from src.shared.python.physics.rust_kernel import is_rust_available
from src.tools.shot_pattern_analysis.matched_config import matched_preset
from src.tools.shot_pattern_analysis.matched_physics import (
    MatchedShotPhysics,
    UnsupportedShot,
)
from src.tools.shot_pattern_analysis.physics import outcome_from_launch

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


def test_contact_law_config_contract() -> None:
    cfg = matched_preset("driver")
    assert cfg.contact_law == "incremental_coulomb_energetic"
    assert cfg.steps_per_compression == 128
    assert cfg.contact_backend == "auto"
    with pytest.raises(ValueError, match="contact law"):
        replace(cfg, contact_law="unknown")
    with pytest.raises(ValueError, match="compression steps"):
        replace(cfg, steps_per_compression=0)
    with pytest.raises(ValueError, match="contact backend"):
        replace(cfg, contact_backend="unknown")


def test_primary_and_comparator_contact_laws_are_explicit(
    engine: MatchedShotPhysics, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.tools.shot_pattern_analysis import matched_physics

    cfg = matched_preset("driver")
    called: list[tuple[str, float | int | str]] = []
    original_incremental = matched_physics.solve_incremental_contact
    original_comparator = matched_physics.solve_rigid_contact

    def incremental(state, **kwargs):
        called.append(("incremental", kwargs["steps_per_compression"]))
        assert kwargs["energetic_restitution"] == cfg.cor
        assert kwargs["backend"] == "python_reference"
        return original_incremental(state, **kwargs)

    def comparator(state):
        called.append(("comparator", state.restitution))
        return original_comparator(state)

    monkeypatch.setattr(matched_physics, "solve_incremental_contact", incremental)
    monkeypatch.setattr(matched_physics, "solve_rigid_contact", comparator)
    primary = engine.simulate_delivery(
        0, 0, replace(cfg, contact_backend="python_reference", steps_per_compression=32)
    )
    assert primary.outcome.carry_x_m > 0
    assert called == [("incremental", 32)]
    called.clear()
    comparison = engine.simulate_delivery(
        0, 0, replace(cfg, contact_law="maximum_dissipation_newton")
    )
    assert comparison.outcome.carry_x_m > 0
    assert called == [("comparator", cfg.cor)]


def test_zero_spin_airborne_shot_is_supported(engine: MatchedShotPhysics) -> None:
    from src.shared.python.physics.ball_simulator import BallFlightSimulator

    direct = outcome_from_launch(
        np.array([60.0, 0.0, 10.0]),
        np.zeros(3),
        flight=BallFlightSimulator(),
        max_time_s=12.0,
        dt_s=0.02,
    )
    assert direct.carry_x_m > 0
    assert direct.spin_rpm == 0
    assert direct.spin_axis_tilt_deg == 0
    assert not direct.spin_axis_defined
    zero_friction = engine.simulate_delivery(
        0, 0, replace(matched_preset("driver"), friction_coefficient=0)
    ).outcome
    assert zero_friction.carry_x_m > 0
    assert zero_friction.spin_rpm == 0
    assert not zero_friction.spin_axis_defined


def test_numerical_zero_spin_threshold_has_defined_axis_boundary() -> None:
    from src.shared.python.physics.ball_simulator import BallFlightSimulator

    flight = BallFlightSimulator()

    def shot(spin_rad_s: float):
        return outcome_from_launch(
            np.array([60.0, 0.0, 10.0]),
            np.array([0.0, -spin_rad_s, 0.0]),
            flight=flight,
            max_time_s=12.0,
            dt_s=0.02,
        )

    below = shot(0.5e-10)
    above = shot(2e-10)
    assert below.spin_rpm == 0
    assert not below.spin_axis_defined
    assert above.spin_rpm > 0
    assert above.spin_axis_defined
    assert above.spin_axis_tilt_deg == pytest.approx(0)
