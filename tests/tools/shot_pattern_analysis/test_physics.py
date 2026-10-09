"""Native impact-to-flight checks for face/path sign and numerical stability."""

from __future__ import annotations

import math

import pytest

from src.shared.python.physics.rust_kernel import is_rust_available
from src.tools.shot_pattern_analysis.core import AnalysisConfig
from src.tools.shot_pattern_analysis.physics import ShotPhysics

pytestmark = [pytest.mark.integration, pytest.mark.scientific]


@pytest.fixture(scope="module")
def physics() -> ShotPhysics:
    if not is_rust_available():
        pytest.skip("upstream_physics Rust wheel is required for full trajectories")
    return ShotPhysics()


def test_straight_draw_fade_launch_and_bend(physics: ShotPhysics) -> None:
    cfg = AnalysisConfig(n_shots=2)
    straight = physics.simulate(face_deg=0, path_deg=0, config=cfg)
    draw = physics.simulate(face_deg=1.5, path_deg=3, config=cfg)
    fade = physics.simulate(face_deg=-1.5, path_deg=-3, config=cfg)
    assert straight.launch_azimuth_deg == pytest.approx(0, abs=1e-9)
    assert abs(straight.carry_y_m) < 1e-7
    assert straight.spin_rpm > 0
    assert straight.carry_x_m > 100
    assert draw.launch_azimuth_deg > 0
    assert fade.launch_azimuth_deg < 0
    assert draw.spin_axis_tilt_deg > 0
    assert fade.spin_axis_tilt_deg < 0
    draw_launch_ray_y = draw.carry_x_m * math.tan(math.radians(draw.launch_azimuth_deg))
    fade_launch_ray_y = fade.carry_x_m * math.tan(math.radians(fade.launch_azimuth_deg))
    assert draw.carry_y_m < draw_launch_ray_y
    assert fade.carry_y_m > fade_launch_ray_y
    assert draw.carry_y_m == pytest.approx(-fade.carry_y_m, abs=0.05)


def test_dt_refinement_and_landing_horizon(physics: ShotPhysics) -> None:
    coarse = AnalysisConfig(n_shots=2, dt_s=0.02)
    fine = AnalysisConfig(n_shots=2, dt_s=0.01)
    for center in (0.0, 1.5, -1.5):
        path = center * 2
        for deviation in (-3.0, -1.0, 0.0, 1.0, 3.0):
            a = physics.simulate(
                face_deg=center + deviation, path_deg=path, config=coarse
            )
            b = physics.simulate(
                face_deg=center + deviation, path_deg=path, config=fine
            )
            assert abs(a.carry_x_m - b.carry_x_m) < 0.05
            assert abs(a.carry_y_m - b.carry_y_m) < 0.05
    too_short = AnalysisConfig(n_shots=2, max_time_s=0.1)
    with pytest.raises(RuntimeError, match="max_time_s"):
        physics.simulate(face_deg=0, path_deg=0, config=too_short)


def test_shaft_rotation_matches_nominal_loft_then_couples_face_error(
    physics: ShotPhysics,
) -> None:
    fixed = AnalysisConfig(n_shots=2, delivery_mode="fixed_loft")
    coupled = AnalysisConfig(n_shots=2, delivery_mode="shaft_rotation", lie_deg=58.5)
    for nominal, path in ((0.0, 0.0), (1.5, 3.0), (-1.5, -3.0)):
        baseline = physics.simulate(face_deg=nominal, path_deg=path, config=fixed)
        match = physics.simulate(
            face_deg=nominal,
            path_deg=path,
            nominal_face_deg=nominal,
            config=coupled,
        )
        assert match.carry_x_m == pytest.approx(baseline.carry_x_m, abs=1e-8)
        assert match.carry_y_m == pytest.approx(baseline.carry_y_m, abs=1e-8)
        off_fixed = physics.simulate(face_deg=nominal - 1, path_deg=path, config=fixed)
        off_coupled = physics.simulate(
            face_deg=nominal - 1,
            path_deg=path,
            nominal_face_deg=nominal,
            config=coupled,
        )
        assert abs(off_fixed.carry_x_m - off_coupled.carry_x_m) > 0.1


def test_nonfinite_angles_fail_before_solver(physics: ShotPhysics) -> None:
    with pytest.raises(ValueError, match="finite"):
        physics.simulate(
            face_deg=float("nan"), path_deg=0, config=AnalysisConfig(n_shots=2)
        )
