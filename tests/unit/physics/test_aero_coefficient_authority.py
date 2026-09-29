"""Golden regression gate for aero-coefficient authority (issue #8978).

Two ball-flight model families are LIVE **by design**:

* **Core simulator** (``ball_properties.py`` + ``ball_simulator.py`` /
  ``ball_flight_physics.py``): the canonical, doc-gated authority
  (cd1 = 0.25, lift cap ``MAX_LIFT_COEFFICIENT`` = 0.26; pinned against the
  calc sheet by ``tests/docs/test_ball_flight_calc_sheet_parity.py``).
* **Multi-model comparison framework** (``flight_models.py``): a distinct,
  constant-spin (no spin decay) family used by the REST route
  ``/tools/ball-flight/simulate`` and the Shot Tracer launcher. Its
  ``WaterlooPennerModel`` shares the Penner lift *shape* with the core set
  but carries a deliberately different calibration
  (``WATERLOO_PENNER_COEFFICIENTS``: cd1 = 0.05, cl_max = 0.155).

This module is the gate that prevents silent re-divergence:

1. each public entry path (core simulator, REST route, shot tracer/registry)
   uses its **declared** coefficient set;
2. paths that claim the same model produce identical trajectories;
3. paths that claim different models are explicitly different and every
   result is attributable via ``FlightResult.coefficients`` / the REST
   ``coefficients`` field.
"""

from __future__ import annotations

import math

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes.ball_flight import router as ball_flight_router
from src.shared.python.physics import ball_properties as bp
from src.shared.python.physics import flight_models as fm

pytestmark = [pytest.mark.unit, pytest.mark.scientific]

# Reference driver launch (metric, shared verbatim by every entry path).
BALL_SPEED_MPS = 74.0
LAUNCH_ANGLE_DEG = 10.9
SPIN_RATE_RPM = 2686.0
MAX_TIME_S = 12.0
TIME_STEP_S = 0.01


def _registry_launch() -> fm.UnifiedLaunchConditions:
    return fm.UnifiedLaunchConditions(
        ball_speed=BALL_SPEED_MPS,
        launch_angle=math.radians(LAUNCH_ANGLE_DEG),
        spin_rate=SPIN_RATE_RPM,
    )


def _waterloo_penner_result() -> fm.FlightResult:
    model = fm.FlightModelRegistry.get_model(fm.FlightModelType.WATERLOO_PENNER)
    return model.simulate(_registry_launch(), max_time=MAX_TIME_S, dt=TIME_STEP_S)


class TestDeclaredCoefficientSets:
    """Each live coefficient set matches its declared authority."""

    def test_core_authority_matches_doc_gated_values(self) -> None:
        ball = bp.BallProperties()
        assert ball.cd1 == pytest.approx(0.25)
        assert pytest.approx(0.26) == bp.MAX_LIFT_COEFFICIENT
        s = 0.10
        expected_cl = min(
            bp.MAX_LIFT_COEFFICIENT,
            bp.PENNER_LIFT_SCALE * s**bp.PENNER_LIFT_EXPONENT,
        )
        assert ball.calculate_cl(s) == pytest.approx(expected_cl)

    def test_waterloo_penner_declares_its_own_named_set(self) -> None:
        coeffs = fm.WATERLOO_PENNER_COEFFICIENTS
        assert fm.WaterlooPennerModel().coefficients == coeffs.as_dict()
        # The Penner lift shape has exactly ONE source: ball_properties.
        assert coeffs.lift_scale == bp.PENNER_LIFT_SCALE
        assert coeffs.lift_exponent == bp.PENNER_LIFT_EXPONENT
        # The two live sets are deliberately different and must stay declared.
        assert coeffs.cd1 != bp.BallProperties().cd1
        assert coeffs.cl_max != bp.MAX_LIFT_COEFFICIENT
        assert coeffs.provenance, "coefficient set must document provenance"

    def test_lift_cap_constant_is_the_named_set_cap(self) -> None:
        assert (
            pytest.approx(fm.WATERLOO_PENNER_COEFFICIENTS.cl_max)
            == fm.MAX_GOLF_BALL_LIFT_COEFFICIENT
        )

    def test_registry_default_is_the_declared_waterloo_penner(self) -> None:
        fm.FlightModelRegistry.reset()
        model = fm.FlightModelRegistry.get_model(fm.FlightModelType.WATERLOO_PENNER)
        assert model.coefficients == fm.WATERLOO_PENNER_COEFFICIENTS.as_dict()


class TestEntryPathAttribution:
    """Golden reference trajectory through every public entry path."""

    def test_shot_tracer_path_is_the_registry_path(self) -> None:
        pytest.importorskip("PyQt6")
        from src.launchers import _shot_tracer_gui

        assert _shot_tracer_gui.FlightModelRegistry is fm.FlightModelRegistry

    def test_flight_result_carries_model_identity(self) -> None:
        result = _waterloo_penner_result()
        assert result.model_name == "Waterloo/Penner"
        assert result.coefficients == fm.WATERLOO_PENNER_COEFFICIENTS.as_dict()

    def test_rest_route_matches_direct_model_and_declares_set(self) -> None:
        direct = _waterloo_penner_result()

        app = FastAPI()
        app.include_router(ball_flight_router)
        client = TestClient(app)
        response = client.post(
            "/tools/ball-flight/simulate",
            json={
                "ball_speed_mps": BALL_SPEED_MPS,
                "launch_angle_deg": LAUNCH_ANGLE_DEG,
                "spin_rate_rpm": SPIN_RATE_RPM,
                "model_name": fm.FlightModelType.WATERLOO_PENNER.value,
                "max_time_s": MAX_TIME_S,
                "time_step_s": TIME_STEP_S,
            },
        )
        assert response.status_code == 200
        data = response.json()

        # Same claimed model => identical trajectory metrics.
        assert data["model_name"] == direct.model_name
        assert data["summary"]["carry_m"] == pytest.approx(
            direct.carry_distance, rel=1e-9
        )
        assert data["summary"]["apex_m"] == pytest.approx(direct.max_height, rel=1e-9)
        # Attribution: the response declares the coefficient set it used.
        assert data["coefficients"] == fm.WATERLOO_PENNER_COEFFICIENTS.as_dict()
        assert data["results"][0]["coefficients"] == data["coefficients"]

    def test_core_simulator_is_explicitly_a_different_model(self) -> None:
        from src.shared.python.physics.ball_flight_physics import (
            EnhancedBallFlightSimulator,
            LaunchConditions,
        )

        simulator = EnhancedBallFlightSimulator()
        trajectory = simulator.simulate_trajectory(
            LaunchConditions(
                velocity=BALL_SPEED_MPS,
                launch_angle=math.radians(LAUNCH_ANGLE_DEG),
                spin_rate=SPIN_RATE_RPM,
            ),
            max_time=MAX_TIME_S,
            dt=TIME_STEP_S,
        )
        analysis = simulator.analyze_trajectory(trajectory)
        framework = _waterloo_penner_result()

        # Both models land in a plausible driver window ...
        assert 180.0 < analysis["carry_distance"] < 320.0
        assert 180.0 < framework.carry_distance < 320.0
        # ... but they are DIFFERENT models and must not silently converge
        # into looking like copies: their coefficient claims differ, and the
        # trajectories differ measurably for the same launch.
        wp = fm.WATERLOO_PENNER_COEFFICIENTS
        core = bp.BallProperties()
        s = 0.10
        assert core.calculate_cd(s) != pytest.approx(
            wp.cd0 + wp.cd1 * s + wp.cd2 * s**2
        )
        assert analysis["carry_distance"] != pytest.approx(
            framework.carry_distance, rel=1e-3
        )
