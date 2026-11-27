"""Flight omission must have a reason at the headless model boundary."""

from collections.abc import Callable
from dataclasses import replace
from unittest.mock import Mock

import pytest

from bunkershot3d.ball import compute_bunker_launch
from src.tools.bunker_shot_gui import model as model_module
from src.tools.bunker_shot_gui.design import (
    SandCondition,
    SolverSetup,
    SwingSetup,
    WedgeDesign,
)
from src.tools.bunker_shot_gui.flight import BunkerFlight
from src.tools.bunker_shot_gui.model import ShotOutcome, WorkbenchModel
from src.tools.bunker_shot_gui.uncertainty import CarryEstimate
from tests.bunkershot3d.ball.test_pipeline_handoff import nominal_state

pytestmark = pytest.mark.unit
FlightCase = tuple[Callable[[], ShotOutcome], Mock, Mock, CarryEstimate, BunkerFlight]


@pytest.fixture
def flight_case(
    monkeypatch: pytest.MonkeyPatch,
    nominal_design: WedgeDesign,
    firm_sand: SandCondition,
) -> FlightCase:
    """Run real sand reduction with a controlled launch/flight service boundary.

    These tests qualify orchestration and diagnostics, not native flight physics.
    The measured-delivery boundary is injected to avoid unrelated divot coverage.
    """
    state = nominal_state()
    launch = compute_bunker_launch(state)
    estimate = CarryEstimate(5.0, launch.verdict, launch=launch)
    flight = BunkerFlight(((0.0, 0.0, 0.0), (3.0, 4.0, 0.0)), ((3.0, 4.0, 0.0),), 5.0)
    model = WorkbenchModel(
        SolverSetup(n_profile_points=12, n_stations=5, flight_time_step_s=0.007)
    )
    carry_service = Mock(return_value=estimate)
    flight_service = Mock(return_value=flight)
    monkeypatch.setattr(model, "carry_estimate", carry_service)
    monkeypatch.setattr(model_module, "bunker_flight", flight_service)
    monkeypatch.setattr(model_module, "_measured_divot", lambda _: None)
    monkeypatch.setattr(model_module, "_sand_delivery", lambda *_: state.delivery)

    def run() -> ShotOutcome:
        return model.run_shot(
            nominal_design.geometry(), firm_sand.sand_state(), SwingSetup()
        )

    return run, carry_service, flight_service, estimate, flight


@pytest.mark.parametrize("without_launch", [False, True])
def test_silent_trace_omission_gets_an_explicit_reason(
    flight_case: FlightCase, without_launch: bool
) -> None:
    run, carry_service, flight_service, estimate, _ = flight_case
    flight_service.return_value = None
    if without_launch:
        carry_service.return_value = replace(estimate, launch=None)
    shot = run()
    assert shot.flight is None
    assert shot.carry_m == 5.0
    reasons = [
        reason for reason in shot.unavailable if reason.startswith("ball flight:")
    ]
    assert len(reasons) == 1
    assert "usable flight and settled green rollout" in reasons[0]
    assert flight_service.call_count == (0 if without_launch else 1)


def test_success_forwards_trace_and_configured_time_step(
    flight_case: FlightCase,
) -> None:
    run, _, flight_service, estimate, flight = flight_case
    shot = run()
    assert shot.flight is flight
    flight_service.assert_called_once_with(estimate.launch, refused=False, dt=0.007)
    assert not any(reason.startswith("ball flight:") for reason in shot.unavailable)


@pytest.mark.parametrize("error", [ValueError, RuntimeError, ImportError])
def test_adapter_error_reason_is_preserved_without_duplicate(
    flight_case: FlightCase, error: type[Exception]
) -> None:
    run, _, flight_service, _, _ = flight_case
    flight_service.side_effect = error("specific adapter failure")
    shot = run()
    assert shot.flight is None
    reasons = [
        reason for reason in shot.unavailable if reason.startswith("ball flight:")
    ]
    assert reasons == ["ball flight: specific adapter failure"]


def test_missing_carry_does_not_attempt_flight_or_add_a_second_failure(
    flight_case: FlightCase,
) -> None:
    run, carry_service, flight_service, _, _ = flight_case
    carry_service.side_effect = ValueError("launch unavailable")
    shot = run()
    assert shot.flight is None
    assert shot.carry_m is None
    flight_service.assert_not_called()
    assert "carry: launch unavailable" in shot.unavailable
    assert not any(reason.startswith("ball flight:") for reason in shot.unavailable)
