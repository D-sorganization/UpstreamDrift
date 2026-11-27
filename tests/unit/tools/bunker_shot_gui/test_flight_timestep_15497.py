"""Invalid integration steps must never reach the bunker flight simulator."""

from unittest.mock import patch

import pytest

from bunkershot3d.ball import compute_bunker_launch
from tests.bunkershot3d.ball.test_pipeline_handoff import nominal_state
from src.tools.bunker_shot_gui.flight import bunker_flight

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("dt", [0.0, -0.01, float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("contract_level", ["off", "strict"])
def test_invalid_timestep_is_rejected_before_simulation(
    dt: float, contract_level: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DBC_LEVEL", contract_level)
    launch = compute_bunker_launch(nominal_state())
    with patch("src.tools.bunker_shot_gui.flight.BallFlightSimulator") as simulator:
        with pytest.raises(ValueError, match="dt must be finite and strictly positive"):
            bunker_flight(launch, refused=False, dt=dt)
        simulator.assert_not_called()


@pytest.mark.parametrize("dt", [0.001, 0.01, 0.02])
def test_positive_timestep_is_forwarded_unchanged(dt: float) -> None:
    launch = compute_bunker_launch(nominal_state())
    with patch("src.tools.bunker_shot_gui.flight.BallFlightSimulator") as simulator:
        # An incomplete trajectory is unavailable; no green solver is needed.
        simulator.return_value.simulate_trajectory.return_value = []
        assert bunker_flight(launch, refused=False, dt=dt) is None
        assert simulator.return_value.simulate_trajectory.call_args.kwargs["dt"] == dt


def test_refused_shot_does_not_validate_or_simulate_timestep() -> None:
    launch = compute_bunker_launch(nominal_state())
    with patch("src.tools.bunker_shot_gui.flight.BallFlightSimulator") as simulator:
        assert bunker_flight(launch, refused=True, dt=0.0) is None
        simulator.assert_not_called()
