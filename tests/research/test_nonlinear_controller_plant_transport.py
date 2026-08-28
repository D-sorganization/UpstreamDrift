"""Qualification gates for the controller-facing double-pendulum plant map."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.nonlinear_controller_plant_transport import (
    REPORT_PATH,
    RegisteredDoublePendulumPlant,
    build_transport_qualification,
    validate_transport_qualification,
)
from src.shared.python.simulation_backends.factory import make_backend
from src.shared.python.simulation_backends.model_params import GolfModelParams
from src.shared.python.simulation_backends.protocol import SimState

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.scientific


def _reference_step(
    state: np.ndarray, control: np.ndarray, step_s: float
) -> np.ndarray:
    backend = make_backend("ode", GolfModelParams.default())
    backend.reset(SimState(q=state[:2], v=state[2:], time=0.0))
    backend.set_control(control)
    backend.step(step_s)
    result = backend.get_state()
    return np.concatenate((result.q, result.v))


@pytest.mark.parametrize("step_s", [0.0005, 0.001, 0.002])
@pytest.mark.parametrize(
    ("state", "control"),
    [
        ([-2.20, -1.57, 0.00, 0.00], [0.00, 0.00]),
        ([-1.40, -1.10, 3.50, -2.00], [60.00, -15.00]),
        ([-0.35, -0.65, 12.00, 8.00], [60.00, 15.00]),
        ([0.10, 0.20, -4.00, 3.00], [-60.00, 15.00]),
    ],
)
def test_controller_plant_matches_canonical_ode_backend(
    step_s: float, state: list[float], control: list[float]
) -> None:
    state_array = np.asarray(state, dtype=float)
    control_array = np.asarray(control, dtype=float)
    plant = RegisteredDoublePendulumPlant(GolfModelParams.default(), step_s=step_s)
    observed = plant(state_array, control_array)
    np.testing.assert_allclose(
        observed,
        _reference_step(state_array, control_array, step_s),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_array_equal(plant(state_array, control_array), observed)


@pytest.mark.parametrize(
    ("state", "control"),
    [
        ([0.0, 0.0, 0.0], [0.0, 0.0]),
        ([0.0, 0.0, 0.0, 0.0], [0.0]),
        ([0.0, 0.0, np.nan, 0.0], [0.0, 0.0]),
        ([0.0, 0.0, 0.0, 0.0], [0.0, np.inf]),
    ],
)
def test_controller_plant_rejects_invalid_inputs(
    state: list[float], control: list[float]
) -> None:
    plant = RegisteredDoublePendulumPlant(GolfModelParams.default(), step_s=0.001)
    with pytest.raises(ValueError):
        plant(np.asarray(state, dtype=float), np.asarray(control, dtype=float))


def test_committed_transport_report_is_deterministic_and_scope_limited() -> None:
    expected = build_transport_qualification(ROOT)
    committed = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    assert committed == expected
    assert validate_transport_qualification(committed, ROOT) == {
        "parity_case_count": 12,
        "controller_evaluation_count": 0,
        "ranking_eligible_count": 0,
    }
    assert committed["step_sizes_s"] == [0.0005, 0.001, 0.002]
    assert committed["maximum_state_parity_error"] <= 1.0e-12
    assert committed["controller_evaluation_count"] == 0
    assert committed["eligible_for_ranking"] is False
