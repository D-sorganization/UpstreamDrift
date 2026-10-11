"""Unit tests for the per-frame static optimisation (issue #11617, phase 2)."""

from __future__ import annotations

import numpy as np
import pytest
from src.engines.physics_engines.opensim.python import musculoskeletal_static_opt as so

pytestmark = pytest.mark.unit


def _problem() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    active = np.array([500.0, 800.0, 300.0])
    passive = np.array([5.0, 0.0, 2.0])
    moment = np.array([[0.04, 0.0], [0.05, -0.03], [0.0, 0.06]])
    return active, passive, moment


def test_solve_frame_meets_feasible_demand_without_reserve() -> None:
    active, passive, moment = _problem()
    truth = np.array([0.2, 0.3, 0.1])
    tau = moment.T @ (passive + truth * active)
    sol = so.solve_frame(active, passive, moment, tau)
    assert sol.success
    assert np.all(sol.activation >= 0) and np.all(sol.activation <= 1)
    np.testing.assert_allclose(
        moment.T @ (passive + sol.activation * active) + sol.reserve, tau
    )
    assert np.abs(sol.reserve).max() < 1.0  # penalised reserves stay near zero
    assert sol.cost < 1.0


def test_solve_frame_uses_reserve_when_muscles_saturate() -> None:
    active, passive, moment = _problem()
    tau = np.array([1000.0, 0.0])  # far above what the muscles can give
    sol = so.solve_frame(active, passive, moment, tau)
    assert sol.activation.max() == pytest.approx(1.0, abs=1e-6)
    assert sol.reserve[0] > 100.0
    np.testing.assert_allclose(
        moment.T @ (passive + sol.activation * active) + sol.reserve, tau, atol=1e-8
    )


def test_solve_frame_zero_demand_gives_zero_activation_when_passive_balanced() -> None:
    active, _, moment = _problem()
    sol = so.solve_frame(active, np.zeros(3), moment, np.zeros(2))
    np.testing.assert_allclose(sol.activation, 0.0, atol=1e-9)


def test_solve_frame_validates_inputs() -> None:
    active, passive, moment = _problem()
    with pytest.raises(ValueError):
        so.solve_frame(active, passive, moment, np.zeros(3))
    with pytest.raises(ValueError):
        so.solve_frame(active, passive, moment, np.zeros(2), reserve_weight=0.0)
    bad = active.copy()
    bad[0] = np.nan
    with pytest.raises(ValueError):
        so.solve_frame(bad, passive, moment, np.zeros(2))


def test_leg_coordinates_order_and_sides() -> None:
    names = so.leg_coordinates()
    assert len(names) == 14 and names[0] == "hip_flexion_r"
    assert so.leg_coordinates("l")[-1] == "mtp_angle_l"
