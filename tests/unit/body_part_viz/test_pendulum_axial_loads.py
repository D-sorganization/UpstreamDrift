"""Concrete analytical source qualification, including inverted compression."""

import numpy as np
import pytest

from src.shared.python.pendulum_simulator.force_colors import pendulum_axial_loads
from src.shared.python.pendulum_simulator.physics import PendulumParams
from src.shared.python.pendulum_simulator.simulation import SimulationResult

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("angle,sign", [(0.0, 1.0), (np.pi, -1.0)])
def test_double_static_equilibria(angle, sign):
    result = SimulationResult(
        np.array([0.0, 1.0]),
        np.array([[angle, 0, 0, 0], [angle, 0, 0, 0]]),
        PendulumParams(m1=2, m2=3, L1=1, L2=1),
        lambda t: (0.0, 0.0),
    )
    before = result.states.copy()
    frame = pendulum_axial_loads(result, 0)
    assert frame.values_n["arm"] == pytest.approx(sign * 5 * result.params.g)
    assert frame.values_n["club"] == pytest.approx(sign * 3 * result.params.g)
    np.testing.assert_array_equal(result.states, before)


def test_unsupported_result_is_not_inferred_from_names():
    assert pendulum_axial_loads(object(), 0) is None


@pytest.mark.parametrize("angle,sign", [(0.0, 1.0), (np.pi, -1.0)])
def test_triple_static_equilibria(angle, sign):
    from src.shared.python.pendulum_simulator.physics_triple import TriplePendulumParams
    from src.shared.python.pendulum_simulator.simulation_triple import (
        TripleSimulationResult,
    )

    result = TripleSimulationResult(
        np.array([0.0, 1.0]),
        np.array([[angle, 0, 0, 0, 0, 0], [angle, 0, 0, 0, 0, 0]]),
        TriplePendulumParams(m1=2, m2=3, m3=4, L1=1, L2=1, L3=1),
        lambda t: (0.0, 0.0, 0.0),
    )
    frame = pendulum_axial_loads(result, 0)
    for segment, mass in (("arm", 9), ("forearm", 7), ("club", 4)):
        assert frame.values_n[segment] == pytest.approx(sign * mass * result.params.g)
