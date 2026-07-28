"""Public pendulum simulator contracts must survive optimized Python."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.pendulum_simulator.physics import PendulumParams
from src.shared.python.pendulum_simulator.physics_base import (
    clamp_torque_ndof,
    kinetic_energy_from_M,
)
from src.shared.python.pendulum_simulator.physics_golfer import GolferParams, N_DOF
from src.shared.python.pendulum_simulator.physics_triple import TriplePendulumParams
from src.shared.python.pendulum_simulator.simulation import (
    SimulationResult,
    run_simulation as run_double_simulation,
)
from src.shared.python.pendulum_simulator.simulation_golfer import (
    run_simulation as run_golfer_simulation,
)
from src.shared.python.pendulum_simulator.simulation_triple import (
    run_simulation as run_triple_simulation,
)

pytestmark = pytest.mark.unit


def _double_params() -> PendulumParams:
    return PendulumParams(m1=5.0, m2=0.3, L1=0.65, L2=1.1)


def _triple_params() -> TriplePendulumParams:
    return TriplePendulumParams(m1=5.0, m2=0.3, m3=0.05, L1=0.65, L2=1.1, L3=0.1)


def _golfer_params() -> GolferParams:
    return GolferParams(
        m_hub=0.1,
        m_r_upper=2.0,
        m_r_fore=1.5,
        m_l_upper=2.0,
        m_l_fore=1.5,
        m_club=0.4,
        L_hub=0.2,
        L_r_upper=0.3,
        L_r_fore=0.25,
        L_l_upper=0.3,
        L_l_fore=0.25,
        L_club=1.0,
        d_rs=0.2,
        d_ls=0.2,
        grip_right=0.2,
        grip_left=0.3,
    )


def _zero_double_torque(_t: float) -> tuple[float, float]:
    return 0.0, 0.0


def _zero_triple_torque(_t: float) -> tuple[float, float, float]:
    return 0.0, 0.0, 0.0


def _zero_golfer_torque(_t: float) -> np.ndarray:
    return np.zeros(7)


def test_double_runner_rejects_invalid_public_inputs_with_value_error() -> None:
    with pytest.raises(ValueError, match=r"Initial state shape must be \(4,\)"):
        run_double_simulation(_double_params(), np.zeros(3), 0.2, _zero_double_torque)

    with pytest.raises(ValueError, match="Initial state must be finite"):
        run_double_simulation(
            _double_params(),
            np.array([0.0, 0.0, np.inf, 0.0]),
            0.2,
            _zero_double_torque,
        )

    with pytest.raises(ValueError, match="t_end must be positive"):
        run_double_simulation(_double_params(), np.zeros(4), -1.0, _zero_double_torque)

    with pytest.raises(ValueError, match=r"dt must be in \(0, t_end\)"):
        run_double_simulation(
            _double_params(), np.zeros(4), 0.2, _zero_double_torque, dt=0.3
        )


def test_triple_runner_rejects_invalid_public_inputs_with_value_error() -> None:
    with pytest.raises(ValueError, match=r"Initial state shape must be \(6,\)"):
        run_triple_simulation(_triple_params(), np.zeros(4), 0.2, _zero_triple_torque)

    with pytest.raises(ValueError, match="Initial state must be finite"):
        state = np.zeros(6)
        state[1] = np.nan
        run_triple_simulation(_triple_params(), state, 0.2, _zero_triple_torque)


def test_golfer_runner_rejects_invalid_public_inputs_with_value_error() -> None:
    with pytest.raises(
        ValueError, match=f"Initial state shape must be \\({2 * N_DOF},\\)"
    ):
        run_golfer_simulation(_golfer_params(), np.zeros(4), 0.2, _zero_golfer_torque)

    with pytest.raises(ValueError, match="Initial state must be finite"):
        state = np.zeros(2 * N_DOF)
        state[-1] = np.inf
        run_golfer_simulation(_golfer_params(), state, 0.2, _zero_golfer_torque)


def test_physics_base_rejects_invalid_public_inputs_with_value_error() -> None:
    with pytest.raises(ValueError, match="M shape"):
        kinetic_energy_from_M(np.eye(2), np.zeros(3))

    with pytest.raises(ValueError, match="Mass matrix has non-finite values"):
        kinetic_energy_from_M(np.array([[np.inf]]), np.zeros(1))

    with pytest.raises(ValueError, match="Torque limits must be positive"):
        clamp_torque_ndof(np.array([1.0]), np.array([0.0]))


def test_result_container_rejects_invalid_public_inputs_with_value_error() -> None:
    params = _double_params()

    with pytest.raises(ValueError, match="states must have width 4"):
        SimulationResult(
            t=np.array([0.0]),
            states=np.zeros((1, 3)),
            params=params,
            torque_func=_zero_double_torque,
        )

    result = SimulationResult(
        t=np.array([0.0, 0.1]),
        states=np.zeros((2, 4)),
        params=params,
        torque_func=_zero_double_torque,
    )
    with pytest.raises(ValueError, match="idx must be provided"):
        result.energy_at(None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"Index 2 out of range"):
        result.energy_at(2)
