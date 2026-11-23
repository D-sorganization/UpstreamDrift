"""Fresh-state and timing contracts for the Slice 4 replay runner."""

import numpy as np
import pytest

from scripts.replay_optimized_candidate import replay_candidate
from src.shared.python.simulation_backends.protocol import (
    BackendCapabilities,
    SimState,
)

pytestmark = pytest.mark.unit


class Plant:
    """One-dimensional unit-mass plant with explicit Euler stepping."""

    capabilities = BackendCapabilities(name="test-unit-mass")

    def __init__(self):
        self.resets = 0
        self.state = SimState([99.0], [99.0])

    def reset(self, state):
        self.resets += 1
        self.state = state.copy()

    def get_state(self):
        return self.state  # Deliberately exposes storage to test isolation.

    def set_control(self, control):
        self.control = control.copy()

    def step(self, dt):
        self.state.q += dt * self.state.v
        self.state.v += dt * self.control
        self.state.time += dt


def run(**kwargs):
    options = {
        "backend_factory": Plant,
        "controller_factory": lambda: lambda state: -state.q - state.v,
        "initial": SimState([1.0], [0.0]),
        "steps": 2,
        "dt": 0.1,
        "substeps": 2,
    }
    options.update(kwargs)
    return replay_candidate(**options)


def test_feedback_uses_live_state_and_holds_control_between_updates():
    plant = Plant()
    seen = []

    def control(state):
        seen.append(state.copy())
        return -state.q - state.v

    trace = run(backend_factory=lambda: plant, controller_factory=lambda: control)
    assert plant.resets == 1
    assert len(seen) == 2
    np.testing.assert_allclose(trace.t, [0, 0.1, 0.2])
    np.testing.assert_allclose(trace.q[:, 0], [1, 0.9975, 0.98525625])
    np.testing.assert_allclose(trace.v[:, 0], [0, -0.1, -0.18975])
    np.testing.assert_allclose(trace.u[:, 0], [-1, -0.8975, 0])
    assert trace.meta["control_sampling"] == "left_endpoint_zoh"
    assert trace.meta["terminal_control"] == "unused_zero_padding"
    assert trace.torques is None  # Commanded controls do not attest applied torque.


def test_every_replay_constructs_new_plant_and_controller():
    plants, controllers = [], []

    def backend():
        plants.append(Plant())
        return plants[-1]

    def controller():
        calls = []
        controllers.append(calls)

        def evaluate(state):
            calls.append(state.time)
            return np.array([len(calls)], dtype=float)

        return evaluate

    a = run(backend_factory=backend, controller_factory=controller)
    b = run(backend_factory=backend, controller_factory=controller)
    assert len(plants) == len(controllers) == 2
    np.testing.assert_array_equal(a.q, b.q)
    assert controllers == [[0, 0.1], [0, 0.1]]


def test_controller_cannot_rewrite_live_plant_or_initial_state():
    initial = SimState([1.0], [0.0], time=3.0)

    def control(state):
        state.q[:] = 999
        return np.zeros(1)

    trace = run(initial=initial, controller_factory=lambda: control)
    np.testing.assert_array_equal(initial.q, [1])
    np.testing.assert_array_equal(trace.q, [[1], [1], [1]])
    np.testing.assert_allclose(trace.t, [3, 3.1, 3.2])


@pytest.mark.parametrize(
    "key,value",
    [
        ("steps", 0),
        ("steps", True),
        ("steps", 1.5),
        ("substeps", 0),
        ("substeps", False),
        ("dt", 0),
        ("dt", np.nan),
        ("dt", np.inf),
    ],
)
def test_bad_schedule_fails_before_constructing_backend(key, value):
    def forbidden():
        pytest.fail("backend must not be constructed")

    with pytest.raises(ValueError):
        run(backend_factory=forbidden, **{key: value})


@pytest.mark.parametrize("control", [[np.nan], [np.inf], [], [[1]], [1, 2]])
def test_invalid_control_is_rejected(control):
    with pytest.raises(ValueError, match="control"):
        run(controller_factory=lambda: lambda state: np.array(control))


@pytest.mark.parametrize("failure", ["clock", "nonfinite", "shape"])
def test_invalid_backend_state_is_rejected(failure):
    class Broken(Plant):
        def step(self, dt):
            super().step(dt)
            if failure == "clock":
                self.state.time = 0
            elif failure == "nonfinite":
                self.state.q[:] = np.nan
            else:
                self.state.q = np.zeros(2)

    with pytest.raises(ValueError, match="state|clock"):
        run(backend_factory=Broken)


def test_nonfinite_initial_state_is_rejected():
    with pytest.raises(ValueError, match="state"):
        run(initial=SimState([np.nan], [0]))


def test_reset_that_ignores_initial_state_is_rejected():
    class BadReset(Plant):
        def reset(self, state):
            pass

    with pytest.raises(ValueError, match="reset"):
        run(backend_factory=BadReset)


def test_real_ode_backend_reproduces_fresh_closed_loop_rollout():
    from src.shared.python.simulation_backends.model_params import GolfModelParams
    from src.shared.python.simulation_backends.ode_backend import ODEBackend

    def backend():
        return ODEBackend(GolfModelParams.default())

    options = {
        "backend_factory": backend,
        "initial": SimState([0.2, -0.1], [0.0, 0.0]),
        "steps": 10,
        "dt": 0.001,
    }
    a, b = run(**options), run(**options)
    np.testing.assert_allclose(a.q, b.q, rtol=0, atol=1e-14)
    assert not np.array_equal(a.q[0], a.q[-1])
    assert a.backend == "ode"
