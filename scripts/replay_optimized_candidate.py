"""Fresh closed-loop forward replay orchestration for optimization Slice 4.

The caller decodes a bundle and reconstructs its plant/controller factories.
Recorded rollout states and torques are deliberately absent from this API.
This returns a new trace, not a replay-comparison or physical-acceptance verdict.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from src.shared.python.simulation_backends.protocol import (
    SimState,
    SimulationBackend,
    Trace,
)

Controller = Callable[[SimState], np.ndarray]


def _snapshot(state: SimState, shape: tuple[int, ...], time: float) -> SimState:
    """Copy and validate the live state before exposing it outside the plant."""
    if (
        state.q.shape != shape
        or state.v.shape != shape
        or not np.all(np.isfinite(state.q))
        or not np.all(np.isfinite(state.v))
    ):
        raise ValueError("state must contain finite vectors of the initial shape")
    if not math.isfinite(state.time) or not math.isclose(
        state.time, time, rel_tol=0, abs_tol=1e-10
    ):
        raise ValueError("backend clock disagrees with replay schedule")
    return state.copy()


def replay_candidate(
    *,
    backend_factory: Callable[[], SimulationBackend],
    controller_factory: Callable[[], Controller],
    initial: SimState,
    steps: int,
    dt: float,
    substeps: int = 1,
) -> Trace:
    """Re-simulate a candidate with fixed-rate feedback and ZOH controls.

    Preconditions: fresh factories reconstruct the pinned model and controller
    (including internal state), and controller output has the initial state's
    dimension. Only equal position/velocity/control dimensions are supported by
    this SimulationBackend adapter. dt is the controller period in seconds;
    each period contains substeps equal integration steps. The controller owns
    feedforward, PD and saturation semantics. No reference state is injected
    into the plant after the single initial reset.

    Postcondition: a finite trace with steps+1 states at control boundaries.
    u[i] is the command held over interval i; the final unused row is zero.
    Commands are not measured applied torques. Errors propagate without issuing
    a partial trace or a success verdict. Factories must not return cached state.
    """
    for name, value in (("steps", steps), ("substeps", substeps)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not math.isfinite(dt) or dt <= 0 or dt / substeps <= 0:
        raise ValueError("dt must be finite, positive and representable per substep")
    start = initial.time
    end = start + steps * dt
    if not math.isfinite(end) or end <= start or start + dt / substeps <= start:
        raise ValueError("replay clock must advance and remain finite")
    shape = initial.q.shape
    first = _snapshot(initial, shape, start)
    backend = backend_factory()
    controller = controller_factory()
    backend.reset(first.copy())
    state = _snapshot(backend.get_state(), shape, start)
    if not np.array_equal(state.q, first.q) or not np.array_equal(state.v, first.v):
        raise ValueError("backend reset did not restore the complete initial state")
    states = [state]
    commands = []
    for interval in range(steps):
        command = np.asarray(controller(state.copy()), dtype=float)
        if command.shape != shape or not np.all(np.isfinite(command)):
            raise ValueError("control must be a finite vector of the initial shape")
        commands.append(command.copy())
        backend.set_control(command.copy())
        for substep in range(substeps):
            backend.step(dt / substeps)
            expected_time = start + (interval + (substep + 1) / substeps) * dt
            state = _snapshot(backend.get_state(), shape, expected_time)
        states.append(state)
    commands.append(np.zeros(shape))
    return Trace(
        t=np.array([s.time for s in states]),
        q=np.stack([s.q for s in states]),
        v=np.stack([s.v for s in states]),
        u=np.stack(commands),
        dt=dt / substeps,
        backend=backend.capabilities.name,
        meta={
            "replay_mode": "closed_loop_forward_resimulation",
            "control_sampling": "left_endpoint_zoh",
            "control_period_s": dt,
            "integration_substeps": substeps,
            "terminal_control": "unused_zero_padding",
            "control_quantity": "commanded_not_measured_applied",
        },
    )
