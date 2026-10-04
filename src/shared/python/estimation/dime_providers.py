"""DIME Dynamics Providers (#11421, #11423).

Provides concrete dynamics providers implementing the DIME DynamicsProvider protocol:
- DeterministicFakeProvider: synthetic test provider satisfying contracts without native qualification.
- AnalyticPendulumProvider: single-DOF analytic harmonic pendulum provider using synthetic fixtures.
- UnderactuatedAnalyticProvider: underactuated two-link fixture provider with passive root DOF.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.shared.python.contracts import require
from src.shared.python.estimation.dime_manifest import (
    CANONICAL_DIME_UNITS,
    CapabilityStatus,
)
from src.shared.python.estimation.synthetic_fixtures import (
    FixedBasePendulumFixture,
    UnderactuatedAnalyticFixture,
    make_fixed_base_pendulum_fixture,
    make_underactuated_analytic_fixture,
)
from src.shared.python.motion_matching.counterfactual import (
    AccelerationDecomposition,
)
from src.shared.python.estimation.dime_contracts import (
    DIME_CONTRACTS_VERSION,
    ContactPolicy,
    ControlChannelSpec,
    DimeCompleteState,
    DimeFullStepRequest,
    DimeFullStepResult,
    DimeZeroInputProposal,
    DynamicsProvider,
    ProviderCapability,
    ProviderSnapshot,
    SE3Manifold,
    VectorSpaceManifold,
    _validate_units_dict,
)


def rollout_zero_input_proposal(
    provider: DynamicsProvider, state: DimeCompleteState, duration: float, dt: float
) -> DimeZeroInputProposal:
    """Propagate forward dynamics under zero applied control to evaluate passive drift and ZTCF."""
    if hasattr(provider, "_validate_incoming_state"):
        provider._validate_incoming_state(state)  # type: ignore[attr-defined]

    n_steps = max(1, int(round(duration / dt)))
    times = [state.t + i * dt for i in range(n_steps + 1)]
    states = [state]
    accels = []
    decomps = []

    zero_u = np.zeros(len(provider.capability.control_channels), dtype=np.float64)
    curr_state = state
    curr_decomp = provider.compute_acceleration_decomposition(curr_state, zero_u)
    accels.append(curr_decomp.ztcf)
    decomps.append(curr_decomp)

    for _ in range(n_steps):
        step_req = DimeFullStepRequest(
            state=curr_state,
            controls=zero_u,
            dt=dt,
            model_hash=provider.model_hash,
        )
        res = provider.step(step_req)
        curr_state = res.next_state
        states.append(curr_state)
        decomp = provider.compute_acceleration_decomposition(curr_state, zero_u)
        accels.append(decomp.ztcf)
        decomps.append(decomp)

    return DimeZeroInputProposal(
        t_start=times[0],
        t_end=times[-1],
        times=np.array(times, dtype=np.float64),
        states=tuple(states),
        accelerations=np.array(accels, dtype=np.float64),
        decompositions=tuple(decomps),
        source_intervention="ZTCF",
    )


class _BaseDynamicsProvider:
    """Base class providing shared state validation, snapshot memory, and ZTCF proposals."""

    def __init__(
        self,
        model_hash: str,
        capability: ProviderCapability,
        initial_state: DimeCompleteState,
    ) -> None:
        self._model_hash = model_hash
        self._capability = capability
        self._state = initial_state
        self._solver_memory: dict[str, Any] = {}

    @property
    def model_hash(self) -> str:
        return self._model_hash

    @property
    def capability(self) -> ProviderCapability:
        return self._capability

    def get_state(self) -> DimeCompleteState:
        return self._state

    def set_state(self, state: DimeCompleteState) -> None:
        self._validate_incoming_state(state)
        self._state = state

    def snapshot(self) -> ProviderSnapshot:
        return ProviderSnapshot(
            provider_id=self._capability.provider_id,
            model_hash=self._model_hash,
            timestamp=self._state.t,
            state=self._state,
            solver_memory=dict(self._solver_memory),
        )

    def restore(self, snapshot: ProviderSnapshot) -> None:
        require(snapshot.model_hash == self._model_hash, "Snapshot model_hash mismatch")
        self._state = snapshot.state
        self._solver_memory = dict(snapshot.solver_memory)

    def _validate_incoming_state(self, state: DimeCompleteState) -> None:
        require(self._capability.status != "unavailable", "Provider is unavailable")
        require(
            state.model_hash == self._model_hash,
            "State model_hash does not match provider",
        )
        require(len(state.q) == self._capability.n_q, "q dimension mismatch")
        require(len(state.v) == self._capability.n_v, "v dimension mismatch")
        _validate_units_dict(state.units, "incoming state units")

    def compute_zero_input_proposal(
        self, state: DimeCompleteState, duration: float, dt: float
    ) -> DimeZeroInputProposal:
        return rollout_zero_input_proposal(self, state, duration, dt)  # type: ignore[arg-type]

    def _finalize_step(
        self,
        next_state: DimeCompleteState,
        decomp: AccelerationDecomposition,
        reaction_forces: np.ndarray | None = None,
    ) -> DimeFullStepResult:
        self._state = next_state
        rf = (
            np.zeros(3, dtype=np.float64)
            if reaction_forces is None
            else reaction_forces
        )
        return DimeFullStepResult(
            next_state=next_state,
            accelerations=decomp.a_grav + decomp.a_drift + decomp.a_ctrl,
            reaction_forces=rf,
            decomposition=decomp,
        )


class DeterministicFakeProvider(_BaseDynamicsProvider):
    """Deterministic fake provider satisfying identical contracts without claiming qualification."""

    def __init__(
        self,
        n_q: int = 1,
        n_v: int = 1,
        model_hash: str = "fake-provider-hash",
        status: CapabilityStatus = "implemented",
    ) -> None:
        manifold = VectorSpaceManifold(n_q) if n_q == n_v else SE3Manifold()
        cap = ProviderCapability(
            provider_id="fake-provider-deterministic",
            version=DIME_CONTRACTS_VERSION,
            status=status,
            n_q=n_q,
            n_v=n_v,
            manifold=manifold,
            control_channels=(
                ControlChannelSpec(
                    name="actuator_0",
                    physical_type="torque",
                    units="N*m",
                    selection_map=(0,),
                    limits=(-100.0, 100.0),
                ),
            ),
            contact_policy=ContactPolicy.NATIVE_ELIMINATED,
            retained_passive_loads=(),
            is_qualified=False,
        )
        init_state = DimeCompleteState(
            t=0.0,
            q=np.zeros(n_q, dtype=np.float64),
            v=np.zeros(n_v, dtype=np.float64),
            model_hash=model_hash,
            units=dict(CANONICAL_DIME_UNITS),
        )
        super().__init__(model_hash, cap, init_state)
        self._inject_failure = False

    def inject_step_failure(self, fail: bool) -> None:
        self._inject_failure = fail

    def step(self, request: DimeFullStepRequest) -> DimeFullStepResult:
        if self.capability.status == "unavailable":
            raise RuntimeError("Provider is unavailable")
        require(request.model_hash == self.model_hash, "Request model_hash mismatch")
        self._validate_incoming_state(request.state)
        require(
            len(request.controls) == len(self.capability.control_channels),
            "Controls count mismatch",
        )

        pre_snap = self.snapshot()
        try:
            if self._inject_failure:
                raise RuntimeError("Simulated engine failure")

            dt = request.dt
            nv = self.capability.n_v
            decomp = self.compute_acceleration_decomposition(
                request.state, request.controls
            )
            a_full = decomp.a_grav + decomp.a_drift + decomp.a_ctrl
            v_next = request.state.v + a_full * dt
            q_next = (
                request.state.q[:nv] + request.state.v * dt + 0.5 * a_full * (dt**2)
            )
            if len(q_next) < self.capability.n_q:
                q_next = np.concatenate([q_next, request.state.q[len(q_next) :]])

            next_state = DimeCompleteState(
                t=request.state.t + dt,
                q=q_next,
                v=v_next,
                v_dot=a_full,
                internal_state=dict(request.state.internal_state),
                model_hash=self.model_hash,
                units=dict(request.state.units),
            )
            return self._finalize_step(next_state, decomp)
        except Exception:
            self.restore(pre_snap)
            raise

    def compute_acceleration_decomposition(
        self, state: DimeCompleteState, controls: np.ndarray
    ) -> AccelerationDecomposition:
        nv = self.capability.n_v
        a_grav = -9.81 * np.ones(nv, dtype=np.float64)
        a_drift = np.zeros(nv, dtype=np.float64)
        a_ctrl = np.zeros(nv, dtype=np.float64)
        if len(controls) > 0:
            a_ctrl[0] = controls[0]
        return AccelerationDecomposition(a_grav=a_grav, a_drift=a_drift, a_ctrl=a_ctrl)


class AnalyticPendulumProvider(_BaseDynamicsProvider):
    """Deterministic single-DOF pendulum provider implementing identical contracts."""

    def __init__(self, fixture: FixedBasePendulumFixture | None = None) -> None:
        self._fixture = fixture or make_fixed_base_pendulum_fixture()
        model_hash = "sha256-pendulum-model-v1"
        cap = ProviderCapability(
            provider_id="analytic-fixed-base-pendulum",
            version=DIME_CONTRACTS_VERSION,
            status="implemented",
            n_q=1,
            n_v=1,
            manifold=VectorSpaceManifold(1),
            control_channels=(
                ControlChannelSpec(
                    name="pivot_torque",
                    physical_type="torque",
                    units="N*m",
                    selection_map=(0,),
                    limits=(-50.0, 50.0),
                ),
            ),
            contact_policy=ContactPolicy.NATIVE_ELIMINATED,
            retained_passive_loads=(),
            is_qualified=False,
        )
        q0, v0 = self._fixture.initial_state
        init_state = DimeCompleteState(
            t=0.0,
            q=np.array([q0], dtype=np.float64),
            v=np.array([v0], dtype=np.float64),
            model_hash=model_hash,
            units=dict(CANONICAL_DIME_UNITS),
        )
        super().__init__(model_hash, cap, init_state)

    def step(self, request: DimeFullStepRequest) -> DimeFullStepResult:
        require(request.model_hash == self.model_hash, "Request model_hash mismatch")
        self._validate_incoming_state(request.state)
        require(len(request.controls) == 1, "Controls length must be 1")

        pre_snap = self.snapshot()
        try:
            dt = request.dt
            theta = float(request.state.q[0])
            omega = float(request.state.v[0])

            decomp = self.compute_acceleration_decomposition(
                request.state, request.controls
            )
            a_full = decomp.a_grav + decomp.a_drift + decomp.a_ctrl

            omega_next = omega + float(a_full[0]) * dt
            theta_next = theta + omega_next * dt

            next_state = DimeCompleteState(
                t=request.state.t + dt,
                q=np.array([theta_next], dtype=np.float64),
                v=np.array([omega_next], dtype=np.float64),
                v_dot=a_full,
                model_hash=self.model_hash,
                units=dict(request.state.units),
            )
            return self._finalize_step(next_state, decomp)
        except Exception:
            self.restore(pre_snap)
            raise

    def compute_acceleration_decomposition(
        self, state: DimeCompleteState, controls: np.ndarray
    ) -> AccelerationDecomposition:
        theta = float(state.q[0])
        tau = float(controls[0]) if len(controls) > 0 else 0.0
        g = self._fixture.gravity_m_s2
        length = self._fixture.length_m
        mass = self._fixture.mass_kg

        a_grav = np.array([-(g / length) * np.sin(theta)], dtype=np.float64)
        a_drift = np.zeros(1, dtype=np.float64)
        a_ctrl = np.array([tau / (mass * length**2)], dtype=np.float64)
        return AccelerationDecomposition(a_grav=a_grav, a_drift=a_drift, a_ctrl=a_ctrl)


class UnderactuatedAnalyticProvider(_BaseDynamicsProvider):
    """Underactuated two-link fixture dynamics provider with unactuated root joint."""

    def __init__(self, fixture: UnderactuatedAnalyticFixture | None = None) -> None:
        self._fixture = fixture or make_underactuated_analytic_fixture()
        model_hash = "sha256-underactuated-model-v1"
        cap = ProviderCapability(
            provider_id="underactuated-analytic-provider",
            version=DIME_CONTRACTS_VERSION,
            status="implemented",
            n_q=2,
            n_v=2,
            manifold=VectorSpaceManifold(2),
            control_channels=(
                ControlChannelSpec(
                    name="actuated_joint_1_torque",
                    physical_type="torque",
                    units="N*m",
                    selection_map=(1,),
                    limits=(-20.0, 20.0),
                ),
            ),
            contact_policy=ContactPolicy.NATIVE_ELIMINATED,
            retained_passive_loads=(),
            is_qualified=False,
        )
        init_q, init_v = self._fixture.initial_state
        init_state = DimeCompleteState(
            t=0.0,
            q=np.array(init_q[:2], dtype=np.float64),
            v=np.array(init_v[:2], dtype=np.float64),
            model_hash=model_hash,
            units=dict(CANONICAL_DIME_UNITS),
        )
        super().__init__(model_hash, cap, init_state)

    def step(self, request: DimeFullStepRequest) -> DimeFullStepResult:
        require(request.model_hash == self.model_hash, "Model hash mismatch")
        self._validate_incoming_state(request.state)
        require(len(request.controls) == 1, "Expected 1 control for actuated DOF 1")
        pre_snap = self.snapshot()
        try:
            dt = request.dt
            v = request.state.v

            decomp = self.compute_acceleration_decomposition(
                request.state, request.controls
            )
            a_full = decomp.a_grav + decomp.a_drift + decomp.a_ctrl

            v_next = v + a_full * dt
            q_next = request.state.q + v_next * dt

            next_state = DimeCompleteState(
                t=request.state.t + dt,
                q=q_next,
                v=v_next,
                v_dot=a_full,
                model_hash=self.model_hash,
                units=dict(request.state.units),
            )
            return self._finalize_step(next_state, decomp)
        except Exception:
            self.restore(pre_snap)
            raise

    def compute_acceleration_decomposition(
        self, state: DimeCompleteState, controls: np.ndarray
    ) -> AccelerationDecomposition:
        q = state.q
        v = state.v
        tau1 = float(controls[0]) if len(controls) > 0 else 0.0
        a_grav = -9.81 * np.array([np.sin(q[0]), np.sin(q[1])], dtype=np.float64)
        a_drift = np.array([-0.1 * v[0], -0.1 * v[1]], dtype=np.float64)
        a_ctrl = np.array([0.0, tau1 * 0.5], dtype=np.float64)
        return AccelerationDecomposition(a_grav=a_grav, a_drift=a_drift, a_ctrl=a_ctrl)
