"""NSX-30: counterfactual rollouts integrate the real (constrained) dynamics.

The ``m_eff = 1.0`` mock and the ``5 sin(t)`` placeholder are gone: every rollout
is a forward simulation of a provider, and null-space exploration samples ``z``
from the NSX-15 decomposition, so a sampled allocation reproduces the baseline
motion (the family is invisible to the acceleration).
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from src.shared.python.loop_nullspace.decomposition import AllocationMetric, decompose
from src.shared.python.loop_nullspace.fixtures import (
    PlanarGolferConfig,
    load_golden,
    planar_golfer_snapshot,
)
from src.shared.python.loop_nullspace.planar_model import (
    PlanarGolferParams,
    PlanarGolferProvider,
)
from src.shared.python.motion_matching import counterfactual as cf
from src.shared.python.motion_matching.candidate import (
    CandidateAuxiliary,
    CandidateMarkers,
    CandidateMetadata,
    CandidateProfile,
    MatchedSwingCandidate,
)
from src.shared.python.motion_matching.candidate_session import CandidateSession
from src.shared.python.motion_matching.counterfactual import (
    CounterfactualDynamics,
    CounterfactualStrategy,
    NullspaceExploration,
    create_counterfactual_rollout,
)
from src.shared.python.simulation_backends.constrained_ztcf_zvcf import (
    constrained_acceleration_at,
)

pytestmark = pytest.mark.unit

NAMES = ("lead_shoulder", "lead_elbow", "club_angle", "trail_shoulder", "trail_elbow")
DT = 1e-3
FRAMES = 12


def _session(t, q, v, a, tau, names=NAMES) -> CandidateSession:
    meta = CandidateMetadata(profile=CandidateProfile.DYNAMIC, coordinate_names=names)
    cand = MatchedSwingCandidate(
        metadata=meta,
        time_s=t,
        q=q,
        v=v,
        tau=tau,
        markers=CandidateMarkers(),
        auxiliary=CandidateAuxiliary(),
    )
    return CandidateSession(
        candidate=cand,
        specification={"coordinate_order": list(names)},
        candidate_sha256="nsx30_sha",
        model_sha256="nsx30_model",
        coordinate_names=names,
        time_s=t,
        q=q,
        v=v,
        a=a,
        tau=tau,
        is_accepted=True,
    )


@pytest.fixture(scope="module")
def planar():
    """A consistent planar swing: states, per-frame NSX-15 decompositions."""
    case = planar_golfer_snapshot()
    golden = load_golden("planar_two_arm_full")
    provider = PlanarGolferProvider(PlanarGolferParams.from_golden(golden))
    s0 = case.snapshot
    q, v = s0.q.copy(), s0.v.copy()
    qs, vs, acc, taus, decs = [], [], [], [], []
    metric = AllocationMetric.identity(5)
    for _ in range(FRAMES):
        # ``a`` must satisfy the closure to build a snapshot: solve it first
        # with a placeholder load, then take the allocation that reproduces it.
        a, _ = constrained_acceleration_at(provider, q, v, np.zeros(5))
        snap = planar_golfer_snapshot(PlanarGolferConfig(q, v, a)).snapshot
        dec = decompose(snap, weights=metric)
        tau_a = dec.torque(np.zeros(dec.nullity))
        a_true, _ = constrained_acceleration_at(provider, q, v, snap.S.T @ tau_a)
        qs.append(q.copy())
        vs.append(v.copy())
        acc.append(a_true)
        taus.append(snap.S.T @ tau_a)
        decs.append(dec)
        # advance with the cf integrator's own scheme: one RK4 step, ZOH load
        q, v = _rk4(provider, q, v, taus[-1], DT)
    t = np.arange(FRAMES) * DT
    return provider, _session(t, np.array(qs), np.array(vs), np.array(acc), np.array(taus)), decs


def _rk4(provider, q, v, tau, dt):
    def f(qq, vv):
        return vv, constrained_acceleration_at(provider, qq, vv, tau)[0]

    k1 = f(q, v)
    k2 = f(q + 0.5 * dt * k1[0], v + 0.5 * dt * k1[1])
    k3 = f(q + 0.5 * dt * k2[0], v + 0.5 * dt * k2[1])
    k4 = f(q + dt * k3[0], v + dt * k3[1])
    return (
        q + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
        v + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]),
    )


class _DiagProvider:
    def __init__(self, m: np.ndarray, h: np.ndarray) -> None:
        self._m, self._h = np.diag(m), h

    def mass_matrix(self, q):
        return self._m

    def bias_forces(self, q, v):
        return self._h


def test_mock_is_removed() -> None:
    src = inspect.getsource(cf)
    assert not hasattr(cf, "m_eff")
    assert "m_eff" not in src
    assert "np.sin" not in src


def test_missing_dynamics_fails_closed(planar) -> None:
    _, session, _ = planar
    with pytest.raises(ValueError, match="dynamics"):
        create_counterfactual_rollout(session=session, fork_frame_idx=0)


def test_unconstrained_rollout_is_the_exact_forward_dynamics() -> None:
    n = 5
    t = np.arange(n) * 0.1
    q = np.zeros((n, 2))
    session = _session(t, q, np.zeros((n, 2)), np.zeros((n, 2)), np.ones((n, 2)), ("a", "b"))
    dyn = CounterfactualDynamics(_DiagProvider(np.array([2.0, 4.0]), np.zeros(2)))
    fork = create_counterfactual_rollout(
        session=session,
        fork_frame_idx=0,
        strategy=CounterfactualStrategy.CUSTOM,
        control_override=np.ones((n, 2)),
        dynamics=dyn,
    )
    np.testing.assert_allclose(fork.a, np.tile([0.5, 0.25], (n, 1)), atol=1e-14)
    np.testing.assert_allclose(fork.q[-1], 0.5 * np.array([0.5, 0.25]) * 0.4**2, atol=1e-12)


def test_constrained_rollout_reproduces_the_baseline_motion(planar) -> None:
    provider, session, _ = planar
    fork = create_counterfactual_rollout(
        session=session,
        fork_frame_idx=0,
        strategy=CounterfactualStrategy.CUSTOM,
        control_override=np.array(session.tau),
        dynamics=CounterfactualDynamics(provider),
    )
    scale = max(1.0, float(np.max(np.abs(session.q))))
    assert np.max(np.abs(fork.q - session.q)) / scale < 1e-9
    assert fork.divergence_rms < 1e-9


def test_closure_drift_stays_below_tolerance_over_the_horizon(planar) -> None:
    provider, session, _ = planar
    fork = create_counterfactual_rollout(
        session=session,
        fork_frame_idx=0,
        strategy=CounterfactualStrategy.CLAMPED_ACTUATOR_TORQUE,
        clamp_limits=(-0.5, 0.5),
        dynamics=CounterfactualDynamics(provider),
    )
    phi = np.array([provider.closure_error(qi) for qi in fork.q])
    assert np.max(np.abs(phi)) < 1e-7
    assert fork.constraint_status["closure_velocity_residual_max"] < 1e-6
    assert np.max(np.abs(fork.altered_tau)) <= 0.5 + 1e-12


def test_constrained_rollout_differs_from_the_unconstrained_one(planar) -> None:
    provider, session, _ = planar
    free = CounterfactualDynamics(
        _DiagProvider(np.diag(provider.mass_matrix(session.q[0])), provider.bias_forces(session.q[0], session.v[0]))
    )
    kwargs = dict(
        session=session,
        fork_frame_idx=0,
        strategy=CounterfactualStrategy.CUSTOM,
        control_override=np.array(session.tau),
    )
    closed = create_counterfactual_rollout(**kwargs, dynamics=CounterfactualDynamics(provider))
    open_ = create_counterfactual_rollout(**kwargs, dynamics=free)
    assert np.max(np.abs(closed.a - open_.a)) > 1e-3


def test_nullspace_exploration_samples_z_and_keeps_the_motion(planar) -> None:
    provider, session, decs = planar
    exploration = NullspaceExploration(
        decompositions=tuple(decs), actuated_indices=(0, 1, 2, 3, 4), seed=7, scale=3.0
    )
    fork = create_counterfactual_rollout(
        session=session,
        fork_frame_idx=0,
        strategy=CounterfactualStrategy.NULLSPACE_EXPLORATION,
        dynamics=CounterfactualDynamics(provider, nullspace=exploration),
    )
    # the sampled allocation really differs from the measured one ...
    assert np.max(np.abs(fork.altered_tau - np.array(session.tau))) > 1e-3
    # ... yet every family member produces the measured acceleration at the
    # frame state it was sampled for (R9)
    np.testing.assert_allclose(fork.a[0], session.a[0], rtol=0, atol=1e-9)
    # Family membership is instantaneous: the load is held across each step
    # while the state moves, so later frames drift by O(dt), not by the size
    # of the sampled null-space torque.
    scale = float(np.max(np.abs(session.a)))
    assert np.max(np.abs(fork.a - session.a)) / scale < 1e-2
    assert fork.divergence_rms < 1e-6
    off_family = create_counterfactual_rollout(
        session=session,
        fork_frame_idx=0,
        strategy=CounterfactualStrategy.CUSTOM,
        control_override=np.array(session.tau) + 0.3,
        dynamics=CounterfactualDynamics(provider),
    )
    assert off_family.divergence_rms > 10 * fork.divergence_rms
    # deterministic in the seed
    again = create_counterfactual_rollout(
        session=session,
        fork_frame_idx=0,
        strategy=CounterfactualStrategy.NULLSPACE_EXPLORATION,
        dynamics=CounterfactualDynamics(provider, nullspace=exploration),
    )
    np.testing.assert_array_equal(again.altered_tau, fork.altered_tau)


def test_explicit_z_is_used_verbatim(planar) -> None:
    provider, session, decs = planar
    nullity = decs[0].nullity
    z = np.full(nullity, 1.5)
    exploration = NullspaceExploration(
        decompositions=tuple(decs), actuated_indices=(0, 1, 2, 3, 4), z=z
    )
    fork = create_counterfactual_rollout(
        session=session,
        fork_frame_idx=0,
        strategy=CounterfactualStrategy.NULLSPACE_EXPLORATION,
        dynamics=CounterfactualDynamics(provider, nullspace=exploration),
    )
    np.testing.assert_allclose(fork.altered_tau[0], decs[0].torque(z), atol=1e-12)


def test_nullspace_strategy_without_exploration_fails_closed(planar) -> None:
    provider, session, _ = planar
    with pytest.raises(ValueError, match="nullspace"):
        create_counterfactual_rollout(
            session=session,
            fork_frame_idx=0,
            strategy=CounterfactualStrategy.NULLSPACE_EXPLORATION,
            dynamics=CounterfactualDynamics(provider),
        )


def test_nullspace_frame_count_mismatch_is_rejected(planar) -> None:
    provider, session, decs = planar
    bad = NullspaceExploration(decompositions=tuple(decs[:3]), actuated_indices=(0, 1, 2, 3, 4))
    with pytest.raises(ValueError, match="decompositions"):
        create_counterfactual_rollout(
            session=session,
            fork_frame_idx=0,
            strategy=CounterfactualStrategy.NULLSPACE_EXPLORATION,
            dynamics=CounterfactualDynamics(provider, nullspace=bad),
        )
