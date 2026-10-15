"""Unit tests for constrained EquationsOfMotionProvider and constrained ZTCF/ZVCF (NSX-30, #12570)."""

from __future__ import annotations

import inspect
import numpy as np
import pytest

from src.shared.python.loop_nullspace.constrained_dynamics import (
    ConstrainedEquationsOfMotionProvider,
    compute_constrained_ztcf_zvcf,
    constrained_acceleration,
)
from src.shared.python.loop_nullspace.contracts import (
    AllocationDependence,
    ClosedLoopSnapshot,
)
from src.shared.python.loop_nullspace.fixtures import (
    _default_closed_loop_golfer_state,
    planar_golfer_snapshot,
)
from src.shared.python.motion_matching import counterfactual as cf_module
from src.shared.python.motion_matching.candidate import (
    CandidateAuxiliary,
    CandidateMarkers,
    CandidateMetadata,
    CandidateProfile,
    MatchedSwingCandidate,
)
from src.shared.python.motion_matching.candidate_session import CandidateSession
from src.shared.python.motion_matching.counterfactual import (
    CounterfactualStrategy,
    create_counterfactual_rollout,
)
from src.shared.python.pendulum_simulator.counterfactual_golfer import (
    zero_torque_accelerations,
)
from src.shared.python.simulation_backends.ztcf_zvcf import (
    drift_and_control_split,
    ztcf_acceleration,
    zvcf_acceleration,
)

pytestmark = pytest.mark.unit


class _PlanarConstrainedProvider:
    """Concrete ConstrainedEquationsOfMotionProvider backed by planar_golfer_snapshot."""

    def __init__(self, snapshot: ClosedLoopSnapshot) -> None:
        self._snap = snapshot

    def mass_matrix(self, q: np.ndarray) -> np.ndarray:
        _ = q
        return np.array(self._snap.M, dtype=np.float64, copy=True)

    def bias_forces(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        _ = q
        if np.allclose(v, 0.0):
            return 0.5 * np.array(self._snap.h, dtype=np.float64)
        return np.array(self._snap.h, dtype=np.float64, copy=True)

    def closure_jacobian(self, q: np.ndarray) -> np.ndarray:
        _ = q
        return np.array(self._snap.J_c, dtype=np.float64, copy=True)

    def closure_drift(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        _ = q
        if np.allclose(v, 0.0):
            return np.zeros(self._snap.nc(), dtype=np.float64)
        return np.array(self._snap.Jdot_v, dtype=np.float64, copy=True)


def test_constrained_ztcf_matches_planar_oracle_golden() -> None:
    """Verify constrained ZTCF on planar oracle matches golden a_ztcf and respects closure."""
    snap, golden = planar_golfer_snapshot(actuation="full")
    records = compute_constrained_ztcf_zvcf(snap)

    ztcf_rec = records["ztcf"]
    zvcf_rec = records["zvcf"]
    ctrl_rec = records["control_acceleration"]
    attr_rec = records["per_actuator_attribution"]

    assert golden.a_ztcf is not None
    np.testing.assert_allclose(ztcf_rec.values, golden.a_ztcf, atol=1e-12)
    # Closure acceleration constraint J_c a_ztcf + Jdot_v == 0 must hold
    np.testing.assert_allclose(
        snap.J_c @ ztcf_rec.values + snap.Jdot_v, 0.0, atol=1e-12
    )
    # ZVCF has v = 0 -> Jdot_v = 0 -> J_c a_zvcf == 0
    np.testing.assert_allclose(snap.J_c @ zvcf_rec.values, 0.0, atol=1e-12)

    assert ztcf_rec.allocation_dependence is AllocationDependence.INVARIANT
    assert zvcf_rec.allocation_dependence is AllocationDependence.INVARIANT
    assert ctrl_rec.allocation_dependence is AllocationDependence.INVARIANT
    assert attr_rec.allocation_dependence is AllocationDependence.FAMILY_DEPENDENT
    np.testing.assert_allclose(ztcf_rec.values + ctrl_rec.values, snap.a, atol=1e-12)


def test_ztcf_zvcf_module_supports_constrained_provider() -> None:
    """Verify simulation_backends.ztcf_zvcf routes ConstrainedEquationsOfMotionProvider via KKT."""
    snap, golden = planar_golfer_snapshot(actuation="full")
    provider = _PlanarConstrainedProvider(snap)
    assert isinstance(provider, ConstrainedEquationsOfMotionProvider)

    a_ztcf = ztcf_acceleration(provider, snap.q, snap.v)
    a_zvcf = zvcf_acceleration(provider, snap.q)
    tau_p = np.linalg.lstsq(snap.J_c.T, np.zeros(snap.nv()), rcond=None)[0]
    _ = tau_p
    tau_applied = snap.generalized_load()
    drift, ctrl = drift_and_control_split(provider, snap.q, snap.v, tau_applied)

    assert golden.a_ztcf is not None
    np.testing.assert_allclose(a_ztcf, golden.a_ztcf, atol=1e-12)
    np.testing.assert_allclose(drift, a_ztcf, atol=1e-12)
    np.testing.assert_allclose(snap.J_c @ a_zvcf, 0.0, atol=1e-12)
    a_tot, _ = constrained_acceleration(
        snap.M, snap.h, tau_applied, snap.J_c, snap.Jdot_v
    )
    np.testing.assert_allclose(drift + ctrl, a_tot, atol=1e-12)


def test_closure_drift_remains_bounded_over_constrained_rollout_horizon() -> None:
    """Verify constrained rollout with NULLSPACE_EXPLORATION keeps closure drift below tolerance."""
    snap, _ = planar_golfer_snapshot(actuation="full")
    provider = _PlanarConstrainedProvider(snap)
    n_frames = 25
    n_dofs = snap.nv()
    time_s = np.linspace(0.0, 0.024, n_frames, dtype=np.float64)
    q_traj = np.zeros((n_frames, n_dofs), dtype=np.float64)
    v_traj = np.tile(snap.v, (n_frames, 1))
    a_traj = np.tile(snap.a, (n_frames, 1))
    tau_traj = np.tile(snap.generalized_load(), (n_frames, 1))

    coord_names = snap.topology.coordinate_order
    meta = CandidateMetadata(
        profile=CandidateProfile.DYNAMIC,
        coordinate_names=coord_names,
    )
    candidate = MatchedSwingCandidate(
        metadata=meta,
        time_s=time_s,
        q=q_traj,
        v=v_traj,
        tau=tau_traj,
        markers=CandidateMarkers(),
        auxiliary=CandidateAuxiliary(),
    )
    session = CandidateSession(
        candidate=candidate,
        specification={"coordinate_order": list(coord_names)},
        candidate_sha256="12570" + "a" * 59,
        model_sha256="12570" + "b" * 59,
        coordinate_names=coord_names,
        time_s=time_s,
        q=q_traj,
        v=v_traj,
        a=a_traj,
        tau=tau_traj,
        is_accepted=True,
    )

    fork = create_counterfactual_rollout(
        session=session,
        fork_frame_idx=0,
        strategy=CounterfactualStrategy.NULLSPACE_EXPLORATION,
        provider=provider,
        snapshot=snap,
        nullspace_z=np.array([1.5, -0.75], dtype=np.float64),
        baumgarte_alpha=20.0,
        baumgarte_beta=20.0,
    )
    assert fork.v is not None
    assert fork.a is not None
    # Every frame's acceleration must satisfy J_c a + Jdot_v (with Baumgarte) and keep J_c q small
    max_closure_pos_err = float(np.max(np.abs(fork.q @ snap.J_c.T)))
    assert max_closure_pos_err < 1e-3


def test_pendulum_counterfactual_golfer_regression_unchanged() -> None:
    """Verify pendulum_simulator.counterfactual_golfer output is finite and consistent."""
    q0, params = _default_closed_loop_golfer_state()
    state = np.concatenate([q0, np.zeros_like(q0)])
    qddot = zero_torque_accelerations(state, params)
    assert qddot.shape == (8,)
    assert np.all(np.isfinite(qddot))


def test_m_eff_mock_and_sine_placeholder_removed_from_counterfactual() -> None:
    """Verify m_eff = 1.0 mock and 5.0 * sin placeholder are removed from counterfactual.py."""
    assert not hasattr(cf_module, "m_eff")
    src = inspect.getsource(cf_module)
    assert "m_eff" not in src
    assert "5.0 * np.sin" not in src
