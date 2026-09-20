"""Unit tests for counterfactual semantics and acceleration decomposition (MV-06, #10482)."""

from __future__ import annotations

import hashlib
from typing import Any
import numpy as np
import pytest

from src.shared.python.contracts import ContractViolationError
from src.shared.python.motion_matching.candidate import (
    CandidateAuxiliary,
    CandidateMarkers,
    CandidateMetadata,
    CandidateProfile,
    MatchedSwingCandidate,
)
from src.shared.python.motion_matching.candidate_session import CandidateSession
from src.shared.python.motion_matching.counterfactual import (
    AccelerationDecomposition,
    CounterfactualFork,
    CounterfactualStrategy,
    create_counterfactual_rollout,
)

pytestmark = pytest.mark.unit


def _create_mock_session(
    n_frames: int = 100,
    n_dofs: int = 4,
    profile: CandidateProfile = CandidateProfile.DYNAMIC,
    has_tau: bool = True,
    is_accepted: bool = True,
) -> CandidateSession:
    """Create a mock CandidateSession for counterfactual testing."""
    time_s = np.linspace(0.0, 1.0, n_frames, dtype=np.float64)
    q = np.sin(np.outer(time_s, np.arange(1, n_dofs + 1)))
    v = np.cos(np.outer(time_s, np.arange(1, n_dofs + 1)))
    a = -np.sin(np.outer(time_s, np.arange(1, n_dofs + 1)))
    tau = 10.0 * np.cos(np.outer(time_s, np.arange(1, n_dofs + 1))) if has_tau else None

    coord_names = (
        "pelvis_tilt",
        "torso_rotation",
        "right_shoulder_pitch",
        "right_elbow_flexion",
    )
    meta = CandidateMetadata(
        profile=profile,
        coordinate_names=coord_names,
    )
    candidate = MatchedSwingCandidate(
        metadata=meta,
        time_s=time_s,
        q=q,
        v=v,
        tau=tau,
        markers=CandidateMarkers(),
        auxiliary=CandidateAuxiliary(),
    )
    spec: dict[str, Any] = {
        "coordinate_order": list(coord_names),
        "joint_limits": {
            "pelvis_tilt": [-2.0, 2.0],
            "torso_rotation": [-3.14, 3.14],
        },
    }
    return CandidateSession(
        candidate=candidate,
        specification=spec,
        candidate_sha256="test_candidate_sha256",
        model_sha256="test_model_sha256",
        coordinate_names=coord_names,
        time_s=time_s,
        q=q,
        v=v,
        a=a,
        tau=tau,
        is_accepted=is_accepted,
    )


def test_acceleration_decomposition_invariants() -> None:
    """Verify exact relationship a_tot = ztcf + a_ctrl and zvcf = a_grav + a_ctrl."""
    a_grav = np.array([0.0, 0.0, -9.81], dtype=np.float64)
    a_drift = np.array([1.2, -0.5, 0.3], dtype=np.float64)
    a_ctrl = np.array([5.0, -3.0, 10.0], dtype=np.float64)

    decomp = AccelerationDecomposition(a_grav=a_grav, a_drift=a_drift, a_ctrl=a_ctrl)

    np.testing.assert_allclose(decomp.ztcf, a_grav + a_drift)
    np.testing.assert_allclose(decomp.zvcf, a_grav + a_ctrl)
    np.testing.assert_allclose(decomp.total_accel, a_grav + a_drift + a_ctrl)
    assert decomp.verify_decomposition(a_grav + a_drift + a_ctrl)
    assert not decomp.verify_decomposition(a_grav + a_drift)


def test_acceleration_decomposition_rejects_invalid() -> None:
    """Ensure non-finite values or shape mismatches trigger ValueError."""
    with pytest.raises(ValueError, match="finite"):
        AccelerationDecomposition(
            a_grav=np.array([np.nan, 0.0]),
            a_drift=np.array([0.0, 1.0]),
            a_ctrl=np.array([1.0, 2.0]),
        )

    with pytest.raises(ValueError, match="shape mismatch"):
        AccelerationDecomposition(
            a_grav=np.array([0.0, 0.0]),
            a_drift=np.array([0.0, 1.0, 2.0]),
            a_ctrl=np.array([1.0, 2.0]),
        )


def test_acceleration_decomposition_immutability() -> None:
    """Ensure components in AccelerationDecomposition cannot be modified in place."""
    decomp = AccelerationDecomposition(
        a_grav=np.array([0.0, -9.81]),
        a_drift=np.array([0.1, 0.2]),
        a_ctrl=np.array([1.0, 2.0]),
    )
    with pytest.raises(ValueError):
        decomp.a_grav[0] = 5.0  # type: ignore[index]


def test_counterfactual_rollout_baseline_immutability() -> None:
    """CRITICAL: Baseline candidate session arrays must remain byte-identical after rollout."""
    session = _create_mock_session()
    q_digest_before = hashlib.sha256(session.q.tobytes()).hexdigest()
    v_digest_before = hashlib.sha256(session.v.tobytes()).hexdigest()  # type: ignore[union-attr]
    tau_digest_before = hashlib.sha256(session.tau.tobytes()).hexdigest()  # type: ignore[union-attr]

    fork = create_counterfactual_rollout(
        session=session,
        fork_frame_idx=20,
        strategy=CounterfactualStrategy.ZERO_TRAIL_ARM_TORQUE,
        duration_frames=30,
    )

    q_digest_after = hashlib.sha256(session.q.tobytes()).hexdigest()
    v_digest_after = hashlib.sha256(session.v.tobytes()).hexdigest()  # type: ignore[union-attr]
    tau_digest_after = hashlib.sha256(session.tau.tobytes()).hexdigest()  # type: ignore[union-attr]

    assert q_digest_before == q_digest_after
    assert v_digest_before == v_digest_after
    assert tau_digest_before == tau_digest_after
    assert isinstance(fork, CounterfactualFork)
    assert fork.fork_frame_idx == 20
    assert fork.frame_count == 30


def test_counterfactual_rollout_zero_trail_arm_divergence() -> None:
    """Zeroing trail arm torque should diverge from baseline trajectory."""
    session = _create_mock_session()
    fork = create_counterfactual_rollout(
        session=session,
        fork_frame_idx=10,
        strategy=CounterfactualStrategy.ZERO_TRAIL_ARM_TORQUE,
        duration_frames=40,
    )

    assert fork.divergence_rms > 0.0
    # Coordinates for right_shoulder_pitch and right_elbow_flexion should have altered torques = 0
    assert fork.altered_tau is not None
    assert np.all(fork.altered_tau[:, 2] == 0.0)
    assert np.all(fork.altered_tau[:, 3] == 0.0)
    # Non-trail coordinates should retain baseline torques
    np.testing.assert_allclose(
        fork.altered_tau[:, 0],
        session.tau[10:50, 0],  # type: ignore[index]
    )


def test_counterfactual_rollout_clamped_torque() -> None:
    """Clamped torque strategy limits peak efforts and diverges."""
    session = _create_mock_session()
    clamp_limit = 2.0
    fork = create_counterfactual_rollout(
        session=session,
        fork_frame_idx=5,
        strategy=CounterfactualStrategy.CLAMPED_ACTUATOR_TORQUE,
        duration_frames=20,
        clamp_limits=(-clamp_limit, clamp_limit),
    )
    assert fork.divergence_rms > 0.0
    assert fork.altered_tau is not None
    assert np.all(fork.altered_tau <= clamp_limit + 1e-9)
    assert np.all(fork.altered_tau >= -clamp_limit - 1e-9)


def test_counterfactual_rollout_unsupported_session_rejected() -> None:
    """Kinematic or rejected sessions without forces must reject rollout requests."""
    kinematic_session = _create_mock_session(
        profile=CandidateProfile.KINEMATIC, has_tau=False
    )
    with pytest.raises(ContractViolationError):
        create_counterfactual_rollout(
            session=kinematic_session,
            fork_frame_idx=10,
        )

    rejected_session = _create_mock_session(is_accepted=False)
    with pytest.raises(ContractViolationError):
        create_counterfactual_rollout(
            session=rejected_session,
            fork_frame_idx=10,
        )
