"""Unit tests for force/torque and counterfactual accessors on CandidateSession (MV-06, #10482)."""

from __future__ import annotations

from typing import Any
import numpy as np
import pytest

from src.shared.python.motion_matching.candidate import (
    CandidateAuxiliary,
    CandidateMarkers,
    CandidateMetadata,
    CandidateProfile,
    MatchedSwingCandidate,
)
from src.shared.python.motion_matching.candidate_session import CandidateSession
from src.shared.python.motion_matching.counterfactual import (
    CounterfactualFork,
    CounterfactualStrategy,
)
from src.shared.python.motion_matching.force_torque import SpatialWrench

pytestmark = pytest.mark.unit


def _build_test_session(
    *,
    has_forces: bool = False,
    has_tau: bool = False,
    has_residuals: bool = False,
    fz_value: float = 750.0,
) -> CandidateSession:
    time_s = np.array([0.0, 0.01, 0.02, 0.03], dtype=np.float64)
    q = np.zeros((4, 2), dtype=np.float64)
    v = np.zeros((4, 2), dtype=np.float64)
    coord_names = ("hip_flexion", "knee_angle")

    tau = (
        np.array([[10.0, -5.0], [12.0, -4.0], [14.0, -3.0], [16.0, -2.0]])
        if has_tau
        else None
    )
    ext_forces = (
        np.array(
            [
                [10.0, 5.0, fz_value, 2.0, 1.0, 0.5],
                [12.0, 4.0, fz_value, 2.1, 1.1, 0.6],
                [14.0, 3.0, fz_value, 2.2, 1.2, 0.7],
                [16.0, 2.0, fz_value, 2.3, 1.3, 0.8],
            ],
            dtype=np.float64,
        )
        if has_forces
        else None
    )
    diagnostics: dict[str, Any] = {}
    if has_residuals:
        diagnostics["closure_residuals"] = [0.001, 0.002, 0.0015, 0.0012]

    meta = CandidateMetadata(
        profile=CandidateProfile.DYNAMIC if has_tau else CandidateProfile.KINEMATIC,
        coordinate_names=coord_names,
    )
    cand = MatchedSwingCandidate(
        metadata=meta,
        time_s=time_s,
        q=q,
        v=v,
        tau=tau,
        markers=CandidateMarkers(),
        auxiliary=CandidateAuxiliary(external_forces=ext_forces),
    )
    return CandidateSession(
        candidate=cand,
        specification={"coordinate_order": list(coord_names)},
        candidate_sha256="session_test_candidate_sha256",
        model_sha256="session_test_model_sha256",
        coordinate_names=coord_names,
        time_s=time_s,
        q=q,
        v=v,
        tau=tau,
        external_forces=ext_forces,
        diagnostics=diagnostics,
        is_accepted=True,
    )


def test_get_wrench_at_absent_channel() -> None:
    """Session without forces channel must return None, never fabricating zeros."""
    session = _build_test_session(has_forces=False)
    assert session.get_wrench_at(0) is None
    assert session.get_center_of_pressure(0) is None


def test_get_wrench_at_present_channel() -> None:
    """Session with forces channel returns properly configured SpatialWrench."""
    session = _build_test_session(has_forces=True, fz_value=800.0)
    wrench = session.get_wrench_at(1, contact_name="lead_foot")
    assert wrench is not None
    assert isinstance(wrench, SpatialWrench)
    assert wrench.application_frame == "lead_foot"
    assert wrench.force_n == (12.0, 4.0, 800.0)
    assert wrench.torque_nm == (2.1, 1.1, 0.6)


def test_get_wrench_at_index_error() -> None:
    """Invalid frame index raises IndexError."""
    session = _build_test_session(has_forces=True)
    with pytest.raises(IndexError):
        session.get_wrench_at(99)
    with pytest.raises(IndexError):
        session.get_wrench_at(-1)


def test_get_center_of_pressure_thresholding() -> None:
    """CoP returns coordinates when Fz > threshold, and None when Fz <= threshold."""
    session_high = _build_test_session(has_forces=True, fz_value=700.0)
    cop = session_high.get_center_of_pressure(0, fz_threshold=5.0)
    assert cop is not None
    assert isinstance(cop, tuple)
    assert len(cop) == 2

    session_low = _build_test_session(has_forces=True, fz_value=2.0)
    assert session_low.get_center_of_pressure(0, fz_threshold=5.0) is None


def test_get_joint_torques_at() -> None:
    """Joint torques return dict of coordinate names to torque or None when absent."""
    session_kin = _build_test_session(has_tau=False)
    assert session_kin.get_joint_torques_at(0) is None

    session_dyn = _build_test_session(has_tau=True)
    torques = session_dyn.get_joint_torques_at(1)
    assert torques == {"hip_flexion": 12.0, "knee_angle": -4.0}

    with pytest.raises(IndexError):
        session_dyn.get_joint_torques_at(10)


def test_get_closure_residual_at() -> None:
    """Kinematic closure residual returns value when recorded, or None if absent."""
    session_no_res = _build_test_session(has_residuals=False)
    assert session_no_res.get_closure_residual_at(0) is None

    session_res = _build_test_session(has_residuals=True)
    assert session_res.get_closure_residual_at(2) == 0.0015


def test_create_counterfactual_fork_delegation() -> None:
    """Session method create_counterfactual_fork generates a valid fork."""
    session = _build_test_session(has_tau=True, has_forces=True)
    fork = session.create_counterfactual_fork(
        fork_frame_idx=1,
        strategy=CounterfactualStrategy.CLAMPED_ACTUATOR_TORQUE,
        duration_frames=2,
    )
    assert isinstance(fork, CounterfactualFork)
    assert fork.fork_frame_idx == 1
    assert fork.frame_count == 2
