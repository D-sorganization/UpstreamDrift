"""Unit tests for candidate force telemetry and counterfactual API routes (MV-06, #10482)."""

from __future__ import annotations

from typing import Any
from fastapi import FastAPI
from fastapi.testclient import TestClient
import numpy as np
import pytest

from src.api.dependencies import get_simulation_service
from src.api.routes.analysis import router
from src.shared.python.motion_matching.candidate import (
    CandidateAuxiliary,
    CandidateMarkers,
    CandidateMetadata,
    CandidateProfile,
    MatchedSwingCandidate,
)
from src.shared.python.motion_matching.candidate_session import CandidateSession

pytestmark = pytest.mark.unit


def _create_candidate_session(
    is_dynamic: bool = True,
    has_forces: bool = True,
    is_accepted: bool = True,
) -> CandidateSession:
    time_s = np.array([0.0, 0.01, 0.02, 0.03, 0.04], dtype=np.float64)
    q = np.zeros((5, 2), dtype=np.float64)
    v = np.zeros((5, 2), dtype=np.float64) if is_dynamic else None
    tau = np.ones((5, 2), dtype=np.float64) * 10.0 if is_dynamic else None
    ext_forces = (
        np.array(
            [
                [1.0, 2.0, 700.0, 0.1, 0.2, 0.3],
                [1.5, 2.5, 710.0, 0.1, 0.2, 0.3],
                [2.0, 3.0, 720.0, 0.1, 0.2, 0.3],
                [2.5, 3.5, 730.0, 0.1, 0.2, 0.3],
                [3.0, 4.0, 740.0, 0.1, 0.2, 0.3],
            ],
            dtype=np.float64,
        )
        if has_forces
        else None
    )
    coord_names = ("shoulder_elevation", "elbow_flexion")
    meta = CandidateMetadata(
        profile=CandidateProfile.DYNAMIC if is_dynamic else CandidateProfile.KINEMATIC,
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
        candidate_sha256="cand_sha256_mock",
        model_sha256="model_sha256_mock",
        coordinate_names=coord_names,
        time_s=time_s,
        q=q,
        v=v,
        tau=tau,
        external_forces=ext_forces,
        is_accepted=is_accepted,
    )


class MockSimulationService:
    def __init__(self, session: CandidateSession | None = None) -> None:
        self.active_candidate_session = session

    def get_candidate_forces(self) -> dict[str, Any]:
        if self.active_candidate_session is None:
            raise ValueError("No active candidate session loaded")
        session = self.active_candidate_session
        if not session.supports_forces:
            return {
                "session_available": True,
                "supports_forces": False,
                "frame_count": session.frame_count,
                "time_s": session.time_s.tolist(),
                "coordinate_names": list(session.coordinate_names),
                "torques": None,
                "external_forces": None,
                "center_of_pressure": None,
            }
        return {
            "session_available": True,
            "supports_forces": True,
            "frame_count": session.frame_count,
            "time_s": session.time_s.tolist(),
            "coordinate_names": list(session.coordinate_names),
            "torques": session.tau.tolist() if session.tau is not None else None,
            "external_forces": (
                session.external_forces.tolist()
                if session.external_forces is not None
                else None
            ),
            "center_of_pressure": [
                session.get_center_of_pressure(i) for i in range(session.frame_count)
            ],
        }

    def run_candidate_counterfactual(
        self,
        fork_frame_idx: int,
        strategy: str = "zero_trail_arm_torque",
        duration_frames: int | None = None,
    ) -> dict[str, Any]:
        if self.active_candidate_session is None:
            raise ValueError("No active candidate session loaded")
        from src.shared.python.motion_matching.counterfactual import (
            CounterfactualStrategy,
            create_counterfactual_rollout,
        )

        fork = create_counterfactual_rollout(
            session=self.active_candidate_session,
            fork_frame_idx=fork_frame_idx,
            strategy=CounterfactualStrategy(strategy),
            duration_frames=duration_frames,
        )
        return {
            "fork_id": fork.fork_id,
            "baseline_candidate_sha256": fork.baseline_candidate_sha256,
            "strategy": fork.strategy.value,
            "fork_frame_idx": fork.fork_frame_idx,
            "divergence_rms": fork.divergence_rms,
            "is_accepted": fork.is_accepted,
        }


@pytest.fixture
def app() -> FastAPI:
    test_app = FastAPI()
    test_app.include_router(router)
    return test_app


def test_get_candidate_forces_no_session(app: FastAPI) -> None:
    """GET /analysis/candidate/forces fails closed with 409 when no session is loaded."""
    mock_service = MockSimulationService(session=None)
    app.dependency_overrides[get_simulation_service] = lambda: mock_service
    client = TestClient(app)

    response = client.get("/analysis/candidate/forces")
    assert response.status_code == 409
    assert "No active candidate session" in response.json()["detail"]


def test_get_candidate_forces_success(app: FastAPI) -> None:
    """GET /analysis/candidate/forces returns 200 with forces, torques, and CoP."""
    session = _create_candidate_session(is_dynamic=True, has_forces=True)
    mock_service = MockSimulationService(session=session)
    app.dependency_overrides[get_simulation_service] = lambda: mock_service
    client = TestClient(app)

    response = client.get("/analysis/candidate/forces")
    assert response.status_code == 200
    data = response.json()
    assert data["session_available"] is True
    assert data["supports_forces"] is True
    assert data["frame_count"] == 5
    assert len(data["center_of_pressure"]) == 5
    assert data["torques"] is not None


def test_post_candidate_counterfactual_no_session(app: FastAPI) -> None:
    """POST /analysis/candidate/counterfactual fails closed with 409 when no session is loaded."""
    mock_service = MockSimulationService(session=None)
    app.dependency_overrides[get_simulation_service] = lambda: mock_service
    client = TestClient(app)

    response = client.post(
        "/analysis/candidate/counterfactual",
        json={"fork_frame_idx": 1, "strategy": "zero_trail_arm_torque"},
    )
    assert response.status_code == 409
    assert "No active candidate session" in response.json()["detail"]


def test_post_candidate_counterfactual_unsupported_session(app: FastAPI) -> None:
    """POST /analysis/candidate/counterfactual fails closed with 409 on kinematic session."""
    session = _create_candidate_session(is_dynamic=False, has_forces=False)
    mock_service = MockSimulationService(session=session)
    app.dependency_overrides[get_simulation_service] = lambda: mock_service
    client = TestClient(app)

    response = client.post(
        "/analysis/candidate/counterfactual",
        json={"fork_frame_idx": 1, "strategy": "zero_trail_arm_torque"},
    )
    assert response.status_code == 409
    assert "does not support counterfactual rollouts" in response.json()["detail"]


def test_post_candidate_counterfactual_success(app: FastAPI) -> None:
    """POST /analysis/candidate/counterfactual returns 200 and rollout fork data."""
    session = _create_candidate_session(is_dynamic=True, has_forces=True)
    mock_service = MockSimulationService(session=session)
    app.dependency_overrides[get_simulation_service] = lambda: mock_service
    client = TestClient(app)

    response = client.post(
        "/analysis/candidate/counterfactual",
        json={
            "fork_frame_idx": 1,
            "strategy": "clamped_actuator_torque",
            "duration_frames": 3,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "fork_id" in data
    assert data["strategy"] == "clamped_actuator_torque"
    assert data["fork_frame_idx"] == 1
    assert "divergence_rms" in data
