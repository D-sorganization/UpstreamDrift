"""NSX-30: a candidate session's own model specification supplies the dynamics."""

from __future__ import annotations

import json
from pathlib import Path

from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.dependencies import get_simulation_service
from src.api.routes.analysis import router
from src.api.services.simulation_service import SimulationService
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
from src.shared.python.motion_matching.counterfactual_provider import (
    resolve_session_dynamics,
)

pytestmark = pytest.mark.unit

EVIDENCE = (
    Path(__file__).resolve().parents[3]
    / "docs/development/full_body_models/evidence/ground_support"
)
SPEC_PATH = EVIDENCE / "full_body_spec_hipcal_scaled.json"
FRAMES = 4


def _session(spec: dict, names: tuple[str, ...], q: np.ndarray) -> CandidateSession:
    n = len(names)
    t = np.arange(FRAMES) * 1e-3
    qs = np.tile(q, (FRAMES, 1))
    v = np.zeros((FRAMES, n))
    tau = np.zeros((FRAMES, n))
    meta = CandidateMetadata(profile=CandidateProfile.DYNAMIC, coordinate_names=names)
    cand = MatchedSwingCandidate(
        metadata=meta,
        time_s=t,
        q=qs,
        v=v,
        tau=tau,
        markers=CandidateMarkers(),
        auxiliary=CandidateAuxiliary(),
    )
    return CandidateSession(
        candidate=cand,
        specification=spec,
        candidate_sha256="nsx30_sha",
        model_sha256="nsx30_model_" + str(len(names)),
        coordinate_names=names,
        time_s=t,
        q=qs,
        v=v,
        tau=tau,
        is_accepted=True,
    )


@pytest.fixture(scope="module")
def anthro():
    pytest.importorskip("pinocchio")
    from src.shared.python.motion_matching.same_input import (
        VectorPlant,
        project_to_closure,
    )

    spec = json.loads(SPEC_PATH.read_bytes())
    plant = VectorPlant("pinocchio", SPEC_PATH.read_bytes())
    with np.load(EVIDENCE / "dynamics_record.npz") as rec:
        q0, v0 = rec["q"][400], rec["v"][400]
    proj = project_to_closure(plant, q0, v0)
    names = tuple(spec["coordinate_order"])
    qn = plant._named(proj.q)
    return spec, names, np.array([qn[n] for n in names])


@pytest.mark.requires_pinocchio
def test_session_spec_gives_a_constrained_provider(anthro) -> None:
    spec, names, q = anthro
    dyn = resolve_session_dynamics(_session(spec, names, q))
    assert hasattr(dyn.provider, "closure_jacobian")


@pytest.mark.requires_pinocchio
def test_rollout_without_explicit_dynamics_uses_the_real_model(anthro) -> None:
    pin = pytest.importorskip("pinocchio")
    spec, names, q = anthro
    session = _session(spec, names, q)
    fork = create_counterfactual_rollout(
        session=session,
        fork_frame_idx=0,
        strategy=CounterfactualStrategy.CUSTOM,
        control_override=np.zeros((FRAMES, len(names))),
    )
    from src.engines.physics_engines.pinocchio.python.force_adapter import (
        PinocchioForceAdapter,
    )

    adapter = PinocchioForceAdapter(SPEC_PATH.read_bytes())
    plant = adapter._plant
    native = list(adapter.coordinate_order)
    q_native = np.array([q[names.index(n)] for n in native])
    model, cons = plant.model, plant.constraints
    data = model.createData()
    cdata = [c.createData() for c in cons]
    pin.initConstraintDynamics(model, data, cons, cdata)
    zeros = np.zeros(len(names))
    ref_native = np.array(
        pin.constraintDynamics(model, data, q_native, zeros, zeros, cons, cdata)
    )
    ref = np.array([ref_native[native.index(n)] for n in names])
    assert native != list(names)  # the reordering path is really exercised
    assert np.max(np.abs(fork.a[0] - ref)) / max(1.0, np.max(np.abs(ref))) < 1e-9
    assert "closure_velocity_residual_max" in fork.constraint_status


@pytest.mark.requires_pinocchio
def test_route_returns_200_with_real_dynamics(anthro) -> None:
    spec, names, q = anthro
    service = SimulationService(engine_manager=MagicMock())
    service.set_active_candidate_session(_session(spec, names, q))
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_simulation_service] = lambda: service
    resp = TestClient(app).post(
        "/analysis/candidate/counterfactual",
        json={"fork_frame_idx": 0, "strategy": "clamped_actuator_torque"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["frame_count"] == FRAMES


def test_route_returns_400_naming_the_missing_model() -> None:
    names = ("a", "b")
    session = _session({"coordinate_order": list(names)}, names, np.zeros(2))
    service = SimulationService(engine_manager=MagicMock())
    service.set_active_candidate_session(session)
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_simulation_service] = lambda: service
    resp = TestClient(app).post(
        "/analysis/candidate/counterfactual",
        json={"fork_frame_idx": 0, "strategy": "clamped_actuator_torque"},
    )
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert "model specification" in detail and "bodies" in detail


@pytest.mark.requires_pinocchio
def test_coordinate_order_mismatch_is_rejected(anthro) -> None:
    spec, names, q = anthro
    session = _session(spec, tuple(reversed(names)), q)
    with pytest.raises(ValueError, match="coordinate order"):
        resolve_session_dynamics(session)
